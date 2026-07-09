/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

use std::sync::Arc;

use carbide_uuid::machine::MachineId;
use nv_redfish::core::{Bmc, ToSnakeCase};
use nv_redfish::{Resource, ServiceRoot};

use crate::HealthError;
use crate::api_client::ApiClientWrapper;
use crate::collectors::runtime::{IterationResult, PeriodicCollector};
use crate::endpoint::{BmcEndpoint, EndpointMetadata};
use crate::sink::{
    Classification, CollectorEvent, DataSink, EventContext, HealthReport, HealthReportAlert,
    HealthReportSuccess, HealthReportTarget, Probe, ReportSource,
};

/// Resolved expected-GPU state for the endpoint's assigned SKU.
#[derive(Clone, Debug)]
enum Expected {
    /// No SKU assigned yet — nothing to validate.
    NoSku,
    /// The assigned SKU id, but its manifest could not be found.
    SkuMissing(String),
    /// Expected GPU count from the SKU manifest.
    Count(u32),
}

pub struct GpuInventoryCollectorConfig {
    pub data_sink: Option<Arc<dyn DataSink>>,
    pub api_client: Arc<ApiClientWrapper>,
}

pub struct GpuInventoryCollector<B: Bmc> {
    endpoint: Arc<BmcEndpoint>,
    bmc: Arc<B>,
    event_context: EventContext,
    data_sink: Option<Arc<dyn DataSink>>,
    api_client: Arc<ApiClientWrapper>,
    /// Machine id for this endpoint. The assigned SKU is re-read live each iteration
    /// (not cached) so SKU assignments/changes after start are honored.
    machine_id: Option<MachineId>,
}

impl<B: Bmc + 'static> GpuInventoryCollector<B> {
    /// Resolve the expected GPU count from the machine's currently-assigned SKU.
    /// Re-reads the SKU live every call (no caching) so assignments/changes after
    /// the collector starts are picked up.
    async fn resolve_expected(&self) -> Result<Expected, HealthError> {
        let Some(machine_id) = self.machine_id else {
            return Ok(Expected::NoSku);
        };
        let Some(sku_id) = self.api_client.machine_hw_sku(machine_id).await? else {
            return Ok(Expected::NoSku);
        };
        let skus = self
            .api_client
            .find_skus_by_ids(vec![sku_id.clone()])
            .await?;
        Ok(match skus.into_iter().next() {
            None => Expected::SkuMissing(sku_id),
            Some(sku) => Expected::Count(
                sku.components
                    .map(|c| c.gpus.iter().map(|g| g.count).sum())
                    .unwrap_or(0),
            ),
        })
    }

    /// Count GPUs out-of-band via Redfish.
    ///
    /// GPUs surface in two ways depending on how they attach, so we count both and
    /// take the max — no vendor table, works across Dell / Lenovo / Supermicro /
    /// NVIDIA regardless of GPU form:
    /// - **SXM / HGX baseboards** (H100 SXM, GB200, GB300, GH200) → one
    ///   `HGX_GPU_*` chassis per GPU (`count_hgx_gpu_chassis`).
    /// - **PCIe cards** (L40 / L40S, H100 PCIe) → Redfish `Processors` with
    ///   `ProcessorType == GPU` (`count_gpu_processors`).
    ///
    /// Some platforms (e.g. GB200) expose the *same* GPUs as both chassis and
    /// processors, so we take the max rather than the sum to avoid double-counting.
    /// `max` is deliberate: `min` would false-alert on platforms that populate only
    /// one view, and `sum` would double-count dual-view platforms.
    ///
    /// Limitations:
    /// - Trusts the BMC's Redfish inventory: if a failed GPU still appears (stale
    ///   inventory), the count won't drop and no shortage alert fires. The SEL
    ///   GPU-fault processor is the complementary signal for present-but-faulty GPUs.
    /// - GPUs exposed ONLY under `PCIeDevices` (neither HGX chassis nor a GPU
    ///   Processor) are not yet counted. nv-redfish exposes no PCI device-class, so
    ///   such a path would have to match on model/name — add it once a platform that
    ///   needs it is confirmed via Redfish.
    async fn count_gpus(&self) -> Result<u32, HealthError> {
        let root = ServiceRoot::new(self.bmc.clone()).await?;
        let chassis_gpus = Self::count_hgx_gpu_chassis(&root).await?;
        let processor_gpus = Self::count_gpu_processors(&root).await?;
        Ok(chassis_gpus.max(processor_gpus))
    }

    /// HGX path: each GPU module is one `HGX_GPU_*` chassis
    /// (`HGX_GPU_SXM_*` on Viking/H100, `HGX_GPU_*` on GH200/GB200/GB300).
    /// NVSwitch trays excluded.
    async fn count_hgx_gpu_chassis(root: &ServiceRoot<B>) -> Result<u32, HealthError> {
        let Some(chassis_list) = root.chassis().await? else {
            return Ok(0);
        };
        let mut count = 0u32;
        for c in chassis_list.members().await? {
            let id = c.id().into_inner();
            if id.starts_with("HGX_GPU_") && !id.contains("NVSwitch") {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Standard-server path: count GPUs exposed as Redfish Processors
    /// (`ProcessorType == GPU`). Vendor-neutral across Dell / Lenovo / HPE /
    /// Supermicro, and model-agnostic — it counts every GPU (H100 PCIe, L40 /
    /// L40S, A100, …), which is what "fewer GPUs than expected" needs.
    ///
    /// Caveat: this relies on the BMC exposing GPUs as `Processors`. A BMC that
    /// only lists GPUs under `PCIeDevices` would count 0 here; add a PCIe fallback
    /// if such firmware shows up in the fleet. Confirmed on hardware that the known
    /// PCIe fleet platforms do expose GPUs this way — e.g. a Lenovo ThinkSystem
    /// SR670 V2 with 8x NVIDIA L40 reports 8 `ProcessorType=GPU` processors via XCC.
    async fn count_gpu_processors(root: &ServiceRoot<B>) -> Result<u32, HealthError> {
        let Some(systems) = root.systems().await? else {
            return Ok(0);
        };
        let mut count = 0u32;
        for system in systems.members().await? {
            let processors = system.processors().await?.unwrap_or_default();
            for processor in processors {
                let is_gpu = processor
                    .raw()
                    .processor_type
                    .flatten()
                    .is_some_and(|pt| pt.to_snake_case() == "gpu");
                if is_gpu {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    fn emit_alert(&self, message: String) {
        tracing::warn!(bmc = %self.endpoint.addr.mac, %message, "GPU inventory alert");
        let report = HealthReport {
            source: ReportSource::GpuInventory,
            target: Some(HealthReportTarget::Machine),
            observed_at: Some(chrono::Utc::now()),
            successes: Vec::new(),
            alerts: vec![HealthReportAlert {
                probe_id: Probe::GpuInventory,
                target: None,
                message,
                classifications: vec![Classification::PreventAllocations],
            }],
        };
        self.emit(report);
    }

    fn emit(&self, report: HealthReport) {
        if let Some(sink) = &self.data_sink {
            sink.handle_event(
                &self.event_context,
                &CollectorEvent::HealthReport(Arc::new(report)),
            );
        }
    }
}

/// Build the health report for a GPU-count comparison — the core of issue #301:
/// alert when the BMC sees fewer GPUs than the SKU expects, success otherwise.
fn gpu_count_report(expected: u32, actual: u32) -> HealthReport {
    if actual < expected {
        HealthReport {
            source: ReportSource::GpuInventory,
            target: Some(HealthReportTarget::Machine),
            observed_at: Some(chrono::Utc::now()),
            successes: Vec::new(),
            alerts: vec![HealthReportAlert {
                probe_id: Probe::GpuInventory,
                target: None,
                message: format!(
                    "Expected gpu count ({expected}) does not match actual ({actual}) \
                     as seen out-of-band via BMC"
                ),
                classifications: vec![Classification::PreventAllocations],
            }],
        }
    } else {
        HealthReport {
            source: ReportSource::GpuInventory,
            target: Some(HealthReportTarget::Machine),
            observed_at: Some(chrono::Utc::now()),
            successes: vec![HealthReportSuccess {
                probe_id: Probe::GpuInventory,
                target: None,
            }],
            alerts: Vec::new(),
        }
    }
}

impl<B: Bmc + 'static> PeriodicCollector<B> for GpuInventoryCollector<B> {
    type Config = GpuInventoryCollectorConfig;

    fn new_runner(
        bmc: Arc<B>,
        endpoint: Arc<BmcEndpoint>,
        config: Self::Config,
    ) -> Result<Self, HealthError> {
        let event_context =
            EventContext::from_endpoint(endpoint.as_ref(), "gpu_inventory_collector");
        let machine_id = match &endpoint.metadata {
            Some(EndpointMetadata::Machine(m)) => Some(m.machine_id),
            _ => None,
        };
        Ok(Self {
            endpoint,
            bmc,
            event_context,
            data_sink: config.data_sink,
            api_client: config.api_client,
            machine_id,
        })
    }

    async fn run_iteration(&mut self) -> Result<IterationResult, HealthError> {
        let expected_count = match self.resolve_expected().await? {
            // No SKU assigned, or the SKU declares zero GPUs (e.g. a CPU-only node):
            // nothing to validate. Emit a success so any prior shortage alert on
            // this machine clears (recovery), rather than lingering forever.
            Expected::NoSku | Expected::Count(0) => {
                self.emit(gpu_count_report(0, 0));
                return Ok(IterationResult {
                    refresh_triggered: false,
                    entity_count: None,
                    fetch_failures: 0,
                });
            }
            Expected::SkuMissing(sku_id) => {
                self.emit_alert(format!("The assigned sku {sku_id} does not exist"));
                return Ok(IterationResult {
                    refresh_triggered: false,
                    entity_count: Some(0),
                    fetch_failures: 0,
                });
            }
            Expected::Count(n) => n,
        };

        let actual = self.count_gpus().await?;
        let report = gpu_count_report(expected_count, actual);
        if !report.alerts.is_empty() {
            tracing::warn!(
                bmc = %self.endpoint.addr.mac,
                expected = expected_count,
                actual,
                "GPU count below SKU expectation"
            );
        }
        self.emit(report);

        Ok(IterationResult {
            refresh_triggered: false,
            entity_count: Some(actual as usize),
            fetch_failures: 0,
        })
    }

    fn collector_type(&self) -> &'static str {
        "gpu_inventory_collector"
    }
}

/// Integration tests that drive the GPU-counting logic against realistic Redfish
/// trees served in-process by `bmc-mock` (no socket, no reqwest).
///
/// `count_gpus` takes `max(count_hgx_gpu_chassis, count_gpu_processors)`, so both
/// counters are exercised here:
/// - `count_hgx_gpu_chassis` (`HGX_GPU_*`) is the same code for H100 SXM, GH200,
///   GB200 and GB300 — GB300 pins it to an exact count. No in-process H100/GH200
///   helper exists (H100 is a Rust fixture without a helper; GH200 is a tarball
///   served over HTTP), so those are covered by the shared predicate.
/// - `count_gpu_processors` (`ProcessorType == GPU`) is validated against GB200,
///   the only in-process fixture that models GPUs as Redfish Processors — the same
///   signal PCIe GPUs (L40/L40S, H100 PCIe) produce on Dell/Lenovo/HPE.
#[cfg(test)]
mod bmc_mock_integration_tests {
    use bmc_mock::test_support::{
        TestBmc, dell_poweredge_r750_bmc, dgx_gb300_bmc, wiwynn_gb200_bmc,
    };

    use super::{GpuInventoryCollector, gpu_count_report};
    use crate::sink::{Classification, Probe};

    #[test]
    fn alerts_when_fewer_gpus_than_sku_expects() {
        // Issue #301 core: BMC sees 6 GPUs, the SKU expects 8 -> shortage alert.
        let report = gpu_count_report(8, 6);
        assert!(report.successes.is_empty());
        assert_eq!(report.alerts.len(), 1);
        let alert = &report.alerts[0];
        assert_eq!(alert.probe_id, Probe::GpuInventory);
        assert_eq!(
            alert.classifications,
            vec![Classification::PreventAllocations]
        );
        assert!(
            alert.message.contains('6') && alert.message.contains('8'),
            "message should name actual and expected: {}",
            alert.message
        );
    }

    #[test]
    fn success_when_gpu_count_matches_or_exceeds_sku() {
        // Exact match -> success, no alert.
        let matched = gpu_count_report(8, 8);
        assert!(matched.alerts.is_empty());
        assert_eq!(matched.successes.len(), 1);
        // More GPUs than the SKU expects is not a shortage -> still success.
        assert!(gpu_count_report(4, 5).alerts.is_empty());
        // Clear/no-op case (no SKU or SKU declares 0 GPUs) -> success, no alert,
        // which clears any prior shortage alert on the machine.
        let cleared = gpu_count_report(0, 0);
        assert!(cleared.alerts.is_empty());
        assert_eq!(cleared.successes.len(), 1);
    }

    #[tokio::test]
    async fn gb300_shortage_vs_sku_raises_alert_end_to_end() {
        // Real Redfish count feeding the #301 decision: DGX GB300 exposes 4 GPU
        // chassis; if its SKU expected 8, that must raise a shortage alert.
        let h = dgx_gb300_bmc().await;
        let actual =
            GpuInventoryCollector::<TestBmc>::count_hgx_gpu_chassis(h.service_root.as_ref())
                .await
                .expect("count");
        assert_eq!(actual, 4);
        assert_eq!(gpu_count_report(8, actual).alerts.len(), 1);
        assert!(gpu_count_report(4, actual).alerts.is_empty());
    }

    #[tokio::test]
    async fn dgx_gb300_counts_four_hgx_gpu_chassis() {
        let h = dgx_gb300_bmc().await;
        let root = h.service_root.as_ref();

        let count = GpuInventoryCollector::<TestBmc>::count_hgx_gpu_chassis(root)
            .await
            .expect("count_hgx_gpu_chassis");
        // The dgx_gb300 fixture models a 4-GPU compute tray (HGX_GPU_0..=3).
        assert_eq!(count, 4, "expected 4 HGX_GPU_ chassis on DGX GB300");
    }

    #[tokio::test]
    async fn gb200_exposes_gpus_as_both_chassis_and_processors() {
        let h = wiwynn_gb200_bmc().await;
        let root = h.service_root.as_ref();

        let chassis = GpuInventoryCollector::<TestBmc>::count_hgx_gpu_chassis(root)
            .await
            .expect("count_hgx_gpu_chassis");
        let processors = GpuInventoryCollector::<TestBmc>::count_gpu_processors(root)
            .await
            .expect("count_gpu_processors");

        // GB200 lists the same GPUs both ways. Both counters must find them; because
        // count_gpus() takes the max (not the sum) they are not double-counted.
        assert!(chassis > 0, "expected HGX_GPU_* chassis, got {chassis}");
        assert!(processors > 0, "expected GPU processors, got {processors}");
    }

    #[tokio::test]
    async fn dell_r750_gpuless_counts_zero_by_both_methods() {
        let h = dell_poweredge_r750_bmc().await;
        let root = h.service_root.as_ref();

        let chassis = GpuInventoryCollector::<TestBmc>::count_hgx_gpu_chassis(root)
            .await
            .expect("count_hgx_gpu_chassis");
        let processors = GpuInventoryCollector::<TestBmc>::count_gpu_processors(root)
            .await
            .expect("count_gpu_processors");

        // The R750 fixture is a GPU-less server: neither method finds a GPU.
        assert_eq!(chassis, 0);
        assert_eq!(processors, 0);
    }
}
