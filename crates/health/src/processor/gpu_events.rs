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

use std::borrow::Cow;
use std::sync::Arc;

use super::{CollectorEvent, EventContext, EventProcessor};
use crate::sink::{
    Classification, HealthReport, HealthReportAlert, HealthReportTarget, LogRecord, Probe,
    ReportSource,
};

/// Substrings that mark a GPU-related fault in a Redfish SEL / LogService entry.
/// Matched case-insensitively against the log message body.
const GPU_FAULT_KEYWORDS: &[&str] = &[
    "gpu",       // "GPU_0 ...", "HGX_GPU_..."
    "nvlink",    // "NVLink Training Error"
    "row remap", // ECC row-remap failures
    "xid",       // driver XID surfaced via SEL on some platforms
    "sxm",       // HGX_GPU_SXM_*
];

#[derive(Default)]
pub struct GpuFaultEventProcessor;

impl GpuFaultEventProcessor {
    pub fn new() -> Self {
        Self
    }

    fn attr<'a>(attributes: &'a [(Cow<'static, str>, String)], key: &str) -> Option<&'a str> {
        attributes
            .iter()
            .find(|(name, _)| name.as_ref() == key)
            .map(|(_, value)| value.as_str())
    }

    /// Severities worth alerting on; "ok"/"info" GPU log lines are ignored.
    fn is_actionable(severity: &str) -> bool {
        matches!(
            severity.to_ascii_uppercase().as_str(),
            "WARN" | "WARNING" | "ERROR" | "FATAL" | "CRITICAL"
        )
    }

    /// Returns the fault message if this log record looks like a GPU fault.
    fn gpu_fault(record: &LogRecord) -> Option<String> {
        if !Self::is_actionable(&record.severity) {
            return None;
        }
        let haystack = record.body.to_ascii_lowercase();
        if GPU_FAULT_KEYWORDS.iter().any(|kw| haystack.contains(kw)) {
            Some(record.body.clone())
        } else {
            None
        }
    }
}

impl EventProcessor for GpuFaultEventProcessor {
    fn processor_type(&self) -> &'static str {
        "gpu_fault_event_processor"
    }

    fn process_event(
        &self,
        _context: &EventContext,
        event: &CollectorEvent,
    ) -> Vec<CollectorEvent> {
        let CollectorEvent::Log(record) = event else {
            return Vec::new();
        };

        let Some(message) = Self::gpu_fault(record) else {
            return Vec::new();
        };

        // Attach the SEL entry id as the target if present, else the BMC.
        let target = Self::attr(&record.attributes, "entry_id")
            .map(|id| format!("GPU/{id}"))
            .or_else(|| Some("HostBMC".to_string()));

        let report = HealthReport {
            source: ReportSource::GpuFault,
            target: Some(HealthReportTarget::Machine),
            observed_at: Some(chrono::Utc::now()),
            successes: Vec::new(),
            alerts: vec![HealthReportAlert {
                probe_id: Probe::GpuFault,
                target,
                message,
                classifications: vec![Classification::GpuFault],
            }],
        };

        vec![CollectorEvent::HealthReport(Arc::new(report))]
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::str::FromStr;

    use mac_address::MacAddress;

    use super::*;
    use crate::endpoint::BmcAddr;

    fn context() -> EventContext {
        EventContext {
            endpoint_key: "42:9e:b1:bd:9d:dd".to_string(),
            addr: BmcAddr {
                ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
                port: Some(443),
                mac: MacAddress::from_str("42:9e:b1:bd:9d:dd").expect("valid mac"),
            },
            collector_type: "test",
            metadata: None,
            rack_id: None,
        }
    }

    fn log_event(body: &str, severity: &str) -> CollectorEvent {
        CollectorEvent::Log(Box::new(LogRecord {
            body: body.to_string(),
            severity: severity.to_string(),
            attributes: Vec::new(),
            diagnostic_record: None,
        }))
    }

    fn process(event: CollectorEvent) -> Vec<CollectorEvent> {
        GpuFaultEventProcessor::new().process_event(&context(), &event)
    }

    #[test]
    fn alerts_on_gpu_fault_at_actionable_severity() {
        let emitted = process(log_event("GPU_0 NVLink Training Error", "ERROR"));
        assert_eq!(emitted.len(), 1);

        let CollectorEvent::HealthReport(report) = &emitted[0] else {
            panic!("expected health report");
        };
        assert_eq!(report.source, ReportSource::GpuFault);
        assert_eq!(report.target, Some(HealthReportTarget::Machine));

        let alert = report.alerts.first().expect("one alert");
        assert_eq!(alert.probe_id, Probe::GpuFault);
        assert_eq!(alert.classifications, vec![Classification::GpuFault]);
        assert!(alert.message.contains("NVLink"));
    }

    #[test]
    fn ignores_non_actionable_severity() {
        assert!(process(log_event("GPU_0 temperature normal", "INFO")).is_empty());
        assert!(process(log_event("GPU_0 ok", "OK")).is_empty());
    }

    #[test]
    fn alerts_on_warning_severity() {
        // The SSE log stream emits "Warning" (not "WARN"); both must alert.
        assert_eq!(process(log_event("GPU_0 NVLink error", "Warning")).len(), 1);
        assert_eq!(process(log_event("GPU_0 NVLink error", "WARN")).len(), 1);
    }

    #[test]
    fn ignores_logs_without_gpu_keywords() {
        assert!(process(log_event("Fan1 speed warning", "WARN")).is_empty());
    }

    #[test]
    fn ignores_non_log_events() {
        assert!(process(CollectorEvent::CollectorRemoved).is_empty());
    }

    #[test]
    fn matches_various_gpu_fault_keywords() {
        for body in [
            "GPU_0 error",
            "NVLink down",
            "ECC row remap failure",
            "XID 79 fault",
            "HGX_GPU_SXM_1 fault",
        ] {
            assert_eq!(
                process(log_event(body, "ERROR")).len(),
                1,
                "should alert on: {body}"
            );
        }
    }
}
