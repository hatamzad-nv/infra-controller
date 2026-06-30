// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package handler

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/labstack/echo/v4"
	"github.com/rs/zerolog"

	"github.com/NVIDIA/infra-controller/rest-api/api/internal/config"
	"github.com/NVIDIA/infra-controller/rest-api/api/pkg/api/handler/util/common"
	"github.com/NVIDIA/infra-controller/rest-api/api/pkg/api/model"
	sc "github.com/NVIDIA/infra-controller/rest-api/api/pkg/client/site"
	auth "github.com/NVIDIA/infra-controller/rest-api/auth/pkg/authorization"
	cutil "github.com/NVIDIA/infra-controller/rest-api/common/pkg/util"
	cdb "github.com/NVIDIA/infra-controller/rest-api/db/pkg/db"
	cdbm "github.com/NVIDIA/infra-controller/rest-api/db/pkg/db/model"
	cwssaws "github.com/NVIDIA/infra-controller/rest-api/workflow-schema/schema/site-agent/workflows/v1"
)

func logAPIError(logger zerolog.Logger, apiErr *cutil.APIError, msg string) {
	if apiErr.Data != nil {
		logger.Error().Err(apiErr.Data).Msg(msg)
		return
	}
	logger.Error().Err(apiErr).Msg(msg)
}

type MachinePowerControlHandler struct {
	dbSession  *cdb.Session
	scp        *sc.ClientPool
	tracerSpan *cutil.TracerSpan
}

func NewMachinePowerControlHandler(dbSession *cdb.Session, scp *sc.ClientPool, _ *config.Config) MachinePowerControlHandler {
	return MachinePowerControlHandler{
		dbSession:  dbSession,
		scp:        scp,
		tracerSpan: cutil.NewTracerSpan(),
	}
}

// Handle godoc
// @Summary Machine Power Control
// @Description Power control a Machine through NICo Core. Provider Admin only.
// @Tags machine-power
// @Accept json
// @Produce json
// @Security ApiKeyAuth
// @Param org path string true "Name of NGC organization"
// @Param machineId path string true "ID of Machine"
// @Param request body model.APIMachinePowerControlRequest true "Power control request"
// @Success 200 {object} model.APIMachinePowerControlResponse
// @Router /v2/org/{org}/nico/machine/{machineId}/power [patch]
func (h MachinePowerControlHandler) Handle(c echo.Context) error {
	org, dbUser, ctx, logger, handlerSpan := common.SetupHandler("MachinePower", "Control", c, h.tracerSpan)
	if handlerSpan != nil {
		defer handlerSpan.End()
	}

	if dbUser == nil {
		logger.Error().Msg("invalid User object found in request context")
		return cutil.NewAPIErrorResponse(c, http.StatusInternalServerError, "Failed to retrieve current user", nil)
	}

	// Validate org
	ok, err := auth.ValidateOrgMembership(dbUser, org)
	if !ok {
		if err != nil {
			logger.Error().Err(err).Msg("error validating org membership for User in request")
		} else {
			logger.Warn().Msg("could not validate org membership for user, access denied")
		}
		return cutil.NewAPIErrorResponse(c, http.StatusForbidden, fmt.Sprintf("Failed to validate membership for org: %s", org), nil)
	}

	// Validate role: Provider Admins or Viewers, or privileged Tenant Admins (TargetedInstanceCreation; see filters below).
	provider, _, apiError := common.IsProviderOrTenant(ctx, logger, h.dbSession, org, dbUser, true, true)
	if apiError != nil {
		return cutil.NewAPIErrorResponse(c, apiError.Code, apiError.Message, apiError.Data)
	}

	if provider == nil {
		logger.Warn().Msg("user does not have Provider role, access denied")
		return cutil.NewAPIErrorResponse(c, http.StatusForbidden, "User does not have Provider role with org", nil)
	}

	var apiReq model.APIMachinePowerControlRequest
	if err := c.Bind(&apiReq); err != nil {
		return cutil.NewAPIErrorResponse(c, http.StatusBadRequest, "Invalid request body", nil)
	}

	if err := apiReq.Validate(); err != nil {
		return cutil.NewAPIErrorResponse(c, http.StatusBadRequest, err.Error(), nil)
	}

	// Retrieve Machine details
	machineDAO := cdbm.NewMachineDAO(h.dbSession)
	machine, err := machineDAO.GetByID(ctx, nil, apiReq.MachineID, []string{cdbm.SiteRelationName}, false)
	if err != nil {
		if errors.Is(err, cdb.ErrDoesNotExist) {
			return cutil.NewAPIErrorResponse(c, http.StatusNotFound, "Could not find Machine with specified ID", nil)
		}
		logger.Error().Err(err).Msg("failed to retrieve Machine details from DB")
		return cutil.NewAPIErrorResponse(c, http.StatusInternalServerError, "Failed to retrieve Machine details, DB error", nil)
	}

	// Check Site presence in Machine object
	if machine.Site == nil {
		logger.Error().Msg("Related Site was not returned for Machine DB entity")
		return cutil.NewAPIErrorResponse(c, http.StatusInternalServerError, "Failed to retrieve Site details for Machine, DB error", nil)
	}

	site := machine.Site

	if machine.InfrastructureProviderID != provider.ID {
		logger.Error().Msg("Machine doesn't belong to org's Infrastructure provider")
		// Return 404 instead of 403 to avoid leaking information about the existence of the machine
		return cutil.NewAPIErrorResponse(c, http.StatusNotFound, "Could not find Machine with specified ID", nil)
	}

	if machine.IsAssigned && (apiReq.AcknowledgeAttachedInstance == nil || !*apiReq.AcknowledgeAttachedInstance) {
		logger.Error().Msg("Machine is currently in use by an Instance and cannot be power controlled without acknowledging the attached Instance")
		return cutil.NewAPIErrorResponse(c, http.StatusBadRequest, "Machine is currently in use by an Instance, set acknowledgeAttachedInstance to true to proceed", nil)
	}

	if site.Status != cdbm.SiteStatusRegistered {
		logger.Warn().Msg("Site specified in request data is not in Registered state")
		return cutil.NewAPIErrorResponse(c, http.StatusBadRequest, "Site specified in request data is not in Registered state, cannot execute power control", nil)
	}

	// Get Temporal client for Site
	stc, err := h.scp.GetClientByID(site.ID)
	if err != nil {
		logger.Error().Err(err).Msg("failed to retrieve Temporal client for Site")
		return cutil.NewAPIErrorResponse(c, http.StatusInternalServerError, "Failed to retrieve workflow client for Site", nil)
	}

	logger.Info().Str("machineID", apiReq.MachineID).Str("action", string(apiReq.Action)).Str("siteID", site.ID.String()).Msg("Sending power control request to Site via Core proxy")

	coreResp := &cwssaws.AdminPowerControlResponse{}
	apiErr := common.ExecuteCoreGRPC(ctx, stc, cwssaws.Forge_AdminPowerControl_FullMethodName, apiReq.ToProto(), coreResp, site.ID.String())
	if apiErr != nil {
		logAPIError(logger, apiErr, "failed to execute power control request on Site via Core proxy")
		return cutil.NewAPIErrorResponse(c, apiErr.Code, apiErr.Message, nil)
	}

	// Build response
	response := model.APIMachinePowerControlResponse{
		MachineID: apiReq.MachineID,
		Action:    apiReq.Action,
		Message:   coreResp.Msg,
	}

	// Return success response
	return c.JSON(http.StatusOK, response)
}
