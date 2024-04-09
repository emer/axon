// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start prjntypes

// PrjnTypes is an axon-specific prjn type enum,
// that encompasses all the different algorithm types supported.
// Class parameter styles automatically key off of these types.
// The first entries must be kept synchronized with the emer.PrjnType.
type PrjnTypes int32 //enums:enum

// The projection types
const (
	// Forward is a feedforward, bottom-up projection from sensory inputs to higher layers
	ForwardPrjn PrjnTypes = iota

	// Back is a feedback, top-down projection from higher layers back to lower layers
	BackPrjn

	// Lateral is a lateral projection within the same layer / area
	LateralPrjn

	// Inhib is an inhibitory projection that drives inhibitory
	// synaptic conductances instead of the default excitatory ones.
	InhibPrjn

	// CTCtxt are projections from Superficial layers to CT layers that
	// send Burst activations drive updating of CtxtGe excitatory conductance,
	// at end of plus (51B Bursting) phase.  Biologically, this projection
	// comes from the PT layer 5IB neurons, but it is simpler to use the
	// Super neurons directly, and PT are optional for most network types.
	// These projections also use a special learning rule that
	// takes into account the temporal delays in the activation states.
	// Can also add self context from CT for deeper temporal context.
	CTCtxtPrjn

	// RWPrjn does dopamine-modulated learning for reward prediction:
	// Da * Send.CaSpkP (integrated current spiking activity).
	// Uses RLPredPrjn parameters.
	// Use in RWPredLayer typically to generate reward predictions.
	// If the Da sign is positive, the first recv unit learns fully;
	// for negative, second one learns fully.  Lower lrate applies for
	// opposite cases.  Weights are positive-only.
	RWPrjn

	// TDPredPrjn does dopamine-modulated learning for reward prediction:
	// DWt = Da * Send.SpkPrv (activity on *previous* timestep)
	// Uses RLPredPrjn parameters.
	// Use in TDPredLayer typically to generate reward predictions.
	// If the Da sign is positive, the first recv unit learns fully;
	// for negative, second one learns fully.  Lower lrate applies for
	// opposite cases.  Weights are positive-only.
	TDPredPrjn

	// BLAPrjn implements the Rubicon BLA learning rule:
	// dW = ACh * X_t-1 * (Y_t - Y_t-1)
	// The recv delta is across trials, where the US should activate on trial
	// boundary, to enable sufficient time for gating through to OFC, so
	// BLA initially learns based on US present - US absent.
	// It can also learn based on CS onset if there is a prior CS that predicts that.
	BLAPrjn

	HipPrjn

	// VSPatchPrjn implements the VSPatch learning rule:
	// dW = ACh * DA * X * Y
	// where DA is D1 vs. D2 modulated DA level, X = sending activity factor,
	// Y = receiving activity factor, and ACh provides overall modulation.
	VSPatchPrjn

	// VSMatrixPrjn is for ventral striatum matrix (SPN / MSN) neurons
	// supporting trace-based learning, where an initial
	// trace of synaptic co-activity is formed, and then modulated
	// by subsequent phasic dopamine & ACh when an outcome occurs.
	// This bridges the temporal gap between gating activity
	// and subsequent outcomes, and is based biologically on synaptic tags.
	// Trace is reset at time of reward based on ACh level (from CINs in biology).
	VSMatrixPrjn

	// DSMatrixPrjn is for dorsal striatum matrix (SPN / MSN) neurons
	// supporting trace-based learning, where an initial
	// trace of synaptic co-activity is formed, and then modulated
	// by subsequent phasic dopamine & ACh when an outcome occurs.
	// This bridges the temporal gap between gating activity
	// and subsequent outcomes, and is based biologically on synaptic tags.
	// Trace is reset at time of reward based on ACh level (from CINs in biology).
	DSMatrixPrjn
)

//gosl: end prjntypes
