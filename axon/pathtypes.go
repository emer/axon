// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl:start

// PathTypes enumerates all the different types of axon pathways,
// for the different algorithm types supported.
// Class parameter styles automatically key off of these types.
type PathTypes int32 //enums:enum

// The pathway types
const (
	// Forward is a feedforward, bottom-up pathway from sensory inputs to higher layers
	ForwardPath PathTypes = iota

	// Back is a feedback, top-down pathway from higher layers back to lower layers
	BackPath

	// Lateral is a lateral pathway within the same layer / area
	LateralPath

	// Inhib is an inhibitory pathway that drives inhibitory
	// synaptic conductances instead of the default excitatory ones.
	InhibPath

	// CTCtxt are pathways from Superficial layers to CT layers that
	// send Burst activations drive updating of CtxtGe excitatory conductance,
	// at end of plus (51B Bursting) phase.  Biologically, this pathway
	// comes from the PT layer 5IB neurons, but it is simpler to use the
	// Super neurons directly, and PT are optional for most network types.
	// These pathways also use a special learning rule that
	// takes into account the temporal delays in the activation states.
	// Can also add self context from CT for deeper temporal context.
	CTCtxtPath

	// DSPatchPath implements the DSPatch learning rule:
	// dW = ACh * DA * X * Y
	// where DA is D1 vs. D2 modulated DA level, X = sending activity factor,
	// Y = receiving activity factor, and ACh provides overall modulation.
	DSPatchPath

	// VSPatchPath implements the VSPatch learning rule:
	// dW = ACh * DA * X * Y
	// where DA is D1 vs. D2 modulated DA level, X = sending activity factor,
	// Y = receiving activity factor, and ACh provides overall modulation.
	VSPatchPath

	// VSMatrixPath is for ventral striatum matrix (SPN / MSN) neurons
	// supporting trace-based learning, where an initial
	// trace of synaptic co-activity is formed, and then modulated
	// by subsequent phasic dopamine & ACh when an outcome occurs.
	// This bridges the temporal gap between gating activity
	// and subsequent outcomes, and is based biologically on synaptic tags.
	// Trace is reset at time of reward based on ACh level (from CINs in biology).
	VSMatrixPath

	// DSMatrixPath is for dorsal striatum matrix (SPN / MSN) neurons
	// supporting trace-based learning, where an initial
	// trace of synaptic co-activity is formed, and then modulated
	// by subsequent phasic dopamine & ACh when an outcome occurs.
	// This bridges the temporal gap between gating activity
	// and subsequent outcomes, and is based biologically on synaptic tags.
	// Trace is reset at time of reward based on ACh level (from CINs in biology).
	DSMatrixPath

	// CNiPredToOutPath is the prediction to output pathway in the cerebellar
	// nucleus system, which is inhibitory and learns drive the output neurons at
	// their target baseline activity level, when both excitatory and inhibitory
	// input is present there.
	CNiPredToOutPath

	// RWPath does dopamine-modulated learning for reward prediction:
	// Da * Send.CaP (integrated current spiking activity).
	// Uses RLPredPath parameters.
	// Use in RWPredLayer typically to generate reward predictions.
	// If the Da sign is positive, the first recv unit learns fully;
	// for negative, second one learns fully.  Lower lrate applies for
	// opposite cases.  Weights are positive-only.
	RWPath

	// TDPredPath does dopamine-modulated learning for reward prediction:
	// DWt = Da * Send.CaDPrev (activity on *previous* timestep)
	// Uses RLPredPath parameters.
	// Use in TDPredLayer typically to generate reward predictions.
	// If the Da sign is positive, the first recv unit learns fully;
	// for negative, second one learns fully.  Lower lrate applies for
	// opposite cases.  Weights are positive-only.
	TDPredPath

	// BLAPath implements the Rubicon BLA learning rule:
	// dW = ACh * X_t-1 * (Y_t - Y_t-1)
	// The recv delta is across trials, where the US should activate on trial
	// boundary, to enable sufficient time for gating through to OFC, so
	// BLA initially learns based on US present - US absent.
	// It can also learn based on CS onset if there is a prior CS that predicts that.
	BLAPath

	// HipPath is a special pathway for the hippocampus. TODO: fixme.
	HipPath
)

//gosl:end
