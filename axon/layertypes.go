// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/goki/ki/kit"
)

//gosl: start layertypes

// LayerTypes is an axon-specific layer type enum,
// that encompasses all the different algorithm types supported.
// Class parameter styles automatically key off of these types.
// The first entries must be kept synchronized with the emer.LayerType,
// although we replace Hidden -> Super.
type LayerTypes int32

// note: we need to add the Layer extension to avoid naming
// conflicts between layer, projection and other things.

// The layer types
const (
	// Super is a superficial cortical layer (lamina 2-3-4)
	// which does not receive direct input or targets.
	// In more generic models, it should be used as a Hidden layer,
	// and maps onto the Hidden type in emer.LayerType.
	SuperLayer LayerTypes = iota

	// Input is a layer that receives direct external input
	// in its Ext inputs.  Biologically, it can be a primary
	// sensory layer, or a thalamic layer.
	InputLayer

	// Target is a layer that receives direct external target inputs
	// used for driving plus-phase learning.
	// Simple target layers are generally not used in more biological
	// models, which instead use predictive learning via Pulvinar
	// or related mechanisms.
	TargetLayer

	// Compare is a layer that receives external comparison inputs,
	// which drive statistics but do NOT drive activation
	// or learning directly.  It is rarely used in axon.
	CompareLayer

	/////////////
	// Deep

	// CT are layer 6 corticothalamic projecting neurons,
	// which drive "top down" predictions in Pulvinar layers.
	// They maintain information over time via stronger NMDA
	// channels and use maintained prior state information to
	// generate predictions about current states forming on Super
	// layers that then drive PT (5IB) bursting activity, which
	// are the plus-phase drivers of Pulvinar activity.
	CTLayer

	// Pulvinar are thalamic relay cell neurons in the higher-order
	// Pulvinar nucleus of the thalamus, and functionally isomorphic
	// neurons in the MD thalamus, and potentially other areas.
	// These cells alternately reflect predictions driven by CT projections,
	// and actual outcomes driven by 5IB Burst activity from corresponding
	// PT or Super layer neurons that provide strong driving inputs.
	PulvinarLayer

	// TRNLayer is thalamic reticular nucleus layer for inhibitory competition
	// within the thalamus.
	TRNLayer

	// PTMaintLayer implements the pyramidal tract layer 5 intrinsic bursting
	// (5IB) deep neurons, which provide the main output signal from cortex,
	// specifically the subset of PT neurons that are gated by the BG to
	// drive sustained active maintenance, via strong NMDA channels.
	// Set projections from thalamus to be modulatory, and use Act.Dend.ModGain
	// to set extra strength these inputs which are only briefly active.
	PTMaintLayer

	/////////////
	// RL

	// RewLayer represents positive or negative reward values across 2 units,
	// showing spiking rates for each, and Act always represents signed value.
	RewLayer

	// RSalienceAChLayer reads Max layer activity from specified source layer(s)
	// and optionally the global Context.NeuroMod.Rew or RewPred state variables,
	// and updates the global ACh = Max of all as the positively-rectified,
	// non-prediction-discounted reward salience signal.
	// Acetylcholine (ACh) is known to represent something like this signal.
	RSalienceAChLayer

	// RWPredLayer computes reward prediction for a simple Rescorla-Wagner
	// learning dynamic (i.e., PV learning in the PVLV framework).
	// Activity is computed as linear function of excitatory conductance
	// (which can be negative -- there are no constraints).
	// Use with RWPrjn which does simple delta-rule learning on minus-plus.
	RWPredLayer

	// RWDaLayer computes a dopamine (DA) signal based on a simple Rescorla-Wagner
	// learning dynamic (i.e., PV learning in the PVLV framework).
	// It computes difference between r(t) and RWPred values.
	// r(t) is accessed directly from a Rew layer -- if no external input then no
	// DA is computed -- critical for effective use of RW only for PV cases.
	// RWPred prediction is also accessed directly from Rew layer to avoid any issues.
	RWDaLayer

	// TDPredLayer is the temporal differences reward prediction layer.
	// It represents estimated value V(t) in the minus phase, and computes
	// estimated V(t+1) based on its learned weights in plus phase,
	// using the TDPredPrjn projection type for DA modulated learning.
	TDPredLayer

	// TDIntegLayer is the temporal differences reward integration layer.
	// It represents estimated value V(t) from prior time step in the minus phase,
	// and estimated discount * V(t+1) + r(t) in the plus phase.
	// It gets Rew, PrevPred from Context.NeuroMod, and Special
	// LayerVals from TDPredLayer.
	TDIntegLayer

	// TDDaLayer computes a dopamine (DA) signal as the temporal difference (TD)
	// between the TDIntegLayer activations in the minus and plus phase.
	// These are retrieved from Special LayerVals.
	TDDaLayer

	/////////////
	// PVLV

	// BLALayer represents a basolateral amygdala layer
	// which learns to associate arbitrary stimuli (CSs)
	// with behaviorally salient outcomes (USs)
	BLALayer

	// CeMLayer represents a central nucleus of the amygdala layer.
	CeMLayer

	// PPTgLayer represents a pedunculopontine tegmental nucleus layer.
	// it subtracts prior trial's excitatory conductance to
	// compute the temporal derivative over time, with a positive
	// rectification.
	// also sets Act to the exact differenence.
	PPTgLayer

	/////////////////////////////
	// PCORE Basal Ganglia (BG)

	// MatrixLayer represents the matrisome medium spiny neurons (MSNs)
	// that are the main Go / NoGo gating units in BG.
	// These are strongly modulated by phasic dopamine: D1 = Go, D2 = NoGo.
	MatrixLayer

	// STNLayer represents subthalamic nucleus neurons, with two subtypes:
	// STNp are more strongly driven and get over bursting threshold, driving strong,
	// rapid activation of the KCa channels, causing a long pause in firing, which
	// creates a window during which GPe dynamics resolve Go vs. No balance.
	// STNs are more weakly driven and thus more slowly activate KCa, resulting in
	// a longer period of activation, during which the GPi is inhibited to prevent
	// premature gating based only MtxGo inhibition -- gating only occurs when
	// GPeIn signal has had a chance to integrate its MtxNo inputs.
	STNLayer

	// GPLayer represents a globus pallidus layer in the BG, including:
	// GPeOut, GPeIn, GPeTA (arkypallidal), and GPi.
	// Typically just a single unit per Pool representing a given stripe.
	GPLayer

	// VThalLayer represents a BG gated thalamic layer,
	// which receives BG gating in the form of an
	// inhibitory projection from GPi.  Located
	// mainly in the Ventral thalamus: VA / VM / VL,
	// and also parts of MD mediodorsal thalamus.
	VThalLayer

	LayerTypesN
)

// IsExtLayerType returns true if the layer type deals with external input:
// Input, Target, Compare
func IsExtLayerType(lt LayerTypes) bool {
	if lt == InputLayer || lt == TargetLayer || lt == CompareLayer {
		return true
	}
	return false
}

//gosl: end layertypes

// IsExt returns true if the layer type deals with external input:
// Input, Target, Compare
func (lt LayerTypes) IsExt() bool {
	if lt == InputLayer || lt == TargetLayer || lt == CompareLayer {
		return true
	}
	return false
}

//go:generate stringer -type=LayerTypes

var KiT_LayerTypes = kit.Enums.AddEnum(LayerTypesN, kit.NotBitFlag, nil)

func (ev LayerTypes) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *LayerTypes) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }
