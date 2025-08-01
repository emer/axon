// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl:start

// LayerTypes enumerates all the different types of layers,
// for the different algorithm types supported.
// Class parameter styles automatically key off of these types.
type LayerTypes int32 //enums:enum

// note: we need to add the Layer extension to avoid naming
// conflicts between layer, pathway and other things.

// The layer types
const (
	// Super is a superficial cortical layer (lamina 2-3-4)
	// which does not receive direct input or targets.
	// In more generic models, it should be used as a Hidden layer,
	// and maps onto the Hidden type in LayerTypes.
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

	//////// Deep

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
	// These cells alternately reflect predictions driven by CT pathways,
	// and actual outcomes driven by 5IB Burst activity from corresponding
	// PT or Super layer neurons that provide strong driving inputs.
	PulvinarLayer

	// TRNLayer is thalamic reticular nucleus layer for inhibitory competition
	// within the thalamus.
	TRNLayer

	// PTMaintLayer implements the subset of pyramidal tract (PT)
	// layer 5 intrinsic bursting (5IB) deep neurons that exhibit
	// robust, stable maintenance of activity over the duration of a
	// goal engaged window, modulated by basal ganglia (BG) disinhibitory
	// gating, supported by strong MaintNMDA channels and recurrent excitation.
	// The lateral PTSelfMaint pathway uses MaintG to drive GMaintRaw input
	// that feeds into the stronger, longer MaintNMDA channels,
	// and the ThalToPT ModulatoryG pathway from BGThalamus multiplicatively
	// modulates the strength of other inputs, such that only at the time of
	// BG gating are these strong enough to drive sustained active maintenance.
	// Use Act.Dend.ModGain to parameterize.
	PTMaintLayer

	// PTPredLayer implements the subset of pyramidal tract (PT)
	// layer 5 intrinsic bursting (5IB) deep neurons that combine
	// modulatory input from PTMaintLayer sustained maintenance and
	// CTLayer dynamic predictive learning that helps to predict
	// state changes during the period of active goal maintenance.
	// This layer provides the primary input to VSPatch US-timing
	// prediction layers, and other layers that require predictive dynamic
	PTPredLayer

	//////// PCore Basal Ganglia (BG)

	// DSMatrixLayer represents the matrisome spiny projection neurons
	// (SPNs, MSNs) that are the main Go / No gating units in BG,
	// and are modulated by phasic dopamine: D1 = Go, D2 = No.
	// These are for dorsal striatum, which interact with matrisomes and
	// receive PF (parafasciculus) feedback signals.
	DSMatrixLayer

	// VSMatrixLayer represents the matrisome spiny projection neurons
	// (SPNs, MSNs) that are the main Go / No gating units in BG,
	// and are modulated by phasic dopamine: D1 = Go, D2 = No.
	// These are for ventral striatum, which drive goal-selection
	// gating signals through the MD thalamus, and activate instinctive
	// behaviors based on learned inputs projecting to various output
	// pathways.
	VSMatrixLayer

	// DSPatchLayer represents the dorsolateral striosomal spiny neurons
	// that modulate the activity of SNc dopamine to a given Pool.
	DSPatchLayer

	// STNLayer represents subthalamic nucleus neurons, with two subtypes:
	// STNp are more strongly driven and get over bursting threshold, driving strong,
	// rapid activation of the KCa channels, causing a long pause in firing, which
	// creates a window during which GPe dynamics resolve Go vs. No balance.
	// STNs are more weakly driven and thus more slowly activate KCa, resulting in
	// a longer period of activation, during which the GPi is inhibited to prevent
	// premature gating based only MtxGo inhibition -- gating only occurs when
	// GPePr signal has had a chance to integrate its MtxNo inputs.
	STNLayer

	// GPLayer represents a globus pallidus layer in the BG, including:
	// GPePr, GPeAk (arkypallidal), and GPi / SNr.
	// Has intrinsic activity.
	GPLayer

	// BGThalLayer represents a BG gated thalamic layer,
	// which receives BG gating in the form of an
	// inhibitory pathway from GPi. Located
	// mainly in the Ventral thalamus: VA / VM / VL,
	// and also parts of MD mediodorsal thalamus.
	BGThalLayer

	// VSGated represents explicit coding of VS gating status:
	// JustGated and HasGated (since last US or failed predicted US),
	// For visualization and / or motor action signaling.
	VSGatedLayer

	//////// Rubicon

	// BLALayer represents a basolateral amygdala layer
	// which learns to associate arbitrary stimuli (CSs)
	// with behaviorally salient outcomes (USs)
	BLALayer

	// CeMLayer represents a central nucleus of the amygdala layer.
	CeMLayer

	// VSPatchLayer represents a ventral striatum patch layer,
	// which learns to represent the expected amount of dopamine reward
	// and projects both directly with shunting inhibition to the VTA
	// and indirectly via the LHb / RMTg to cancel phasic dopamine firing
	// to expected rewards (i.e., reward prediction error).
	VSPatchLayer

	// LHbLayer represents the lateral habenula, which drives dipping
	// in the VTA.  It tracks the Global LHb values for
	// visualization purposes -- updated by VTALayer.
	LHbLayer

	// DrivesLayer represents the Drives in .Rubicon framework.
	// It tracks the Global Drives values for
	// visualization and predictive learning purposes.
	DrivesLayer

	// UrgencyLayer represents the Urgency factor in Rubicon framework.
	// It tracks the Global Urgency.Urge value for
	// visualization and predictive learning purposes.
	UrgencyLayer

	// USLayer represents a US unconditioned stimulus layer (USpos or USneg).
	// It tracks the Global USpos or USneg, for visualization
	// and predictive learning purposes. Actual US inputs are set in Rubicon.
	USLayer

	// PVLayer represents a PV primary value layer (PVpos or PVneg) representing
	// the total primary value as a function of US inputs, drives, and effort.
	// It tracks the Global VTA.PVpos, PVneg values for
	// visualization and predictive learning purposes.
	PVLayer

	// LDTLayer represents the laterodorsal tegmentum layer, which
	// is the primary limbic ACh (acetylcholine) driver to other ACh:
	// BG cholinergic interneurons (CIN) and nucleus basalis ACh areas.
	// The phasic ACh release signals reward salient inputs from CS, US
	// and US omssion, and it drives widespread disinhibition of BG gating
	// and VTA DA firing.
	// It receives excitation from superior colliculus which computes
	// a temporal derivative (stimulus specific adaptation, SSA)
	// of sensory inputs, and inhibitory input from OFC, ACC driving
	// suppression of distracting inputs during goal-engaged states.
	LDTLayer

	// VTALayer represents the ventral tegmental area, which releases
	// dopamine.  It computes final DA value from Rubicon-computed
	// LHb PVDA (primary value DA), updated at start of each trial from
	// updated US, Effort, etc state, and cycle-by-cycle LV learned value
	// state reflecting CS inputs, in the Amygdala (CeM).
	// Its activity reflects this DA level, which is effectively broadcast
	// vial Global state values to all layers.
	VTALayer

	//////// RL

	// RewLayer represents positive (first unit) or negative (second unit)
	// reward values, showing spiking rates for each, and Act always represents
	// the signed value.
	RewLayer

	// RWPredLayer computes reward prediction for a simple Rescorla-Wagner
	// learning dynamic (i.e., PV learning in the Rubicon framework).
	// Activity is computed as linear function of excitatory conductance.
	// The first unit in the layer represents positive reward, second negative.
	// Use with RWPath which does simple delta-rule learning on minus-plus.
	RWPredLayer

	// RWDaLayer computes a dopamine (DA) signal based on a simple Rescorla-Wagner
	// learning dynamic (i.e., PV learning in the Rubicon framework).
	// It computes difference between r(t) and RWPred values.
	// r(t) is accessed directly from a Rew layer -- if no external input then no
	// DA is computed -- critical for effective use of RW only for PV cases.
	// RWPred prediction is also accessed directly from Rew layer to avoid any issues.
	RWDaLayer

	// TDPredLayer is the temporal differences reward prediction layer.
	// It represents estimated value V(t) in the minus phase, and computes
	// estimated V(t+1) based on its learned weights in plus phase,
	// using the TDPredPath pathway type for DA modulated learning.
	// The first unit in the layer represents positive reward, second negative.
	TDPredLayer

	// TDIntegLayer is the temporal differences reward integration layer.
	// It represents estimated value V(t) from prior time step in the minus phase,
	// and estimated discount * V(t+1) + r(t) in the plus phase.
	// It gets Rew, PrevPred from Context.NeuroMod, and Special
	// LayerValues from TDPredLayer.
	// The first unit in the layer represents positive reward, second negative.
	TDIntegLayer

	// TDDaLayer computes a dopamine (DA) signal as the temporal difference (TD)
	// between the TDIntegLayer activations in the minus and plus phase.
	// These are retrieved from Special LayerValues.
	TDDaLayer
)

// IsExtLayerType returns true if the layer type deals with external input:
// Input, Target, Compare
func IsExtLayerType(lt LayerTypes) bool {
	if lt == InputLayer || lt == TargetLayer || lt == CompareLayer || lt == RewLayer {
		return true
	}
	return false
}

//gosl:end

// IsExt returns true if the layer type deals with external input:
// Input, Target, Compare
func (lt LayerTypes) IsExt() bool {
	if lt == InputLayer || lt == TargetLayer || lt == CompareLayer || lt == RewLayer {
		return true
	}
	return false
}
