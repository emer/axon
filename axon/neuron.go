// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"

	"github.com/emer/emergent/v2/netview"
)

//gosl: start neuron

// NeuronFlags are bit-flags encoding relevant binary state for neurons
type NeuronFlags int32 //enums:enum

// The neuron flags
const (
	// NeuronOff flag indicates that this neuron has been turned off (i.e., lesioned)
	NeuronOff NeuronFlags = 1

	// NeuronHasExt means the neuron has external input in its Ext field
	NeuronHasExt NeuronFlags = 2

	// NeuronHasTarg means the neuron has external target input in its Target field
	NeuronHasTarg NeuronFlags = 4

	// NeuronHasCmpr means the neuron has external comparison input in its Target field -- used for computing
	// comparison statistics but does not drive neural activity ever
	NeuronHasCmpr NeuronFlags = 8
)

// NeuronVars are the neuron variables representing current active state,
// specific to each input data state.
// See NeuronAvgVars for vars shared across data.
type NeuronVars int32 //enums:enum

const (
	/////////////////////////////////////////
	// Spiking, Activation

	// Spike is whether neuron has spiked or not on this cycle (0 or 1)
	Spike NeuronVars = iota

	// Spiked is 1 if neuron has spiked within the last 10 cycles (msecs), corresponding to a nominal max spiking rate of 100 Hz, 0 otherwise -- useful for visualization and computing activity levels in terms of average spiked levels.
	Spiked

	// Act is rate-coded activation value reflecting instantaneous estimated rate of spiking, based on 1 / ISIAvg.  This drives feedback inhibition in the FFFB function (todo: this will change when better inhibition is implemented), and is integrated over time for ActInt which is then used for performance statistics and layer average activations, etc.  Should not be used for learning or other computations.
	Act

	// ActInt is integrated running-average activation value computed from Act with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall activation state across the ThetaCycle time scale, as the overall response of network to current input state -- this is copied to ActM and ActP at the ends of the minus and plus phases, respectively, and used in computing performance-level statistics (which are typically based on ActM).  Should not be used for learning or other computations.
	ActInt

	// ActM is ActInt activation state at end of third quarter, representing the posterior-cortical minus phase activation -- used for statistics and monitoring network performance. Should not be used for learning or other computations.
	ActM

	// ActP is ActInt activation state at end of fourth quarter, representing the posterior-cortical plus_phase activation -- used for statistics and monitoring network performance.  Should not be used for learning or other computations.
	ActP

	// Ext is external input: drives activation of unit from outside influences (e.g., sensory input)
	Ext

	// Target is the target value: drives learning to produce this activation value
	Target

	/////////////////////////////////////////
	// Major conductances, Vm

	// Ge is total excitatory conductance, including all forms of excitation (e.g., NMDA) -- does *not* include Gbar.E
	Ge

	// Gi is total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I
	Gi

	// Gk is total potassium conductance, typically reflecting sodium-gated potassium currents involved in adaptation effects -- does *not* include Gbar.K
	Gk

	// Inet is net current produced by all channels -- drives update of Vm
	Inet

	// Vm is membrane potential -- integrates Inet current over time
	Vm

	// VmDend is dendritic membrane potential -- has a slower time constant, is not subject to the VmR reset after spiking
	VmDend

	// ISI is current inter-spike-interval -- counts up since last spike.  Starts at -1 when initialized.
	ISI

	// ISIAvg is average inter-spike-interval -- average time interval between spikes, integrated with ISITau rate constant (relatively fast) to capture something close to an instantaneous spiking rate.  Starts at -1 when initialized, and goes to -2 after first spike, and is only valid after the second spike post-initialization.
	ISIAvg

	/////////////////////////////////////////
	// Calcium for learning

	// CaSpkP is continuous cascaded integration of CaSpkM at PTau time constant (typically 40), representing neuron-level purely spiking version of plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act.
	CaSpkP

	// CaSpkD is continuous cascaded integration CaSpkP at DTau time constant (typically 40), representing neuron-level purely spiking version of minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act.
	CaSpkD

	// CaSyn is spike-driven calcium trace for synapse-level Ca-driven learning: exponential integration of SpikeG * Spike at SynTau time constant (typically 30).  Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for the synaptic trace driving credit assignment in learning. Time constant reflects binding time of Glu to NMDA and Ca buffering postsynaptically, and determines time window where pre * post spiking must overlap to drive learning.
	CaSyn

	// CaSpkM is spike-driven calcium trace used as a neuron-level proxy for synpatic credit assignment factor based on continuous time-integrated spiking: exponential integration of SpikeG * Spike at MTau time constant (typically 5).  Simulates a calmodulin (CaM) like signal at the most abstract level.
	CaSpkM

	// CaSpkPM is minus-phase snapshot of the CaSpkP value -- similar to ActM but using a more directly spike-integrated value.
	CaSpkPM

	// CaLrn is recv neuron calcium signal used to drive temporal error difference component of standard learning rule, combining NMDA (NmdaCa) and spiking-driven VGCC (VgccCaInt) calcium sources (vs. CaSpk* which only reflects spiking component).  This is integrated into CaM, CaP, CaD, and temporal derivative is CaP - CaD (CaMKII - DAPK1).  This approximates the backprop error derivative on net input, but VGCC component adds a proportion of recv activation delta as well -- a balance of both works best.  The synaptic-level trace multiplier provides the credit assignment factor, reflecting coincident activity and potentially integrated over longer multi-trial timescales.
	CaLrn

	// NrnCaM is integrated CaLrn at MTau timescale (typically 5), simulating a calmodulin (CaM) like signal, which then drives CaP, CaD for delta signal driving error-driven learning.
	NrnCaM

	// NrnCaP is cascaded integration of CaM at PTau time constant (typically 40), representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule.
	NrnCaP

	// NrnCaD is cascaded integratoin of CaP at DTau time constant (typically 40), representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule.
	NrnCaD

	// CaDiff is difference between CaP - CaD -- this is the error signal that drives error-driven learning.
	CaDiff

	// Attn is Attentional modulation factor, which can be set by special layers such as the TRC -- multiplies Ge
	Attn

	// RLRate is recv-unit based learning rate multiplier, reflecting the sigmoid derivative computed from the CaSpkD of recv unit, and the normalized difference CaSpkP - CaSpkD / MAX(CaSpkP - CaSpkD).
	RLRate

	/////////////////////////////////////////
	// Stats, aggregate values

	// SpkMaxCa is Ca integrated like CaSpkP but only starting at MaxCycStart cycle, to prevent inclusion of carryover spiking from prior theta cycle trial -- the PTau time constant otherwise results in significant carryover.  This is the input to SpkMax
	SpkMaxCa

	// SpkMax is maximum CaSpkP across one theta cycle time window (max of SpkMaxCa) -- used for specialized algorithms that have more phasic behavior within a single trial, e.g., BG Matrix layer gating.  Also useful for visualization of peak activity of neurons.
	SpkMax

	// SpkPrv is final CaSpkD activation state at end of previous theta cycle.  used for specialized learning mechanisms that operate on delayed sending activations.
	SpkPrv

	// SpkSt1 is the activation state at specific time point within current state processing window (e.g., 50 msec for beta cycle within standard theta cycle), as saved by SpkSt1() function.  Used for example in hippocampus for CA3, CA1 learning
	SpkSt1

	// SpkSt2 is the activation state at specific time point within current state processing window (e.g., 100 msec for beta cycle within standard theta cycle), as saved by SpkSt2() function.  Used for example in hippocampus for CA3, CA1 learning
	SpkSt2

	/////////////////////////////////////////
	// Noise

	// GeNoiseP is accumulating poisson probability factor for driving excitatory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda.
	GeNoiseP

	// GeNoise is integrated noise excitatory conductance, added into Ge
	GeNoise

	// GiNoiseP is accumulating poisson probability factor for driving inhibitory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda.
	GiNoiseP

	// GiNoise is integrated noise inhibotyr conductance, added into Gi
	GiNoise

	/////////////////////////////////////////
	// Ge, Gi integration

	// GeExt is extra excitatory conductance added to Ge -- from Ext input, GeCtxt etc
	GeExt

	// GeRaw is raw excitatory conductance (net input) received from senders = current raw spiking drive
	GeRaw

	// GeSyn is time-integrated total excitatory synaptic conductance, with an instantaneous rise time from each spike (in GeRaw) and exponential decay with Dt.GeTau, aggregated over projections -- does *not* include Gbar.E
	GeSyn

	// GiRaw is raw inhibitory conductance (net input) received from senders  = current raw spiking drive
	GiRaw

	// GiSyn is time-integrated total inhibitory synaptic conductance, with an instantaneous rise time from each spike (in GiRaw) and exponential decay with Dt.GiTau, aggregated over projections -- does *not* include Gbar.I.  This is added with computed FFFB inhibition to get the full inhibition in Gi
	GiSyn

	// GeInt is integrated running-average activation value computed from Ge with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall Ge level across the ThetaCycle time scale (Ge itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive
	GeInt

	// GeIntNorm is normalized GeInt value (divided by the layer maximum) -- this is used for learning in layers that require learning on subthreshold activity
	GeIntNorm

	// GiInt is integrated running-average activation value computed from GiSyn with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall synaptic Gi level across the ThetaCycle time scale (Gi itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall inhibitory drive
	GiInt

	// GModRaw is raw modulatory conductance, received from GType = ModulatoryG projections
	GModRaw

	// GModSyn is syn integrated modulatory conductance, received from GType = ModulatoryG projections
	GModSyn

	// GMaintRaw is raw maintenance conductance, received from GType = MaintG projections
	GMaintRaw

	// GMaintSyn is syn integrated maintenance conductance, integrated using MaintNMDA params.
	GMaintSyn

	/////////////////////////////////////////
	// SST somatostatin inhibition factors

	// SSGi is SST+ somatostatin positive slow spiking inhibition
	SSGi

	// SSGiDend is amount of SST+ somatostatin positive slow spiking inhibition applied to dendritic Vm (VmDend)
	SSGiDend

	// Gak is conductance of A-type K potassium channels
	Gak

	/////////////////////////////////////////
	// AHP channels: Mahp, Sahp, Gkna

	// MahpN is accumulating voltage-gated gating value for the medium time scale AHP
	MahpN

	// SahpCa is slowly accumulating calcium value that drives the slow AHP
	SahpCa

	// SahpN is sAHP gating value
	SahpN

	// GknaMed is conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick) -- produces accommodation / adaptation of firing
	GknaMed

	// GknaSlow is conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack) -- produces accommodation / adaptation of firing
	GknaSlow

	/////////////////////////////////////////
	// NMDA channels

	// GnmdaSyn is integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant
	GnmdaSyn

	// Gnmda is net postsynaptic (recv) NMDA conductance, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential
	Gnmda

	// GnmdaMaint is net postsynaptic maintenance NMDA conductance, computed from GMaintSyn and GMaintRaw, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential
	GnmdaMaint

	// GnmdaLrn is learning version of integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant -- drives NmdaCa that then drives CaM for learning
	GnmdaLrn

	// NmdaCa is NMDA calcium computed from GnmdaLrn, drives learning via CaM
	NmdaCa

	/////////////////////////////////////////
	// GABA channels

	// GgabaB is net GABA-B conductance, after Vm gating and Gbar + Gbase -- applies to Gk, not Gi, for GIRK, with .1 reversal potential.
	GgabaB

	// GABAB is GABA-B / GIRK activation -- time-integrated value with rise and decay time constants
	GABAB

	// GABABx is GABA-B / GIRK internal drive variable -- gets the raw activation and decays
	GABABx

	/////////////////////////////////////////
	//  VGCC voltage gated calcium channels

	// Gvgcc is conductance (via Ca) for VGCC voltage gated calcium channels
	Gvgcc

	// VgccM is activation gate of VGCC channels
	VgccM

	// VgccH inactivation gate of VGCC channels
	VgccH

	// VgccCa is instantaneous VGCC calcium flux -- can be driven by spiking or directly from Gvgcc
	VgccCa

	// VgccCaInt time-integrated VGCC calcium flux -- this is actually what drives learning
	VgccCaInt

	/////////////////////////////////////////
	//  SKCa small conductance calcium-gated potassium channels

	// SKCaIn is intracellular calcium store level, available to be released with spiking as SKCaR, which can bind to SKCa receptors and drive K current. replenishment is a function of spiking activity being below a threshold
	SKCaIn

	// SKCaR released amount of intracellular calcium, from SKCaIn, as a function of spiking events.  this can bind to SKCa channels and drive K currents.
	SKCaR

	// SKCaM is Calcium-gated potassium channel gating factor, driven by SKCaR via a Hill equation as in chans.SKPCaParams.
	SKCaM

	// Gsk is Calcium-gated potassium channel conductance as a function of Gbar * SKCaM.
	Gsk

	/////////////////////////////////////////
	//  Special Layer Vars

	// Burst is 5IB bursting activation value, computed by thresholding regular CaSpkP value in Super superficial layers
	Burst

	// BurstPrv is previous Burst bursting activation from prior time step -- used for context-based learning
	BurstPrv

	// CtxtGe is context (temporally delayed) excitatory conductance, driven by deep bursting at end of the plus phase, for CT layers.
	CtxtGe

	// CtxtGeRaw is raw update of context (temporally delayed) excitatory conductance, driven by deep bursting at end of the plus phase, for CT layers.
	CtxtGeRaw

	// CtxtGeOrig is original CtxtGe value prior to any decay factor -- updates at end of plus phase.
	CtxtGeOrig

	// NrnFlags are bit flags for binary state variables, which are converted to / from uint32.
	// These need to be in Vars because they can be differential per data (for ext inputs)
	// and are writable (indexes are read only).
	NrnFlags

	// IMPORTANT: if NrnFlags is not the last, need to update gosl defn below
)

// NeuronAvgVars are mostly neuron variables involved in longer-term average activity
// which is aggregated over time and not specific to each input data state,
// along with any other state that is not input data specific.
type NeuronAvgVars int32 //enums:enum

const (
	// ActAvg is average activation (of minus phase activation state) over long time intervals (time constant = Dt.LongAvgTau) -- useful for finding hog units and seeing overall distribution of activation
	ActAvg NeuronAvgVars = iota

	// AvgPct is ActAvg as a proportion of overall layer activation -- this is used for synaptic scaling to match TrgAvg activation -- updated at SlowInterval intervals
	AvgPct

	// TrgAvg is neuron's target average activation as a proportion of overall layer activation, assigned during weight initialization, driving synaptic scaling relative to AvgPct
	TrgAvg

	// DTrgAvg is change in neuron's target average activation as a result of unit-wise error gradient -- acts like a bias weight.  MPI needs to share these across processors.
	DTrgAvg

	// AvgDif is AvgPct - TrgAvg -- i.e., the error in overall activity level relative to set point for this neuron, which drives synaptic scaling -- updated at SlowInterval intervals
	AvgDif

	// GeBase is baseline level of Ge, added to GeRaw, for intrinsic excitability
	GeBase

	// GiBase is baseline level of Gi, added to GiRaw, for intrinsic excitability
	GiBase

	// IMPORTANT: if GiBase is not the last, need to update gosl defn below
)

// NeuronIdxs are the neuron indexes and other uint32 values.
// There is only one of these per neuron -- not data parallel.
// note: Flags are encoded in Vars because they are data parallel and
// writable, whereas indexes are read-only.
type NeuronIdxs int32 //enums:enum

const (
	// NrnNeurIdx is the index of this neuron within its owning layer
	NrnNeurIdx NeuronIdxs = iota

	// NrnLayIdx is the index of the layer that this neuron belongs to,
	// needed for neuron-level parallel code.
	NrnLayIdx

	// NrnSubPool is the index of the sub-level inhibitory pool for this neuron
	// (only for 4D shapes, the pool (unit-group / hypercolumn) structure level).
	// Indicies start at 1 -- 0 is layer-level pool (is 0 if no sub-pools).
	NrnSubPool

	// IMPORTANT: if NrnSubPool is not the last, need to update gosl defn below
)

//gosl: end neuron

//gosl: hlsl neuron
/*
static const NeuronVars NeuronVarsN = NrnFlags + 1;
static const NeuronAvgVars NeuronAvgVarsN = GiBase + 1;
static const NeuronIdxs NeuronIdxsN = NrnSubPool + 1;
*/
//gosl: end neuron

//gosl: start neuron

////////////////////////////////////////////////
// 	Strides

// NeuronVarStrides encodes the stride offsets for neuron variable access
// into network float32 array.  Data is always the inner-most variable.
type NeuronVarStrides struct {

	// neuron level
	Neuron uint32

	// variable level
	Var uint32

	pad, pad1 uint32
}

// Idx returns the index into network float32 array for given neuron, data, and variable
func (ns *NeuronVarStrides) Idx(neurIdx, di uint32, nvar NeuronVars) uint32 {
	return neurIdx*ns.Neuron + uint32(nvar)*ns.Var + di
}

// SetNeuronOuter sets strides with neurons as outer loop:
// [Neurons][Vars][Data], which is optimal for CPU-based computation.
func (ns *NeuronVarStrides) SetNeuronOuter(ndata int) {
	ns.Neuron = uint32(ndata) * uint32(NeuronVarsN)
	ns.Var = uint32(ndata)
}

// SetVarOuter sets strides with vars as outer loop:
// [Vars][Neurons][Data], which is optimal for GPU-based computation.
func (ns *NeuronVarStrides) SetVarOuter(nneur, ndata int) {
	ns.Var = uint32(ndata) * uint32(nneur)
	ns.Neuron = uint32(ndata)
}

////////////////////////////////////////////////
// 	NeuronAvgVars

// NeuronAvgVarStrides encodes the stride offsets for neuron variable access
// into network float32 array.  Data is always the inner-most variable.
type NeuronAvgVarStrides struct {

	// neuron level
	Neuron uint32

	// variable level
	Var uint32

	pad, pad1 uint32
}

// Idx returns the index into network float32 array for given neuron and variable
func (ns *NeuronAvgVarStrides) Idx(neurIdx uint32, nvar NeuronAvgVars) uint32 {
	return neurIdx*ns.Neuron + uint32(nvar)*ns.Var
}

// SetNeuronOuter sets strides with neurons as outer loop:
// [Neurons][Vars], which is optimal for CPU-based computation.
func (ns *NeuronAvgVarStrides) SetNeuronOuter() {
	ns.Neuron = uint32(NeuronAvgVarsN)
	ns.Var = 1
}

// SetVarOuter sets strides with vars as outer loop:
// [Vars][Neurons], which is optimal for GPU-based computation.
func (ns *NeuronAvgVarStrides) SetVarOuter(nneur int) {
	ns.Var = uint32(nneur)
	ns.Neuron = 1
}

////////////////////////////////////////////////
// 	Idxs

// NeuronIdxStrides encodes the stride offsets for neuron index access
// into network uint32 array.
type NeuronIdxStrides struct {

	// neuron level
	Neuron uint32

	// index value level
	Index uint32

	pad, pad1 uint32
}

// Idx returns the index into network uint32 array for given neuron, index value
func (ns *NeuronIdxStrides) Idx(neurIdx uint32, idx NeuronIdxs) uint32 {
	return neurIdx*ns.Neuron + uint32(idx)*ns.Index
}

// SetNeuronOuter sets strides with neurons as outer dimension:
// [Neurons[[Idxs] (outer to inner), which is optimal for CPU-based
// computation.
func (ns *NeuronIdxStrides) SetNeuronOuter() {
	ns.Neuron = uint32(NeuronIdxsN)
	ns.Index = 1
}

// SetIdxOuter sets strides with indexes as outer dimension:
// [Idxs][Neurons] (outer to inner), which is optimal for GPU-based
// computation.
func (ns *NeuronIdxStrides) SetIdxOuter(nneur int) {
	ns.Index = uint32(nneur)
	ns.Neuron = 1
}

//gosl: end neuron

////////////////////////////////////////////////
// 	Props

// NeuronVarProps has all of the display properties for neuron variables, including desc tooltips
var NeuronVarProps = map[string]string{
	/////////////////////////////////////////
	// Spiking, Activation

	"Spike":  `desc:"whether neuron has spiked or not on this cycle (0 or 1)"`,
	"Spiked": `desc:"1 if neuron has spiked within the last 10 cycles (msecs), corresponding to a nominal max spiking rate of 100 Hz, 0 otherwise -- useful for visualization and computing activity levels in terms of average spiked levels."`,
	"Act":    `desc:"rate-coded activation value reflecting instantaneous estimated rate of spiking, based on 1 / ISIAvg.  This drives feedback inhibition in the FFFB function (todo: this will change when better inhibition is implemented), and is integrated over time for ActInt which is then used for performance statistics and layer average activations, etc.  Should not be used for learning or other computations."`,
	"ActInt": `desc:"integrated running-average activation value computed from Act with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall activation state across the ThetaCycle time scale, as the overall response of network to current input state -- this is copied to ActM and ActP at the ends of the minus and plus phases, respectively, and used in computing performance-level statistics (which are typically based on ActM).  Should not be used for learning or other computations."`,
	"ActM":   `desc:"ActInt activation state at end of third quarter, representing the posterior-cortical minus phase activation -- used for statistics and monitoring network performance. Should not be used for learning or other computations."`,
	"ActP":   `desc:"ActInt activation state at end of fourth quarter, representing the posterior-cortical plus_phase activation -- used for statistics and monitoring network performance.  Should not be used for learning or other computations."`,
	"Ext":    `desc:"external input: drives activation of unit from outside influences (e.g., sensory input)"`,
	"Target": `desc:"target value: drives learning to produce this activation value"`,

	/////////////////////////////////////////
	// Major conductances, Vm

	"Ge":     `range:"2" desc:"total excitatory conductance, including all forms of excitation (e.g., NMDA) -- does *not* include Gbar.E"`,
	"Gi":     `auto-scale:"+" desc:"total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I"`,
	"Gk":     `auto-scale:"+" desc:"total potassium conductance, typically reflecting sodium-gated potassium currents involved in adaptation effects -- does *not* include Gbar.K"`,
	"Inet":   `desc:"net current produced by all channels -- drives update of Vm"`,
	"Vm":     `min:"0" max:"1" desc:"membrane potential -- integrates Inet current over time"`,
	"VmDend": `min:"0" max:"1" desc:"dendritic membrane potential -- has a slower time constant, is not subject to the VmR reset after spiking"`,
	"ISI":    `auto-scale:"+" desc:"current inter-spike-interval -- counts up since last spike.  Starts at -1 when initialized."`,
	"ISIAvg": `auto-scale:"+" desc:"average inter-spike-interval -- average time interval between spikes, integrated with ISITau rate constant (relatively fast) to capture something close to an instantaneous spiking rate.  Starts at -1 when initialized, and goes to -2 after first spike, and is only valid after the second spike post-initialization."`,

	/////////////////////////////////////////
	// Calcium for learning

	"CaSyn":   `desc:"spike-driven calcium trace for synapse-level Ca-driven learning: exponential integration of SpikeG * Spike at SynTau time constant (typically 30).  Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for the synaptic trace driving credit assignment in learning. Time constant reflects binding time of Glu to NMDA and Ca buffering postsynaptically, and determines time window where pre * post spiking must overlap to drive learning."`,
	"CaSpkM":  `desc:"spike-driven calcium trace used as a neuron-level proxy for synpatic credit assignment factor based on continuous time-integrated spiking: exponential integration of SpikeG * Spike at MTau time constant (typically 5).  Simulates a calmodulin (CaM) like signal at the most abstract level."`,
	"CaSpkP":  `desc:"continuous cascaded integration of CaSpkM at PTau time constant (typically 40), representing neuron-level purely spiking version of plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act."`,
	"CaSpkD":  `desc:"continuous cascaded integration CaSpkP at DTau time constant (typically 40), representing neuron-level purely spiking version of minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act."`,
	"CaSpkPM": `desc:"minus-phase snapshot of the CaSpkP value -- similar to ActM but using a more directly spike-integrated value."`,
	"CaLrn":   `desc:"recv neuron calcium signal used to drive temporal error difference component of standard learning rule, combining NMDA (NmdaCa) and spiking-driven VGCC (VgccCaInt) calcium sources (vs. CaSpk* which only reflects spiking component).  This is integrated into CaM, CaP, CaD, and temporal derivative is CaP - CaD (CaMKII - DAPK1).  This approximates the backprop error derivative on net input, but VGCC component adds a proportion of recv activation delta as well -- a balance of both works best.  The synaptic-level trace multiplier provides the credit assignment factor, reflecting coincident activity and potentially integrated over longer multi-trial timescales."`,
	"NrnCaM":  `desc:"integrated CaLrn at MTau timescale (typically 5), simulating a calmodulin (CaM) like signal, which then drives CaP, CaD for delta signal driving error-driven learning."`,
	"NrnCaP":  `desc:"cascaded integration of CaM at PTau time constant (typically 40), representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule."`,
	"NrnCaD":  `desc:"cascaded integratoin of CaP at DTau time constant (typically 40), representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule."`,
	"CaDiff":  `desc:"difference between CaP - CaD -- this is the error signal that drives error-driven learning."`,
	"RLRate":  `auto-scale:"+" desc:"recv-unit based learning rate multiplier, reflecting the sigmoid derivative computed from the CaSpkD of recv unit, and the normalized difference CaSpkP - CaSpkD / MAX(CaSpkP - CaSpkD)."`,
	"Attn":    `desc:"Attentional modulation factor, which can be set by special layers such as the TRC -- multiplies Ge"`,

	/////////////////////////////////////////
	// Stats, aggregate values

	"SpkMaxCa": `desc:"Ca integrated like CaSpkP but only starting at MaxCycStart cycle, to prevent inclusion of carryover spiking from prior theta cycle trial -- the PTau time constant otherwise results in significant carryover.  This is the input to SpkMax"`,
	"SpkMax":   `desc:"maximum CaSpkP across one theta cycle time window (max of SpkMaxCa) -- used for specialized algorithms that have more phasic behavior within a single trial, e.g., BG Matrix layer gating.  Also useful for visualization of peak activity of neurons."`,
	"SpkPrv":   `desc:"final CaSpkD activation state at end of previous theta cycle.  used for specialized learning mechanisms that operate on delayed sending activations."`,
	"SpkSt1":   `desc:"the activation state at specific time point within current state processing window (e.g., 50 msec for beta cycle within standard theta cycle), as saved by SpkSt1() function.  Used for example in hippocampus for CA3, CA1 learning"`,
	"SpkSt2":   `desc:"the activation state at specific time point within current state processing window (e.g., 100 msec for beta cycle within standard theta cycle), as saved by SpkSt2() function.  Used for example in hippocampus for CA3, CA1 learning"`,
	"DASign":   `desc:"sign of dopamine-based learning effects for this neuron -- 1 = D1, -1 = D2"`,

	/////////////////////////////////////////
	// Noise

	"GeNoiseP": `desc:"accumulating poisson probability factor for driving excitatory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda."`,
	"GeNoise":  `desc:"integrated noise excitatory conductance, added into Ge"`,
	"GiNoiseP": `desc:"accumulating poisson probability factor for driving inhibitory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda."`,
	"GiNoise":  `desc:"integrated noise inhibotyr conductance, added into Gi"`,

	/////////////////////////////////////////
	// Ge, Gi integration

	"GeExt":     `desc:"extra excitatory conductance added to Ge -- from Ext input, GeCtxt etc"`,
	"GeRaw":     `desc:"raw excitatory conductance (net input) received from senders = current raw spiking drive"`,
	"GeSyn":     `range:"2" desc:"time-integrated total excitatory synaptic conductance, with an instantaneous rise time from each spike (in GeRaw) and exponential decay with Dt.GeTau, aggregated over projections -- does *not* include Gbar.E"`,
	"GiRaw":     `desc:"raw inhibitory conductance (net input) received from senders  = current raw spiking drive"`,
	"GiSyn":     `desc:"time-integrated total inhibitory synaptic conductance, with an instantaneous rise time from each spike (in GiRaw) and exponential decay with Dt.GiTau, aggregated over projections -- does *not* include Gbar.I.  This is added with computed FFFB inhibition to get the full inhibition in Gi"`,
	"GeInt":     `range:"2" desc:"integrated running-average activation value computed from Ge with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall Ge level across the ThetaCycle time scale (Ge itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive"`,
	"GeIntNorm": `range:"1" desc:"GeIntNorm is normalized GeInt value (divided by the layer maximum) -- this is used for learning in layers that require learning on subthreshold activity."`,
	"GiInt":     `range:"2" desc:"integrated running-average activation value computed from GiSyn with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall synaptic Gi level across the ThetaCycle time scale (Gi itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall inhibitory drive"`,
	"GModRaw":   `desc:"raw modulatory conductance, received from GType = ModulatoryG projections"`,
	"GModSyn":   `desc:"syn integrated modulatory conductance, received from GType = ModulatoryG projections"`,
	"GMaintRaw": `desc:"raw maintenance conductance, received from GType = MaintG projections"`,
	"GMaintSyn": `desc:"syn integrated maintenance conductance, integrated using MaintNMDA params."`,

	/////////////////////////////////////////
	// SST somatostatin inhibition factors

	"SSGi":     `auto-scale:"+" desc:"SST+ somatostatin positive slow spiking inhibition"`,
	"SSGiDend": `auto-scale:"+" desc:"amount of SST+ somatostatin positive slow spiking inhibition applied to dendritic Vm (VmDend)"`,
	"Gak":      `auto-scale:"+" desc:"conductance of A-type K potassium channels"`,

	/////////////////////////////////////////
	// AHP channels: Mahp, Sahp, Gkna

	"MahpN":    `auto-scale:"+" desc:"accumulating voltage-gated gating value for the medium time scale AHP"`,
	"SahpCa":   `desc:"slowly accumulating calcium value that drives the slow AHP"`,
	"SahpN":    `desc:"sAHP gating value"`,
	"GknaMed":  `auto-scale:"+" desc:"conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick) -- produces accommodation / adaptation of firing"`,
	"GknaSlow": `auto-scale:"+" desc:"conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack) -- produces accommodation / adaptation of firing"`,

	/////////////////////////////////////////
	// NMDA channels

	"GnmdaSyn":   `auto-scale:"+" desc:"integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant"`,
	"Gnmda":      `auto-scale:"+" desc:"net postsynaptic (recv) NMDA conductance, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential"`,
	"GnmdaMaint": `auto-scale:"+" desc:"net postsynaptic maintenance NMDA conductance, computed from GMaintSyn and GMaintRaw, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential"`,
	"GnmdaLrn":   `auto-scale:"+" desc:"learning version of integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant -- drives NmdaCa that then drives CaM for learning"`,
	"NmdaCa":     `auto-scale:"+" desc:"NMDA calcium computed from GnmdaLrn, drives learning via CaM"`,

	/////////////////////////////////////////
	// GABA channels

	"GgabaB": `auto-scale:"+" desc:"net GABA-B conductance, after Vm gating and Gbar + Gbase -- applies to Gk, not Gi, for GIRK, with .1 reversal potential."`,
	"GABAB":  `auto-scale:"+" desc:"GABA-B / GIRK activation -- time-integrated value with rise and decay time constants"`,
	"GABABx": `auto-scale:"+" desc:"GABA-B / GIRK internal drive variable -- gets the raw activation and decays"`,

	/////////////////////////////////////////
	//  VGCC voltage gated calcium channels

	"Gvgcc":     `auto-scale:"+" desc:"conductance (via Ca) for VGCC voltage gated calcium channels"`,
	"VgccM":     `desc:"activation gate of VGCC channels"`,
	"VgccH":     `desc:"inactivation gate of VGCC channels"`,
	"VgccCa":    `auto-scale:"+" desc:"instantaneous VGCC calcium flux -- can be driven by spiking or directly from Gvgcc"`,
	"VgccCaInt": `auto-scale:"+" desc:"time-integrated VGCC calcium flux -- this is actually what drives learning"`,

	/////////////////////////////////////////
	//  SKCa small conductance calcium-gated potassium channels

	"SKCaIn": `desc:"intracellular calcium store level, available to be released with spiking as SKCaR, which can bind to SKCa receptors and drive K current. replenishment is a function of spiking activity being below a threshold"`,
	"SKCaR":  `desc:"released amount of intracellular calcium, from SKCaIn, as a function of spiking events.  this can bind to SKCa channels and drive K currents."`,
	"SKCaM":  `desc:"Calcium-gated potassium channel gating factor, driven by SKCaR via a Hill equation as in chans.SKPCaParams."`,
	"Gsk":    `desc:"Calcium-gated potassium channel conductance as a function of Gbar * SKCaM."`,

	/////////////////////////////////////////
	//  Special Layer Vars

	"Burst":      `desc:"5IB bursting activation value, computed by thresholding regular CaSpkP value in Super superficial layers"`,
	"BurstPrv":   `desc:"previous Burst bursting activation from prior time step -- used for context-based learning"`,
	"CtxtGe":     `desc:"context (temporally delayed) excitatory conductance, driven by deep bursting at end of the plus phase, for CT layers."`,
	"CtxtGeRawa": `desc:"raw update of context (temporally delayed) excitatory conductance, driven by deep bursting at end of the plus phase, for CT layers."`,
	"CtxtGeOrig": `desc:"original CtxtGe value prior to any decay factor -- updates at end of plus phase."`,

	"NrnFlags": `view:"-" desc:"bit flags for external input and other neuron status state"`,

	/////////////////////////////////////////
	// Long-term average activation, set point for synaptic scaling

	"ActAvg":  `desc:"average activation (of minus phase activation state) over long time intervals (time constant = Dt.LongAvgTau) -- useful for finding hog units and seeing overall distribution of activation"`,
	"AvgPct":  `range:"2" desc:"ActAvg as a proportion of overall layer activation -- this is used for synaptic scaling to match TrgAvg activation -- updated at SlowInterval intervals"`,
	"TrgAvg":  `range:"2" desc:"neuron's target average activation as a proportion of overall layer activation, assigned during weight initialization, driving synaptic scaling relative to AvgPct"`,
	"DTrgAvg": `auto-scale:"+" desc:"change in neuron's target average activation as a result of unit-wise error gradient -- acts like a bias weight.  MPI needs to share these across processors."`,
	"AvgDif":  `desc:"AvgPct - TrgAvg -- i.e., the error in overall activity level relative to set point for this neuron, which drives synaptic scaling -- updated at SlowInterval intervals"`,
	"GeBase":  `desc:"baseline level of Ge, added to GeRaw, for intrinsic excitability"`,
	"GiBase":  `desc:"baseline level of Gi, added to GiRaw, for intrinsic excitability"`,
}

var (
	NeuronVarNames []string
	NeuronVarsMap  map[string]int
)

// NeuronLayerVars are layer-level variables displayed as neuron layers.
var (
	NeuronLayerVars  = []string{"DA", "ACh", "NE", "Ser", "Gated"}
	NNeuronLayerVars = len(NeuronLayerVars)
)

func init() {
	netview.NVarCols = 4 // many neurons
	NeuronVarsMap = make(map[string]int, int(NeuronVarsN)+int(NeuronAvgVarsN)+NNeuronLayerVars)
	for i := Spike; i < NeuronVarsN; i++ {
		vnm := i.String()
		NeuronVarNames = append(NeuronVarNames, vnm)
		NeuronVarsMap[vnm] = int(i)
	}
	for i := ActAvg; i < NeuronAvgVarsN; i++ {
		vnm := i.String()
		NeuronVarNames = append(NeuronVarNames, vnm)
		NeuronVarsMap[vnm] = int(NeuronVarsN) + int(i)
	}
	for i, vnm := range NeuronLayerVars {
		NeuronVarNames = append(NeuronVarNames, vnm)
		NeuronVarsMap[vnm] = i + int(NeuronVarsN) + int(NeuronAvgVarsN)
	}
}

// NeuronVarIdxByName returns the index of the variable in the Neuron, or error
func NeuronVarIdxByName(varNm string) (int, error) {
	i, ok := NeuronVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Neuron VarByName: variable name: %s not valid", varNm)
	}
	return i, nil
}
