// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"

	"github.com/emer/emergent/v2/emer"
)

//gosl:start neuron

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

	// Ext is external input: drives activation of unit from outside influences (e.g., sensory input)
	Ext

	// Target is the target value: drives learning to produce this activation value
	Target

	// NrnFlags are bit flags for binary state variables, which are converted to / from uint32.
	// These need to be in Vars because they can be differential per data (for ext inputs)
	// and are writable (indexes are read only).
	NrnFlags

	/////////////////////////////////////////
	// Calcium for learning

	// CaSpkM is spike-driven calcium trace used as a neuron-level proxy for synpatic credit assignment factor based on continuous time-integrated spiking: exponential integration of SpikeG * Spike at MTau time constant (typically 5).  Simulates a calmodulin (CaM) like signal at the most abstract level.
	CaSpkM

	// CaSpkP is continuous cascaded integration of CaSpkM at PTau time constant (typically 40), representing neuron-level purely spiking version of plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act.
	CaSpkP

	// CaSpkD is continuous cascaded integration CaSpkP at DTau time constant (typically 40), representing neuron-level purely spiking version of minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act.
	CaSpkD

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

	// RLRate is recv-unit based learning rate multiplier, reflecting the sigmoid derivative computed from the CaSpkD of recv unit, and the normalized difference CaSpkP - CaSpkD / MAX(CaSpkP - CaSpkD).
	RLRate

	/////////////////////////////////////////
	// NMDA channels

	// GnmdaSyn is integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant
	GnmdaSyn

	// Gnmda is net postsynaptic (recv) NMDA conductance, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential
	Gnmda

	// GnmdaLrn is learning version of integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant -- drives NmdaCa that then drives CaM for learning
	GnmdaLrn

	// GnmdaMaint is net postsynaptic maintenance NMDA conductance, computed from GMaintSyn and GMaintRaw, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential
	GnmdaMaint

	// NmdaCa is NMDA calcium computed from GnmdaLrn, drives learning via CaM
	NmdaCa

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

	/////////////////////////////////////////
	// GABA channels

	// GgabaB is net GABA-B conductance, after Vm gating and Gbar + Gbase -- applies to Gk, not Gi, for GIRK, with .1 reversal potential.
	GgabaB

	// GABAB is GABA-B / GIRK activation -- time-integrated value with rise and decay time constants
	GABAB

	// GABABx is GABA-B / GIRK internal drive variable -- gets the raw activation and decays
	GABABx

	/////////////////////////////////////////
	// SST somatostatin inhibition factors

	// Gak is conductance of A-type K potassium channels
	Gak

	// SSGi is SST+ somatostatin positive slow spiking inhibition
	SSGi

	// SSGiDend is amount of SST+ somatostatin positive slow spiking inhibition applied to dendritic Vm (VmDend)
	SSGiDend

	/////////////////////////////////////////
	// AHP channels: Mahp, Sahp, Gkna

	// MahpN is accumulating voltage-gated gating value for the medium time scale AHP
	MahpN

	// Gmahp is medium time scale AHP conductance
	Gmahp

	// SahpCa is slowly accumulating calcium value that drives the slow AHP
	SahpCa

	// SahpN is the sAHP gating value
	SahpN

	// Gsahp is slow time scale AHP conductance
	Gsahp

	// GknaMed is conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick), which produces accommodation / adaptation of firing
	GknaMed

	// GknaSlow is conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack), which produces accommodation / adaptation of firing
	GknaSlow

	// KirM is the Kir potassium (K) inwardly rectifying gating value
	KirM

	// Gkir is the conductance of the potassium (K) inwardly rectifying channel,
	// which is strongest at low membrane potentials.  Can be modulated by DA.
	Gkir

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
	// Stats, aggregate values

	// ActM is ActInt activation state at end of third quarter, representing the posterior-cortical minus phase activation -- used for statistics and monitoring network performance. Should not be used for learning or other computations.
	ActM

	// ActP is ActInt activation state at end of fourth quarter, representing the posterior-cortical plus_phase activation -- used for statistics and monitoring network performance.  Should not be used for learning or other computations.
	ActP

	// SpkSt1 is the activation state at specific time point within current state processing window (e.g., 50 msec for beta cycle within standard theta cycle), as saved by SpkSt1() function.  Used for example in hippocampus for CA3, CA1 learning
	SpkSt1

	// SpkSt2 is the activation state at specific time point within current state processing window (e.g., 100 msec for beta cycle within standard theta cycle), as saved by SpkSt2() function.  Used for example in hippocampus for CA3, CA1 learning
	SpkSt2

	// SpkMax is maximum CaSpkP across one theta cycle time window (max of SpkMaxCa) -- used for specialized algorithms that have more phasic behavior within a single trial, e.g., BG Matrix layer gating.  Also useful for visualization of peak activity of neurons.
	SpkMax

	// SpkMaxCa is Ca integrated like CaSpkP but only starting at MaxCycStart cycle, to prevent inclusion of carryover spiking from prior theta cycle trial -- the PTau time constant otherwise results in significant carryover.  This is the input to SpkMax
	SpkMaxCa

	// SpkBin has aggregated spikes within 50 msec bins across the theta cycle, for computing synaptic calcium efficiently
	SpkBin0
	SpkBin1
	SpkBin2
	SpkBin3
	SpkBin4
	SpkBin5
	SpkBin6
	SpkBin7

	// SpkPrv is final CaSpkD activation state at end of previous theta cycle.  used for specialized learning mechanisms that operate on delayed sending activations.
	SpkPrv

	/////////////////////////////////////////
	// Noise

	// GeNoise is integrated noise excitatory conductance, added into Ge
	GeNoise

	// GeNoiseP is accumulating poisson probability factor for driving excitatory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on poisson lambda as function of noise firing rate.
	GeNoiseP

	// GiNoise is integrated noise inhibotyr conductance, added into Gi
	GiNoise

	// GiNoiseP is accumulating poisson probability factor for driving inhibitory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on poisson lambda as a function of noise firing rate.
	GiNoiseP

	/////////////////////////////////////////
	// Ge, Gi integration

	// GeExt is extra excitatory conductance added to Ge -- from Ext input, GeCtxt etc
	GeExt

	// GeRaw is raw excitatory conductance (net input) received from senders = current raw spiking drive
	GeRaw

	// GeSyn is time-integrated total excitatory synaptic conductance, with an instantaneous rise time from each spike (in GeRaw) and exponential decay with Dt.GeTau, aggregated over pathways -- does *not* include Gbar.E
	GeSyn

	// GiRaw is raw inhibitory conductance (net input) received from senders  = current raw spiking drive
	GiRaw

	// GiSyn is time-integrated total inhibitory synaptic conductance, with an instantaneous rise time from each spike (in GiRaw) and exponential decay with Dt.GiTau, aggregated over pathways -- does *not* include Gbar.I.  This is added with computed FFFB inhibition to get the full inhibition in Gi
	GiSyn

	// SMaintP is accumulating poisson probability factor for driving self-maintenance by simulating a population of mutually interconnected neurons.  multiply times uniform random deviate at each time step, until it gets below the target threshold based on poisson lambda based on accumulating self maint factor
	SMaintP

	// GeInt is integrated running-average activation value computed from Ge with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall Ge level across the ThetaCycle time scale (Ge itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive
	GeInt

	// GeIntNorm is normalized GeInt value (divided by the layer maximum) -- this is used for learning in layers that require learning on subthreshold activity
	GeIntNorm

	// GiInt is integrated running-average activation value computed from GiSyn with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall synaptic Gi level across the ThetaCycle time scale (Gi itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall inhibitory drive
	GiInt

	// GModRaw is raw modulatory conductance, received from GType = ModulatoryG pathways
	GModRaw

	// GModSyn is syn integrated modulatory conductance, received from GType = ModulatoryG pathways
	GModSyn

	// GMaintRaw is raw maintenance conductance, received from GType = MaintG pathways
	GMaintRaw

	// GMaintSyn is syn integrated maintenance conductance, integrated using MaintNMDA params.
	GMaintSyn

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

// NeuronIndexes are the neuron indexes and other uint32 values.
// There is only one of these per neuron -- not data parallel.
// note: Flags are encoded in Vars because they are data parallel and
// writable, whereas indexes are read-only.
type NeuronIndexes int32 //enums:enum

const (
	// NrnNeurIndex is the index of this neuron within its owning layer
	NrnNeurIndex NeuronIndexes = iota

	// NrnLayIndex is the index of the layer that this neuron belongs to,
	// needed for neuron-level parallel code.
	NrnLayIndex

	// NrnSubPool is the index of the sub-level inhibitory pool for this neuron
	// (only for 4D shapes, the pool (unit-group / hypercolumn) structure level).
	// Indicies start at 1 -- 0 is layer-level pool (is 0 if no sub-pools).
	NrnSubPool

	// IMPORTANT: if NrnSubPool is not the last, need to update gosl defn below
)

//gosl:end neuron

//gosl:hlsl neuron
/*
static const NeuronVars NeuronVarsN = NrnFlags + 1;
static const NeuronAvgVars NeuronAvgVarsN = GiBase + 1;
static const NeuronIndexes NeuronIndexesN = NrnSubPool + 1;
*/
//gosl:end neuron

//gosl:start neuron

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

// Index returns the index into network float32 array for given neuron, data, and variable
func (ns *NeuronVarStrides) Index(neurIndex, di uint32, nvar NeuronVars) uint32 {
	return neurIndex*ns.Neuron + uint32(nvar)*ns.Var + di
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

// Index returns the index into network float32 array for given neuron and variable
func (ns *NeuronAvgVarStrides) Index(neurIndex uint32, nvar NeuronAvgVars) uint32 {
	return neurIndex*ns.Neuron + uint32(nvar)*ns.Var
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
// 	Indexes

// NeuronIndexStrides encodes the stride offsets for neuron index access
// into network uint32 array.
type NeuronIndexStrides struct {

	// neuron level
	Neuron uint32

	// index value level
	Idx uint32

	pad, pad1 uint32
}

// Index returns the index into network uint32 array for given neuron, index value
func (ns *NeuronIndexStrides) Index(neurIdx uint32, idx NeuronIndexes) uint32 {
	return neurIdx*ns.Neuron + uint32(idx)*ns.Idx
}

// SetNeuronOuter sets strides with neurons as outer dimension:
// [Neurons[[Indexes] (outer to inner), which is optimal for CPU-based
// computation.
func (ns *NeuronIndexStrides) SetNeuronOuter() {
	ns.Neuron = uint32(NeuronIndexesN)
	ns.Idx = 1
}

// SetIndexOuter sets strides with indexes as outer dimension:
// [Indexes][Neurons] (outer to inner), which is optimal for GPU-based
// computation.
func (ns *NeuronIndexStrides) SetIndexOuter(nneur int) {
	ns.Idx = uint32(nneur)
	ns.Neuron = 1
}

//gosl:end neuron

////////////////////////////////////////////////
// 	Props

var VarCategories = []emer.VarCategory{
	{"Act", "basic activation variables, including conductances, current, Vm, spiking"},
	{"Learn", "calcium-based learning variables and other related learning factors"},
	{"Excite", "excitatory channels including NMDA, Vgcc and other excitatory inputs"},
	{"Inhib", "inhibitory channels including after hyperpolarization (AHP) and other K channels; GABA inhibition"},
	{"Stats", "statistics and aggregate values"},
	{"Gmisc", "more detailed conductance (G) variables for integration and other computational values"},
	{"Avg", "longer-term average variables and homeostatic regulation"},
	{"Wts", "weights and other synaptic-level variables"},
}

// NeuronVarProps has all of the display properties for neuron variables, including desc tooltips
var NeuronVarProps = map[string]string{
	/////////////////////////////////////////
	// Spiking, Activation, Major conductances, Vm

	"Spike":  `cat:"Act" desc:"whether neuron has spiked or not on this cycle (0 or 1)"`,
	"Spiked": `cat:"Act" desc:"1 if neuron has spiked within the last 10 cycles (msecs), corresponding to a nominal max spiking rate of 100 Hz, 0 otherwise -- useful for visualization and computing activity levels in terms of average spiked levels."`,
	"Act":    `cat:"Act" desc:"rate-coded activation value reflecting instantaneous estimated rate of spiking, based on 1 / ISIAvg.  This drives feedback inhibition in the FFFB function (todo: this will change when better inhibition is implemented), and is integrated over time for ActInt which is then used for performance statistics and layer average activations, etc.  Should not be used for learning or other computations."`,
	"ActInt": `cat:"Act" desc:"integrated running-average activation value computed from Act with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall activation state across the ThetaCycle time scale, as the overall response of network to current input state -- this is copied to ActM and ActP at the ends of the minus and plus phases, respectively, and used in computing performance-level statistics (which are typically based on ActM).  Should not be used for learning or other computations."`,
	"Ge":     `cat:"Act" range:"2" desc:"total excitatory conductance, including all forms of excitation (e.g., NMDA) -- does *not* include Gbar.E"`,
	"Gi":     `cat:"Act" auto-scale:"+" desc:"total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I"`,
	"Gk":     `cat:"Act" auto-scale:"+" desc:"total potassium conductance, typically reflecting sodium-gated potassium currents involved in adaptation effects -- does *not* include Gbar.K"`,
	"Inet":   `cat:"Act" desc:"net current produced by all channels -- drives update of Vm"`,
	"Vm":     `cat:"Act" min:"0" max:"1" desc:"membrane potential -- integrates Inet current over time"`,
	"VmDend": `cat:"Act" min:"0" max:"1" desc:"dendritic membrane potential -- has a slower time constant, is not subject to the VmR reset after spiking"`,
	"ISI":    `cat:"Act" auto-scale:"+" desc:"current inter-spike-interval -- counts up since last spike.  Starts at -1 when initialized."`,
	"ISIAvg": `cat:"Act" auto-scale:"+" desc:"average inter-spike-interval -- average time interval between spikes, integrated with ISITau rate constant (relatively fast) to capture something close to an instantaneous spiking rate.  Starts at -1 when initialized, and goes to -2 after first spike, and is only valid after the second spike post-initialization."`,
	"Ext":    `cat:"Act" desc:"external input: drives activation of unit from outside influences (e.g., sensory input)"`,
	"Target": `cat:"Act" desc:"target value: drives learning to produce this activation value"`,

	/////////////////////////////////////////
	// Calcium for learning

	"CaSpkM":  `cat:"Learn" desc:"spike-driven calcium trace used as a neuron-level proxy for synpatic credit assignment factor based on continuous time-integrated spiking: exponential integration of SpikeG * Spike at MTau time constant (typically 5).  Simulates a calmodulin (CaM) like signal at the most abstract level."`,
	"CaSpkP":  `cat:"Learn" desc:"continuous cascaded integration of CaSpkM at PTau time constant (typically 40), representing neuron-level purely spiking version of plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act."`,
	"CaSpkD":  `cat:"Learn" desc:"continuous cascaded integration CaSpkP at DTau time constant (typically 40), representing neuron-level purely spiking version of minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act."`,
	"CaSpkPM": `cat:"Learn" desc:"minus-phase snapshot of the CaSpkP value -- similar to ActM but using a more directly spike-integrated value."`,
	"CaLrn":   `cat:"Learn" desc:"recv neuron calcium signal used to drive temporal error difference component of standard learning rule, combining NMDA (NmdaCa) and spiking-driven VGCC (VgccCaInt) calcium sources (vs. CaSpk* which only reflects spiking component).  This is integrated into CaM, CaP, CaD, and temporal derivative is CaP - CaD (CaMKII - DAPK1).  This approximates the backprop error derivative on net input, but VGCC component adds a proportion of recv activation delta as well -- a balance of both works best.  The synaptic-level trace multiplier provides the credit assignment factor, reflecting coincident activity and potentially integrated over longer multi-trial timescales."`,
	"NrnCaM":  `cat:"Learn" desc:"integrated CaLrn at MTau timescale (typically 5), simulating a calmodulin (CaM) like signal, which then drives CaP, CaD for delta signal driving error-driven learning."`,
	"NrnCaP":  `cat:"Learn" desc:"cascaded integration of CaM at PTau time constant (typically 40), representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule."`,
	"NrnCaD":  `cat:"Learn" desc:"cascaded integratoin of CaP at DTau time constant (typically 40), representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule."`,
	"CaDiff":  `cat:"Learn" desc:"difference between CaP - CaD -- this is the error signal that drives error-driven learning."`,
	"RLRate":  `cat:"Learn" auto-scale:"+" desc:"recv-unit based learning rate multiplier, reflecting the sigmoid derivative computed from the CaSpkD of recv unit, and the normalized difference CaSpkP - CaSpkD / MAX(CaSpkP - CaSpkD)."`,

	/////////////////////////////////////////
	// NMDA channels

	"GnmdaSyn":   `cat:"Excite" auto-scale:"+" desc:"integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant"`,
	"Gnmda":      `cat:"Excite" auto-scale:"+" desc:"net postsynaptic (recv) NMDA conductance, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential"`,
	"GnmdaLrn":   `cat:"Excite" auto-scale:"+" desc:"learning version of integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant -- drives NmdaCa that then drives CaM for learning"`,
	"GnmdaMaint": `cat:"Excite" auto-scale:"+" desc:"net postsynaptic maintenance NMDA conductance, computed from GMaintSyn and GMaintRaw, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential"`,
	"NmdaCa":     `cat:"Excite" auto-scale:"+" desc:"NMDA calcium computed from GnmdaLrn, drives learning via CaM"`,

	/////////////////////////////////////////
	//  VGCC voltage gated calcium channels

	"Gvgcc":     `cat:"Excite" auto-scale:"+" desc:"conductance (via Ca) for VGCC voltage gated calcium channels"`,
	"VgccM":     `cat:"Excite" desc:"activation gate of VGCC channels"`,
	"VgccH":     `cat:"Excite" desc:"inactivation gate of VGCC channels"`,
	"VgccCa":    `cat:"Excite" auto-scale:"+" desc:"instantaneous VGCC calcium flux -- can be driven by spiking or directly from Gvgcc"`,
	"VgccCaInt": `cat:"Excite" auto-scale:"+" desc:"time-integrated VGCC calcium flux -- this is actually what drives learning"`,

	/////////////////////////////////////////
	//  Misc Excitatory Vars

	"Burst":      `cat:"Excite" desc:"5IB bursting activation value, computed by thresholding regular CaSpkP value in Super superficial layers"`,
	"BurstPrv":   `cat:"Excite" desc:"previous Burst bursting activation from prior time step -- used for context-based learning"`,
	"CtxtGe":     `cat:"Excite" desc:"context (temporally delayed) excitatory conductance, driven by deep bursting at end of the plus phase, for CT layers."`,
	"CtxtGeRaw":  `cat:"Excite" desc:"raw update of context (temporally delayed) excitatory conductance, driven by deep bursting at end of the plus phase, for CT layers."`,
	"CtxtGeOrig": `cat:"Excite" desc:"original CtxtGe value prior to any decay factor -- updates at end of plus phase."`,

	/////////////////////////////////////////
	// GABA channels

	"GgabaB": `cat:"Inhib" auto-scale:"+" desc:"net GABA-B conductance, after Vm gating and Gbar + Gbase -- applies to Gk, not Gi, for GIRK, with .1 reversal potential."`,
	"GABAB":  `cat:"Inhib" auto-scale:"+" desc:"GABA-B / GIRK activation -- time-integrated value with rise and decay time constants"`,
	"GABABx": `cat:"Inhib" auto-scale:"+" desc:"GABA-B / GIRK internal drive variable -- gets the raw activation and decays"`,

	/////////////////////////////////////////
	// SST somatostatin inhibition factors

	"Gak":      `cat:"Inhib" auto-scale:"+" desc:"conductance of A-type K potassium channels"`,
	"SSGi":     `cat:"Inhib" auto-scale:"+" desc:"SST+ somatostatin positive slow spiking inhibition"`,
	"SSGiDend": `cat:"Inhib" auto-scale:"+" desc:"amount of SST+ somatostatin positive slow spiking inhibition applied to dendritic Vm (VmDend)"`,

	/////////////////////////////////////////
	// AHP channels: Mahp, Sahp, Gkna

	"Gmahp":    `cat:"Inhib" auto-scale:"+" desc:"medium time scale AHP conductance"`,
	"MahpN":    `cat:"Inhib" auto-scale:"+" desc:"accumulating voltage-gated gating value for the medium time scale AHP"`,
	"SahpCa":   `cat:"Inhib" desc:"slowly accumulating calcium value that drives the slow AHP"`,
	"SahpN":    `cat:"Inhib" desc:"sAHP gating value"`,
	"Gsahp":    `cat:"Inhib" auto-scale:"+" desc:"slow time scale AHP conductance"`,
	"GknaMed":  `cat:"Inhib" auto-scale:"+" desc:"conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick) -- produces accommodation / adaptation of firing"`,
	"GknaSlow": `cat:"Inhib" auto-scale:"+" desc:"conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack) -- produces accommodation / adaptation of firing"`,
	"KirM":     `cat:"Inhib" desc:"the Kir gating value"`,
	"Gkir":     `cat:"Inhib" desc:"the conductance of the potassium (K) inwardly rectifying channel, which is strongest at low membrane potentials.  Can be modulated by DA."`,

	/////////////////////////////////////////
	//  SKCa small conductance calcium-gated potassium channels

	"SKCaIn": `cat:"Inhib" desc:"intracellular calcium store level, available to be released with spiking as SKCaR, which can bind to SKCa receptors and drive K current. replenishment is a function of spiking activity being below a threshold"`,
	"SKCaR":  `cat:"Inhib" desc:"released amount of intracellular calcium, from SKCaIn, as a function of spiking events.  this can bind to SKCa channels and drive K currents."`,
	"SKCaM":  `cat:"Inhib" desc:"Calcium-gated potassium channel gating factor, driven by SKCaR via a Hill equation as in chans.SKPCaParams."`,
	"Gsk":    `cat:"Inhib" desc:"Calcium-gated potassium channel conductance as a function of Gbar * SKCaM."`,

	/////////////////////////////////////////
	// Stats, aggregate values

	"ActM":     `cat:"Stats" desc:"ActInt activation state at end of third quarter, representing the posterior-cortical minus phase activation -- used for statistics and monitoring network performance. Should not be used for learning or other computations."`,
	"ActP":     `cat:"Stats" desc:"ActInt activation state at end of fourth quarter, representing the posterior-cortical plus_phase activation -- used for statistics and monitoring network performance.  Should not be used for learning or other computations."`,
	"SpkSt1":   `cat:"Stats" desc:"the activation state at specific time point within current state processing window (e.g., 50 msec for beta cycle within standard theta cycle), as saved by SpkSt1() function.  Used for example in hippocampus for CA3, CA1 learning"`,
	"SpkSt2":   `cat:"Stats" desc:"the activation state at specific time point within current state processing window (e.g., 100 msec for beta cycle within standard theta cycle), as saved by SpkSt2() function.  Used for example in hippocampus for CA3, CA1 learning"`,
	"SpkMax":   `cat:"Stats" desc:"maximum CaSpkP across one theta cycle time window (max of SpkMaxCa) -- used for specialized algorithms that have more phasic behavior within a single trial, e.g., BG Matrix layer gating.  Also useful for visualization of peak activity of neurons."`,
	"SpkMaxCa": `cat:"Stats" desc:"Ca integrated like CaSpkP but only starting at MaxCycStart cycle, to prevent inclusion of carryover spiking from prior theta cycle trial -- the PTau time constant otherwise results in significant carryover.  This is the input to SpkMax"`,

	"SpkBin0": `cat:"Stats" min:"0" max:"10" desc:"aggregated spikes within 8 bins across the theta cycle, for computing synaptic calcium efficiently."`,
	"SpkBin1": `cat:"Stats" min:"0" max:"10" desc:"aggregated spikes within 8 bins across the theta cycle, for computing synaptic calcium efficiently."`,
	"SpkBin2": `cat:"Stats" min:"0" max:"10" desc:"aggregated spikes within 8 bins across the theta cycle, for computing synaptic calcium efficiently."`,
	"SpkBin3": `cat:"Stats" min:"0" max:"10" desc:"aggregated spikes within 8 bins across the theta cycle, for computing synaptic calcium efficiently."`,
	"SpkBin4": `cat:"Stats" min:"0" max:"10" desc:"aggregated spikes within 8 bins across the theta cycle, for computing synaptic calcium efficiently."`,
	"SpkBin5": `cat:"Stats" min:"0" max:"10" desc:"aggregated spikes within 8 bins across the theta cycle, for computing synaptic calcium efficiently."`,
	"SpkBin6": `cat:"Stats" min:"0" max:"10" desc:"aggregated spikes within 8 bins across the theta cycle, for computing synaptic calcium efficiently."`,
	"SpkBin7": `cat:"Stats" min:"0" max:"10" desc:"aggregated spikes within 8 bins across the theta cycle, for computing synaptic calcium efficiently."`,

	"SpkPrv": `cat:"Stats" desc:"final CaSpkD activation state at end of previous theta cycle.  used for specialized learning mechanisms that operate on delayed sending activations."`,

	/////////////////////////////////////////
	// Noise

	"GeNoise":  `cat:"Gmisc" desc:"integrated noise excitatory conductance, added into Ge"`,
	"GeNoiseP": `cat:"Gmisc" desc:"accumulating poisson probability factor for driving excitatory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda."`,
	"GiNoise":  `cat:"Gmisc" desc:"integrated noise inhibotyr conductance, added into Gi"`,
	"GiNoiseP": `cat:"Gmisc" desc:"accumulating poisson probability factor for driving inhibitory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda."`,

	/////////////////////////////////////////
	// Ge, Gi integration

	"GeExt":     `cat:"Gmisc" desc:"extra excitatory conductance added to Ge -- from Ext input, GeCtxt etc"`,
	"GeRaw":     `cat:"Gmisc" desc:"raw excitatory conductance (net input) received from senders = current raw spiking drive"`,
	"GeSyn":     `cat:"Gmisc" range:"2" desc:"time-integrated total excitatory synaptic conductance, with an instantaneous rise time from each spike (in GeRaw) and exponential decay with Dt.GeTau, aggregated over pathways -- does *not* include Gbar.E"`,
	"GiRaw":     `cat:"Gmisc" desc:"raw inhibitory conductance (net input) received from senders  = current raw spiking drive"`,
	"GiSyn":     `cat:"Gmisc" desc:"time-integrated total inhibitory synaptic conductance, with an instantaneous rise time from each spike (in GiRaw) and exponential decay with Dt.GiTau, aggregated over pathways -- does *not* include Gbar.I.  This is added with computed FFFB inhibition to get the full inhibition in Gi"`,
	"SMaintP":   `cat:"Gmisc" desc:"accumulating poisson probability factor for driving self-maintenance by simulating a population of mutually interconnected neurons.  multiply times uniform random deviate at each time step, until it gets below the target threshold based on poisson lambda based on accumulating self maint factor"`,
	"GeInt":     `cat:"Gmisc" range:"2" desc:"integrated running-average activation value computed from Ge with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall Ge level across the ThetaCycle time scale (Ge itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive"`,
	"GeIntNorm": `cat:"Gmisc" range:"1" desc:"GeIntNorm is normalized GeInt value (divided by the layer maximum) -- this is used for learning in layers that require learning on subthreshold activity."`,
	"GiInt":     `cat:"Gmisc" range:"2" desc:"integrated running-average activation value computed from GiSyn with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall synaptic Gi level across the ThetaCycle time scale (Gi itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall inhibitory drive"`,
	"GModRaw":   `cat:"Gmisc" desc:"raw modulatory conductance, received from GType = ModulatoryG pathways"`,
	"GModSyn":   `cat:"Gmisc" desc:"syn integrated modulatory conductance, received from GType = ModulatoryG pathways"`,
	"GMaintRaw": `cat:"Gmisc" desc:"raw maintenance conductance, received from GType = MaintG pathways"`,
	"GMaintSyn": `cat:"Gmisc" desc:"syn integrated maintenance conductance, integrated using MaintNMDA params."`,

	/////////////////////////////////////////
	// Long-term average activation, set point for synaptic scaling

	"ActAvg":  `cat:"Avg" desc:"average activation (of minus phase activation state) over long time intervals (time constant = Dt.LongAvgTau) -- useful for finding hog units and seeing overall distribution of activation"`,
	"AvgPct":  `cat:"Avg" range:"2" desc:"ActAvg as a proportion of overall layer activation -- this is used for synaptic scaling to match TrgAvg activation -- updated at SlowInterval intervals"`,
	"TrgAvg":  `cat:"Avg" range:"2" desc:"neuron's target average activation as a proportion of overall layer activation, assigned during weight initialization, driving synaptic scaling relative to AvgPct"`,
	"DTrgAvg": `cat:"Avg" auto-scale:"+" desc:"change in neuron's target average activation as a result of unit-wise error gradient -- acts like a bias weight.  MPI needs to share these across processors."`,
	"AvgDif":  `cat:"Avg" desc:"AvgPct - TrgAvg -- i.e., the error in overall activity level relative to set point for this neuron, which drives synaptic scaling -- updated at SlowInterval intervals"`,
	"GeBase":  `cat:"Avg" desc:"baseline level of Ge, added to GeRaw, for intrinsic excitability"`,
	"GiBase":  `cat:"Avg" desc:"baseline level of Gi, added to GiRaw, for intrinsic excitability"`,

	"DA":    `cat:"Learn" desc:"dopamine neuromodulation (layer-level variable)"`,
	"ACh":   `cat:"Learn" desc:"cholinergic neuromodulation (layer-level variable)"`,
	"NE":    `cat:"Learn" desc:"norepinepherine (noradrenaline) neuromodulation  (layer-level variable)"`,
	"Ser":   `cat:"Learn" desc:"serotonin neuromodulation (layer-level variable)"`,
	"Gated": `cat:"Learn" desc:"signals whether the layer gated"`,

	"NrnFlags": `display:"-" desc:"bit flags for external input and other neuron status state"`,
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

// NeuronVarIndexByName returns the index of the variable in the Neuron, or error
func NeuronVarIndexByName(varNm string) (int, error) {
	i, ok := NeuronVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Neuron VarByName: variable name: %s not valid", varNm)
	}
	return i, nil
}
