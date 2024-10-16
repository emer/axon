// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"strings"

	_ "cogentcore.org/core/tree"
	"github.com/emer/emergent/v2/emer"
)

//gosl:start

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

// Neuron flag functions in act.goal

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

	// GknaMed is conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick), which produces accommodation / adaptation of firing
	GknaMed

	// GknaSlow is conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack), which produces accommodation / adaptation of firing
	GknaSlow

	// Gkir is the conductance of the potassium (K) inwardly rectifying channel,
	// which is strongest at low membrane potentials.  Can be modulated by DA.
	Gkir

	// KirM is the Kir potassium (K) inwardly rectifying gating value
	KirM

	/////////////////////////////////////////
	//  SKCa small conductance calcium-gated potassium channels

	// Gsk is Calcium-gated potassium channel conductance as a function of Gbar * SKCaM.
	Gsk

	// SKCaIn is intracellular calcium store level, available to be released with spiking as SKCaR, which can bind to SKCa receptors and drive K current. replenishment is a function of spiking activity being below a threshold
	SKCaIn

	// SKCaR released amount of intracellular calcium, from SKCaIn, as a function of spiking events.  this can bind to SKCa channels and drive K currents.
	SKCaR

	// SKCaM is Calcium-gated potassium channel gating factor, driven by SKCaR via a Hill equation as in chans.SKPCaParams.
	SKCaM

	/////////////////////////////////////////
	// AHP channels: Mahp, Sahp, Gkna

	// Gmahp is medium time scale AHP conductance
	Gmahp

	// MahpN is accumulating voltage-gated gating value for the medium time scale AHP
	MahpN

	// Gsahp is slow time scale AHP conductance
	Gsahp

	// SahpCa is slowly accumulating calcium value that drives the slow AHP
	SahpCa

	// SahpN is the sAHP gating value
	SahpN

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

	// SMaintP is accumulating poisson probability factor for driving self-maintenance by simulating a population of mutually interconnected neurons.  multiply times uniform random deviate at each time step, until it gets below the target threshold based on poisson lambda based on accumulating self maint factor
	SMaintP

	// GMaintRaw is raw maintenance conductance, received from GType = MaintG pathways
	GMaintRaw

	// GMaintSyn is syn integrated maintenance conductance, integrated using MaintNMDA params.
	GMaintSyn

	// NrnFlags are bit flags for binary state variables, which are converted to / from uint32.
	// These need to be in Vars because they can be differential per data (for ext inputs)
	// and are writable (indexes are read only).
	NrnFlags
)

// NeuronAvgVars are mostly neuron variables involved in longer-term average activity
// which is aggregated over time and not specific to each input data state,
// along with any other state that is not input data specific.
type NeuronAvgVars int32 //enums:enum

const (
	// ActAvg is average activation (of minus phase activation state)
	// over long time intervals (time constant = Dt.LongAvgTau).
	// Useful for finding hog units and seeing overall distribution of activation.
	ActAvg NeuronAvgVars = iota

	// AvgPct is ActAvg as a proportion of overall layer activation.
	// This is used for synaptic scaling to match TrgAvg activation,
	// updated at SlowInterval intervals.
	AvgPct

	// TrgAvg is neuron's target average activation as a proportion
	// of overall layer activation, assigned during weight initialization,
	// driving synaptic scaling relative to AvgPct.
	TrgAvg

	// DTrgAvg is change in neuron's target average activation as a result
	// of unit-wise error gradient. Acts like a bias weight.
	// MPI needs to share these across processors.
	DTrgAvg

	// AvgDif is AvgPct - TrgAvg, i.e., the error in overall activity level
	// relative to set point for this neuron, which drives synaptic scaling.
	// Updated at SlowInterval intervals.
	AvgDif

	// GeBase is baseline level of Ge, added to GeRaw, for intrinsic excitability.
	GeBase

	// GiBase is baseline level of Gi, added to GiRaw, for intrinsic excitability.
	GiBase
)

// NeuronIndexVars are neuron-level indexes used to access layers and pools
// from the individual neuron level.
type NeuronIndexVars int32 //enums:enum

const (
	// NrnNeurIndex is the index of this neuron within its owning layer.
	NrnNeurIndex NeuronIndexVars = iota

	// NrnLayIndex is the index of the layer that this neuron belongs to,
	// needed for neuron-level parallel code.
	NrnLayIndex

	// NrnSubPool is the index of the sub-level inhibitory pool for this neuron
	// (only for 4D shapes, the pool (unit-group / hypercolumn) structure level).
	// Indicies start at 1 -- 0 is layer-level pool (is 0 if no sub-pools).
	NrnSubPool
)

//gosl:end

////////////////////////////////////////////////
// 	Props

var VarCategories = []emer.VarCategory{
	{"Act", "basic activation variables, including conductances, current, Vm, spiking"},
	{"Learn", "calcium-based learning variables and other related learning factors"},
	{"Excite", "excitatory channels including NMDA, Vgcc and other excitatory inputs"},
	{"Inhib", "inhibitory channels including GABA inhibition, after hyperpolarization (AHP) and other K channels"},
	{"Stats", "statistics and aggregate values"},
	{"Gmisc", "more detailed conductance (G) variables for integration and other computational values"},
	{"Avg", "longer-term average variables and homeostatic regulation"},
	{"Wts", "weights and other synaptic-level variables"},
}

// NeuronVarProps has all of the display properties for neuron variables, including desc tooltips
var NeuronVarProps = map[string]string{
	/////////////////////////////////////////
	// Spiking, Activation, Major conductances, Vm

	"Spike":  `cat:"Act"`,
	"Spiked": `cat:"Act"`,
	"Act":    `cat:"Act"`,
	"ActInt": `cat:"Act"`,
	"Ge":     `cat:"Act" range:"2"`,
	"Gi":     `cat:"Act" auto-scale:"+"`,
	"Gk":     `cat:"Act" auto-scale:"+"`,
	"Inet":   `cat:"Act"`,
	"Vm":     `cat:"Act" min:"0" max:"1"`,
	"VmDend": `cat:"Act" min:"0" max:"1"`,
	"ISI":    `cat:"Act" auto-scale:"+"`,
	"ISIAvg": `cat:"Act" auto-scale:"+"`,
	"Ext":    `cat:"Act"`,
	"Target": `cat:"Act"`,

	/////////////////////////////////////////
	// Calcium for learning

	"CaSpkM":  `cat:"Learn"`,
	"CaSpkP":  `cat:"Learn"`,
	"CaSpkD":  `cat:"Learn"`,
	"CaSpkPM": `cat:"Learn"`,
	"CaLrn":   `cat:"Learn"`,
	"NrnCaM":  `cat:"Learn"`,
	"NrnCaP":  `cat:"Learn"`,
	"NrnCaD":  `cat:"Learn"`,
	"CaDiff":  `cat:"Learn"`,
	"RLRate":  `cat:"Learn" auto-scale:"+"`,

	/////////////////////////////////////////
	// NMDA channels

	"GnmdaSyn":   `cat:"Excite" auto-scale:"+"`,
	"Gnmda":      `cat:"Excite" auto-scale:"+"`,
	"GnmdaLrn":   `cat:"Excite" auto-scale:"+"`,
	"GnmdaMaint": `cat:"Excite" auto-scale:"+"`,
	"NmdaCa":     `cat:"Excite" auto-scale:"+"`,

	/////////////////////////////////////////
	//  VGCC voltage gated calcium channels

	"Gvgcc":     `cat:"Excite" auto-scale:"+"`,
	"VgccM":     `cat:"Excite"`,
	"VgccH":     `cat:"Excite"`,
	"VgccCa":    `cat:"Excite" auto-scale:"+"`,
	"VgccCaInt": `cat:"Excite" auto-scale:"+"`,

	/////////////////////////////////////////
	//  Misc Excitatory Vars

	"Burst":      `cat:"Excite"`,
	"BurstPrv":   `cat:"Excite"`,
	"CtxtGe":     `cat:"Excite"`,
	"CtxtGeRaw":  `cat:"Excite"`,
	"CtxtGeOrig": `cat:"Excite"`,

	/////////////////////////////////////////
	// GABA channels

	"GgabaB": `cat:"Inhib" auto-scale:"+"`,
	"GABAB":  `cat:"Inhib" auto-scale:"+"`,
	"GABABx": `cat:"Inhib" auto-scale:"+"`,

	/////////////////////////////////////////
	// SST somatostatin inhibition factors

	"Gak":      `cat:"Inhib" auto-scale:"+"`,
	"SSGi":     `cat:"Inhib" auto-scale:"+"`,
	"SSGiDend": `cat:"Inhib" auto-scale:"+"`,

	"GknaMed":  `cat:"Inhib" auto-scale:"+"`,
	"GknaSlow": `cat:"Inhib" auto-scale:"+"`,
	"Gkir":     `cat:"Inhib"`,
	"KirM":     `cat:"Inhib"`,

	/////////////////////////////////////////
	//  SKCa small conductance calcium-gated potassium channels

	"Gsk":    `cat:"Inhib"`,
	"SKCaIn": `cat:"Inhib"`,
	"SKCaR":  `cat:"Inhib"`,
	"SKCaM":  `cat:"Inhib"`,

	/////////////////////////////////////////
	// AHP channels: Mahp, Sahp

	"Gmahp":  `cat:"Inhib" auto-scale:"+"`,
	"MahpN":  `cat:"Inhib" auto-scale:"+"`,
	"Gsahp":  `cat:"Inhib" auto-scale:"+"`,
	"SahpCa": `cat:"Inhib"`,
	"SahpN":  `cat:"Inhib"`,

	/////////////////////////////////////////
	// Stats, aggregate values

	"ActM":     `cat:"Stats"`,
	"ActP":     `cat:"Stats"`,
	"SpkSt1":   `cat:"Stats"`,
	"SpkSt2":   `cat:"Stats"`,
	"SpkMax":   `cat:"Stats"`,
	"SpkMaxCa": `cat:"Stats"`,

	"SpkBin0": `cat:"Stats" min:"0" max:"10"`,
	"SpkBin1": `cat:"Stats" min:"0" max:"10"`,
	"SpkBin2": `cat:"Stats" min:"0" max:"10"`,
	"SpkBin3": `cat:"Stats" min:"0" max:"10"`,
	"SpkBin4": `cat:"Stats" min:"0" max:"10"`,
	"SpkBin5": `cat:"Stats" min:"0" max:"10"`,
	"SpkBin6": `cat:"Stats" min:"0" max:"10"`,
	"SpkBin7": `cat:"Stats" min:"0" max:"10"`,

	"SpkPrv": `cat:"Stats"`,

	/////////////////////////////////////////
	// Noise

	"GeNoise":  `cat:"Gmisc"`,
	"GeNoiseP": `cat:"Gmisc"`,
	"GiNoise":  `cat:"Gmisc"`,
	"GiNoiseP": `cat:"Gmisc"`,

	/////////////////////////////////////////
	// Ge, Gi integration

	"GeExt":     `cat:"Gmisc"`,
	"GeRaw":     `cat:"Gmisc"`,
	"GeSyn":     `cat:"Gmisc" range:"2"`,
	"GiRaw":     `cat:"Gmisc"`,
	"GiSyn":     `cat:"Gmisc"`,
	"GeInt":     `cat:"Gmisc" range:"2"`,
	"GeIntNorm": `cat:"Gmisc" range:"1"`,
	"GiInt":     `cat:"Gmisc" range:"2"`,
	"GModRaw":   `cat:"Gmisc"`,
	"GModSyn":   `cat:"Gmisc"`,
	"SMaintP":   `cat:"Gmisc"`,
	"GMaintRaw": `cat:"Gmisc"`,
	"GMaintSyn": `cat:"Gmisc"`,

	"NrnFlags": `display:"-"`,

	/////////////////////////////////////////
	// Long-term average activation, set point for synaptic scaling

	"ActAvg":  `cat:"Avg"`,
	"AvgPct":  `cat:"Avg" range:"2"`,
	"TrgAvg":  `cat:"Avg" range:"2"`,
	"DTrgAvg": `cat:"Avg" auto-scale:"+"`,
	"AvgDif":  `cat:"Avg"`,
	"GeBase":  `cat:"Avg"`,
	"GiBase":  `cat:"Avg"`,

	/////////////////////////////////////////
	// Layer-level variables

	"DA":    `cat:"Learn" doc:"dopamine neuromodulation (layer-level variable)"`,
	"ACh":   `cat:"Learn" doc:"cholinergic neuromodulation (layer-level variable)"`,
	"NE":    `cat:"Learn" doc:"norepinepherine (noradrenaline) neuromodulation  (layer-level variable)"`,
	"Ser":   `cat:"Learn" doc:"serotonin neuromodulation (layer-level variable)"`,
	"Gated": `cat:"Learn" doc:"signals whether the layer gated"`,
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
		tag := NeuronVarProps[vnm]
		NeuronVarProps[vnm] = tag + ` doc:"` + strings.ReplaceAll(i.Desc(), "\n", " ") + `"`
	}
	for i := ActAvg; i < NeuronAvgVarsN; i++ {
		vnm := i.String()
		NeuronVarNames = append(NeuronVarNames, vnm)
		NeuronVarsMap[vnm] = int(NeuronVarsN) + int(i)
		tag := NeuronVarProps[vnm]
		NeuronVarProps[vnm] = tag + ` doc:"` + strings.ReplaceAll(i.Desc(), "\n", " ") + `"`
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
