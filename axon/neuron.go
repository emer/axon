// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"unsafe"

	"github.com/emer/emergent/netview"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

//go:generate stringer -type=NeuronFlags
//go:generate stringer -type=NeuronVars
//go:generate stringer -type=NeuronIdxs

var KiT_NeuronFlags = kit.Enums.AddEnum(NeuronFlagsN, kit.BitFlag, nil)

func (ev NeuronFlags) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *NeuronFlags) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

var KiT_NeuronVars = kit.Enums.AddEnum(NeuronVarsN, kit.NotBitFlag, nil)

func (ev NeuronVars) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *NeuronVars) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

var KiT_NeuronIdxs = kit.Enums.AddEnum(NeuronIdxsN, kit.NotBitFlag, nil)

func (ev NeuronIdxs) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *NeuronIdxs) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

//gosl: start neuron

// NeuronFlags are bit-flags encoding relevant binary state for neurons
type NeuronFlags int32

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

// NeuronVars are the neuron variables
type NeuronVars int32

const (
	/////////////////////////////////////////
	// Spiking, Activation

	Spike  NeuronVars = iota // whether neuron has spiked or not on this cycle (0 or 1)
	Spiked                   // 1 if neuron has spiked within the last 10 cycles (msecs), corresponding to a nominal max spiking rate of 100 Hz, 0 otherwise -- useful for visualization and computing activity levels in terms of average spiked levels.
	Act                      // rate-coded activation value reflecting instantaneous estimated rate of spiking, based on 1 / ISIAvg.  This drives feedback inhibition in the FFFB function (todo: this will change when better inhibition is implemented), and is integrated over time for ActInt which is then used for performance statistics and layer average activations, etc.  Should not be used for learning or other computations.
	ActInt                   // integrated running-average activation value computed from Act with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall activation state across the ThetaCycle time scale, as the overall response of network to current input state -- this is copied to ActM and ActP at the ends of the minus and plus phases, respectively, and used in computing performance-level statistics (which are typically based on ActM).  Should not be used for learning or other computations.
	ActM                     // ActInt activation state at end of third quarter, representing the posterior-cortical minus phase activation -- used for statistics and monitoring network performance. Should not be used for learning or other computations.
	ActP                     // ActInt activation state at end of fourth quarter, representing the posterior-cortical plus_phase activation -- used for statistics and monitoring network performance.  Should not be used for learning or other computations.
	Ext                      // external input: drives activation of unit from outside influences (e.g., sensory input)
	Target                   // target value: drives learning to produce this activation value

	/////////////////////////////////////////
	// Major conductances, Vm

	Ge     // total excitatory conductance, including all forms of excitation (e.g., NMDA) -- does *not* include Gbar.E
	Gi     // total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I
	Gk     // total potassium conductance, typically reflecting sodium-gated potassium currents involved in adaptation effects -- does *not* include Gbar.K
	Inet   // net current produced by all channels -- drives update of Vm
	Vm     // membrane potential -- integrates Inet current over time
	VmDend // dendritic membrane potential -- has a slower time constant, is not subject to the VmR reset after spiking

	/////////////////////////////////////////
	// Calcium for learning

	CaSyn   // spike-driven calcium trace for synapse-level Ca-driven learning: exponential integration of SpikeG * Spike at SynTau time constant (typically 30).  Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for the synaptic trace driving credit assignment in learning. Time constant reflects binding time of Glu to NMDA and Ca buffering postsynaptically, and determines time window where pre * post spiking must overlap to drive learning.
	CaSpkM  // spike-driven calcium trace used as a neuron-level proxy for synpatic credit assignment factor based on continuous time-integrated spiking: exponential integration of SpikeG * Spike at MTau time constant (typically 5).  Simulates a calmodulin (CaM) like signal at the most abstract level.
	CaSpkP  // continuous cascaded integration of CaSpkM at PTau time constant (typically 40), representing neuron-level purely spiking version of plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act.
	CaSpkD  // continuous cascaded integration CaSpkP at DTau time constant (typically 40), representing neuron-level purely spiking version of minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act.
	CaSpkPM // minus-phase snapshot of the CaSpkP value -- similar to ActM but using a more directly spike-integrated value.
	CaLrn   // recv neuron calcium signal used to drive temporal error difference component of standard learning rule, combining NMDA (NmdaCa) and spiking-driven VGCC (VgccCaInt) calcium sources (vs. CaSpk* which only reflects spiking component).  This is integrated into CaM, CaP, CaD, and temporal derivative is CaP - CaD (CaMKII - DAPK1).  This approximates the backprop error derivative on net input, but VGCC component adds a proportion of recv activation delta as well -- a balance of both works best.  The synaptic-level trace multiplier provides the credit assignment factor, reflecting coincident activity and potentially integrated over longer multi-trial timescales.
	CaM     // integrated CaLrn at MTau timescale (typically 5), simulating a calmodulin (CaM) like signal, which then drives CaP, CaD for delta signal driving error-driven learning.
	CaP     // cascaded integration of CaM at PTau time constant (typically 40), representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule.
	CaD     // cascaded integratoin of CaP at DTau time constant (typically 40), representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule.
	CaDiff  // difference between CaP - CaD -- this is the error signal that drives error-driven learning.
	RLRate  // recv-unit based learning rate multiplier, reflecting the sigmoid derivative computed from the CaSpkD of recv unit, and the normalized difference CaSpkP - CaSpkD / MAX(CaSpkP - CaSpkD).

	/////////////////////////////////////////
	// Stats, aggregate values

	SpkMaxCa // Ca integrated like CaSpkP but only starting at MaxCycStart cycle, to prevent inclusion of carryover spiking from prior theta cycle trial -- the PTau time constant otherwise results in significant carryover.  This is the input to SpkMax
	SpkMax   // maximum CaSpkP across one theta cycle time window (max of SpkMaxCa) -- used for specialized algorithms that have more phasic behavior within a single trial, e.g., BG Matrix layer gating.  Also useful for visualization of peak activity of neurons.
	SpkPrv   // final CaSpkD activation state at end of previous theta cycle.  used for specialized learning mechanisms that operate on delayed sending activations.
	SpkSt1   // the activation state at specific time point within current state processing window (e.g., 50 msec for beta cycle within standard theta cycle), as saved by SpkSt1() function.  Used for example in hippocampus for CA3, CA1 learning
	SpkSt2   // the activation state at specific time point within current state processing window (e.g., 100 msec for beta cycle within standard theta cycle), as saved by SpkSt2() function.  Used for example in hippocampus for CA3, CA1 learning
	DASign   // sign of dopamine-based learning effects for this neuron -- 1 = D1, -1 = D2

	/////////////////////////////////////////
	// Long-term average activation, set point for synaptic scaling

	ActAvg  // average activation (of minus phase activation state) over long time intervals (time constant = Dt.LongAvgTau) -- useful for finding hog units and seeing overall distribution of activation
	AvgPct  // ActAvg as a proportion of overall layer activation -- this is used for synaptic scaling to match TrgAvg activation -- updated at SlowInterval intervals
	TrgAvg  // neuron's target average activation as a proportion of overall layer activation, assigned during weight initialization, driving synaptic scaling relative to AvgPct
	DTrgAvg // change in neuron's target average activation as a result of unit-wise error gradient -- acts like a bias weight.  MPI needs to share these across processors.
	AvgDif  // AvgPct - TrgAvg -- i.e., the error in overall activity level relative to set point for this neuron, which drives synaptic scaling -- updated at SlowInterval intervals
	Attn    // Attentional modulation factor, which can be set by special layers such as the TRC -- multiplies Ge

	/////////////////////////////////////////
	// ISI for computing rate-code activation

	ISI    // current inter-spike-interval -- counts up since last spike.  Starts at -1 when initialized.
	ISIAvg // average inter-spike-interval -- average time interval between spikes, integrated with ISITau rate constant (relatively fast) to capture something close to an instantaneous spiking rate.  Starts at -1 when initialized, and goes to -2 after first spike, and is only valid after the second spike post-initialization.

	/////////////////////////////////////////
	// Noise

	GeNoiseP // accumulating poisson probability factor for driving excitatory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda.
	GeNoise  // integrated noise excitatory conductance, added into Ge
	GiNoiseP // accumulating poisson probability factor for driving inhibitory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda.
	GiNoise  // integrated noise inhibotyr conductance, added into Gi

	/////////////////////////////////////////
	// Ge, Gi integration

	GeExt     // extra excitatory conductance added to Ge -- from Ext input, GeCtxt etc
	GeRaw     // raw excitatory conductance (net input) received from senders = current raw spiking drive
	GeSyn     // time-integrated total excitatory synaptic conductance, with an instantaneous rise time from each spike (in GeRaw) and exponential decay with Dt.GeTau, aggregated over projections -- does *not* include Gbar.E
	GeBase    // baseline level of Ge, added to GeRaw, for intrinsic excitability
	GiRaw     // raw inhibitory conductance (net input) received from senders  = current raw spiking drive
	GiSyn     // time-integrated total inhibitory synaptic conductance, with an instantaneous rise time from each spike (in GiRaw) and exponential decay with Dt.GiTau, aggregated over projections -- does *not* include Gbar.I.  This is added with computed FFFB inhibition to get the full inhibition in Gi
	GiBase    // baseline level of Gi, added to GiRaw, for intrinsic excitability
	GeInt     // integrated running-average activation value computed from Ge with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall Ge level across the ThetaCycle time scale (Ge itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive
	GeIntMax  // maximum GeInt value across one theta cycle time window.
	GiInt     // integrated running-average activation value computed from GiSyn with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall synaptic Gi level across the ThetaCycle time scale (Gi itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall inhibitory drive
	GModRaw   // raw modulatory conductance, received from GType = ModulatoryG projections
	GModSyn   // syn integrated modulatory conductance, received from GType = ModulatoryG projections
	GMaintRaw // raw maintenance conductance, received from GType = MaintG projections
	GMaintSyn // syn integrated maintenance conductance, integrated using MaintNMDA params.

	/////////////////////////////////////////
	// SST somatostatin inhibition factors

	SSGi     // SST+ somatostatin positive slow spiking inhibition
	SSGiDend // amount of SST+ somatostatin positive slow spiking inhibition applied to dendritic Vm (VmDend)
	Gak      // conductance of A-type K potassium channels

	/////////////////////////////////////////
	// AHP channels: Mahp, Sahp, Gkna

	MahpN    // accumulating voltage-gated gating value for the medium time scale AHP
	SahpCa   // slowly accumulating calcium value that drives the slow AHP
	SahpN    // sAHP gating value
	GknaMed  // conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick) -- produces accommodation / adaptation of firing
	GknaSlow // conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack) -- produces accommodation / adaptation of firing

	/////////////////////////////////////////
	// NMDA channels

	// integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant
	GnmdaSyn
	// net postsynaptic (recv) NMDA conductance, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential
	Gnmda
	// net postsynaptic maintenance NMDA conductance, computed from GMaintSyn and GMaintRaw, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential
	GnmdaMaint
	// learning version of integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant -- drives NmdaCa that then drives CaM for learning
	GnmdaLrn
	// NMDA calcium computed from GnmdaLrn, drives learning via CaM
	NmdaCa

	/////////////////////////////////////////
	// GABA channels

	GgabaB // net GABA-B conductance, after Vm gating and Gbar + Gbase -- applies to Gk, not Gi, for GIRK, with .1 reversal potential.
	GABAB  // GABA-B / GIRK activation -- time-integrated value with rise and decay time constants
	GABABx // GABA-B / GIRK internal drive variable -- gets the raw activation and decays

	/////////////////////////////////////////
	//  VGCC voltage gated calcium channels

	Gvgcc     // conductance (via Ca) for VGCC voltage gated calcium channels
	VgccM     // activation gate of VGCC channels
	VgccH     // inactivation gate of VGCC channels
	VgccCa    // instantaneous VGCC calcium flux -- can be driven by spiking or directly from Gvgcc
	VgccCaInt // time-integrated VGCC calcium flux -- this is actually what drives learning

	/////////////////////////////////////////
	//  SKCa small conductance calcium-gated potassium channels

	SKCaIn // intracellular calcium store level, available to be released with spiking as SKCaR, which can bind to SKCa receptors and drive K current. replenishment is a function of spiking activity being below a threshold
	SKCaR  // released amount of intracellular calcium, from SKCaIn, as a function of spiking events.  this can bind to SKCa channels and drive K currents.
	SKCaM  // Calcium-gated potassium channel gating factor, driven by SKCaR via a Hill equation as in chans.SKPCaParams.
	Gsk    // Calcium-gated potassium channel conductance as a function of Gbar * SKCaM.

	/////////////////////////////////////////
	//  Special Layer Vars

	Burst      // 5IB bursting activation value, computed by thresholding regular CaSpkP value in Super superficial layers
	BurstPrv   // previous Burst bursting activation from prior time step -- used for context-based learning
	CtxtGe     // context (temporally delayed) excitatory conductance, driven by deep bursting at end of the plus phase, for CT layers.
	CtxtGeRaw  // raw update of context (temporally delayed) excitatory conductance, driven by deep bursting at end of the plus phase, for CT layers.
	CtxtGeOrig // original CtxtGe value prior to any decay factor -- updates at end of plus phase.

	NeuronVarsN
)

// NeurVarStrides encodes the stride offsets for neuron variable access
// into network float32 array.
type NeurVarStrides struct {
	Neuron uint32 `desc:"neuron level"`
	Var    uint32 `desc:"variable level"`
	Data   uint32 `desc:"data parallel level"`

	pad uint32
}

// Idx returns the index into network float32 array for given neuron, data, and variable
func (ns *NeurVarStrides) Idx(neurIdx, dataIdx uint32, nvar NeuronVars) uint32 {
	return ns.Neuron*neurIdx + ns.Var*uint32(nvar) + ns.Data*dataIdx
}

// SetNeurVarData sets strides as neurons x vars x data (outer to inner),
// which is likely to be optimal for CPU-based computation.
func (ns *NeurVarStrides) SetNeurVarData(nneur, ndata int) {
	ns.Neuron = uint32(ndata) * uint32(NeuronVarsN)
	ns.Var = uint32(ndata)
	ns.Data = 1
}

// SetVarNeurData sets strides as vars x neurons x data (outer to inner),
// which is likely to be optimal for GPU-based computation.
func (ns *NeurVarStrides) SetVarNeurData(nneur, ndata int) {
	ns.Var = uint32(ndata) * uint32(nneur)
	ns.Neuron = uint32(ndata)
	ns.Data = 1
}

////////////////////////////////////////////////
// Idxs

// NeuronIdxs are the neuron indexes and other uint32 values (flags, etc)
type NeuronIdxs int32

const (
	// bit flags for binary state variables -- note that these are automatically shared across all data parallel vals!
	NidxFlags NeuronIdxs = iota
	// index of this neuron within its owning layer
	NidxNeurIdx
	// index of the layer that this neuron belongs to -- needed for neuron-level parallel code.
	NidxLayIdx
	// index of the sub-level inhibitory pool that this neuron is in (only for 4D shapes, the pool (unit-group / hypercolumn) structure level) -- indicies start at 1 -- 0 is layer-level pool (is 0 if no sub-pools).
	NidxSubPool
	// index in network-wide list of all pools
	NidxSubPoolN

	NeuronIdxsN
)

// NeurIdxStrides encodes the stride offsets for neuron index access
// into network uint32 array.
type NeurIdxStrides struct {
	Neuron uint32 `desc:"neuron level"`
	Idx    uint32 `desc:"index value level"`

	pad, pad2 uint32
}

// Idx returns the index into network uint32 array for given neuron, index value
func (ns *NeurIdxStrides) Idx(neurIdx uint32, idx NeuronIdxs) uint32 {
	return ns.Neuron*neurIdx + ns.Idx*uint32(idx)
}

// SetNeurIdx sets strides as neurons x idxs (outer to inner),
// which is likely to be optimal for CPU-based computation.
func (ns *NeurIdxStrides) SetNeurIdx(nneur int) {
	ns.Neuron = uint32(NeuronIdxsN)
	ns.Idx = 1
}

// SetIdxNeur sets strides as idxs x neurons (outer to inner),
// which is likely to be optimal for GPU-based computation.
func (ns *NeurIdxStrides) SetIdxNeur(nneur int) {
	ns.Idx = uint32(nneur)
	ns.Neuron = 1
}

//gosl: end neuron

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

	/////////////////////////////////////////
	// Calcium for learning

	"CaSyn":   `desc:"spike-driven calcium trace for synapse-level Ca-driven learning: exponential integration of SpikeG * Spike at SynTau time constant (typically 30).  Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for the synaptic trace driving credit assignment in learning. Time constant reflects binding time of Glu to NMDA and Ca buffering postsynaptically, and determines time window where pre * post spiking must overlap to drive learning."`,
	"CaSpkM":  `desc:"spike-driven calcium trace used as a neuron-level proxy for synpatic credit assignment factor based on continuous time-integrated spiking: exponential integration of SpikeG * Spike at MTau time constant (typically 5).  Simulates a calmodulin (CaM) like signal at the most abstract level."`,
	"CaSpkP":  `desc:"continuous cascaded integration of CaSpkM at PTau time constant (typically 40), representing neuron-level purely spiking version of plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act."`,
	"CaSpkD":  `desc:"continuous cascaded integration CaSpkP at DTau time constant (typically 40), representing neuron-level purely spiking version of minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act."`,
	"CaSpkPM": `desc:"minus-phase snapshot of the CaSpkP value -- similar to ActM but using a more directly spike-integrated value."`,
	"CaLrn":   `desc:"recv neuron calcium signal used to drive temporal error difference component of standard learning rule, combining NMDA (NmdaCa) and spiking-driven VGCC (VgccCaInt) calcium sources (vs. CaSpk* which only reflects spiking component).  This is integrated into CaM, CaP, CaD, and temporal derivative is CaP - CaD (CaMKII - DAPK1).  This approximates the backprop error derivative on net input, but VGCC component adds a proportion of recv activation delta as well -- a balance of both works best.  The synaptic-level trace multiplier provides the credit assignment factor, reflecting coincident activity and potentially integrated over longer multi-trial timescales."`,
	"CaM":     `desc:"integrated CaLrn at MTau timescale (typically 5), simulating a calmodulin (CaM) like signal, which then drives CaP, CaD for delta signal driving error-driven learning."`,
	"CaP":     `desc:"cascaded integration of CaM at PTau time constant (typically 40), representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule."`,
	"CaD":     `desc:"cascaded integratoin of CaP at DTau time constant (typically 40), representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule."`,
	"CaDiff":  `desc:"difference between CaP - CaD -- this is the error signal that drives error-driven learning."`,
	"RLRate":  `auto-scale:"+" desc:"recv-unit based learning rate multiplier, reflecting the sigmoid derivative computed from the CaSpkD of recv unit, and the normalized difference CaSpkP - CaSpkD / MAX(CaSpkP - CaSpkD)."`,

	/////////////////////////////////////////
	// Stats, aggregate values

	"SpkMaxCa": `desc:"Ca integrated like CaSpkP but only starting at MaxCycStart cycle, to prevent inclusion of carryover spiking from prior theta cycle trial -- the PTau time constant otherwise results in significant carryover.  This is the input to SpkMax"`,
	"SpkMax":   `desc:"maximum CaSpkP across one theta cycle time window (max of SpkMaxCa) -- used for specialized algorithms that have more phasic behavior within a single trial, e.g., BG Matrix layer gating.  Also useful for visualization of peak activity of neurons."`,
	"SpkPrv":   `desc:"final CaSpkD activation state at end of previous theta cycle.  used for specialized learning mechanisms that operate on delayed sending activations."`,
	"SpkSt1":   `desc:"the activation state at specific time point within current state processing window (e.g., 50 msec for beta cycle within standard theta cycle), as saved by SpkSt1() function.  Used for example in hippocampus for CA3, CA1 learning"`,
	"SpkSt2":   `desc:"the activation state at specific time point within current state processing window (e.g., 100 msec for beta cycle within standard theta cycle), as saved by SpkSt2() function.  Used for example in hippocampus for CA3, CA1 learning"`,
	"DASign":   `desc:"sign of dopamine-based learning effects for this neuron -- 1 = D1, -1 = D2"`,

	/////////////////////////////////////////
	// Long-term average activation, set point for synaptic scaling

	"ActAvg":  `desc:"average activation (of minus phase activation state) over long time intervals (time constant = Dt.LongAvgTau) -- useful for finding hog units and seeing overall distribution of activation"`,
	"AvgPct":  `range:"2" desc:"ActAvg as a proportion of overall layer activation -- this is used for synaptic scaling to match TrgAvg activation -- updated at SlowInterval intervals"`,
	"TrgAvg":  `range:"2" desc:"neuron's target average activation as a proportion of overall layer activation, assigned during weight initialization, driving synaptic scaling relative to AvgPct"`,
	"DTrgAvg": `auto-scale:"+" desc:"change in neuron's target average activation as a result of unit-wise error gradient -- acts like a bias weight.  MPI needs to share these across processors."`,
	"AvgDif":  `desc:"AvgPct - TrgAvg -- i.e., the error in overall activity level relative to set point for this neuron, which drives synaptic scaling -- updated at SlowInterval intervals"`,
	"Attn":    `desc:"Attentional modulation factor, which can be set by special layers such as the TRC -- multiplies Ge"`,

	/////////////////////////////////////////
	// ISI for computing rate-code activation

	"ISI":    `auto-scale:"+" desc:"current inter-spike-interval -- counts up since last spike.  Starts at -1 when initialized."`,
	"ISIAvg": `auto-scale:"+" desc:"average inter-spike-interval -- average time interval between spikes, integrated with ISITau rate constant (relatively fast) to capture something close to an instantaneous spiking rate.  Starts at -1 when initialized, and goes to -2 after first spike, and is only valid after the second spike post-initialization."`,

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
	"GeBase":    `desc:"baseline level of Ge, added to GeRaw, for intrinsic excitability"`,
	"GiRaw":     `desc:"raw inhibitory conductance (net input) received from senders  = current raw spiking drive"`,
	"GiSyn":     `desc:"time-integrated total inhibitory synaptic conductance, with an instantaneous rise time from each spike (in GiRaw) and exponential decay with Dt.GiTau, aggregated over projections -- does *not* include Gbar.I.  This is added with computed FFFB inhibition to get the full inhibition in Gi"`,
	"GiBase":    `desc:"baseline level of Gi, added to GiRaw, for intrinsic excitability"`,
	"GeInt":     `range:"2" desc:"integrated running-average activation value computed from Ge with time constant Act.Dt.IntTau, to produce a longer-term integrated value reflecting the overall Ge level across the ThetaCycle time scale (Ge itself fluctuates considerably) -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive"`,
	"GeIntMax":  `range:"2" desc:"maximum GeInt value across one theta cycle time window."`,
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
}

var NeuronVarNames = []string{}

var NeuronVarsMap map[string]int

// NeuronLayerVars are layer-level variables displayed as neuron layers.
var (
	NeuronLayerVars  = []string{"DA", "ACh", "NE", "Ser", "Gated"}
	NNeuronLayerVars = len(NeuronLayerVars)
)

func init() {
	netview.NVarCols = 4 // many neurons
	NeuronVarsMap = make(map[string]int, len(NeuronVars))
	for i := Spike; i < NeuronVarsN; i++ {
		vnm := i.(String)
		NeuronVarNames = append(NeuronVarNames, vnm)
		NeuronVarsMap[v] = i
	}
	for _, v := range NeuronLayerVars {
		NeuronVarsMap[v] = len(NeuronVars)
		NeuronVars = append(NeuronVars, v)
	}
}

func (nrn *Neuron) VarNames() []string {
	return NeuronVars
}

// NeuronVarIdxByName returns the index of the variable in the Neuron, or error
func NeuronVarIdxByName(varNm string) (int, error) {
	i, ok := NeuronVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Neuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in NeuronVars list)
func (nrn *Neuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(NeuronVarStart*4+4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (nrn *Neuron) VarByName(varNm string) (float32, error) {
	i, err := NeuronVarIdxByName(varNm)
	if err != nil {
		return mat32.NaN(), err
	}
	return nrn.VarByIndex(i), nil
}
