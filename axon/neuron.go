// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/goki/mat32"
)

// NeuronVarStart is the starting *field* index (not byte count!)
// where float32 variables start -- all prior must be 32 bit (uint32, int32),
// Note: all non-float32 infrastructure variables must be at the start!
const NeuronVarStart = 5

//go:generate stringer -type=NeuronFlags

// var KiT_NeurFlags = kit.Enums.AddEnum(NeuronFlagsNum, kit.BitFlag, nil)
// func (ev NeuronFlags) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
// func (ev *NeuronFlags) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

//gosl: start neuron

// NeuronFlags are bit-flags encoding relevant binary state for neurons
type NeuronFlags int32

// The neuron flags
const (
	// NeuronOff flag indicates that this neuron has been turned off (i.e., lesioned)
	NeuronOff NeuronFlags = 1

	// NeuronHasExt means the neuron has external input in its Ext field
	NeuronHasExt NeuronFlags = 1 << 2

	// NeuronHasTarg means the neuron has external target input in its Target field
	NeuronHasTarg NeuronFlags = 1 << 3

	// NeuronHasCmpr means the neuron has external comparison input in its Target field -- used for computing
	// comparison statistics but does not drive neural activity ever
	NeuronHasCmpr NeuronFlags = 1 << 4
)

// axon.Neuron holds all of the neuron (unit) level variables.
// This is the most basic version, without any optional features.
// All variables accessible via Unit interface must be float32
// and start at the top, in contiguous order
type Neuron struct {
	Flags    NeuronFlags `desc:"bit flags for binary state variables"`
	NeurIdx  uint32      `desc:"index of this neuron within its owning layer"`
	LayIdx   uint32      `desc:"index of the layer that this neuron belongs to -- needed for neuron-level parallel code."`
	SubPool  uint32      `desc:"index of the sub-level inhibitory pool that this neuron is in (only for 4D shapes, the pool (unit-group / hypercolumn) structure level) -- indicies start at 1 -- 0 is layer-level pool (is 0 if no sub-pools)."`
	SubPoolG uint32      `desc:"index in global network-wide list of pools"`

	Spike  float32 `desc:"whether neuron has spiked or not on this cycle (0 or 1)"`
	Spiked float32 `desc:"1 if neuron has spiked within the last 10 cycles (msecs), corresponding to a nominal max spiking rate of 100 Hz, 0 otherwise -- useful for visualization and computing activity levels in terms of average spiked levels."`
	Act    float32 `desc:"rate-coded activation value reflecting instantaneous estimated rate of spiking, based on 1 / ISIAvg.  This drives feedback inhibition in the FFFB function (todo: this will change when better inhibition is implemented), and is integrated over time for ActInt which is then used for performance statistics and layer average activations, etc.  Should not be used for learning or other computations."`
	ActInt float32 `desc:"integrated running-average activation value computed from Act to produce a longer-term integrated value reflecting the overall activation state across a reasonable time scale to reflect overall response of network to current input state -- this is copied to ActM and ActP at the ends of the minus and plus phases, respectively, and used in computing performance-level statistics (which are typically based on ActM).  Should not be used for learning or other computations."`
	ActM   float32 `desc:"ActInt activation state at end of third quarter, representing the posterior-cortical minus phase activation -- used for statistics and monitoring network performance. Should not be used for learning or other computations."`
	ActP   float32 `desc:"ActInt activation state at end of fourth quarter, representing the posterior-cortical plus_phase activation -- used for statistics and monitoring network performance.  Should not be used for learning or other computations."`
	Ext    float32 `desc:"external input: drives activation of unit from outside influences (e.g., sensory input)"`
	Target float32 `desc:"target value: drives learning to produce this activation value"`

	Ge     float32 `desc:"total excitatory conductance, including all forms of excitation (e.g., NMDA) -- does *not* include Gbar.E"`
	Gi     float32 `desc:"total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I"`
	Gk     float32 `desc:"total potassium conductance, typically reflecting sodium-gated potassium currents involved in adaptation effects -- does *not* include Gbar.K"`
	Inet   float32 `desc:"net current produced by all channels -- drives update of Vm"`
	Vm     float32 `desc:"membrane potential -- integrates Inet current over time"`
	VmDend float32 `desc:"dendritic membrane potential -- has a slower time constant, is not subject to the VmR reset after spiking"`

	CaSyn   float32 `desc:"spike-driven calcium trace for synapse-level Ca-driven learning: exponential integration of SpikeG * Spike at SynTau time constant (typically 30).  Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for the synaptic trace driving credit assignment in learning. Time constant reflects binding time of Glu to NMDA and Ca buffering postsynaptically, and determines time window where pre * post spiking must overlap to drive learning."`
	CaSpkM  float32 `desc:"spike-driven calcium trace used as a neuron-level proxy for synpatic credit assignment factor based on continuous time-integrated spiking: exponential integration of SpikeG * Spike at MTau time constant (typically 5).  Simulates a calmodulin (CaM) like signal at the most abstract level."`
	CaSpkP  float32 `desc:"continuous cascaded integration of CaSpkM at PTau time constant (typically 40), representing neuron-level purely spiking version of plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act."`
	CaSpkD  float32 `desc:"continuous cascaded integration CaSpkP at DTau time constant (typically 40), representing neuron-level purely spiking version of minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act."`
	CaSpkPM float32 `desc:"minus-phase snapshot of the CaSpkP value -- similar to ActM but using a more directly spike-integrated value."`
	CaLrn   float32 `desc:"recv neuron calcium signal used to drive temporal error difference component of standard learning rule, combining NMDA (NmdaCa) and spiking-driven VGCC (VgccCaInt) calcium sources (vs. CaSpk* which only reflects spiking component).  This is integrated into CaM, CaP, CaD, and temporal derivative is CaP - CaD (CaMKII - DAPK1).  This approximates the backprop error derivative on net input, but VGCC component adds a proportion of recv activation delta as well -- a balance of both works best.  The synaptic-level trace multiplier provides the credit assignment factor, reflecting coincident activity and potentially integrated over longer multi-trial timescales."`
	CaM     float32 `desc:"integrated CaLrn at MTau timescale (typically 5), simulating a calmodulin (CaM) like signal, which then drives CaP, CaD for delta signal driving error-driven learning."`
	CaP     float32 `desc:"cascaded integration of CaM at PTau time constant (typically 40), representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule."`
	CaD     float32 `desc:"cascaded integratoin of CaP at DTau time constant (typically 40), representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule."`
	CaDiff  float32 `desc:"difference between CaP - CaD -- this is the error signal that drives error-driven learning."`

	SpkMaxCa float32 `desc:"Ca integrated like CaSpkP but only starting at MacCycStart cycle, to prevent inclusion of carryover spiking from prior theta cycle trial -- the PTau time constant otherwise results in significant carryover."`
	SpkMax   float32 `desc:"maximum CaSpkP across one theta cycle time window -- used for specialized algorithms that have more phasic behavior within a single trial, e.g., BG Matrix layer gating.  Also useful for visualization of peak activity of neurons."`
	SpkPrv   float32 `desc:"final CaSpkD activation state at end of previous theta cycle.  used for specialized learning mechanisms that operate on delayed sending activations."`
	SpkSt1   float32 `desc:"the activation state at specific time point within current state processing window (e.g., 50 msec for beta cycle within standard theta cycle), as saved by SpkSt1() function.  Used for example in hippocampus for CA3, CA1 learning"`
	SpkSt2   float32 `desc:"the activation state at specific time point within current state processing window (e.g., 100 msec for beta cycle within standard theta cycle), as saved by SpkSt2() function.  Used for example in hippocampus for CA3, CA1 learning"`
	RLRate   float32 `desc:"recv-unit based learning rate multiplier, reflecting the sigmoid derivative computed from the CaSpkD of recv unit, and the normalized difference CaSpkP - CaSpkD / MAX(CaSpkP - CaSpkD)."`

	ActAvg  float32 `desc:"average activation (of minus phase activation state) over long time intervals (time constant = Dt.LongAvgTau) -- useful for finding hog units and seeing overall distribution of activation"`
	AvgPct  float32 `desc:"ActAvg as a proportion of overall layer activation -- this is used for synaptic scaling to match TrgAvg activation -- updated at SlowInterval intervals"`
	TrgAvg  float32 `desc:"neuron's target average activation as a proportion of overall layer activation, assigned during weight initialization, driving synaptic scaling relative to AvgPct"`
	DTrgAvg float32 `desc:"change in neuron's target average activation as a result of unit-wise error gradient -- acts like a bias weight.  MPI needs to share these across processors."`
	AvgDif  float32 `desc:"AvgPct - TrgAvg -- i.e., the error in overall activity level relative to set point for this neuron, which drives synaptic scaling -- updated at SlowInterval intervals"`
	Attn    float32 `desc:"Attentional modulation factor, which can be set by special layers such as the TRC -- multiplies Ge"`

	ISI    float32 `desc:"current inter-spike-interval -- counts up since last spike.  Starts at -1 when initialized."`
	ISIAvg float32 `desc:"average inter-spike-interval -- average time interval between spikes, integrated with ISITau rate constant (relatively fast) to capture something close to an instantaneous spiking rate.  Starts at -1 when initialized, and goes to -2 after first spike, and is only valid after the second spike post-initialization."`

	GeNoiseP float32 `desc:"accumulating poisson probability factor for driving excitatory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda."`
	GeNoise  float32 `desc:"integrated noise excitatory conductance, added into Ge"`
	GiNoiseP float32 `desc:"accumulating poisson probability factor for driving inhibitory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda."`
	GiNoise  float32 `desc:"integrated noise inhibotyr conductance, added into Gi"`

	GeM      float32 `desc:"time-averaged Ge value over the minus phase -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive"`
	GiM      float32 `desc:"time-averaged GiSyn value over the minus phase -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive"`
	MahpN    float32 `desc:"accumulating voltage-gated gating value for the medium time scale AHP"`
	SahpCa   float32 `desc:"slowly accumulating calcium value that drives the slow AHP"`
	SahpN    float32 `desc:"sAHP gating value"`
	GknaMed  float32 `desc:"conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick) -- produces accommodation / adaptation of firing"`
	GknaSlow float32 `desc:"conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack) -- produces accommodation / adaptation of firing"`

	GnmdaSyn float32 `desc:"integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant"`
	Gnmda    float32 `desc:"net postsynaptic (recv) NMDA conductance, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential"`
	GnmdaLrn float32 `desc:"learning version of integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant -- drives NmdaCa that then drives CaM for learning"`
	NmdaCa   float32 `desc:"NMDA calcium computed from GnmdaLrn, drives learning via CaM"`
	SnmdaO   float32 `desc:"Sender-based number of open NMDA channels based on spiking activity and consequent glutamate release for all sending synapses -- this is the presynaptic component of NMDA activation that can be used for computing Ca levels for learning -- increases by (1-SnmdaI)*(1-SnmdaO) with spiking and decays otherwise"`
	SnmdaI   float32 `desc:"Sender-based inhibitory factor on NMDA as a function of sending (presynaptic) spiking history, capturing the allosteric dynamics from Urakubo et al (2008) model.  Increases to 1 with every spike, and decays back to 0 with its own longer decay rate."`

	GgabaB float32 `desc:"net GABA-B conductance, after Vm gating and Gbar + Gbase -- applies to Gk, not Gi, for GIRK, with .1 reversal potential."`
	GABAB  float32 `desc:"GABA-B / GIRK activation -- time-integrated value with rise and decay time constants"`
	GABABx float32 `desc:"GABA-B / GIRK internal drive variable -- gets the raw activation and decays"`

	Gvgcc     float32 `desc:"conductance (via Ca) for VGCC voltage gated calcium channels"`
	VgccM     float32 `desc:"activation gate of VGCC channels"`
	VgccH     float32 `desc:"inactivation gate of VGCC channels"`
	VgccCa    float32 `desc:"instantaneous VGCC calcium flux -- can be driven by spiking or directly from Gvgcc"`
	VgccCaInt float32 `desc:"time-integrated VGCC calcium flux -- this is actually what drives learning"`

	GeExt    float32 `desc:"extra excitatory conductance added to Ge -- from Ext input, deep.GeCtxt etc"`
	GeRaw    float32 `desc:"raw excitatory conductance (net input) received from senders = current raw spiking drive"`
	GeSyn    float32 `desc:"time-integrated total excitatory synaptic conductance, with an instantaneous rise time from each spike (in GeRaw) and exponential decay with Dt.GeTau, aggregated over projections -- does *not* include Gbar.E"`
	GeBase   float32 `desc:"baseline level of Ge, added to GeRaw, for intrinsic excitability"`
	GiRaw    float32 `desc:"raw inhibitory conductance (net input) received from senders  = current raw spiking drive"`
	GiSyn    float32 `desc:"time-integrated total inhibitory synaptic conductance, with an instantaneous rise time from each spike (in GiRaw) and exponential decay with Dt.GiTau, aggregated over projections -- does *not* include Gbar.I.  This is added with computed FFFB inhibition to get the full inhibition in Gi"`
	GiBase   float32 `desc:"baseline level of Gi, added to GiRaw, for intrinsic excitability"`
	GModRaw  float32 `desc:"modulatory conductance, received from GType = ModulatoryG projections"`
	GModSyn  float32 `desc:"modulatory conductance, received from GType = ModulatoryG projections"`
	GeSynMax float32 `desc:"maximum GeSyn value across the ThetaCycle"`
	GeSynPrv float32 `desc:"previous GeSynMax value from the previous ThetaCycle"`

	SSGi     float32 `desc:"SST+ somatostatin positive slow spiking inhibition"`
	SSGiDend float32 `desc:"amount of SST+ somatostatin positive slow spiking inhibition applied to dendritic Vm (VmDend)"`
	Gak      float32 `desc:"conductance of A-type K potassium channels"`

	/////////////////////////////////////////
	//  Special Layer Vars Below

	Burst    float32 `desc:"5IB bursting activation value, computed by thresholding regular CaSpkP value in Super superficial layers"`
	BurstPrv float32 `desc:"previous Burst bursting activation from prior time step -- used for context-based learning"`
	CtxtGe   float32 `desc:"context (temporally delayed) excitatory conductance, driven by deep bursting at end of the plus phase, for CT layers."`
	LearnMod float32 `desc:"learning modulator factor used by special algorithms.  e.g., in MatrixLayer, reflects whether gating happened or not (+1 or -1)"`

	pad, pad1 float32
}

func (nrn *Neuron) HasFlag(flag NeuronFlags) bool {
	return nrn.Flags&flag != 0
}

func (nrn *Neuron) SetFlag(flag NeuronFlags) {
	nrn.Flags |= flag
}

func (nrn *Neuron) ClearFlag(flag NeuronFlags) {
	nrn.Flags &^= flag
}

// IsOff returns true if the neuron has been turned off (lesioned)
func (nrn *Neuron) IsOff() bool {
	return nrn.HasFlag(NeuronOff)
}

//gosl: end neuron

var NeuronVars = []string{}

var NeuronVarsMap map[string]int

var NeuronVarProps = map[string]string{
	"GeSyn":     `range:"2"`,
	"Ge":        `range:"2"`,
	"GeM":       `range:"2"`,
	"Vm":        `min:"0" max:"1"`,
	"VmDend":    `min:"0" max:"1"`,
	"ISI":       `auto-scale:"+"`,
	"ISIAvg":    `auto-scale:"+"`,
	"Gi":        `auto-scale:"+"`,
	"Gk":        `auto-scale:"+"`,
	"ActDel":    `auto-scale:"+"`,
	"ActDiff":   `auto-scale:"+"`,
	"RLRate":    `auto-scale:"+"`,
	"AvgPct":    `range:"2"`,
	"TrgAvg":    `range:"2"`,
	"DTrgAvg":   `auto-scale:"+"`,
	"MahpN":     `auto-scale:"+"`,
	"GknaMed":   `auto-scale:"+"`,
	"GknaSlow":  `auto-scale:"+"`,
	"Gnmda":     `auto-scale:"+"`,
	"GnmdaSyn":  `auto-scale:"+"`,
	"GnmdaLrn":  `auto-scale:"+"`,
	"NmdaCa":    `auto-scale:"+"`,
	"GgabaB":    `auto-scale:"+"`,
	"GABAB":     `auto-scale:"+"`,
	"GABABx":    `auto-scale:"+"`,
	"Gvgcc":     `auto-scale:"+"`,
	"VgccCa":    `auto-scale:"+"`,
	"VgccCaInt": `auto-scale:"+"`,
	"Gak":       `auto-scale:"+"`,
	"SSGi":      `auto-scale:"+"`,
	"SSGiDend":  `auto-scale:"+"`,
}

// NeuronLayerVars are layer-level variables displayed as neuron layers.
var (
	NeuronLayerVars  = []string{"DA", "ACh", "NE"}
	NNeuronLayerVars = len(NeuronLayerVars)
)

func init() {
	NeuronVarsMap = make(map[string]int, len(NeuronVars))
	typ := reflect.TypeOf((*Neuron)(nil)).Elem()
	nf := typ.NumField()
	startIdx := NeuronVarStart
	for i := startIdx; i < nf; i++ {
		fs := typ.FieldByIndex([]int{i})
		v := fs.Name
		if !fs.IsExported() {
			continue
		}
		NeuronVars = append(NeuronVars, v)
		NeuronVarsMap[v] = i - startIdx
		pstr := NeuronVarProps[v]
		if fld, has := typ.FieldByName(v); has {
			if desc, ok := fld.Tag.Lookup("desc"); ok {
				pstr += ` desc:"` + desc + `"`
				NeuronVarProps[v] = pstr
			}
		}
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
