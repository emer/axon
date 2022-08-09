// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/goki/ki/bitflag"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// NeuronVarStart is the byte offset of fields in the Neuron structure
// where the float32 named variables start.
// Note: all non-float32 infrastructure variables must be at the start!
const NeuronVarStart = 8

// axon.Neuron holds all of the neuron (unit) level variables.
// This is the most basic version, without any optional features.
// All variables accessible via Unit interface must be float32
// and start at the top, in contiguous order
type Neuron struct {
	Flags   NeurFlags `desc:"bit flags for binary state variables"`
	SubPool int32     `desc:"index of the sub-level inhibitory pool that this neuron is in (only for 4D shapes, the pool (unit-group / hypercolumn) structure level) -- indicies start at 1 -- 0 is layer-level pool (is 0 if no sub-pools)."`
	Spike   float32   `desc:"whether neuron has spiked or not on this cycle (0 or 1)"`
	Act     float32   `desc:"rate-coded activation value reflecting instantaneous estimated rate of spiking, based on 1 / ISIAvg.  This drives feedback inhibition in the FFFB function, and is integrated over time for ActInt which is then used for performance statistics and layer average activations, etc."`
	GeSyn   float32   `desc:"total excitatory synaptic conductance -- the net excitatory input to the neuron -- does *not* include Gbar.E"`
	Ge      float32   `desc:"total excitatory conductance, including all forms of excitation (e.g., NMDA) -- does *not* include Gbar.E"`
	GiSyn   float32   `desc:"aggregated synaptic inhibition (from Inhib projections) -- time integral of GiRaw -- this is added with computed FFFB inhibition to get the full inhibition in Gi"`
	Gi      float32   `desc:"total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I"`
	Gk      float32   `desc:"total potassium conductance, typically reflecting sodium-gated potassium currents involved in adaptation effects -- does *not* include Gbar.K"`
	Inet    float32   `desc:"net current produced by all channels -- drives update of Vm"`
	Vm      float32   `desc:"membrane potential -- integrates Inet current over time"`
	VmDend  float32   `desc:"dendritic membrane potential -- has a slower time constant, is not subject to the VmR reset after spiking"`

	Targ float32 `desc:"target value: drives learning to produce this activation value"`
	Ext  float32 `desc:"external input: drives activation of unit from outside influences (e.g., sensory input)"`

	CaSyn  float32 `desc:"spike-driven calcium trace for synapse-level Ca-driven learning rules: SynSpkCa"`
	CaM    float32 `desc:"simple spike-driven calcium signal, with immediate impulse rise and exponential decay, simulating a calmodulin (CaM) like signal at the most abstract level for the Kinase learning rule"`
	CaP    float32 `desc:"shorter timescale integrated CaM value, representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule"`
	CaD    float32 `desc:"longer timescale integrated CaP value, representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule"`
	CaDiff float32 `desc:"difference between CaP - CaD -- basic error signal"`
	PctDWt float32 `desc:"for experimental Kinase continuous learning algorithm: percent of synapses that had DWt updated on the current cycle, for sending-neuron"`

	ActInt float32 `desc:"integrated running-average activation value computed from Act to produce a longer-term integrated value reflecting the overall activation state across a reasonable time scale to reflect overall response of network to current input state -- this is copied to ActM and ActP at the ends of the minus and plus phases, respectively, and used in computing performance-level statistics (which are typically based on ActM)"`
	ActSt1 float32 `desc:"the activation state at specific time point within current state processing window (e.g., 50 msec for beta cycle within standard theta cycle), as saved by ActSt1() function.  Used for example in hippocampus for CA3, CA1 learning"`
	ActSt2 float32 `desc:"the activation state at specific time point within current state processing window (e.g., 100 msec for beta cycle within standard theta cycle), as saved by ActSt2() function.  Used for example in hippocampus for CA3, CA1 learning"`
	ActM   float32 `desc:"the activation state at end of third quarter, which is the traditional posterior-cortical minus phase activation"`
	ActP   float32 `desc:"the activation state at end of fourth quarter, which is the traditional posterior-cortical plus_phase activation"`
	ActDif float32 `desc:"ActP - ActM -- difference between plus and minus phase acts -- reflects the individual error gradient for this neuron in standard error-driven learning terms"`
	ActDel float32 `desc:"delta activation: change in Act from one cycle to next -- can be useful to track where changes are taking place"`
	ActPrv float32 `desc:"the final activation state at end of previous state"`
	RLrate float32 `desc:"recv-unit based learning rate computed from the activity dynamics of recv unit -- extra filtering when recv unit is likely close enough"`

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
	GiSelf   float32 `desc:"total amount of self-inhibition -- time-integrated to avoid oscillations"`

	GeM      float32 `desc:"time-averaged Ge value over the minus phase -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive"`
	GiM      float32 `desc:"time-averaged GiSyn value over the minus phase -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive"`
	GknaFast float32 `desc:"conductance of sodium-gated potassium channel (KNa) fast dynamics (M-type) -- produces accommodation / adaptation of firing"`
	GknaMed  float32 `desc:"conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick) -- produces accommodation / adaptation of firing"`
	GknaSlow float32 `desc:"conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack) -- produces accommodation / adaptation of firing"`
	GgabaB   float32 `desc:"net GABA-B conductance, after Vm gating and Gbar + Gbase -- applies to Gk, not Gi, for GIRK, with .1 reversal potential."`
	GABAB    float32 `desc:"GABA-B / GIRK activation -- time-integrated value with rise and decay time constants"`
	GABABx   float32 `desc:"GABA-B / GIRK internal drive variable -- gets the raw activation and decays"`
	Gvgcc    float32 `desc:"conductance (via Ca) for VGCC voltage gated calcium channels"`
	VgccM    float32 `desc:"activation gate of VGCC channels"`
	VgccH    float32 `desc:"inactivation gate of VGCC channels"`
	VgccCa   float32 `desc:"VGCC calcium flux"`
	Gak      float32 `desc:"conductance of A-type K potassium channels"`

	GnmdaSyn float32 `desc:"integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant"`
	Gnmda    float32 `desc:"net postsynaptic (recv) NMDA conductance, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential"`
	RnmdaSyn float32 `desc:"recv-side NMDA for learning, vs activity: integrated NMDA recv synaptic current -- adds GnmdaRaw and decays with time constant"`
	RCa      float32 `desc:"Receiver-based voltage-driven postsynaptic calcium current factor, reflecting Mg block and V-based current drive, both a function of VmDend: Mg * Vca -- RCa * SnmdaO = total synaptic Ca at each moment"`
	SnmdaO   float32 `desc:"Sender-based number of open NMDA channels based on spiking activity and consequent glutamate release for all sending synapses -- this is the presynaptic component of NMDA activation that is used for computing Ca levels for learning -- increases by (1-SnmdaI)*(1-SnmdaO) with spiking and decays otherwise"`
	SnmdaI   float32 `desc:"Sender-based inhibitory factor on NMDA as a function of sending (presynaptic) spiking history, capturing the allosteric dynamics from Urakubo et al (2008) model.  Increases to 1 with every spike, and decays back to 0 with its own longer decay rate."`

	GeRaw float32 `desc:"raw excitatory conductance (net input) received from senders = current raw spiking drive -- always 0 in display because it is reset during computation"`
	GiRaw float32 `desc:"raw inhibitory conductance (net input) received from senders  = current raw spiking drive -- always 0 in display because it is reset during computation"`
}

var NeuronVars = []string{}

var NeuronVarsMap map[string]int

var NeuronVarProps = map[string]string{
	"GeSyn":    `range:"2"`,
	"Ge":       `range:"2"`,
	"GeM":      `range:"2"`,
	"Vm":       `min:"0" max:"1"`,
	"VmDend":   `min:"0" max:"1"`,
	"ISI":      `auto-scale:"+"`,
	"ISIAvg":   `auto-scale:"+"`,
	"Gi":       `auto-scale:"+"`,
	"Gk":       `auto-scale:"+"`,
	"ActDel":   `auto-scale:"+"`,
	"ActDif":   `auto-scale:"+"`,
	"AvgPct":   `range:"2"`,
	"TrgAvg":   `range:"2"`,
	"DTrgAvg":  `auto-scale:"+"`,
	"GknaFast": `auto-scale:"+"`,
	"GknaMed":  `auto-scale:"+"`,
	"GknaSlow": `auto-scale:"+"`,
	"Gnmda":    `auto-scale:"+"`,
	"GnmdaSyn": `auto-scale:"+"`,
	"RnmdaSyn": `auto-scale:"+"`,
	"GgabaB":   `auto-scale:"+"`,
	"GABAB":    `auto-scale:"+"`,
	"GABABx":   `auto-scale:"+"`,
}

func init() {
	NeuronVarsMap = make(map[string]int, len(NeuronVars))
	typ := reflect.TypeOf((*Neuron)(nil)).Elem()
	nf := typ.NumField()
	starti := 2
	for i := starti; i < nf; i++ {
		fs := typ.FieldByIndex([]int{i})
		v := fs.Name
		NeuronVars = append(NeuronVars, v)
		NeuronVarsMap[v] = i - starti
		pstr := NeuronVarProps[v]
		if fld, has := typ.FieldByName(v); has {
			if desc, ok := fld.Tag.Lookup("desc"); ok {
				pstr += ` desc:"` + desc + `"`
				NeuronVarProps[v] = pstr
			}
		}
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
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(NeuronVarStart+4*idx)))
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

func (nrn *Neuron) HasFlag(flag NeurFlags) bool {
	return bitflag.Has32(int32(nrn.Flags), int(flag))
}

func (nrn *Neuron) SetFlag(flag NeurFlags) {
	bitflag.Set32((*int32)(&nrn.Flags), int(flag))
}

func (nrn *Neuron) ClearFlag(flag NeurFlags) {
	bitflag.Clear32((*int32)(&nrn.Flags), int(flag))
}

func (nrn *Neuron) SetMask(mask int32) {
	bitflag.SetMask32((*int32)(&nrn.Flags), mask)
}

func (nrn *Neuron) ClearMask(mask int32) {
	bitflag.ClearMask32((*int32)(&nrn.Flags), mask)
}

// IsOff returns true if the neuron has been turned off (lesioned)
func (nrn *Neuron) IsOff() bool {
	return nrn.HasFlag(NeurOff)
}

// NeurFlags are bit-flags encoding relevant binary state for neurons
type NeurFlags int32

//go:generate stringer -type=NeurFlags

var KiT_NeurFlags = kit.Enums.AddEnum(NeurFlagsN, kit.BitFlag, nil)

func (ev NeurFlags) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *NeurFlags) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The neuron flags
const (
	// NeurOff flag indicates that this neuron has been turned off (i.e., lesioned)
	NeurOff NeurFlags = iota

	// NeurHasExt means the neuron has external input in its Ext field
	NeurHasExt

	// NeurHasTarg means the neuron has external target input in its Targ field
	NeurHasTarg

	// NeurHasCmpr means the neuron has external comparison input in its Targ field -- used for computing
	// comparison statistics but does not drive neural activity ever
	NeurHasCmpr

	NeurFlagsN
)
