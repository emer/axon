// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"unsafe"

	"github.com/goki/ki/bitflag"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// NeuronVarStart is the byte offset of fields in the Neuron structure
// where the float32 named variables start.
// Note: all non-float32 infrastructure variables must be at the start!
const NeuronVarStart = 8

// axon.Neuron holds all of the neuron (unit) level variables -- this is the most basic version with
// rate-code only and no optional features at all.
// All variables accessible via Unit interface must be float32 and start at the top, in contiguous order
type Neuron struct {
	Flags   NeurFlags `desc:"bit flags for binary state variables"`
	SubPool int32     `desc:"index of the sub-level inhibitory pool that this neuron is in (only for 4D shapes, the pool (unit-group / hypercolumn) structure level) -- indicies start at 1 -- 0 is layer-level pool (is 0 if no sub-pools)."`
	Spike   float32   `desc:"whether neuron has spiked or not on this cycle (0 or 1)"`
	ISI     float32   `desc:"current inter-spike-interval -- counts up since last spike.  Starts at -1 when initialized."`
	ISIAvg  float32   `desc:"average inter-spike-interval -- average time interval between spikes, integrated with ISITau rate constant (relatively fast) to capture something close to an instantaneous spiking rate.  Starts at -1 when initialized, and goes to -2 after first spike, and is only valid after the second spike post-initialization."`
	Act     float32   `desc:"rate-coded activation value reflecting instantaneous estimated rate of spiking, based on 1 / ISIAvg.  This drives feedback inhibition in the FFFB function, and is integrated over time for ActInt which is then used for performance statistics and layer average activations, etc."`
	ActInt  float32   `desc:"integrated running-average activation value computed from Act to produce a longer-term integrated value reflecting the overall activation state across a reasonable time scale to reflect overall response of network to current input state -- this is copied to ActM and ActP at the ends of the minus and plus phases, respectively, and used in computing performance-level statistics (which are typically based on ActM)"`
	Ge      float32   `desc:"total excitatory synaptic conductance -- the net excitatory input to the neuron -- does *not* include Gbar.E"`
	Gi      float32   `desc:"total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I"`
	Gk      float32   `desc:"total potassium conductance, typically reflecting sodium-gated potassium currents involved in adaptation effects -- does *not* include Gbar.K"`
	Inet    float32   `desc:"net current produced by all channels -- drives update of Vm"`
	Vm      float32   `desc:"membrane potential -- integrates Inet current over time"`
	VmDend  float32   `desc:"dendritic membrane potential -- subject to multiplier"`

	Targ float32 `desc:"target value: drives learning to produce this activation value"`
	Ext  float32 `desc:"external input: drives activation of unit from outside influences (e.g., sensory input)"`

	AvgSS   float32 `desc:"super-short time-scale average of spiking -- goes up instantaneously when a Spike occurs, and then decays until the next spike -- provides the lowest-level time integration for running-averages that simulate accumulation of Calcium over time"`
	AvgS    float32 `desc:"short time-scale average of spiking, as a running average over AvgSS -- tracks the most recent activation states, and represents the plus phase for learning in error-driven learning (see AvgSLrn)"`
	AvgM    float32 `desc:"medium time-scale average of spiking, as a running average over AvgS -- represents the minus phase for error-driven learning"`
	AvgSLrn float32 `desc:"short time-scale activation average that is used for learning -- typically includes a small contribution from AvgMLrn in addition to mostly AvgS, as determined by LrnActAvgParams.LrnM -- important to ensure that when neuron turns off in plus phase (short time scale), enough medium-phase trace remains so that learning signal doesn't just go all the way to 0, at which point no learning would take place -- AvgS is subject to thresholding prior to mixing so low values become zero"`
	AvgMLrn float32 `desc:"medium time-scale activation average used in learning: subect to thresholding so low values become zero"`

	ActSt1 float32 `desc:"the activation state at specific time point within current state processing window, as saved by ActSt1() function.  Used for example in hippocampus for CA3, CA1 learning"`
	ActSt2 float32 `desc:"the activation state at specific time point within current state processing window, as saved by ActSt2() function.  Used for example in hippocampus for CA3, CA1 learning"`
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

	Noise float32 `desc:"noise value added to unit (ActNoiseParams determines distribution, and when / where it is added)"`

	GiSyn    float32 `desc:"aggregated synaptic inhibition (from Inhib projections) -- time integral of GiRaw -- this is added with computed FFFB inhibition to get the full inhibition in Gi"`
	GiSelf   float32 `desc:"total amount of self-inhibition -- time-integrated to avoid oscillations"`
	GeRaw    float32 `desc:"raw excitatory conductance (net input) received from sending units (send delta's are added to this value)"`
	GiRaw    float32 `desc:"raw inhibitory conductance (net input) received from sending units (send delta's are added to this value)"`
	GeM      float32 `desc:"time-averaged Ge value over the minus phase -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive"`
	GiM      float32 `desc:"time-averaged GiSyn value over the minus phase -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive"`
	GknaFast float32 `desc:"conductance of sodium-gated potassium channel (KNa) fast dynamics (M-type) -- produces accommodation / adaptation of firing"`
	GknaMed  float32 `desc:"conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick) -- produces accommodation / adaptation of firing"`
	GknaSlow float32 `desc:"conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack) -- produces accommodation / adaptation of firing"`
	Gnmda    float32 `desc:"net NMDA conductance, after Vm gating and Gbar -- added directly to Ge as it has the same reversal potential."`
	NMDA     float32 `desc:"NMDA channel activation -- underlying time-integrated value with decay"`
	NMDASyn  float32 `desc:"synaptic NMDA activation directly from projection(s)"`
	GgabaB   float32 `desc:"net GABA-B conductance, after Vm gating and Gbar + Gbase -- set to Gk for GIRK, with .1 reversal potential."`
	GABAB    float32 `desc:"GABA-B / GIRK activation -- time-integrated value with rise and decay time constants"`
	GABABx   float32 `desc:"GABA-B / GIRK internal drive variable -- gets the raw activation and decays"`
	Attn     float32 `desc:"Attentional modulation factor, which can be set by special layers such as the TRC -- multiplies Ge"`
}

var NeuronVars = []string{"Spike", "ISI", "ISIAvg", "Act", "ActInt", "Ge", "Gi", "Gk", "Inet", "Vm", "VmDend", "Targ", "Ext", "AvgSS", "AvgS", "AvgM", "AvgSLrn", "AvgMLrn", "ActSt1", "ActSt2", "ActM", "ActP", "ActDif", "ActDel", "ActPrv", "RLrate", "ActAvg", "AvgPct", "TrgAvg", "DTrgAvg", "AvgDif", "Noise", "GiSyn", "GiSelf", "GeRaw", "GiRaw", "GeM", "GiM", "GknaFast", "GknaMed", "GknaSlow", "Gnmda", "NMDA", "NMDASyn", "GgabaB", "GABAB", "GABABx", "Attn"}

var NeuronVarsMap map[string]int

var NeuronVarProps = map[string]string{
	"Ge":       `min:"-2" max:"2"`,
	"GeM":      `min:"-2" max:"2"`,
	"Vm":       `min:"0" max:"1"`,
	"VmDend":   `min:"0" max:"1"`,
	"ISI":      `auto-scale:"+"`,
	"ISIAvg":   `auto-scale:"+"`,
	"Gi":       `auto-scale:"+"`,
	"Gk":       `auto-scale:"+"`,
	"ActDel":   `auto-scale:"+"`,
	"ActDif":   `auto-scale:"+"`,
	"AvgPct":   `min:"-2" max:"2"`,
	"TrgAvg":   `min:"-2" max:"2"`,
	"DTrgAvg":  `auto-scale:"+"`,
	"GknaFast": `auto-scale:"+"`,
	"GknaMed":  `auto-scale:"+"`,
	"GknaSlow": `auto-scale:"+"`,
	"Gnmda":    `auto-scale:"+"`,
	"NMDA":     `auto-scale:"+"`,
	"GgabaB":   `auto-scale:"+"`,
	"GABAB":    `auto-scale:"+"`,
	"GABABx":   `auto-scale:"+"`,
}

func init() {
	NeuronVarsMap = make(map[string]int, len(NeuronVars))
	for i, v := range NeuronVars {
		NeuronVarsMap[v] = i
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
