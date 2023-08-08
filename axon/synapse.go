// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"

	"github.com/goki/ki/kit"
)

//go:generate stringer -type=SynapseVars
//go:generate stringer -type=SynapseCaVars
//go:generate stringer -type=SynapseIdxs

var KiT_SynapseVars = kit.Enums.AddEnum(SynapseVarsN, kit.NotBitFlag, nil)

func (ev SynapseVars) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *SynapseVars) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

var KiT_SynapseCaVars = kit.Enums.AddEnum(SynapseCaVarsN, kit.NotBitFlag, nil)

func (ev SynapseCaVars) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *SynapseCaVars) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

var KiT_SynapseIdxs = kit.Enums.AddEnum(SynapseIdxsN, kit.NotBitFlag, nil)

func (ev SynapseIdxs) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *SynapseIdxs) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

//gosl: start synapse

// SynapseVars are the neuron variables representing current synaptic state,
// specifically weights.
type SynapseVars int32

const (
	// Wt is effective synaptic weight value, determining how much conductance one spike drives on the receiving neuron, representing the actual number of effective AMPA receptors in the synapse.  Wt = SWt * WtSig(LWt), where WtSig produces values between 0-2 based on LWt, centered on 1.
	Wt SynapseVars = iota

	// LWt is rapidly learning, linear weight value -- learns according to the lrate specified in the connection spec.  Biologically, this represents the internal biochemical processes that drive the trafficking of AMPA receptors in the synaptic density.  Initially all LWt are .5, which gives 1 from WtSig function.
	LWt

	// SWt is slowly adapting structural weight value, which acts as a multiplicative scaling factor on synaptic efficacy: biologically represents the physical size and efficacy of the dendritic spine.  SWt values adapt in an outer loop along with synaptic scaling, with constraints to prevent runaway positive feedback loops and maintain variance and further capacity to learn.  Initial variance is all in SWt, with LWt set to .5, and scaling absorbs some of LWt into SWt.
	SWt

	// DWt is delta (change in) synaptic weight, from learning -- updates LWt which then updates Wt.
	DWt

	// DSWt is change in SWt slow synaptic weight -- accumulates DWt
	DSWt

	SynapseVarsN
)

// SynapseVarStrides encodes the stride offsets for synapse variable access
// into network float32 array.
type SynapseVarStrides struct {

	// synapse level
	Synapse uint32 `desc:"synapse level"`

	// variable level
	Var uint32 `desc:"variable level"`

	pad, pad1 uint32
}

// note: when increasing synapse var capacity beyond 2^31, change back to uint64

// Idx returns the index into network float32 array for given synapse, and variable
func (ns *SynapseVarStrides) Idx(synIdx uint32, nvar SynapseVars) uint32 {
	// return uint64(synIdx)*uint64(ns.Synapse) + uint64(nvar)*uint64(ns.Var)
	return synIdx*ns.Synapse + uint32(nvar)*ns.Var
}

// SetSynapseOuter sets strides with synapses as outer loop:
// [Synapses][Vars], which is optimal for CPU-based computation.
func (ns *SynapseVarStrides) SetSynapseOuter() {
	ns.Synapse = uint32(SynapseVarsN)
	ns.Var = 1
}

// SetVarOuter sets strides with vars as outer loop:
// [Vars][Synapses], which is optimal for GPU-based computation.
func (ns *SynapseVarStrides) SetVarOuter(nsyn int) {
	ns.Var = uint32(nsyn)
	ns.Synapse = 1
}

////////////////////////////////////////////////
// 	SynapseCaVars

// SynapseCaVars are synapse variables for calcium involved in learning,
// which are data parallel input specific.
type SynapseCaVars int32

const (
	// CaM is first stage running average (mean) Ca calcium level (like CaM = calmodulin), feeds into CaP
	CaM SynapseCaVars = iota

	// CaP is shorter timescale integrated CaM value, representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule
	CaP

	// CaD is longer timescale integrated CaP value, representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule
	CaD

	// CaUpT is time in CyclesTotal of last updating of Ca values at the synapse level, for optimized synaptic-level Ca integration -- converted to / from uint32
	CaUpT

	// Tr is trace of synaptic activity over time -- used for credit assignment in learning.  In MatrixPrjn this is a tag that is then updated later when US occurs.
	Tr

	// DTr is delta (change in) Tr trace of synaptic activity over time
	DTr

	// DiDWt is delta weight for each data parallel index (Di) -- this is directly computed from the Ca values (in cortical version) and then aggregated into the overall DWt (which may be further integrated across MPI nodes), which then drives changes in Wt values
	DiDWt

	SynapseCaVarsN
)

// SynapseCaStrides encodes the stride offsets for synapse variable access
// into network float32 array.  Data is always the inner-most variable.
type SynapseCaStrides struct {

	// synapse level
	Synapse uint64 `desc:"synapse level"`

	// variable level
	Var uint64 `desc:"variable level"`
}

// Idx returns the index into network float32 array for given synapse, data, and variable
func (ns *SynapseCaStrides) Idx(synIdx, di uint32, nvar SynapseCaVars) uint64 {
	return uint64(synIdx)*ns.Synapse + uint64(nvar)*ns.Var + uint64(di)
}

// SetSynapseOuter sets strides with synapses as outer loop:
// [Synapses][Vars][Data], which is optimal for CPU-based computation.
func (ns *SynapseCaStrides) SetSynapseOuter(ndata int) {
	ns.Synapse = uint64(ndata) * uint64(SynapseCaVarsN)
	ns.Var = uint64(ndata)
}

// SetVarOuter sets strides with vars as outer loop:
// [Vars][Synapses][Data], which is optimal for GPU-based computation.
func (ns *SynapseCaStrides) SetVarOuter(nsyn, ndata int) {
	ns.Var = uint64(ndata) * uint64(nsyn)
	ns.Synapse = uint64(ndata)
}

////////////////////////////////////////////////
// 	Idxs

// SynapseIdxs are the neuron indexes and other uint32 values (flags, etc).
// There is only one of these per neuron -- not data parallel.
type SynapseIdxs int32

const (
	// SynRecvIdx is receiving neuron index in network's global list of neurons
	SynRecvIdx SynapseIdxs = iota

	// SynSendIdx is sending neuron index in network's global list of neurons
	SynSendIdx

	// SynPrjnIdx is projection index in global list of projections organized as [Layers][RecvPrjns]
	SynPrjnIdx

	SynapseIdxsN
)

// SynapseIdxStrides encodes the stride offsets for synapse index access
// into network uint32 array.
type SynapseIdxStrides struct {

	// synapse level
	Synapse uint32 `desc:"synapse level"`

	// index value level
	Index uint32 `desc:"index value level"`

	pad, pad1 uint32
}

// Idx returns the index into network uint32 array for given synapse, index value
func (ns *SynapseIdxStrides) Idx(synIdx uint32, idx SynapseIdxs) uint32 {
	return synIdx*ns.Synapse + uint32(idx)*ns.Index
}

// SetSynapseOuter sets strides with synapses as outer dimension:
// [Synapses][Idxs] (outer to inner), which is optimal for CPU-based
// computation.
func (ns *SynapseIdxStrides) SetSynapseOuter() {
	ns.Synapse = uint32(SynapseIdxsN)
	ns.Index = 1
}

// SetIdxOuter sets strides with indexes as outer dimension:
// [Idxs][Synapses] (outer to inner), which is optimal for GPU-based
// computation.
func (ns *SynapseIdxStrides) SetIdxOuter(nsyn int) {
	ns.Index = uint32(nsyn)
	ns.Synapse = 1
}

//gosl: end synapse

// SynapseVarProps has all of the display properties for synapse variables, including desc tooltips
var SynapseVarProps = map[string]string{
	"Wt ":   `desc:"effective synaptic weight value, determining how much conductance one spike drives on the receiving neuron, representing the actual number of effective AMPA receptors in the synapse.  Wt = SWt * WtSig(LWt), where WtSig produces values between 0-2 based on LWt, centered on 1."`,
	"LWt":   `desc:"rapidly learning, linear weight value -- learns according to the lrate specified in the connection spec.  Biologically, this represents the internal biochemical processes that drive the trafficking of AMPA receptors in the synaptic density.  Initially all LWt are .5, which gives 1 from WtSig function."`,
	"SWt":   `desc:"slowly adapting structural weight value, which acts as a multiplicative scaling factor on synaptic efficacy: biologically represents the physical size and efficacy of the dendritic spine.  SWt values adapt in an outer loop along with synaptic scaling, with constraints to prevent runaway positive feedback loops and maintain variance and further capacity to learn.  Initial variance is all in SWt, with LWt set to .5, and scaling absorbs some of LWt into SWt."`,
	"DWt":   `auto-scale:"+" desc:"delta (change in) synaptic weight, from learning -- updates LWt which then updates Wt."`,
	"DSWt":  `auto-scale:"+" desc:"change in SWt slow synaptic weight -- accumulates DWt"`,
	"CaM":   `auto-scale:"+" desc:"first stage running average (mean) Ca calcium level (like CaM = calmodulin), feeds into CaP"`,
	"CaP":   `auto-scale:"+"desc:"shorter timescale integrated CaM value, representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule"`,
	"CaD":   `auto-scale:"+" desc:"longer timescale integrated CaP value, representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule"`,
	"Tr":    `auto-scale:"+" desc:"trace of synaptic activity over time -- used for credit assignment in learning.  In MatrixPrjn this is a tag that is then updated later when US occurs."`,
	"DTr":   `auto-scale:"+" desc:"delta (change in) Tr trace of synaptic activity over time"`,
	"DiDWt": `auto-scale:"+" desc:"delta weight for each data parallel index (Di) -- this is directly computed from the Ca values (in cortical version) and then aggregated into the overall DWt (which may be further integrated across MPI nodes), which then drives changes in Wt values"`,
}

var (
	SynapseVarNames []string
	SynapseVarsMap  map[string]int
)

func init() {
	SynapseVarsMap = make(map[string]int, int(SynapseVarsN)+int(SynapseCaVarsN))
	for i := Wt; i < SynapseVarsN; i++ {
		vnm := i.String()
		SynapseVarNames = append(SynapseVarNames, vnm)
		SynapseVarsMap[vnm] = int(i)
	}
	for i := CaM; i < SynapseCaVarsN; i++ {
		vnm := i.String()
		SynapseVarNames = append(SynapseVarNames, vnm)
		SynapseVarsMap[vnm] = int(SynapseVarsN) + int(i)
	}
}

// SynapseVarByName returns the index of the variable in the Synapse, or error
func SynapseVarByName(varNm string) (int, error) {
	i, ok := SynapseVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Synapse VarByName: variable name: %s not valid", varNm)
	}
	return i, nil
}
