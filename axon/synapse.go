// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
)

//gosl: start synapse

// SynapseVars are the neuron variables representing current synaptic state,
// specifically weights.
type SynapseVars int32 //enums:enum

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

	// IMPORTANT: if DSWt is not the last, need to update gosl defn below
)

// SynapseCaVars are synapse variables for calcium involved in learning,
// which are data parallel input specific.
type SynapseCaVars int32 //enums:enum

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

	// IMPORTANT: if DiDWt is not the last, need to update gosl defn below
)

// SynapseIndexes are the neuron indexes and other uint32 values (flags, etc).
// There is only one of these per neuron -- not data parallel.
type SynapseIndexes int32 //enums:enum

const (
	// SynRecvIndex is receiving neuron index in network's global list of neurons
	SynRecvIndex SynapseIndexes = iota

	// SynSendIndex is sending neuron index in network's global list of neurons
	SynSendIndex

	// SynPrjnIndex is projection index in global list of projections organized as [Layers][RecvPrjns]
	SynPrjnIndex

	// IMPORTANT: if SynPrjnIndex is not the last, need to update gosl defn below
)

//gosl: end synapse

//gosl: hlsl synapse
/*
static const SynapseVars SynapseVarsN = DSWt + 1;
static const SynapseCaVars SynapseCaVarsN = DiDWt + 1;
static const SynapseIndexes SynapseIndexesN = SynPrjnIndex + 1;
*/
//gosl: end synapse

//gosl: start synapse

////////////////////////////////////////////////
// 	Strides

// SynapseVarStrides encodes the stride offsets for synapse variable access
// into network float32 array.
type SynapseVarStrides struct {

	// synapse level
	Synapse uint32

	// variable level
	Var uint32

	pad, pad1 uint32
}

// note: when increasing synapse var capacity beyond 2^31, change back to uint64

// Index returns the index into network float32 array for given synapse, and variable
func (ns *SynapseVarStrides) Index(synIndex uint32, nvar SynapseVars) uint32 {
	// return uint64(synIndex)*uint64(ns.Synapse) + uint64(nvar)*uint64(ns.Var)
	return synIndex*ns.Synapse + uint32(nvar)*ns.Var
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

// SynapseCaStrides encodes the stride offsets for synapse variable access
// into network float32 array.  Data is always the inner-most variable.
type SynapseCaStrides struct {

	// synapse level
	Synapse uint64

	// variable level
	Var uint64
}

// Index returns the index into network float32 array for given synapse, data, and variable
func (ns *SynapseCaStrides) Index(synIndex, di uint32, nvar SynapseCaVars) uint64 {
	return uint64(synIndex)*ns.Synapse + uint64(nvar)*ns.Var + uint64(di)
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
// 	Indexes

// SynapseIndexStrides encodes the stride offsets for synapse index access
// into network uint32 array.
type SynapseIndexStrides struct {

	// synapse level
	Synapse uint32

	// index value level
	Index uint32

	pad, pad1 uint32
}

// Idx returns the index into network uint32 array for given synapse, index value
func (ns *SynapseIndexStrides) Idx(synIdx uint32, idx SynapseIndexes) uint32 {
	return synIdx*ns.Synapse + uint32(idx)*ns.Index
}

// SetSynapseOuter sets strides with synapses as outer dimension:
// [Synapses][Indexes] (outer to inner), which is optimal for CPU-based
// computation.
func (ns *SynapseIndexStrides) SetSynapseOuter() {
	ns.Synapse = uint32(SynapseIndexesN)
	ns.Index = 1
}

// SetIndexOuter sets strides with indexes as outer dimension:
// [Indexes][Synapses] (outer to inner), which is optimal for GPU-based
// computation.
func (ns *SynapseIndexStrides) SetIndexOuter(nsyn int) {
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
