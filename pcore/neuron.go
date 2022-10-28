// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"fmt"
	"unsafe"

	"github.com/emer/axon/axon"
)

var (
	// NeuronVars are extra neuron variables for pcore -- union across all types
	NeuronVars = []string{"Burst", "BurstPrv", "CtxtGe", "DA", "DALrn", "ACh", "Gated", "SKCai", "SKCaM", "Gsk"}

	// NeuronVarsAll is the pcore collection of all neuron-level vars
	NeuronVarsAll []string

	// SynVarsAll is the pcore collection of all synapse-level vars (includes TraceSynVars)
	SynVarsAll []string
)

func init() {
	ln := len(axon.NeuronVars)
	NeuronVarsAll = make([]string, len(NeuronVars)+ln)
	copy(NeuronVarsAll, axon.NeuronVars)
	copy(NeuronVarsAll[ln:], NeuronVars)

	ln = len(axon.SynapseVars)
	SynVarsAll = make([]string, len(TraceSynVars)+ln)
	copy(SynVarsAll, axon.SynapseVars)
	copy(SynVarsAll[ln:], TraceSynVars)

	STNNeuronVarsMap = make(map[string]int, len(STNNeuronVars))
	for i, v := range STNNeuronVars {
		STNNeuronVarsMap[v] = i
	}
}

//////////////////////////////////////////////////////////////////////
// STN neurons

// STNNeuron holds the extra neuron (unit) level variables for STN computation.
type STNNeuron struct {
	SKCai float32 `desc:"intracellular Calcium concentration for activation of SKCa channels, driven by VGCC activation from spiking and decaying / buffererd relatively slowly."`
	SKCaM float32 `desc:"Calcium-gated potassium channel gating factor, driven by SKCai via a Hill equation as in chans.SKPCaParams."`
	Gsk   float32 `desc:"Calcium-gated potassium channel conductance as a function of Gbar * SKCaM."`
}

var (
	STNNeuronVars    = []string{"SKCai", "SKCaM", "Gsk"}
	STNNeuronVarsMap map[string]int
)

func (nrn *STNNeuron) VarNames() []string {
	return STNNeuronVars
}

// STNNeuronVarIdxByName returns the index of the variable in the STNNeuron, or error
func STNNeuronVarIdxByName(varNm string) (int, error) {
	i, ok := STNNeuronVarsMap[varNm]
	if !ok {
		return 0, fmt.Errorf("STNNeuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in STNNeuronVars list)
func (nrn *STNNeuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (nrn *STNNeuron) VarByName(varNm string) (float32, error) {
	i, err := STNNeuronVarIdxByName(varNm)
	if err != nil {
		return 0, err
	}
	return nrn.VarByIndex(i), nil
}
