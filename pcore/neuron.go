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
	NeuronVars = []string{"DA", "ActLrn", "PhasicMax", "DALrn", "ACh", "Ca", "KCa"}

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
// Base pcore neurons

// PCoreNeuron holds the extra neuron (unit) level variables
// for pcore computation.
type PCoreNeuron struct {
	ActLrn    float32 `desc:"learning activity value -- based on PhasicMax activation plus other potential factors depending on layer type."`
	PhasicMax float32 `desc:"maximum phasic activation value during a gating window."`
}

var (
	PCoreNeuronVars    = []string{"ActLrn", "PhasicMax"}
	PCoreNeuronVarsMap map[string]int
)

func (nrn *PCoreNeuron) VarNames() []string {
	return PCoreNeuronVars
}

// PCoreNeuronVarIdxByName returns the index of the variable in the PCoreNeuron, or error
func PCoreNeuronVarIdxByName(varNm string) (int, error) {
	i, ok := PCoreNeuronVarsMap[varNm]
	if !ok {
		return 0, fmt.Errorf("PCoreNeuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in PCoreNeuronVars list)
func (nrn *PCoreNeuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (nrn *PCoreNeuron) VarByName(varNm string) (float32, error) {
	i, err := PCoreNeuronVarIdxByName(varNm)
	if err != nil {
		return 0, err
	}
	return nrn.VarByIndex(i), nil
}

//////////////////////////////////////////////////////////////////////
// STN neurons

// STNNeuron holds the extra neuron (unit) level variables for STN computation.
type STNNeuron struct {
	Ca  float32 `desc:"intracellular Calcium concentration -- increased by bursting and elevated levels of activation, drives KCa currents that result in hyperpolarization / inhibition."`
	KCa float32 `desc:"Calcium-gated potassium channel conductance level, computed using function from gillies & Willshaw 2006 as function of Ca."`
}

var (
	STNNeuronVars    = []string{"Ca", "KCa"}
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
