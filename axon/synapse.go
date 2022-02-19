// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"reflect"
	"unsafe"
)

// SynapseVarStart is the byte offset of fields in the Synapse structure
// where the float32 named variables start.
// Note: all non-float32 infrastructure variables must be at the start!
const SynapseVarStart = 4

// axon.Synapse holds state for the synaptic connection between neurons
type Synapse struct {
	SpikeT int32   `desc:"time of last spiking at the synapse level, for optimized synaptic-level Ca integration"`
	Wt     float32 `desc:"effective synaptic weight value, determining how much conductance one spike drives on the receiving neuron.  Wt = SWt * WtSig(LWt), where WtSig produces values between 0-2 based on LWt, centered on 1"`
	SWt    float32 `desc:"slowly adapting structural weight value, which acts as a multiplicative scaling factor on synaptic efficacy: biologically represents the physical size and efficacy of the dendritic spine, while the LWt reflects the AMPA receptor efficacy and number.  SWt values adapt in an outer loop along with synaptic scaling, with constraints to prevent runaway positive feedback loops and maintain variance and further capacity to learn.  Initial variance is all in SWt, with LWt set to .5, and scaling absorbs some of LWt into SWt."`
	LWt    float32 `desc:"rapidly learning, linear weight value -- learns according to the lrate specified in the connection spec.  Initially all LWt are .5, which gives 1 from WtSig function, "`
	DWt    float32 `desc:"change in synaptic weight, from learning"`
	DSWt   float32 `desc:"change in SWt slow synaptic weight -- accumulates DWt"`
	CaM    float32 `desc:"first stage running average (mean) Ca calcium level (like CaM = calmodulin), feeds into CaP, for Kinase based learning -- for SynNMDACa = send.SnmdaO * recv.RCa"`
	CaP    float32 `desc:"shorter timescale integrated CaM value, representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule"`
	CaD    float32 `desc:"longer timescale integrated CaP value, representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule"`
	DWtRaw float32 `desc:"raw change in synaptic weight, from learning -- temporary for Kinase analysis"`
}

func (sy *Synapse) VarNames() []string {
	return SynapseVars
}

var SynapseVars = []string{"Wt", "SWt", "LWt", "DWt", "DSWt", "CaM", "CaP", "CaD", "DWtRaw"}

var SynapseVarProps = map[string]string{
	"DWt":    `auto-scale:"+"`,
	"DSWt":   `auto-scale:"+"`,
	"CaM":    `auto-scale:"+"`,
	"CaP":    `auto-scale:"+"`,
	"CaD":    `auto-scale:"+"`,
	"DWtRaw": `auto-scale:"+"`,
}

var SynapseVarsMap map[string]int

func init() {
	SynapseVarsMap = make(map[string]int, len(SynapseVars))
	typ := reflect.TypeOf((*Synapse)(nil)).Elem()
	for i, v := range SynapseVars {
		SynapseVarsMap[v] = i
		pstr := SynapseVarProps[v]
		if fld, has := typ.FieldByName(v); has {
			if desc, ok := fld.Tag.Lookup("desc"); ok {
				pstr += ` desc:"` + desc + `"`
				SynapseVarProps[v] = pstr
			}
		}
	}
}

// SynapseVarByName returns the index of the variable in the Synapse, or error
func SynapseVarByName(varNm string) (int, error) {
	i, ok := SynapseVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Synapse VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in SynapseVars list)
func (sy *Synapse) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(sy)) + uintptr(SynapseVarStart+4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (sy *Synapse) VarByName(varNm string) (float32, error) {
	i, err := SynapseVarByName(varNm)
	if err != nil {
		return 0, err
	}
	return sy.VarByIndex(i), nil
}

func (sy *Synapse) SetVarByIndex(idx int, val float32) {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(sy)) + uintptr(SynapseVarStart+4*idx)))
	*fv = val
}

// SetVarByName sets synapse variable to given value
func (sy *Synapse) SetVarByName(varNm string, val float32) error {
	i, err := SynapseVarByName(varNm)
	if err != nil {
		return err
	}
	sy.SetVarByIndex(i, val)
	return nil
}
