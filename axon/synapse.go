// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"strings"
)

//gosl:start

// SynapseVars are the synapse variables representing synaptic weights, etc.
// These do not depend on the data parallel index (di).
// See [SynapseTraceVars] for variables that do depend on di.
type SynapseVars int32 //enums:enum

const (
	// Wt is the effective synaptic weight value, determining how much conductance
	// one presynaptic spike drives into the receiving neuron. Biologically it represents
	// the number of effective AMPA receptors in the synapse.
	// Wt = [SWt] * WtSig([LWt]), where WtSig is the sigmoidal constrast enhancement
	// function that produces values between 0-2 based on LWt, centered on 1.
	Wt SynapseVars = iota

	// LWt is the rapid, online learning, linear weight value. It learns on every
	// trial according	to the learning rate (LRate) parameter. Biologically,
	// this represents the internal biochemical processes that drive the trafficking
	// of AMPA receptors in the synaptic density.
	LWt

	// SWt is a slowly adapting structural weight value, which acts as a
	// multiplicative scaling factor on net synaptic efficacy [Wt].
	// Biologically it represents the physical size and efficacy of the dendritic spine.
	// SWt values adapt in a slower outer loop along with synaptic scaling,
	// with constraints to prevent runaway positive feedback loops and maintain
	// variance and further capacity to learn. Initial weight variance is partially or
	// fully captured in the SWt values, with LWt capturing the remainder.
	SWt

	// DWt is delta (change in) synaptic weight, from learning. This updates [LWt]
	// on every trial. It is reset to 0 after it is applied, but the network view
	// captures this value just prior to application.
	DWt

	// DSWt is the accumulated change in the [SWt] slow structural weight, computed
	// as the accumulation of [DWt] values over the longer slow weight update window.
	DSWt
)

// SynapseTraceVars are synaptic variables that depend on the data
// parallel index, for accumulating learning traces and weight changes per data.
type SynapseTraceVars int32 //enums:enum

const (
	// Tr is trace of synaptic activity over time, which is used for
	// credit assignment in learning.
	// In MatrixPath this is a tag that is then updated later when US occurs.
	Tr SynapseTraceVars = iota

	// DTr is delta (change in) Tr trace of synaptic activity over time.
	DTr

	// DiDWt is delta weight for each data parallel index (Di).
	// This is directly computed from the Ca values (in cortical version)
	// and then aggregated into the overall DWt (which may be further
	// integrated across MPI nodes), which then drives changes in Wt values.
	DiDWt
)

// SynapseIndexVars are synapse-level indexes used to access neurons and paths
// from the individual synapse level of processing.
type SynapseIndexVars int32 //enums:enum

const (
	// SynRecvIndex is receiving neuron index in network's global list of neurons
	SynRecvIndex SynapseIndexVars = iota

	// SynSendIndex is sending neuron index in network's global list of neurons
	SynSendIndex

	// SynPathIndex is pathway index in global list of pathways organized as [Layers][RecvPaths]
	SynPathIndex
)

//gosl:end

// SynapseVarProps has all of the display properties for synapse variables, including desc tooltips
var SynapseVarProps = map[string]string{
	"Wt":    `cat:"Wts"`,
	"LWt":   `cat:"Wts"`,
	"SWt":   `cat:"Wts"`,
	"DWt":   `cat:"Wts" auto-scale:"+"`,
	"DSWt":  `cat:"Wts" auto-scale:"+"`,
	"Tr":    `cat:"Wts" auto-scale:"+"`,
	"DTr":   `cat:"Wts" auto-scale:"+"`,
	"DiDWt": `cat:"Wts" auto-scale:"+"`,
}

var (
	SynapseVarNames []string
	SynapseVarsMap  map[string]int
)

func init() {
	SynapseVarsMap = make(map[string]int, int(SynapseVarsN)+int(SynapseTraceVarsN))
	for i := Wt; i < SynapseVarsN; i++ {
		vnm := i.String()
		SynapseVarNames = append(SynapseVarNames, vnm)
		SynapseVarsMap[vnm] = int(i)
		tag := SynapseVarProps[vnm]
		SynapseVarProps[vnm] = tag + ` doc:"` + strings.ReplaceAll(i.Desc(), "\n", " ") + `"`
	}
	for i := Tr; i < SynapseTraceVarsN; i++ {
		vnm := i.String()
		SynapseVarNames = append(SynapseVarNames, vnm)
		SynapseVarsMap[vnm] = int(SynapseVarsN) + int(i)
		tag := SynapseVarProps[vnm]
		SynapseVarProps[vnm] = tag + ` doc:"` + strings.ReplaceAll(i.Desc(), "\n", " ") + `"`
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
