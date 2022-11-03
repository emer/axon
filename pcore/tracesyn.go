// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import "github.com/goki/mat32"

// note: DTr is just for display purposes and can be removed to optimize later

// TraceSyn holds extra synaptic state for trace projections
type TraceSyn struct {
	DTr float32 `desc:"delta trace = send * recv -- increments to Tr when a gating event happens."`
}

// VarByName returns synapse variable by name
func (sy *TraceSyn) VarByName(varNm string) float32 {
	switch varNm {
	case "DTr":
		return sy.DTr
	}
	return mat32.NaN()
}

// VarByIndex returns synapse variable by index
func (sy *TraceSyn) VarByIndex(varIdx int) float32 {
	switch varIdx {
	case 0:
		return sy.DTr
	}
	return mat32.NaN()
}

var TraceSynVars = []string{"DTr"}
