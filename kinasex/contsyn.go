// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinasex

import "cogentcore.org/core/math32"

// ContSyn holds extra synaptic state for continuous learning
type ContSyn struct {

	// transitional, temporary DWt value, which is updated in a window after synaptic activity when Ca levels are still elevated, and added to the DWt value after a longer break of spiking where there is enough time for CaMKII driven AMPA receptor trafficking to take place
	TDWt float32

	// maximum CaD value since last DWt change -- DWt occurs when current CaD has decreased by a given proportion from this recent peak
	CaDMax float32
}

// VarByName returns synapse variable by name
func (sy *ContSyn) VarByName(varNm string) float32 {
	switch varNm {
	case "TDWt":
		return sy.TDWt
	case "CaDMax":
		return sy.CaDMax
	}
	return math32.NaN()
}

// VarByIndex returns synapse variable by index
func (sy *ContSyn) VarByIndex(varIndex int) float32 {
	switch varIndex {
	case 0:
		return sy.TDWt
	case 1:
		return sy.CaDMax
	}
	return math32.NaN()
}

var ContSynVars = []string{"TDWt", "CaDMax"}
