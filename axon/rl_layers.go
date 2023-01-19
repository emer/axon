// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/emer/etable/minmax"

//gosl: start rl_layers

// RWPredParams parameterizes reward prediction for a simple Rescorla-Wagner
// learning dynamic (i.e., PV learning in the PVLV framework).
type RWPredParams struct {
	PredRange minmax.F32 `desc:"default 0.1..0.99 range of predictions that can be represented -- having a truncated range preserves some sensitivity in dopamine at the extremes of good or poor performance"`
}

func (rp *RWPredParams) Defaults() {
	rp.PredRange.Set(0.01, 0.99)
}

func (rp *RWPredParams) Update() {

}

// RWDaParams computes a dopamine (DA) signal using simple Rescorla-Wagner
// learning dynamic (i.e., PV learning in the PVLV framework).
type RWDaParams struct {
	RewLayIdx    uint32 `desc:"reward layer index"`
	RWPredLayIdx uint32 `desc:"RWPredLayer layer index"`

	pad, pad1 uint32
}

func (rp *RWDaParams) Defaults() {

}

func (rp *RWDaParams) Update() {

}

// RWDaVals are values computed in CPU and available to
// the GPU during update of RWDaLayer in the Time struct
type RWDaVals struct {
	RewLayIdx    uint32 `desc:"reward layer index"`
	RWPredLayIdx uint32 `desc:"RWPredLayer layer index"`

	pad, pad1 uint32
}

//gosl: end rl_layers

// note: Defaults not called on GPU

func (ly *LayerParams) RWPredLayerDefaults() {
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Act.Dt.GeTau = 40
}
