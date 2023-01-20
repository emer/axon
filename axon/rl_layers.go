// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"

	"github.com/emer/etable/minmax"
)

//gosl: start rl_layers

// note: RewLayer gets Rew value from Context

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
// This entire computation happens CPU-side in Layer.PostCycle, via
// direct access to other layer state.
type RWDaParams struct {
	RWPredLayIdx uint32 `desc:"RWPredLayer layer index"`

	pad, pad1, pad2 uint32
}

func (rp *RWDaParams) Defaults() {

}

func (rp *RWDaParams) Update() {

}

// TDIntegParams are params for reward integrator layer
type TDIntegParams struct {
	Discount     float32 `desc:"discount factor -- how much to discount the future prediction from RewPred"`
	PredGain     float32 `desc:"gain factor on rew pred activations"`
	TDPredLayIdx uint32  `desc:"idx of TDPredLayer to get reward prediction from "`

	pad uint32
}

func (tp *TDIntegParams) Defaults() {
	tp.Discount = 0.9
	tp.PredGain = 1
}

func (tp *TDIntegParams) Update() {
}

//gosl: end rl_layers

// note: Defaults not called on GPU

func (ly *LayerParams) RWPredLayerDefaults() {
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Act.Dt.GeTau = 40
}

// RWDaPostBuild does post-Build config of Pulvinar based on BuildConfig options
func (ly *Layer) RWDaPostBuild() {
	rnm, err := ly.BuildConfigByName("RWPredLayName")
	if err != nil {
		return
	}
	dly, err := ly.Network.LayerByNameTry(rnm)
	if err != nil {
		log.Println(err)
		return
	}
	ly.Params.RWDa.RWPredLayIdx = uint32(dly.Index())
}

func (ly *LayerParams) TDIntegLayerDefaults() {
	ly.Inhib.ActAvg.Nominal = .5
}
