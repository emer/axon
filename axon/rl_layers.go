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
type RWDaParams struct {
	TonicGe      float32 `desc:"tonic baseline Ge level for DA = 0 -- +/- are between 0 and 2*TonicGe -- just for spiking display of computed DA value"`
	RWPredLayIdx uint32  `inactive:"+" desc:"idx of RWPredLayer to get reward prediction from -- set during Build from BuildConfig RWPredLayName"`

	pad, pad1 uint32
}

func (rp *RWDaParams) Defaults() {
	rp.TonicGe = 0.3
}

func (rp *RWDaParams) Update() {
}

// GeFmDA returns excitatory conductance from DA dopamine value
func (rp *RWDaParams) GeFmDA(da float32) float32 {
	return rp.TonicGe * (1.0 + da)
}

// TDIntegParams are params for reward integrator layer
type TDIntegParams struct {
	Discount     float32 `desc:"discount factor -- how much to discount the future prediction from TDPred"`
	PredGain     float32 `desc:"gain factor on TD rew pred activations"`
	TDPredLayIdx uint32  `inactive:"+" desc:"idx of TDPredLayer to get reward prediction from -- set during Build from BuildConfig TDPredLayName"`

	pad uint32
}

func (tp *TDIntegParams) Defaults() {
	tp.Discount = 0.9
	tp.PredGain = 1
}

func (tp *TDIntegParams) Update() {
}

// TDDaParams are params for dopamine (DA) signal as the temporal difference (TD)
// between the TDIntegLayer activations in the minus and plus phase.
type TDDaParams struct {
	TonicGe       float32 `desc:"tonic baseline Ge level for DA = 0 -- +/- are between 0 and 2*TonicGe -- just for spiking display of computed DA value"`
	TDIntegLayIdx uint32  `inactive:"+" desc:"idx of TDIntegLayer to get reward prediction from -- set during Build from BuildConfig TDIntegLayName"`

	pad, pad1 uint32
}

func (tp *TDDaParams) Defaults() {
	tp.TonicGe = 0.3
}

func (tp *TDDaParams) Update() {
}

// GeFmDA returns excitatory conductance from DA dopamine value
func (tp *TDDaParams) GeFmDA(da float32) float32 {
	return tp.TonicGe * (1.0 + da)
}

//gosl: end rl_layers

// note: Defaults not called on GPU

func (ly *LayerParams) RWLayerDefaults() {
	ly.Inhib.ActAvg.Nominal = .5
}

func (ly *LayerParams) RWPredLayerDefaults() {
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Act.Dt.GeTau = 40
}

// RWDaPostBuild does post-Build config
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

func (ly *LayerParams) TDLayerDefaults() {
	ly.Inhib.ActAvg.Nominal = .5
}

func (ly *LayerParams) TDPredLayerDefaults() {
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Act.Dt.GeTau = 40
}

// TDIntegPostBuild does post-Build config
func (ly *Layer) TDIntegPostBuild() {
	rnm, err := ly.BuildConfigByName("TDPredLayName")
	if err != nil {
		return
	}
	dly, err := ly.Network.LayerByNameTry(rnm)
	if err != nil {
		log.Println(err)
		return
	}
	ly.Params.TDInteg.TDPredLayIdx = uint32(dly.Index())
}

// TDDaPostBuild does post-Build config
func (ly *Layer) TDDaPostBuild() {
	rnm, err := ly.BuildConfigByName("TDIntegLayName")
	if err != nil {
		return
	}
	dly, err := ly.Network.LayerByNameTry(rnm)
	if err != nil {
		log.Println(err)
		return
	}
	ly.Params.TDDa.TDIntegLayIdx = uint32(dly.Index())
}
