// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/etable/minmax"
)

//gosl: start rl_layers

// RWPredParams parameterizes reward prediction for a simple Rescorla-Wagner
// learning dynamic (i.e., PV learning in the PVLV framework).
type RWPredParams struct {

	// default 0.1..0.99 range of predictions that can be represented -- having a truncated range preserves some sensitivity in dopamine at the extremes of good or poor performance
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

	// tonic baseline Ge level for DA = 0 -- +/- are between 0 and 2*TonicGe -- just for spiking display of computed DA value
	TonicGe float32 `desc:"tonic baseline Ge level for DA = 0 -- +/- are between 0 and 2*TonicGe -- just for spiking display of computed DA value"`

	// idx of RWPredLayer to get reward prediction from -- set during Build from BuildConfig RWPredLayName
	RWPredLayIdx int32 `inactive:"+" desc:"idx of RWPredLayer to get reward prediction from -- set during Build from BuildConfig RWPredLayName"`

	pad, pad1 uint32
}

func (rp *RWDaParams) Defaults() {
	rp.TonicGe = 0.2
}

func (rp *RWDaParams) Update() {
}

// GeFmDA returns excitatory conductance from DA dopamine value
func (rp *RWDaParams) GeFmDA(da float32) float32 {
	ge := rp.TonicGe * (1.0 + da)
	if ge < 0 {
		ge = 0
	}
	return ge
}

// TDIntegParams are params for reward integrator layer
type TDIntegParams struct {

	// discount factor -- how much to discount the future prediction from TDPred
	Discount float32 `desc:"discount factor -- how much to discount the future prediction from TDPred"`

	// gain factor on TD rew pred activations
	PredGain float32 `desc:"gain factor on TD rew pred activations"`

	// idx of TDPredLayer to get reward prediction from -- set during Build from BuildConfig TDPredLayName
	TDPredLayIdx int32 `inactive:"+" desc:"idx of TDPredLayer to get reward prediction from -- set during Build from BuildConfig TDPredLayName"`

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

	// tonic baseline Ge level for DA = 0 -- +/- are between 0 and 2*TonicGe -- just for spiking display of computed DA value
	TonicGe float32 `desc:"tonic baseline Ge level for DA = 0 -- +/- are between 0 and 2*TonicGe -- just for spiking display of computed DA value"`

	// idx of TDIntegLayer to get reward prediction from -- set during Build from BuildConfig TDIntegLayName
	TDIntegLayIdx int32 `inactive:"+" desc:"idx of TDIntegLayer to get reward prediction from -- set during Build from BuildConfig TDIntegLayName"`

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

func (ly *LayerParams) RWDefaults() {
	ly.Inhib.ActAvg.Nominal = .5
}

func (ly *LayerParams) RWPredDefaults() {
	ly.Acts.Decay.Act = 1
	ly.Acts.Decay.Glong = 1
	ly.Acts.Dt.GeTau = 40
}

// RWDaPostBuild does post-Build config
func (ly *Layer) RWDaPostBuild() {
	ly.Params.RWDa.RWPredLayIdx = ly.BuildConfigFindLayer("RWPredLayName", true)
}

func (ly *LayerParams) TDDefaults() {
	ly.Inhib.ActAvg.Nominal = .5
}

func (ly *LayerParams) TDPredDefaults() {
	ly.Acts.Decay.Act = 1
	ly.Acts.Decay.Glong = 1
	ly.Acts.Dt.GeTau = 40
}

func (ly *Layer) LDTPostBuild() {
	ly.Params.LDT.SrcLay1Idx = ly.BuildConfigFindLayer("SrcLay1Name", false) // optional
	ly.Params.LDT.SrcLay2Idx = ly.BuildConfigFindLayer("SrcLay2Name", false) // optional
	ly.Params.LDT.SrcLay3Idx = ly.BuildConfigFindLayer("SrcLay3Name", false) // optional
	ly.Params.LDT.SrcLay4Idx = ly.BuildConfigFindLayer("SrcLay4Name", false) // optional
}

// TDIntegPostBuild does post-Build config
func (ly *Layer) TDIntegPostBuild() {
	ly.Params.TDInteg.TDPredLayIdx = ly.BuildConfigFindLayer("TDPredLayName", true)
}

// TDDaPostBuild does post-Build config
func (ly *Layer) TDDaPostBuild() {
	ly.Params.TDDa.TDIntegLayIdx = ly.BuildConfigFindLayer("TDIntegLayName", true)
}
