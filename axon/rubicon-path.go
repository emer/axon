// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl:start rubicon_paths

// BLAPathParams has parameters for basolateral amygdala learning.
// Learning is driven by the Tr trace as function of ACh * Send Act
// recorded prior to US, and at US, recv unit delta: CaP - CaDPrev
// times normalized GeIntNorm for recv unit credit assignment.
type BLAPathParams struct {

	// use 0.01 for acquisition (don't unlearn) and 1 for extinction.
	// negative delta learning rate multiplier
	NegDeltaLRate float32 `default:"0.01,1"`

	// threshold on this layer's ACh level for trace learning updates
	AChThr float32 `default:"0.1"`

	// proportion of US time stimulus activity to use for the trace component of
	USTrace float32 `default:"0,0.5"`

	pad float32
}

func (bp *BLAPathParams) Defaults() {
	bp.NegDeltaLRate = 0.01
	bp.AChThr = 0.1
	bp.USTrace = 0.5
}

func (bp *BLAPathParams) Update() {

}

//gosl:end rubicon_paths

func (pj *PathParams) BLADefaults() {
	pj.SWts.Adapt.On.SetBool(false)
	pj.SWts.Adapt.SigGain = 1
	pj.SWts.Init.SPct = 0
	pj.SWts.Init.Mean = 0.1
	pj.SWts.Init.Var = 0.05
	pj.SWts.Init.Sym.SetBool(false)
	pj.Learn.DWt.Update()
	pj.Learn.LRate.Base = 0.02
}

func (pj *PathParams) VSPatchDefaults() {
	pj.PathScale.Abs = 4 // needs strong drive in general
	pj.SWts.Adapt.On.SetBool(false)
	pj.SWts.Adapt.SigGain = 1
	pj.SWts.Init.SPct = 0
	pj.SWts.Init.Mean = 0.5
	pj.SWts.Init.Var = 0.25
	pj.SWts.Init.Sym.SetBool(false)
	pj.Learn.DWt.LearnThr = 0 // 0.3
	pj.Learn.DWt.Update()
	pj.Learn.LRate.Base = 0.02 // 0.02 needed for smooth integ on vspatch test
}
