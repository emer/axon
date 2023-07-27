// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start pvlv_prjns

// BLAPrjnParams has parameters for basolateral amygdala learning.
// Learning is driven by the Tr trace as function of ACh * Send Act
// recorded prior to US, and at US, recv unit delta: CaSpkP - SpkPrv
// times normalized GeIntMax for recv unit credit assignment.
// The Learn.Trace.Tau time constant determines trace updating over trials
// when ACh is above threshold -- this determines strength of second-order
// conditioning -- default of 1 means none, but can be increased as needed.
type BLAPrjnParams struct {
	NegDeltaLRate float32 `def:"0.01,1" desc:"use 0.01 for acquisition (don't unlearn) and 1 for extinction -- negative delta learning rate multiplier"`
	AChThr        float32 `def:"0.1" desc:"threshold on this layer's ACh level for trace learning updates"`
	USTrace       float32 `def:"0,0.5" desc:"proportion of US time stimulus activity to use for the trace component of "`

	pad float32
}

func (bp *BLAPrjnParams) Defaults() {
	bp.NegDeltaLRate = 0.01
	bp.AChThr = 0.1
	bp.USTrace = 0
}

func (bp *BLAPrjnParams) Update() {

}

//gosl: end pvlv_prjns

func (pj *PrjnParams) BLADefaults() {
	pj.SWts.Adapt.On.SetBool(false)
	pj.SWts.Adapt.SigGain = 1
	pj.SWts.Init.SPct = 0
	pj.SWts.Init.Mean = 0.1
	pj.SWts.Init.Var = 0.05
	pj.SWts.Init.Sym.SetBool(false)
	pj.Learn.Trace.Tau = 1 // increase for second order conditioning
	pj.Learn.Trace.Update()
	pj.Learn.LRate.Base = 0.02
}

func (pj *PrjnParams) VSPatchDefaults() {
	pj.PrjnScale.Abs = 2 // needs strong drive in general
	pj.SWts.Adapt.On.SetBool(false)
	pj.SWts.Adapt.SigGain = 1
	pj.SWts.Init.SPct = 0
	pj.SWts.Init.Mean = 0.1
	pj.SWts.Init.Var = 0.05
	pj.SWts.Init.Sym.SetBool(false)
	pj.Learn.Trace.Tau = 1
	pj.Learn.Trace.LearnThr = 0.3
	pj.Learn.Trace.Update()
	pj.Learn.LRate.Base = 0.05
}
