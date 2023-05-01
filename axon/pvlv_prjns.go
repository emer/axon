// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start pvlv_prjns

// BLAPrjnParams has parameters for basolateral amygdala learning.
// The Learn.Trace.Tau time constant determines the strength of second-order
// conditioning -- default of 1 means none, but can be increased as needed.
type BLAPrjnParams struct {
	NegDeltaLRate float32 `def:"0.01,1" desc:"use 0.01 for acquisition (don't unlearn) and 1 for extinction -- negative delta learning rate multiplier"`

	pad, pad1, pad2 float32
}

func (bp *BLAPrjnParams) Defaults() {
	bp.NegDeltaLRate = 0.01
}

func (bp *BLAPrjnParams) Update() {

}

//gosl: end pvlv_prjns

func (pj *PrjnParams) BLADefaults() {
	pj.SWt.Adapt.On.SetBool(false)
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.SPct = 0
	pj.SWt.Init.Mean = 0.1
	pj.SWt.Init.Var = 0.05
	pj.SWt.Init.Sym.SetBool(false)
	pj.Learn.Trace.Tau = 1 // increase for second order conditioning
	pj.Learn.Trace.Update()
	pj.Learn.LRate.Base = 0.02
}

func (pj *PrjnParams) VSPatchDefaults() {
	pj.PrjnScale.Abs = 2 // needs strong drive in general
	pj.SWt.Adapt.On.SetBool(false)
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.SPct = 0
	pj.SWt.Init.Mean = 0.1
	pj.SWt.Init.Var = 0.05
	pj.SWt.Init.Sym.SetBool(false)
	pj.Learn.Trace.Tau = 1
	pj.Learn.Trace.LearnThr = 0.3
	pj.Learn.Trace.Update()
	pj.Learn.LRate.Base = 0.05
}
