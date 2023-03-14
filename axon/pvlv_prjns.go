// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start pvlv_prjns

// BLAAcqPrjnParams has parameters for basolateral amygdala acquisition learning.
type BLAAcqPrjnParams struct {
	NegDeltaLRate float32 `def:"0.01" desc:"negative delta learning rate multiplier -- weights go down much more slowly than up -- extinction is separate learning in extinction layer"`

	pad, pad1, pad2 float32
}

func (bp *BLAAcqPrjnParams) Defaults() {
	bp.NegDeltaLRate = 0.01
}

func (bp *BLAAcqPrjnParams) Update() {

}

//gosl: end pvlv_prjns

func (pj *PrjnParams) BLAAcqPrjnDefaults() {
	pj.SWt.Adapt.On.SetBool(false)
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.SPct = 0
	pj.SWt.Init.Mean = 0.1
	pj.SWt.Init.Var = 0.05
	pj.SWt.Init.Sym.SetBool(false)
	pj.Learn.Trace.Tau = 1
	pj.Learn.Trace.Update()
	pj.Learn.LRate.Base = 0.02
}

func (pj *PrjnParams) BLAExtPrjnDefaults() {
	pj.SWt.Adapt.On.SetBool(false)
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.SPct = 0
	pj.SWt.Init.Mean = 0.1
	pj.SWt.Init.Var = 0.05
	pj.SWt.Init.Sym.SetBool(false)
	pj.Learn.Trace.Tau = 1
	pj.Learn.Trace.Update()
	pj.Learn.LRate.Base = 0.1
}

func (pj *PrjnParams) VSPatchPrjnDefaults() {
	pj.SWt.Adapt.On.SetBool(false)
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.SPct = 0
	pj.SWt.Init.Mean = 0.1
	pj.SWt.Init.Var = 0.05
	pj.SWt.Init.Sym.SetBool(false)
	pj.Learn.Trace.Tau = 1
	pj.Learn.Trace.Update()
	pj.Learn.LRate.Base = 0.05
}
