// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start pvlv_prjns

//gosl: end pvlv_prjns

func (pj *PrjnParams) BLAPrjnDefaults() {
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.Mean = 0.1
	pj.SWt.Init.Var = 0.05
	pj.SWt.Init.Sym.SetBool(false)
	pj.Learn.Trace.Tau = 1
	pj.Learn.Trace.Update()
}

func (pj *PrjnParams) VSPatchPrjnDefaults() {
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.Mean = 0.1
	pj.SWt.Init.Var = 0.05
	pj.SWt.Init.Sym.SetBool(false)
	pj.Learn.Trace.Tau = 1
	pj.Learn.Trace.Update()
}
