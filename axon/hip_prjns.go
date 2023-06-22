// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start hip_prjns

type HipPrjnParams struct {
	LRate float32 `def:"0.01,1" desc:"use 0.01 for acquisition (don't unlearn) and 1 for extinction -- negative delta learning rate multiplier"`
}

func (hp *HipPrjnParams) Defaults() {
	hp.LRate = 0.01
}

func (hp *HipPrjnParams) Update() {

}

// gosl: end hip_prjns

func (pj *PrjnParams) HipDefaults() {
	pj.SWts.Init.Sym.SetBool(false)
}
