// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start hip_prjns

// HipPrjnParams define behavior of hippocampus prjns, which have special learning rules
type HipPrjnParams struct {
	Hebb     float32 `def:"0" desc:"Hebbian learning proportion"`
	Err      float32 `def:"1" desc:"EDL proportion"`
	SAvgCor  float32 `def:"0.4:0.8" min:"0" max:"1" desc:"proportion of correction to apply to sending average activation for hebbian learning component (0=none, 1=all, .5=half, etc)"`
	SAvgThr  float32 `def:"0.01" min:"0" desc:"threshold of sending average activation below which learning does not occur (prevents learning when there is no input)"`
	SNominal float32 `def:"0.1" min:"0" desc:"sending layer Nominal (need to manually set it to be the same as the sending layer)"`

	pad, pad1, pad2 float32
}

func (hp *HipPrjnParams) Defaults() {
	hp.Hebb = 0
	hp.Err = 1
	hp.SAvgCor = 0.4
	hp.SAvgThr = 0.01
	hp.SNominal = 0.1
}

func (hp *HipPrjnParams) Update() {

}

//gosl: end hip_prjns

func (pj *PrjnParams) HipDefaults() {
	pj.SWts.Init.Sym.SetBool(false)
}
