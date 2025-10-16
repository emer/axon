// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl:start

// HipPathParams define behavior of hippocampus paths, which have special learning rules
type HipPathParams struct {

	// Hebbian learning proportion
	Hebb float32 `default:"0"`

	// EDL proportion
	Err float32 `default:"1"`

	// proportion of correction to apply to sending average activation for hebbian learning component (0=none, 1=all, .5=half, etc)
	SAvgCor float32 `default:"0.4:0.8" min:"0" max:"1"`

	// threshold of sending average activation below which learning does not occur (prevents learning when there is no input)
	SAvgThr float32 `default:"0.01" min:"0"`

	// sending layer Nominal (need to manually set it to be the same as the sending layer)
	SNominal float32 `default:"0.1" min:"0"`

	pad, pad1, pad2 float32
}

func (hp *HipPathParams) Defaults() {
	hp.Hebb = 0
	hp.Err = 1
	hp.SAvgCor = 0.4
	hp.SAvgThr = 0.01
	hp.SNominal = 0.1
}

func (hp *HipPathParams) Update() {

}

//gosl:end

func (pj *PathParams) HipDefaults() {
	pj.SWts.Init.Sym.SetBool(false)
}
