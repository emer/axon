// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start pcore_prjns

// MatrixPrjnParams for trace-based learning in the MatrixPrjn.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is applied to DWt and reset at the time of reward.
type MatrixPrjnParams struct {

	// [def: 0.01] learning rate for when ACh was elevated but no gating took place, in proportion to the level of ACh that indicates the salience of the event.  A low level of this learning prevents the highly maladaptive situation where the BG is not gating and thus no learning can occur.
	NoGateLRate float32 `def:"0.01" desc:"learning rate for when ACh was elevated but no gating took place, in proportion to the level of ACh that indicates the salience of the event.  A low level of this learning prevents the highly maladaptive situation where the BG is not gating and thus no learning can occur."`

	pad, pad1, pad2 float32
}

func (tp *MatrixPrjnParams) Defaults() {
	tp.NoGateLRate = 0.01
}

func (tp *MatrixPrjnParams) Update() {
}

//gosl: end pcore_pjrns

func (pj *PrjnParams) MatrixDefaults() {
	pj.SWts.Adapt.On.SetBool(false)
	pj.SWts.Adapt.SigGain = 6 // not 1 -- could be for some cases
	pj.SWts.Init.Sym.SetBool(false)
	pj.SWts.Init.SPct = 0
	pj.SWts.Init.Mean = 0.5
	pj.SWts.Init.Var = 0.4
	pj.Learn.LRate.Base = 0.02
	pj.Learn.Trace.LearnThr = 0.75
}
