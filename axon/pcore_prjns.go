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

	// weight for trace activity that is a function of the minus-plus delta
	// activity signal on the receiving MSN neuron, independent of PF modulation.
	Delta float32 `default:"1"`

	// proportion of trace activity driven by non-delta activity of receiving neuron,
	// which is multiplied by the PF modulatotory inputs, for strong credit assignment
	// learning of the rewarded action.
	NonDelta float32 `default:"0.2"`

	pad, pad1 float32
}

func (tp *MatrixPrjnParams) Defaults() {
	tp.Delta = 1
	tp.NonDelta = 0.2
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
	pj.Learn.LRate.Base = 0.01
	pj.Learn.Trace.LearnThr = 0.1 // note: higher values prevent ability to learn to gate again after extinction
}
