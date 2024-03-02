// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/emer/gosl/v2/slbool"

//gosl: start pcore_prjns

// MatrixPrjnParams for trace-based learning in the MatrixPrjn.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is applied to DWt and reset at the time of reward.
type MatrixPrjnParams struct {

	// proportion of trace activity driven by the basic credit assignment factor
	// based on the PF modulatory inputs and activity of the receiving neuron,
	// relative to the delta factor which is generally going to be smaller
	// because it is an activity difference.
	Credit float32 `default:"0.6"`

	// baseline amount of PF activity that modulates credit assignment learning,
	// for neurons with zero PF modulatory activity.
	// These were not part of the actual motor action, but can still get some
	// smaller amount of credit learning.
	BasePF float32 `default:"0.005"`

	// weight for trace activity that is a function of the minus-plus delta
	// activity signal on the receiving MSN neuron, independent of PF modulation.
	// This should always be 1 except for testing disabling: adjust NonDelta
	// relative to it, and the overall learning rate.
	Delta float32 `default:"1"`

	// for ventral striatum, learn based on activity at time of reward.
	// otherwise, only uses accumulated trace but doesn't include rew-time activity.
	VSRewLearn slbool.Bool `default:"true"`
}

func (tp *MatrixPrjnParams) Defaults() {
	tp.Credit = 0.6
	tp.BasePF = 0.005
	tp.Delta = 1
	tp.VSRewLearn.SetBool(true)
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
