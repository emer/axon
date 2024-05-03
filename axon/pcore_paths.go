// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "cogentcore.org/core/vgpu/gosl/slbool"

//gosl:start pcore_paths

// MatrixPathParams for trace-based learning in the MatrixPath.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is applied to DWt and reset at the time of reward.
type MatrixPathParams struct {

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

	// for ventral striatum, learn based on activity at time of reward,
	// in inverse proportion to the GoalMaint activity: i.e., if there was no
	// goal maintenance, learn at reward to encourage goal engagement next time,
	// but otherwise, do not further reinforce at time of reward, because the
	// actual goal gating learning trace is a better learning signal.
	// Otherwise, only uses accumulated trace but doesn't include rew-time activity,
	// e.g., for testing cases that do not have GoalMaint.
	VSRewLearn slbool.Bool `default:"true"`
}

func (tp *MatrixPathParams) Defaults() {
	tp.Credit = 0.6
	tp.BasePF = 0.005
	tp.Delta = 1
	tp.VSRewLearn.SetBool(true)
}

func (tp *MatrixPathParams) Update() {
}

//gosl:end pcore_pjrns

func (pj *PathParams) MatrixDefaults() {
	pj.SWts.Adapt.On.SetBool(false)
	pj.SWts.Adapt.SigGain = 6 // not 1 -- could be for some cases
	pj.SWts.Init.Sym.SetBool(false)
	pj.SWts.Init.SPct = 0
	pj.SWts.Init.Mean = 0.5
	pj.SWts.Init.Var = 0.4
	pj.Learn.LRate.Base = 0.01
	pj.Learn.Trace.LearnThr = 0.1 // note: higher values prevent ability to learn to gate again after extinction
}
