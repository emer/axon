// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "cogentcore.org/lab/gosl/slbool"

//gosl:start

// DSMatrixPathParams for trace-based learning in the MatrixPath.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is applied to DWt and reset at the time of reward.
type DSMatrixPathParams struct {

	// PatchDA is proportion of Credit trace factor for learning
	// to modulate by PatchDA versus just standard s*r activity factor.
	PatchDA float32 `default:"0.5"`

	// Credit is proportion of trace activity driven by the credit assignment factor
	// based on the PF modulatory inputs, synaptic activity (send * recv),
	// and Patch DA, which indicates extent to which gating at this time is net
	// associated with subsequent reward or not.
	Credit float32 `default:"0.6"`

	// Delta is weight for trace activity that is a function of the minus-plus delta
	// activity signal on the receiving MSN neuron, independent of PF modulation.
	// This should always be 1 except for testing disabling: adjust NonDelta
	// relative to it, and the overall learning rate.
	Delta float32 `default:"1"`

	// D2Scale is a scaling factor for the DAD2 learning factor relative to
	// the DAD1 contribution (which is 1 - DAD1).
	D2Scale float32 `default:"1"`

	// OffTrace is a multiplier on trace contribution when action output
	// communicated by PF is not above threshold.
	OffTrace float32 `default:"0.1"`

	pad, pad1, pad2 float32
}

func (tp *DSMatrixPathParams) Defaults() {
	tp.PatchDA = 0.5
	tp.Credit = 0.6
	tp.Delta = 1
	tp.D2Scale = 1
	tp.OffTrace = 0.1
}

func (tp *DSMatrixPathParams) Update() {
}

// VSMatrixPathParams for trace-based learning in the VSMatrixPath,
// for ventral striatum paths.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is applied to DWt and reset at the time of reward.
type VSMatrixPathParams struct {

	// RewActLearn makes learning based on activity at time of reward,
	// in inverse proportion to the GoalMaint activity: i.e., if there was no
	// goal maintenance, learn at reward to encourage goal engagement next time,
	// but otherwise, do not further reinforce at time of reward, because the
	// actual goal gating learning trace is a better learning signal.
	// Otherwise, only uses accumulated trace but doesn't include rew-time activity,
	// e.g., for testing cases that do not have GoalMaint.
	RewActLearn slbool.Bool `default:"true"`

	pad, pad1, pad2 float32
}

func (tp *VSMatrixPathParams) Defaults() {
	tp.RewActLearn.SetBool(true)
}

func (tp *VSMatrixPathParams) Update() {
}

// DSPatchPathParams for trace-based learning in the DSPatchPath,
// for ventral striatum paths.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is applied to DWt and reset at the time of reward.
type DSPatchPathParams struct {

	// PFSyn uses modulatory synapse-level version of PF activation
	// (representing ACh from CINs) instead of layer-level
	PFSyn slbool.Bool `default:"true"`

	// BasePF is the baseline level of PF (ACh) activity, providing a baseline
	// level of trace learning even for pools not driven by PF output.
	BasePF float32 `default:"0.005"`

	pad, pad1 float32
}

func (tp *DSPatchPathParams) Defaults() {
	tp.PFSyn.SetBool(false)
	tp.BasePF = 0.005
}

func (tp *DSPatchPathParams) Update() {
}

//gosl:end

func (pj *PathParams) MatrixDefaults() {
	pj.SWts.Adapt.On.SetBool(false)
	pj.SWts.Adapt.SigGain = 6 // not 1 -- could be for some cases
	pj.SWts.Init.Sym.SetBool(false)
	pj.SWts.Init.SPct = 0
	pj.SWts.Init.Mean = 0.5
	pj.SWts.Init.Var = 0.4
	pj.Learn.LRate.Base = 0.01
	pj.Learn.DWt.LearnThr = 0.1 // note: higher values prevent ability to learn to gate again after extinction
}
