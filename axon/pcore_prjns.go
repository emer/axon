// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/goki/gosl/slbool"

//gosl: start pcore_prjns

// MatrixPrjnParams for trace-based learning in the MatrixPrjn.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is reset at time of reward based on ACh level from CINs.
type MatrixPrjnParams struct {
	CurTrlDA    slbool.Bool `def:"true" desc:"if true, current trial DA dopamine can drive learning (i.e., synaptic co-activity trace is updated prior to DA-driven dWt), otherwise DA is applied to existing trace before trace is updated, meaning that at least one trial must separate gating activity and DA"`
	NoGateLRate float32     `def:"0,0.005" desc:"learning rate for when no gating took place, in proportion to the level of ACh that indicates the salience of the event.  A low level of this learning prevents the highly maladaptive situation where the BG is not gating and thus no learning can occur."`
	AChDecay    float32     `def:"2" min:"0" desc:"multiplier on CIN ACh level for decaying prior traces -- decay never exceeds 1, so a larger number ensures complete decay with lower ACh levels."`
	UseHasRew   slbool.Bool `desc:"use Context.HasRew as a unique learning and clearing signal -- if set, then AChDecay is not used.  This is mainly for debugging -- ACh is the appropriate biological mechanism."`
}

func (tp *MatrixPrjnParams) Defaults() {
	tp.CurTrlDA.SetBool(true)
	tp.NoGateLRate = 0.005
	tp.AChDecay = 2
}

func (tp *MatrixPrjnParams) Update() {
}

// TraceDecay returns the decay factor as a function of ach level and context
func (tp *MatrixPrjnParams) TraceDecay(ctx *Context, ach float32) float32 {
	if tp.UseHasRew.IsTrue() {
		return float32(ctx.NeuroMod.HasRew)
	}
	dk := ach * tp.AChDecay
	if dk > 1 {
		dk = 1
	}
	return dk
}

//gosl: end pcore_pjrns

func (pj *PrjnParams) MatrixDefaults() {
	pj.SWt.Adapt.On.SetBool(false)
	pj.SWt.Adapt.SigGain = 6 // not 1 -- could be for some cases
	pj.SWt.Init.Sym.SetBool(false)
	pj.SWt.Init.SPct = 0
}
