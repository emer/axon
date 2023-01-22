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
	CurTrlDA slbool.Bool `def:"true" desc:"if true, current trial DA dopamine can drive learning (i.e., synaptic co-activity trace is updated prior to DA-driven dWt), otherwise DA is applied to existing trace before trace is updated, meaning that at least one trial must separate gating activity and DA"`
	AChDecay float32     `def:"2" min:"0" desc:"multiplier on CIN ACh level for decaying prior traces -- decay never exceeds 1, so a larger number ensures complete decay with lower ACh levels."`

	pad, pad1 float32
}

func (tp *MatrixPrjnParams) Defaults() {
	tp.CurTrlDA.SetBool(true)
	tp.AChDecay = 2
}

func (tp *MatrixPrjnParams) Update() {
}

// TraceDecay returns the decay factor as a function of ach level
func (tp *MatrixPrjnParams) TraceDecay(ach float32) float32 {
	dk := ach * tp.AChDecay
	if dk > 1 {
		dk = 1
	}
	return dk
}

//gosl: end pcore_pjrns
