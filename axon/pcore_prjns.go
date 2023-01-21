// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start pcore_prjns

// MatrixTraceParams for for trace-based learning in the MatrixPrjn.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is reset at time of reward based on ACh level from CINs.
type MatrixTraceParams struct {
	CurTrlDA  bool    `def:"true" desc:"if true, current trial DA dopamine can drive learning (i.e., synaptic co-activity trace is updated prior to DA-driven dWt), otherwise DA is applied to existing trace before trace is updated, meaning that at least one trial must separate gating activity and DA"`
	Decay     float32 `def:"2" min:"0" desc:"multiplier on CIN ACh level for decaying prior traces -- decay never exceeds 1.  larger values drive strong credit assignment for any US outcome."`
	NoACh     bool    `desc:"ignore ACh for learning modulation -- only used for reset if so -- otherwise ACh directly multiplies dWt"`
	Modulator bool    `desc:"this projection is a modulator -- the conductance here multiplies other inputs -- it must be active for anything else to activate"`
}

func (tp *MatrixTraceParams) Defaults() {
	tp.CurTrlDA = true
	tp.Decay = 2
}

//gosl: end pcore_pjrns
