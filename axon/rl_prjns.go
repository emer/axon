// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start rl_prjns

// RLPredPrjnParams does dopamine-modulated learning for reward prediction: Da * Send.Act
// Used by RWPrjn and TDPredPrjn within corresponding RWPredLayer or TDPredLayer
// to generate reward predictions based on its incoming weights, using linear activation
// function. Has no weight bounds or limits on sign etc.
type RLPredPrjnParams struct {
	OppSignLRate float32 `desc:"how much to learn on opposite DA sign coding neuron (0..1)"`
	DaTol        float32 `desc:"tolerance on DA -- if below this abs value, then DA goes to zero and there is no learning -- prevents prediction from exactly learning to cancel out reward value, retaining a residual valence of signal"`

	pad, pad1 float32
}

func (pj *RLPredPrjnParams) Defaults() {
	pj.OppSignLRate = 1.0
}

func (pj *RLPredPrjnParams) Update() {
}

//gosl: end rl_prjns

func (pj *PrjnParams) RLPredDefaults() {
	pj.SWts.Adapt.SigGain = 1
	pj.SWts.Init.Mean = 0
	pj.SWts.Init.Var = 0
	pj.SWts.Init.Sym.SetBool(false)
}
