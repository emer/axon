// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start rl_prjns

// RWPrjnParams does dopamine-modulated learning for reward prediction: Da * Send.Act
// Use in RWPredLayer typically to generate reward predictions.
// Has no weight bounds or limits on sign etc.
type RWPrjnParams struct {
	DaTol        float32 `desc:"tolerance on DA -- if below this abs value, then DA goes to zero and there is no learning -- prevents prediction from exactly learning to cancel out reward value, retaining a residual valence of signal"`
	OppSignLRate float32 `desc:"how much to learn on opposite DA sign coding neuron (0..1)"`

	pad, pad1 float32
}

func (pj *RWPrjnParams) Defaults() {
	pj.OppSignLRate = 1.0
}

func (pj *RWPrjnParams) Update() {
}

func (pj *PrjnParams) RWPrjnDefaults() {
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.Mean = 0
	pj.SWt.Init.Var = 0
	pj.SWt.Init.Sym.SetBool(false)
}
