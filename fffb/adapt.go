// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fffb

// Adapt has parameters for adapting the multiplier on inhibitory gain value (Gi)
// to keep overall layer activation within a given target range as specified
// by AvgAct.Init
type Adapt struct {
	On       bool    `desc:"enable adaptive layer inhibition gain as stored in layer GiCur value"`
	HiTol    float32 `def:"0" viewif:"On=true" desc:"tolerance for higher than target average activation (AvgAct.Init) as a proportion of that target value (0 = exactly the target, 0.2 = 20% higher than target) -- only once activations move outside this tolerance are inhibitory values adapted"`
	LoTol    float32 `def:"0.8" viewif:"On=true" desc:"tolerance for lower than target average activation (AvgAct.Init) as a proportion of that target value (0 = exactly the target, 0.5 = 50% lower than target) -- only once activations move outside this tolerance are inhibitory values adapted"`
	Interval int     `def:"100" viewif:"On=true" desc:"interval in trials between updates of the adaptive inhibition values -- only check and update this often"`
	Tau      float32 `def:"2" viewif:"On=true" desc:"time constant for rate of updating the inhibitory gain value, in terms of Interval periods, to soften the dynamics -- adaptation rate is (1/Tau) * (ActMAvg - AvgAct.Init) / AvgAct.Init"`

	Dt float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (ad *Adapt) Update() {
	ad.Dt = 1 / ad.Tau
}

func (ad *Adapt) Defaults() {
	ad.On = false
	ad.HiTol = 0
	ad.LoTol = 0.8
	ad.Interval = 100
	ad.Tau = 2
	ad.Update()
}

// Adapt adapts the given gi multiplier factor as function of target and actual
// average activation, given current params.
func (ad *Adapt) Adapt(gimult *float32, trg, act float32) bool {
	del := (act - trg) / trg
	if del < -ad.LoTol || del > ad.HiTol {
		*gimult += ad.Dt * del
		return true
	}
	return false
}
