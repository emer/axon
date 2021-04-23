// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fffb

// Adapt has parameters for adapting the multiplier on inhibitory gain value (Gi)
// to keep overall layer activation within a given target range as specified
// by AvgAct.Init
type Adapt struct {
	On       bool    `desc:"enable adaptive layer inhibition gain as stored in layer GiCur value"`
	LoTolPct float32 `def:"0.5" viewif:"On=true" desc:"tolerance for lower than target average activation of AvgAct.Init as a proportion of that target value -- only once activations move outside this tolerance are inhibitory values adapted"`
	HiTolPct float32 `def:"0.1" viewif:"On=true" desc:"tolerance for higher than target average activation of AvgAct.Init as a proportion of that target value -- only once activations move outside this tolerance are inhibitory values adapted"`
	Interval int     `viewif:"On=true" desc:"interval in trials between updates of the adaptive inhibition values -- only check and update this often -- typically the same order as the number of trials per epoch used in training the model"`
	Tau      float32 `def:"200,500" viewif:"On=true" desc:"time constant for rate of updating the inhibitory gain value, in terms of trial_interval periods (e.g., 100 = adapt gain over 100 trial intervals) -- adaptation rate is 1/tau * (ActMAvg - AvgAct.Init) / AvgAct.Init"`

	Dt float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (ad *Adapt) Update() {
	ad.Dt = 1 / ad.Tau
}

func (ad *Adapt) Defaults() {
	ad.On = false
	ad.LoTolPct = 0.5
	ad.HiTolPct = 0.1
	ad.Interval = 100
	ad.Tau = 200
	ad.Update()
}

func (ad *Adapt) Adapt(gimult *float32, trg, act float32) bool {
	del := (act - trg) / trg
	if del < -ad.LoTolPct {
		*gimult += ad.Dt * del
		return true
	} else if del > ad.HiTolPct {
		*gimult += ad.Dt * del
		return true
	}
	return false
}
