// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "github.com/goki/mat32"

// MAHPParams implements an M-type medium afterhyperpolarizing (mAHP) channel,
// where m also stands for muscarinic due to the ACh inactivation of this channel.
// It has a slow activation and deactivation time constant, and opens at a lowish
// membrane potential.
// There is one gating variable n updated over time with a tau that is also voltage dependent.
// The infinite-time value of n is voltage dependent according to a logistic function
// of the membrane potential, centered at Voff with slope Vslope.
type MAHPParams struct {
	Gbar   float32 `def:"0.1,0.01" desc:"strength of mAHP current"`
	Voff   float32 `def:"-30" desc:"voltage offset (threshold) in biological units for infinite time N gating function -- where the gate is at 50% strength"`
	Vslope float32 `def:"9" desc:"slope of the arget (infinite time) gating function"`
	TauMax float32 `def:"1000" desc:"maximum slow rate in msec for activation / deactivation"`
	Tadj   float32 `desc:"temperature adjustment factor: assume temp = 37 C, whereas original units were at 23 C"`
	DtMax  float32 `view:"-" desc:"1/Tau"`
}

// Defaults sets the parameters
func (mp *MAHPParams) Defaults() {
	mp.Gbar = 0.1
	mp.Voff = -30
	mp.Vslope = 9
	mp.TauMax = 1000
	mp.Tadj = mat32.Pow(2.3, (37.0-23.0)/10.0) // 3.2 basically
	mp.Update()
}

func (mp *MAHPParams) Update() {
	mp.DtMax = 1.0 / mp.TauMax
}

// NinfTauFmV returns the target infinite-time N gate value and
// voltage-dependent time constant tau, from vbio
func (mp *MAHPParams) NinfTauFmV(vbio float32) (ninf, tau float32) {
	if vbio == mp.Voff {
		vbio = vbio - 0.01
	}
	vo := vbio - mp.Voff

	a := mp.DtMax * vo / (1.0 - mat32.FastExp(-vo/mp.Vslope))
	b := -mp.DtMax * vo / (1.0 - mat32.FastExp(vo/mp.Vslope))
	tau = 1.0 / (a + b)
	ninf = a * tau // a / (a+b)
	tau /= mp.Tadj

	return
}

// NinfTauFmV returns the target infinite-time N gate value and
// voltage-dependent time constant tau, from normalized vm
func (mp *MAHPParams) NinfTauFmVnorm(v float32) (ninf, tau float32) {
	return mp.NinfTauFmV(VToBio(v))
}

// DNFmV returns the change in gating factor N based on normalized Vm
func (mp *MAHPParams) DNFmV(v, n float32) float32 {
	ninf, tau := mp.NinfTauFmVnorm(v)
	// dt := 1.0 - mat32.FastExp(-mp.Tadj/tau) // Mainen comments out this form; Poirazi uses
	// dt := mp.Tadj / tau // simple linear fix
	dn := (ninf - n) / tau
	return dn
}

// GmAHP returns the conductance as a function of n
// GBar * MFmVnorm(v)
func (mp *MAHPParams) GmAHP(n float32) float32 {
	return mp.Tadj * mp.Gbar * n
}
