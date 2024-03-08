// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "cogentcore.org/core/mat32"

//gosl: start chans

// MahpParams implements an M-type medium afterhyperpolarizing (mAHP) channel,
// where m also stands for muscarinic due to the ACh inactivation of this channel.
// It has a slow activation and deactivation time constant, and opens at a lowish
// membrane potential.
// There is one gating variable n updated over time with a tau that is also voltage dependent.
// The infinite-time value of n is voltage dependent according to a logistic function
// of the membrane potential, centered at Voff with slope Vslope.
type MahpParams struct {

	// strength of mAHP current
	Gbar float32

	// voltage offset (threshold) in biological units for infinite time N gating function -- where the gate is at 50% strength
	Voff float32 `default:"-30"`

	// slope of the arget (infinite time) gating function
	Vslope float32 `default:"9"`

	// maximum slow rate time constant in msec for activation / deactivation.  The effective Tau is much slower -- 1/20th in original temp, and 1/60th in standard 37 C temp
	TauMax float32 `default:"1000"`

	// temperature adjustment factor: assume temp = 37 C, whereas original units were at 23 C
	Tadj float32 `view:"-" edit:"-"`

	// 1/Tau
	DtMax float32 `view:"-" edit:"-"`

	pad, pad2 int32
}

// Defaults sets the parameters
func (mp *MahpParams) Defaults() {
	mp.Gbar = 0.02
	mp.Voff = -30
	mp.Vslope = 9
	mp.TauMax = 1000
	mp.Tadj = mat32.Pow(2.3, (37.0-23.0)/10.0) // 3.2 basically
	mp.Update()
}

func (mp *MahpParams) Update() {
	mp.DtMax = 1.0 / mp.TauMax
}

func (mp *MahpParams) ShouldShow(field string) bool {
	switch field {
	case "Gbar":
		return true
	default:
		return mp.Gbar > 0
	}
}

// EFun handles singularities in an elegant way -- from Mainen impl
func (mp *MahpParams) EFun(z float32) float32 {
	if mat32.Abs(z) < 1.0e-4 {
		return 1.0 - 0.5*z
	}
	return z / (mat32.FastExp(z) - 1.0)
}

// NinfTauFmV returns the target infinite-time N gate value and
// voltage-dependent time constant tau, from vbio
func (mp *MahpParams) NinfTauFmV(vbio float32, ninf, tau *float32) {
	vo := vbio - mp.Voff

	// logical functions, but have signularity at Voff (vo = 0)
	// a := mp.DtMax * vo / (1.0 - mat32.FastExp(-vo/mp.Vslope))
	// b := -mp.DtMax * vo / (1.0 - mat32.FastExp(vo/mp.Vslope))

	a := mp.DtMax * mp.Vslope * mp.EFun(-vo/mp.Vslope)
	b := mp.DtMax * mp.Vslope * mp.EFun(vo/mp.Vslope)
	*tau = 1.0 / (a + b)
	*ninf = a * *tau // a / (a+b)
	*tau /= mp.Tadj  // correct right away..

	return
}

// NinfTauFmV returns the target infinite-time N gate value and
// voltage-dependent time constant tau, from normalized vm
func (mp *MahpParams) NinfTauFmVnorm(v float32, ninf, tau *float32) {
	mp.NinfTauFmV(VToBio(v), ninf, tau)
}

// DNFmV returns the change in gating factor N based on normalized Vm
func (mp *MahpParams) DNFmV(v, n float32) float32 {
	var ninf, tau float32
	mp.NinfTauFmVnorm(v, &ninf, &tau)
	// dt := 1.0 - mat32.FastExp(-mp.Tadj/tau) // Mainen comments out this form; Poirazi uses
	// dt := mp.Tadj / tau // simple linear fix
	dn := (ninf - n) / tau
	return dn
}

// GmAHP returns the conductance as a function of n
func (mp *MahpParams) GmAHP(v float32, n *float32) float32 {
	dn := mp.DNFmV(v, *n)
	*n += dn
	g := mp.Tadj * mp.Gbar * *n
	return g
}

//gosl: end chans
