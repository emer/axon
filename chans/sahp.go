// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "github.com/goki/mat32"

//gosl: start chans

// SahpParams implements a slow afterhyperpolarizing (sAHP) channel,
// It has a slowly accumulating calcium value, aggregated at the
// theta cycle level, that then drives the logistic gating function,
// so that it only activates after a significant accumulation.
// After which point it decays.
// For the theta-cycle updating, the normal m-type tau is all within
// the scope of a single theta cycle, so we just omit the time integration
// of the n gating value, but tau is computed in any case.
type SahpParams struct {
	Gbar   float32 `def:"0.05,0.1" desc:"strength of sAHP current"`
	CaTau  float32 `viewif:"Gbar>0" def:"5,10" desc:"time constant for integrating Ca across theta cycles"`
	Off    float32 `viewif:"Gbar>0" def:"0.8" desc:"integrated Ca offset (threshold) for infinite time N gating function -- where the gate is at 50% strength"`
	Slope  float32 `viewif:"Gbar>0" def:"0.02" desc:"slope of the infinite time logistic gating function"`
	TauMax float32 `viewif:"Gbar>0" def:"1" desc:"maximum slow rate time constant in msec for activation / deactivation.  The effective Tau is much slower -- 1/20th in original temp, and 1/60th in standard 37 C temp"`

	CaDt  float32 `view:"-" inactive:"+" desc:"1/Tau"`
	DtMax float32 `view:"-" inactive:"+" desc:"1/Tau"`

	pad int32
}

// Defaults sets the parameters
func (mp *SahpParams) Defaults() {
	mp.Gbar = 0.05
	mp.CaTau = 5
	mp.Off = 0.8
	mp.Slope = 0.02
	mp.TauMax = 1
	mp.Update()
}

func (mp *SahpParams) Update() {
	mp.DtMax = 1.0 / mp.TauMax
	mp.CaDt = 1.0 / mp.CaTau
}

// EFun handles singularities in an elegant way -- from Mainen impl
func (mp *SahpParams) EFun(z float32) float32 {
	if mat32.Abs(z) < 1.0e-4 {
		return 1.0 - 0.5*z
	}
	return z / (mat32.FastExp(z) - 1.0)
}

// NinfTauFmCa returns the target infinite-time N gate value and
// time constant tau, from integrated Ca value
func (mp *SahpParams) NinfTauFmCa(ca float32, ninf, tau *float32) {
	co := ca - mp.Off

	// logical functions, but have signularity at Voff (vo = 0)
	// a := mp.DtMax * vo / (1.0 - mat32.FastExp(-vo/mp.Vslope))
	// b := -mp.DtMax * vo / (1.0 - mat32.FastExp(vo/mp.Vslope))

	a := mp.DtMax * mp.Slope * mp.EFun(-co/mp.Slope)
	b := mp.DtMax * mp.Slope * mp.EFun(co/mp.Slope)
	*tau = 1.0 / (a + b)
	*ninf = a * *tau // a / (a+b)
	return
}

// CaInt returns the updated time-integrated Ca value from current value and current Ca
func (mp *SahpParams) CaInt(caInt, ca float32) float32 {
	caInt += mp.CaDt * (ca - caInt)
	return caInt
}

// DNFmCa returns the change in gating factor N based on integrated Ca
// Omit this and just use ninf directly for theta-cycle updating.
func (mp *SahpParams) DNFmV(ca, n float32) float32 {
	var ninf, tau float32
	mp.NinfTauFmCa(ca, &ninf, &tau)
	dn := (ninf - n) / tau
	return dn
}

// GsAHP returns the conductance as a function of n
func (mp *SahpParams) GsAHP(n float32) float32 {
	return mp.Gbar * n
}

//gosl: end chans
