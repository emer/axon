// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"cogentcore.org/core/math32"
)

//gosl:start chans

// SKCaParams describes the small-conductance calcium-activated potassium channel,
// activated by intracellular stores in a way that drives pauses in firing,
// and can require inactivity to recharge the Ca available for release.
// These intracellular stores can release quickly, have a slow decay once released,
// and the stores can take a while to rebuild, leading to rapidly triggered,
// long-lasting pauses that don't recur until stores have rebuilt, which is the
// observed pattern of firing of STNp pausing neurons.
// CaIn = intracellular stores available for release; CaR = released amount from stores
// CaM = K channel conductance gating factor driven by CaR binding,
// computed using the Hill equations described in Fujita et al (2012), Gunay et al (2008)
// (also Muddapu & Chakravarthy, 2021): X^h / (X^h + C50^h) where h ~= 4 (hard coded)
type SKCaParams struct {

	// overall strength of sKCa current -- inactive if 0
	Gbar float32 `default:"0,2,3"`

	// 50% Ca concentration baseline value in Hill equation -- set this to level that activates at reasonable levels of SKCaR
	C50 float32 `default:"0.4,0.5"`

	// K channel gating factor activation time constant -- roughly 5-15 msec in literature
	ActTau float32 `default:"15"`

	// K channel gating factor deactivation time constant -- roughly 30-50 msec in literature
	DeTau float32 `default:"30"`

	// proportion of CaIn intracellular stores that are released per spike, going into CaR
	KCaR float32 `default:"0.4,0.8"`

	// SKCaR released calcium decay time constant
	CaRDecayTau float32 `default:"150,200"`

	// level of time-integrated spiking activity (CaSpkD) below which CaIn intracelluar stores are replenished -- a low threshold can be used to require minimal activity to recharge -- set to a high value (e.g., 10) for constant recharge.
	CaInThr float32 `default:"0.01"`

	// time constant in msec for storing CaIn when activity is below CaInThr
	CaInTau float32 `default:"50"`

	// rate = 1 / tau
	ActDt float32 `view:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	DeDt float32 `view:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	CaRDecayDt float32 `view:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	CaInDt float32 `view:"-" json:"-" xml:"-"`
}

func (sp *SKCaParams) Defaults() {
	sp.Gbar = 0.0
	sp.C50 = 0.5
	sp.ActTau = 15
	sp.DeTau = 30
	sp.KCaR = 0.8
	sp.CaRDecayTau = 150
	sp.CaInThr = 0.01
	sp.CaInTau = 50
	sp.Update()
}

func (sp *SKCaParams) Update() {
	sp.ActDt = 1.0 / sp.ActTau
	sp.DeDt = 1.0 / sp.DeTau
	sp.CaRDecayDt = 1.0 / sp.CaRDecayTau
	sp.CaInDt = 1.0 / sp.CaInTau
}

func (sp *SKCaParams) ShouldShow(field string) bool {
	switch field {
	case "Gbar":
		return true
	default:
		return sp.Gbar > 0
	}
}

// MAsympHill gives the asymptotic (driving) gating factor M as a function of CAi
// for the Hill equation version used in Fujita et al (2012)
func (sp *SKCaParams) MAsympHill(cai float32) float32 {
	cai /= sp.C50
	capow := cai * cai * cai * cai
	return capow / (1 + capow)
}

// MAsympGW06 gives the asymptotic (driving) gating factor M as a function of CAi
// for the GilliesWillshaw06 equation version -- not used by default.
// this is a log-saturating function
func (sp *SKCaParams) MAsympGW06(cai float32) float32 {
	if cai < 0.001 {
		cai = 0.001
	}
	return 0.81 / (1.0 + math32.FastExp(-(math32.Log(cai)+0.3))/0.46)
}

// CaInRFromSpike updates CaIn, CaR from Spiking and CaD time-integrated spiking activity
func (sp *SKCaParams) CaInRFromSpike(spike, caD float32, caIn, caR *float32) {
	*caR -= *caR * sp.CaRDecayDt
	if spike > 0 {
		x := *caIn * sp.KCaR
		*caR += x
		*caIn -= x
	}
	if caD < sp.CaInThr {
		*caIn += sp.CaInDt * (1.0 - *caIn)
	}
}

// MFromCa returns updated m gating value as a function of current CaR released Ca
// and the current m gating value, with activation and deactivation time constants.
func (sp *SKCaParams) MFromCa(caR, mcur float32) float32 {
	mas := sp.MAsympHill(caR)
	if mas > mcur {
		return mcur + sp.ActDt*(mas-mcur)
	}
	return mcur + sp.DeDt*(mas-mcur)
}

//gosl:end chans
