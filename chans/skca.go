// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"cogentcore.org/core/math32"
)

//gosl:start

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

	// Gk is the strength of the SKCa conductance contribution to Gk(t) factor
	// (which is then multiplied by Gbar.K that provides pA unit scaling).
	Gk float32 `default:"0,2,3"`

	// C50 is the 50% Ca concentration baseline value in Hill equation.
	// Set this to level that activates at reasonable levels of SKCaR.
	C50 float32 `default:"0.4,0.5"`

	// ActTau is the K channel gating factor activation time constant,
	// roughly 5-15 msec in literature.
	ActTau float32 `default:"15"`

	// DeTau is the K channel gating factor deactivation time constant,
	// roughly 30-50 ms in literature.
	DeTau float32 `default:"30"`

	// KCaR is the proportion of CaIn intracellular stores that are released
	// per spike, going into CaR.
	KCaR float32 `default:"0.4,0.8"`

	// CaRDecayTau is the SKCaR released calcium decay time constant.
	CaRDecayTau float32 `default:"150,200"`

	// CaInThr is the level of time-integrated spiking activity (CaD) below which CaIn
	// intracelluar stores are replenished. A low threshold can be used to
	// require minimal activity to recharge. Set to a high value (e.g., 10)
	// for constant recharge.
	CaInThr float32 `default:"0.01"`

	// CaInTau is the time constant in msec for storing CaIn when activity
	//  is below CaInThr.
	CaInTau float32 `default:"50"`

	// ActDT = 1 / tau
	ActDt float32 `display:"-" json:"-" xml:"-"`

	// DeDt = 1 / tau
	DeDt float32 `display:"-" json:"-" xml:"-"`

	// CaRDecayDt = 1 / tau
	CaRDecayDt float32 `display:"-" json:"-" xml:"-"`

	// CaInDt = 1 / tau
	CaInDt float32 `display:"-" json:"-" xml:"-"`
}

func (sp *SKCaParams) Defaults() {
	sp.Gk = 0.0
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

func (sp *SKCaParams) ShouldDisplay(field string) bool {
	switch field {
	case "Gk":
		return true
	default:
		return sp.Gk > 0
	}
}

// MAsympHill gives the asymptotic (driving) gating factor M as a function of CAi
// for the Hill equation version used in Fujita et al (2012)
func (sp *SKCaParams) MAsympHill(cai float32) float32 {
	caia := cai / sp.C50
	capow := caia * caia * caia * caia
	return capow / (1 + capow)
}

// MAsympGW06 gives the asymptotic (driving) gating factor M as a function of CAi
// for the GilliesWillshaw06 equation version -- not used by default.
// this is a log-saturating function
func (sp *SKCaParams) MAsympGW06(cai float32) float32 {
	caia := max(cai, 0.001)
	return 0.81 / (1.0 + math32.FastExp(-(math32.Log(caia)+0.3))/0.46)
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

//gosl:end
