// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"github.com/goki/mat32"
)

//gosl: start chans

// SKCaParams describes the small-conductance calcium-activated potassium channel,
// activated by intracellular stores in a way that drives pauses in firing, and requires
// inactivity to recharge the Ca available for release.
// SKCaIn = intracellular stores available for release; SKCaR = released amount from stores
// SKCaM = K channel conductance gating factor driven by SKCaR binding,
// computed using the Hill equations described in Fujita et al (2012), Gunay et al (2008)
// (also Muddapu & Chakravarthy, 2021): X^h / (X^h + C50^h) where h ~= 4.
type SKCaParams struct {
	Gbar        float32 `def:"0,2" desc:"overall strength of sKCa current -- inactive if 0"`
	Hill        float32 `viewif:"Gbar>0" def:"4" desc:"Hill coefficient (exponent) for x^h / (x^h + c50^h) function describing the asymptotic gating factor m as a function of Ca -- there are 4 relevant states so a factor around 4 makes sense and is empirically observed"`
	C50         float32 `viewif:"Gbar>0" def:"0.6" desc:"50% Ca concentration baseline value in Hill equation -- set this to level that activates at reasonable levels of SKCaR"`
	ActTau      float32 `viewif:"Gbar>0" def:"5" desc:"K channel gating factor activation time constant -- roughly 5-15 msec in literature"`
	DeTau       float32 `viewif:"Gbar>0" def:"30,50" desc:"K channel gating factor deactivation time constant -- roughly 30-50 msec in literature"`
	KCaR        float32 `viewif:"Gbar>0" def:".5" desc:"proportion of SKCaIn intracellular stores that are released per spike, int SKCaR variable"`
	CaRDecayTau float32 `viewif:"Gbar>0" def:"200" desc:"SKCaR released calcium decay time constant"`
	CaInThr     float32 `viewif:"Gbar>0" def:"0.05" desc:"level of CaSpkD below which SKCaIn intracelluar stores are replenished -- typically a low threshold requiring minimal activity to recharge."`
	CaInTau     float32 `viewif:"Gbar>0" def:"100" desc:"time constant in msec for storing SKCaIn whe activity is below CaInThr"`

	ActDt      float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	DeDt       float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	CaRDecayDt float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	CaInDt     float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	C50Hill    float32 `view:"-" json:"-" xml:"-" desc:"C50 ^ Hill precomputed"`

	pad, pad1 float32
}

func (sp *SKCaParams) Defaults() {
	sp.Gbar = 0.0
	sp.Hill = 4
	sp.C50 = 0.6
	sp.ActTau = 10
	sp.DeTau = 30
	sp.CaRDecayTau = 200
	sp.CaInThr = 0.05
	sp.CaInTau = 100
	sp.Update()
}

func (sp *SKCaParams) Update() {
	sp.C50Hill = mat32.Pow(sp.C50, sp.Hill)
	sp.ActDt = 1.0 / sp.ActTau
	sp.DeDt = 1.0 / sp.DeTau
	sp.CaDecayDt = 1.0 / sp.CaDecayTau
	sp.CaInDt = 1.0 / sp.CaInTau
}

// MAsympHill gives the asymptotic (driving) gating factor M as a function of CAi
// for the Hill equation version used in Fujita et al (2012)
func (sp *SKCaParams) MAsympHill(cai float32) float32 {
	capow := mat32.Pow(cai, sp.Hill)
	return capow / (capow + sp.C50Hill)
}

// MAsympGW06 gives the asymptotic (driving) gating factor M as a function of CAi
// for the GilliesWillshaw06 equation version -- not used by default.
// this is a log-saturating function
func (sp *SKCaParams) MAsympGW06(cai float32) float32 {
	if cai < 0.001 {
		cai = 0.001
	}
	return 0.81 / (1.0 + mat32.FastExp(-(mat32.Log(cai)+0.3))/0.46)
}

// CaInR updates CaIn, CaR from Spiking and CaD time-integrated activity
func (sp *SKCaParams) CaInRFmSpike(spike, cad float32) {
	ca *= sp.CaScale
	if ca > skca {
		return ca
	}
	skca += sp.CaDecayDt * (ca - skca)
	return skca
}

// MFmCa returns updated m gating value as a function of current intracellular Ca
// and the previous intracellular Ca -- the time constant tau is based on previous.
func (sp *SKCaParams) MFmCa(cai, mcur float32) float32 {
	mas := sp.MAsympHill(cai)
	if mas > mcur {
		return mcur + sp.ActDt*(mas-mcur)
	}
	return mcur + sp.DeDt*(mas-mcur)
}

//gosl: end chans
