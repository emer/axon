// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"github.com/goki/gosl/slbool"
	"github.com/goki/mat32"
)

//gosl: start chans

// SKCaParams describes the small-conductance calcium-activated potassium channel
// using the equations described in Fujita et al (2012) based on Gunay et al (2008)
// (also Muddapu & Chakravarthy, 2021)
// There is a gating factor M that depends on the Ca concentration, modeled using
// an X / (X + C50) form Hill equation
type SKCaParams struct {
	Gbar    float32     `def:"0,2" desc:"overall strength of sKCa current -- inactive if 0"`
	CaD     slbool.Bool `viewif:"Gbar>0" desc:"use CaD timescale (delayed) calcium signal -- for STNs -- else use CaP (faster) for STNp"`
	CaScale float32     `viewif:"Gbar>0" def:"3" desc:"scaling factor applied to input Ca to bring into proper range of these dynamics"`
	Hill    float32     `viewif:"Gbar>0" def:"4" desc:"Hill coefficient (exponent) for x^h / (x^h + c50^h) function describing the asymptotic gating factor m as a function of Ca -- there are 4 relevant states so a factor around 4 makes sense and is empirically observed"`
	C50     float32     `viewif:"Gbar>0" def:"0.6" desc:"50% Ca concentration baseline value in Hill equation -- values from .3 to .6 are present in the literature"`
	ActTau  float32     `viewif:"Gbar>0" def:"10" desc:"activation time constant -- roughly 5-15 msec in literature"`
	DeTau   float32     `viewif:"Gbar>0" def:"30,50" desc:"deactivation time constant -- roughly 30-50 msec in literature"`

	ActDt   float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	DeDt    float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	C50Hill float32 `view:"-" json:"-" xml:"-" desc:"C50 ^ Hill precomputed"`

	pad, pad1 float32
}

func (sp *SKCaParams) Defaults() {
	sp.CaScale = 3
	sp.Gbar = 0.0
	sp.Hill = 4
	sp.C50 = 0.6
	sp.ActTau = 10
	sp.DeTau = 50
	sp.Update()
}

func (sp *SKCaParams) Update() {
	sp.C50Hill = mat32.Pow(sp.C50, sp.Hill)
	sp.ActDt = 1.0 / sp.ActTau
	sp.DeDt = 1.0 / sp.DeTau
}

// MAsympHill gives the asymptotic (driving) gating factor M as a function of CAi
// for the Hill equation version used in Fujita et al (2012)
func (sp *SKCaParams) MAsympHill(cai float32) float32 {
	capow := mat32.Pow(cai, sp.Hill)
	return capow / (capow + sp.C50Hill)
}

// MAsympGW06 gives the asymptotic (driving) gating factor M as a function of CAi
// for the GilliesWillshaw06 equation version
// this is a log-saturating function
func (sp *SKCaParams) MAsympGW06(cai float32) float32 {
	if cai < 0.001 {
		cai = 0.001
	}
	return 0.81 / (1.0 + mat32.FastExp(-(mat32.Log(cai)+0.3))/0.46)
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
