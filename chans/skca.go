// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "github.com/goki/mat32"

// SKCaParams describes the small-conductance calcium-activated potassium channel
// using the equations described in Fujita et al (2012) based on Gunay et al (2008)
// (also Muddapu & Chakravarthy, 2021)
// There is a gating factor M that depends on the Ca concentration, modeled using
// an X / (X + C50) form Hill equation
type SKCaParams struct {
	Gbar   float32 `desc:"strength of sKCa current"`
	Coeff  float32 `def:"4.6" desc:"Hill coefficient for asymptotic level of the gating current m"`
	C50    float32 `def:"0.35" desc:"50% Ca concentration baseline value in Hill equation"`
	CFast  float32 `def:"5" desc:"concentration of Ca at and above which time constant is fastest (Tau1)"`
	Tau0   float32 `def:"76" desc:"slow time constant, operative when no Ca is present"`
	Tau1   float32 `def:"4" desc:"fast time constant achieved when Ca is >= CFast"`
	C50Pow float32 `inactive:"+" desc:"C50 to the Coeff power"`
}

func (sp *SKCaParams) Defaults() {
	sp.Gbar = 0.1
	sp.Coeff = 4.6
	sp.C50 = 0.35
	sp.CFast = 5
	sp.Tau0 = 76
	sp.Tau1 = 4
	sp.Update()
}

func (sp *SKCaParams) Update() {
	sp.C50Pow = mat32.Pow(sp.C50, sp.Coeff)
}

// MAsympHill gives the asymptotic (driving) gating factor M as a function of CAi
// for the Hill equation version used in Fujita et al (2012)
func (sp *SKCaParams) MAsympHill(cai float32) float32 {
	capow := mat32.Pow(cai, sp.Coeff)
	return capow / (capow + sp.C50Pow)
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
func (sp *SKCaParams) MFmCa(cai, mcur float32) float32 {
	mas := sp.MAsympHill(cai)
	tau := float32(76.0)
	if cai >= sp.CFast {
		tau = sp.Tau1
	} else {
		tau = sp.Tau0 - (sp.Tau0-sp.Tau1)*(cai/sp.CFast)
	}
	return (mas - mcur) / tau
}
