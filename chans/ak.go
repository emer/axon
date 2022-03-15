// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "github.com/goki/mat32"

// AKParams control an A-type K channel, which is voltage gated with maximal
// activation around -37 mV.  It has two state variables, M (v-gated opening)
// and H (v-gated closing), which integrate with fast and slow time constants,
// respectively.  H relatively quickly hits an asymptotic level of inactivation
// for sustained activity patterns.
// It is particularly important for counteracting the excitatory effects of
// voltage gated calcium channels which can otherwise drive runaway excitatory currents.
// See AKsParams for a much simpler version that works fine when full AP-like spikes are
// not simulated, as in our standard axon models.
type AKParams struct {
	Gbar float32 `def:"0.01" desc:"strength of AK current"`
	Beta float32 `def:"0.01446,02039" desc:"multiplier for the beta term; 0.01446 for distal, 0.02039 for proximal dendrites"`
	Dm   float32 `def:"0.5,0.25" desc:"Dm factor: 0.5 for distal, 0.25 for proximal"`
	Koff float32 `def:"1.8,1.5" desc:"offset for K, 1.8 for distal, 1.5 for proximal"`
	Voff float32 `def:"1,11" desc:"voltage offset for alpha and beta functions: 1 for distal, 11 for proximal"`
	Hf   float32 `def:"0.1133,0.1112" desc:"h multiplier factor, 0.1133 for distal, 0.1112 for proximal"`
}

// Defaults sets the parameters for distal dendrites
func (ap *AKParams) Defaults() {
	ap.Gbar = 0.01
	ap.Distal()
}

func (ap *AKParams) Update() {
}

// Distal sets the parameters for distal dendrites
func (ap *AKParams) Distal() {
	ap.Beta = 0.01446
	ap.Dm = 0.5
	ap.Koff = 1.8
	ap.Voff = 1
	ap.Hf = 0.1133
}

// Proximal sets parameters for proximal dendrites
func (ap *AKParams) Proximal() {
	ap.Beta = 0.02039
	ap.Dm = 0.25
	ap.Koff = 1.5
	ap.Voff = 11
	ap.Hf = 0.1112
}

// AlphaFmVK returns the Alpha function from vbio (not normalized, must not exceed 0)
func (ap *AKParams) AlphaFmVK(vbio, k float32) float32 {
	return mat32.FastExp(0.03707 * k * (vbio - ap.Voff))
}

// BetaFmVK returns the Beta function from vbio (not normalized, must not exceed 0)
func (ap *AKParams) BetaFmVK(vbio, k float32) float32 {
	return mat32.FastExp(ap.Beta * k * (vbio - ap.Voff))
}

// KFmV returns the K value from vbio (not normalized, must not exceed 0)
func (ap *AKParams) KFmV(vbio float32) float32 {
	return -ap.Koff - 1.0/(1.0+mat32.FastExp((vbio+40)/5))
}

// HFmV returns the H gate value from vbio (not normalized, must not exceed 0)
func (ap *AKParams) HFmV(vbio float32) float32 {
	return 1.0 / (1.0 + mat32.FastExp(ap.Hf*(vbio+56)))
}

// HTauFmV returns the HTau rate constant in msec from vbio (not normalized, must not exceed 0)
func (ap *AKParams) HTauFmV(vbio float32) float32 {
	tau := 0.26 * (vbio + 50)
	if tau < 2 {
		tau = 2
	}
	return tau
}

// MFmAlpha returns the M gate factor from alpha
func (ap *AKParams) MFmAlpha(alpha float32) float32 {
	return 1.0 / (1.0 + alpha)
}

// MTauFmAlphaBeta returns the MTau rate constant in msec from alpha, beta
func (ap *AKParams) MTauFmAlphaBeta(alpha, beta float32) float32 {
	return 1 + beta/(ap.Dm*(1+alpha)) // minimum of 1 msec
}

// DMHFmV returns the change at msec update scale in M, H factors
// as a function of V normalized (0-1)
func (ap *AKParams) DMHFmV(v, m, h float32) (float32, float32) {
	vbio := VToBio(v)
	if vbio > 0 {
		vbio = 0
	}
	k := ap.KFmV(vbio)
	a := ap.AlphaFmVK(vbio, k)
	b := ap.BetaFmVK(vbio, k)
	mt := ap.MTauFmAlphaBeta(a, b)
	ht := ap.HTauFmV(vbio)
	dm := (ap.MFmAlpha(a) - m) / mt
	dh := (ap.HFmV(vbio) - h) / ht
	return dm, dh
}

// Gak returns the AK net conductance from m, h gates
func (ap *AKParams) Gak(m, h float32) float32 {
	return ap.Gbar * m * h
}

//////////////////////////////////////////////////////////////////////
//  Simplified AK

// AKsParams provides a highly simplified stateless A-type K channel
// that only has the voltage-gated activation (M) dynamic with a cutoff
// that ends up capturing a close approximation to the much more complex AK function.
// This is voltage gated with maximal activation around -37 mV.
// It is particularly important for counteracting the excitatory effects of
// voltage gated calcium channels which can otherwise drive runaway excitatory currents.
type AKsParams struct {
	Gbar float32 `def:"0.01" desc:"strength of AK current"`
	Hf   float32 `def:"0.076" desc:"H factor as a constant multiplier on overall M factor result -- rescales M to level consistent with H being present at full strength"`
	Mf   float32 `def:"0.075" desc:"multiplier for M -- determines slope of function"`
	Voff float32 `def:"2" desc:"voltage offset in biological units for M function"`
	Vmax float32 `def:-37" desc:"voltage level of maximum channel opening -- stays flat above that"`
}

// Defaults sets the parameters for distal dendrites
func (ap *AKsParams) Defaults() {
	ap.Gbar = 0.01
	ap.Hf = 0.076
	ap.Mf = 0.075
	ap.Voff = 2
	ap.Vmax = -37
}

func (ap *AKsParams) Update() {
}

// MFmV returns the M gate function from vbio
func (ap *AKsParams) MFmV(vbio float32) float32 {
	if vbio > ap.Vmax {
		vbio = ap.Vmax
	}
	return ap.Hf / (1.0 + mat32.FastExp(-ap.Mf*(vbio+ap.Voff)))
}

// MFmVnorm returns the M gate function from vnorm
func (ap *AKsParams) MFmVnorm(v float32) float32 {
	return ap.MFmV(VToBio(v))
}

// Gak returns the conductance as a function of normalized Vm
// GBar * MFmVnorm(v)
func (ap *AKsParams) Gak(v float32) float32 {
	return ap.Gbar * ap.MFmVnorm(v)
}
