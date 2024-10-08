// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "cogentcore.org/core/math32"

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

	// strength of AK current
	Gbar float32 `default:"1,0.1,0.01"`

	// multiplier for the beta term; 0.01446 for distal, 0.02039 for proximal dendrites
	Beta float32 `default:"0.01446,02039"`

	// Dm factor: 0.5 for distal, 0.25 for proximal
	Dm float32 `default:"0.5,0.25"`

	// offset for K, 1.8 for distal, 1.5 for proximal
	Koff float32 `default:"1.8,1.5"`

	// voltage offset for alpha and beta functions: 1 for distal, 11 for proximal
	Voff float32 `default:"1,11"`

	// h multiplier factor, 0.1133 for distal, 0.1112 for proximal
	Hf float32 `default:"0.1133,0.1112"`

	pad, pad1 float32
}

// Defaults sets the parameters for distal dendrites
func (ap *AKParams) Defaults() {
	ap.Gbar = 0.01
	ap.Distal()
}

func (ap *AKParams) Update() {
}

func (ap *AKParams) ShouldDisplay(field string) bool {
	switch field {
	case "Gbar":
		return true
	default:
		return ap.Gbar > 0
	}
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

// AlphaFromVK returns the Alpha function from vbio (not normalized, must not exceed 0)
func (ap *AKParams) AlphaFromVK(vbio, k float32) float32 {
	return math32.FastExp(0.03707 * k * (vbio - ap.Voff))
}

// BetaFromVK returns the Beta function from vbio (not normalized, must not exceed 0)
func (ap *AKParams) BetaFromVK(vbio, k float32) float32 {
	return math32.FastExp(ap.Beta * k * (vbio - ap.Voff))
}

// KFromV returns the K value from vbio (not normalized, must not exceed 0)
func (ap *AKParams) KFromV(vbio float32) float32 {
	return -ap.Koff - 1.0/(1.0+math32.FastExp((vbio+40)/5))
}

// HFromV returns the H gate value from vbio (not normalized, must not exceed 0)
func (ap *AKParams) HFromV(vbio float32) float32 {
	return 1.0 / (1.0 + math32.FastExp(ap.Hf*(vbio+56)))
}

// HTauFromV returns the HTau rate constant in msec from vbio (not normalized, must not exceed 0)
func (ap *AKParams) HTauFromV(vbio float32) float32 {
	tau := 0.26 * (vbio + 50)
	if tau < 2 {
		tau = 2
	}
	return tau
}

// MFromAlpha returns the M gate factor from alpha
func (ap *AKParams) MFromAlpha(alpha float32) float32 {
	return 1.0 / (1.0 + alpha)
}

// MTauFromAlphaBeta returns the MTau rate constant in msec from alpha, beta
func (ap *AKParams) MTauFromAlphaBeta(alpha, beta float32) float32 {
	return 1 + beta/(ap.Dm*(1+alpha)) // minimum of 1 msec
}

// DMHFromV returns the change at msec update scale in M, H factors
// as a function of V normalized (0-1)
func (ap *AKParams) DMHFromV(v, m, h float32) (float32, float32) {
	vbio := VToBio(v)
	if vbio > 0 {
		vbio = 0
	}
	k := ap.KFromV(vbio)
	a := ap.AlphaFromVK(vbio, k)
	b := ap.BetaFromVK(vbio, k)
	mt := ap.MTauFromAlphaBeta(a, b)
	ht := ap.HTauFromV(vbio)
	dm := (ap.MFromAlpha(a) - m) / mt
	dh := (ap.HFromV(vbio) - h) / ht
	return dm, dh
}

// Gak returns the AK net conductance from m, h gates
func (ap *AKParams) Gak(m, h float32) float32 {
	return ap.Gbar * m * h
}

//////////////////////////////////////////////////////////////////////
//  Simplified AK

//gosl:start chans

// AKsParams provides a highly simplified stateless A-type K channel
// that only has the voltage-gated activation (M) dynamic with a cutoff
// that ends up capturing a close approximation to the much more complex AK function.
// This is voltage gated with maximal activation around -37 mV.
// It is particularly important for counteracting the excitatory effects of
// voltage gated calcium channels which can otherwise drive runaway excitatory currents.
type AKsParams struct {

	// strength of AK current
	Gbar float32 `default:"2,0.1,0.01"`

	// H factor as a constant multiplier on overall M factor result -- rescales M to level consistent with H being present at full strength
	Hf float32 `default:"0.076"`

	// multiplier for M -- determines slope of function
	Mf float32 `default:"0.075"`

	// voltage offset in biological units for M function
	Voff float32 `default:"2"`

	//
	Vmax float32 `default:-37" desc:"voltage level of maximum channel opening -- stays flat above that"`

	pad, pad1, pad2 int32
}

// Defaults sets the parameters for distal dendrites
func (ap *AKsParams) Defaults() {
	ap.Gbar = 0.1
	ap.Hf = 0.076
	ap.Mf = 0.075
	ap.Voff = 2
	ap.Vmax = -37
}

func (ap *AKsParams) Update() {
}

func (ap *AKsParams) ShouldDisplay(field string) bool {
	switch field {
	case "Gbar":
		return true
	default:
		return ap.Gbar > 0
	}
}

// MFromV returns the M gate function from vbio
func (ap *AKsParams) MFromV(vbio float32) float32 {
	av := vbio
	if vbio > ap.Vmax {
		av = ap.Vmax
	}
	return ap.Hf / (1.0 + math32.FastExp(-ap.Mf*(av+ap.Voff)))
}

// MFromVnorm returns the M gate function from vnorm
func (ap *AKsParams) MFromVnorm(v float32) float32 {
	return ap.MFromV(VToBio(v))
}

// Gak returns the conductance as a function of normalized Vm
// GBar * MFromVnorm(v)
func (ap *AKsParams) Gak(v float32) float32 {
	return ap.Gbar * ap.MFromVnorm(v)
}

//gosl:end chans
