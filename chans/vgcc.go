// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"cogentcore.org/core/math32"
)

//gosl:start

// VGCCParams control the standard L-type Ca channel
// All functions based on Urakubo et al (2008).
// Source code available at http://kurodalab.bs.s.u-tokyo.ac.jp/info/STDP/Urakubo2008.tar.gz.
// In particular look at the file MODEL/Poirazi_cell/CaL.g.
type VGCCParams struct {

	// Ge is the strength of the VGCC contribution to Ge(t) excitary
	// conductance. Ge(t) is later multiplied by Gbar.E for pA unit scaling.
	// The 0.12 value is from Urakubo et al (2008) model, which best fits actual model
	// behavior using axon equations (1.5 nominal in that model).
	// 0.02 works better in practice for not getting stuck in high plateau firing.
	Ge float32 `default:"0.02,0.12"`

	// calcium from conductance factor. Important for learning contribution of VGCC.
	Ca float32 `default:"0.25"`

	pad, pad1 int32
}

func (np *VGCCParams) Defaults() {
	np.Ge = 0.02
	np.Ca = 0.25
}

func (np *VGCCParams) Update() {
}

func (np *VGCCParams) ShouldDisplay(field string) bool {
	switch field {
	case "Ge":
		return true
	default:
		return np.Ge > 0
	}
}

// GFromV returns the VGCC conductance as a function of normalized membrane potential
// Based on Urakubo's calculation of `max` in CaL.g in the section commented 'i gate'.
func (np *VGCCParams) GFromV(v float32) float32 {
	if v > -0.5 && v < 0.5 { // this avoids divide by 0, and numerical instability around 0
		return 1.0 / (0.0756 * (1 + 0.0378*v))
	}
	return -v / (1.0 - math32.FastExp(0.0756*v))
}

// MFromV returns the M gate function from potential V.
// Based on Urakubo's calculation of `max` in CaL.g in the section commented 'm gate'.
func (np *VGCCParams) MFromV(v float32) float32 {
	// approximate values at the asymptotes for performance
	if v < -60 {
		return 0
	}
	if v > -10 {
		return 1
	}
	return 1.0 / (1.0 + math32.FastExp(-(v + 37)))
}

// HFromV returns the H gate function from potential V.
// Based on Urakubo's calculation of `max` in CaL.g in the section commented 'h gate'.
func (np *VGCCParams) HFromV(v float32) float32 {
	// approximate values at the asymptotes for performance
	if v < -50 {
		return 1
	}
	if v > -10 {
		return 0
	}
	return 1.0 / (1.0 + math32.FastExp((v+41)*2))
}

// DeltaMFromV returns the change at msec update scale in M factor
// as a function of V
func (np *VGCCParams) DeltaMFromV(v, m float32) float32 {
	vb := min(v, 0.0)
	return (np.MFromV(vb) - m) / 3.6
}

// DeltaHFromV returns the change at msec update scale in H factor
// as a function of V
func (np *VGCCParams) DeltaHFromV(v, h float32) float32 {
	vb := min(v, 0.0)
	return (np.HFromV(vb) - h) / 29.0
}

// Gvgcc returns the VGCC net conductance from m, h activation and v.
func (np *VGCCParams) Gvgcc(v, m, h float32) float32 {
	return np.Ge * np.GFromV(v) * m * m * m * h
}

// CaFromG returns the Ca from Gvgcc conductance, current Ca level, and v.
func (np *VGCCParams) CaFromG(v, g, ca float32) float32 {
	return -v * np.Ca * g
}

//gosl:end
