// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"github.com/goki/mat32"
)

//gosl: start chans

// VGCCParams control the standard L-type Ca channel
// All functions based on Urakubo et al (2008).
// Source code available at http://kurodalab.bs.s.u-tokyo.ac.jp/info/STDP/Urakubo2008.tar.gz.
// In particular look at the file MODEL/Poirazi_cell/CaL.g.
type VGCCParams struct {

	// [def: 0.02,0.12] strength of VGCC current -- 0.12 value is from Urakubo et al (2008) model -- best fits actual model behavior using axon equations (1.5 nominal in that model), 0.02 works better in practice for not getting stuck in high plateau firing
	Gbar float32 `def:"0.02,0.12" desc:"strength of VGCC current -- 0.12 value is from Urakubo et al (2008) model -- best fits actual model behavior using axon equations (1.5 nominal in that model), 0.02 works better in practice for not getting stuck in high plateau firing"`

	// [def: 25] [viewif: Gbar>0] calcium from conductance factor -- important for learning contribution of VGCC
	Ca float32 `viewif:"Gbar>0" def:"25" desc:"calcium from conductance factor -- important for learning contribution of VGCC"`

	pad, pad1 int32
}

func (np *VGCCParams) Defaults() {
	np.Gbar = 0.02
	np.Ca = 25
}

func (np *VGCCParams) Update() {
}

// GFmV returns the VGCC conductance as a function of normalized membrane potential
// Based on Urakubo's calculation of `max` in CaL.g in the section commented 'i gate'.
func (np *VGCCParams) GFmV(v float32) float32 {
	vbio := VToBio(v)
	if vbio > -0.5 && vbio < 0.5 { // this avoids divide by 0, and numerical instability around 0
		return 1.0 / (0.0756 * (1 + 0.0378*vbio))
	}
	return -vbio / (1.0 - mat32.FastExp(0.0756*vbio))
}

// MFmV returns the M gate function from vbio (not normalized, must not exceed 0).
// Based on Urakubo's calculation of `max` in CaL.g in the section commented 'm gate'.
func (np *VGCCParams) MFmV(vbio float32) float32 {
	// approximate values at the asymptotes for performance
	if vbio < -60 {
		return 0
	}
	if vbio > -10 {
		return 1
	}
	return 1.0 / (1.0 + mat32.FastExp(-(vbio + 37)))
}

// HFmV returns the H gate function from vbio (not normalized, must not exceed 0)
// Based on Urakubo's calculation of `max` in CaL.g in the section commented 'h gate'.
func (np *VGCCParams) HFmV(vbio float32) float32 {
	// approximate values at the asymptotes for performance
	if vbio < -50 {
		return 1
	}
	if vbio > -10 {
		return 0
	}
	return 1.0 / (1.0 + mat32.FastExp((vbio+41)*2))
}

// DMHFmV returns the change at msec update scale in M, H factors
// as a function of V normalized (0-1)
func (np *VGCCParams) DMHFmV(v, m, h float32, dm, dh *float32) {
	vbio := VToBio(v)
	if vbio > 0 {
		vbio = 0
	}
	*dm = (np.MFmV(vbio) - m) / 3.6
	*dh = (np.HFmV(vbio) - h) / 29.0
}

// Gvgcc returns the VGCC net conductance from m, h activation and vm
func (np *VGCCParams) Gvgcc(vm, m, h float32) float32 {
	return np.Gbar * np.GFmV(vm) * m * m * m * h
}

// CaFmG returns the Ca from Gvgcc conductance, current Ca level,
// and normalized membrane potential.
func (np *VGCCParams) CaFmG(v, g, ca float32) float32 {
	vbio := VToBio(v)
	return -vbio * np.Ca * g
}

//gosl: end chans
