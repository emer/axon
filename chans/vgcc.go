// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "github.com/goki/mat32"

// VGCCParams control the standard L-type Ca channel
type VGCCParams struct {
	Gbar float32 `def:"0.12" desc:"strength of VGCC current -- 0.12 value from Urakubo et al (2008) model"`
}

func (np *VGCCParams) Defaults() {
	np.Gbar = 0.12
}

func (np *VGCCParams) Update() {
}

// GFmV returns the VGCC conductance as a function of normalized membrane potential
func (np *VGCCParams) GFmV(v float32) float32 {
	vbio := VToBio(v)
	if vbio > -0.1 && vbio < 0.1 {
		return 1.0 / (0.0756 + 0.5*vbio)
	}
	return -vbio / (1.0 - mat32.FastExp(0.0756*vbio))
}

// MFmV returns the M gate function from vbio (not normalized, must not exceed 0)
func (np *VGCCParams) MFmV(vbio float32) float32 {
	return 1.0 / (1.0 + mat32.FastExp(-(vbio + 37)))
}

// HFmV returns the H gate function from vbio (not normalized, must not exceed 0)
func (np *VGCCParams) HFmV(vbio float32) float32 {
	return 1.0 / (1.0 + mat32.FastExp((vbio+41)*2))
}

// DMHFmV returns the change at msec update scale in M, H factors
// as a function of V normalized (0-1)
func (np *VGCCParams) DMHFmV(v, m, h float32) (float32, float32) {
	vbio := VToBio(v)
	if vbio > 0 {
		vbio = 0
	}
	dm := (np.MFmV(vbio) - m) / 3.6
	dh := (np.HFmV(vbio) - h) / 29.0
	return dm, dh
}

// Gvgcc returns the VGCC net conductance from m, h activation and vm
func (np *VGCCParams) Gvgcc(vm, m, h float32) float32 {
	return np.Gbar * np.GFmV(vm) * m * m * m * h
}
