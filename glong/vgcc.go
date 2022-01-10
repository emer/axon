// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import "github.com/goki/mat32"

// VGCCParams control the standard L-type Ca channel
type VGCCParams struct {
	Gbar float32 `def:"0.01" desc:"strength of VGCC current"`
}

func (np *VGCCParams) Defaults() {
	np.Gbar = 0.01
}

// GFmV returns the VGCC conductance as a function of normalized membrane potential
func (np *VGCCParams) GFmV(v float32) float32 {
	vbio := mat32.Min(v*100-100, 0) // critical to not go past 0
	return -vbio / (1.0 - mat32.FastExp(0.0756*vbio))
}

// DMHFmV returns the change in M, H factors as a function of V
func (np *VGCCParams) DMHFmV(v, m, h float32) (float32, float32) {
	vbio := mat32.Min(v*100-100, 0) // critical to not go past 0
	dm := ((1.0 / (1.0 + mat32.FastExp(-(vbio + 37)))) - m) / 0.0036
	dh := ((1.0 / (1.0 + mat32.FastExp((vbio+41)/0.5))) - h) / 0.029
	return dm, dh
}

// Gvgcc returns the VGCC net conductance from m, h activation and vm
func (np *VGCCParams) Gvgcc(vm, m, h float32) float32 {
	return np.Gbar * np.GFmV(vm) * m * m * m * h
}
