// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"goki.dev/mat32"
)

//gosl: start chans

// KirParams control the Kir potassium inwardly rectifying current
type KirParams struct {

	// overall strength multiplier of Kir current.
	Gbar float32 `def:"0,0.012,0.015"`

	Q float32 `def:"3"`

	MinfOff float32 `def:"-102"`

	MinfTau float32 `def:"13"`

	RiseOff float32 `def:"-60"`

	RiseTau float32 `def:"14"`

	DecayOff float32 `def:"-31"`

	DecayTau float32 `def:"23"`
}

func (kp *KirParams) Defaults() {
	kp.Gbar = 0.0
	kp.Q = 3
	kp.MinfOff = -102
	kp.MinfTau = 13
	kp.RiseOff = -60
	kp.RiseTau = 14
	kp.DecayOff = -31
	kp.DecayTau = 23
	kp.Update()
}

func (kp *KirParams) Update() {
}

// DM computes the change in M gating parameter, updating m
func (kp *KirParams) DM(vbio float32, m *float32) float32 {
	var minf, mtau float32
	kp.MRates(vbio, &minf, &mtau)
	dm := (minf - *m) / (mtau * kp.Q)
	*m += dm
	return dm
}

// MRates returns minf and mtau as a function of bio voltage
func (kp *KirParams) MRates(vbio float32, minf, mtau *float32) {
	*minf = 1.0 / (1 + mat32.FastExp((vbio-(kp.MinfOff))/kp.MinfTau))
	alpha := 0.1 * mat32.FastExp((vbio-(kp.RiseOff))/(-kp.RiseTau))
	beta := 0.27 / (1 + mat32.FastExp((vbio-(kp.DecayOff))/(-kp.DecayTau)))
	sum := alpha + beta
	*mtau = 1.0 / sum
	return
}

// Gkir returns the overall net Kir conductance, and updates the m gating parameter
func (kp *KirParams) Gkir(gabaB, v float32, m *float32) float32 {
	vbio := VToBio(v)
	// if vbio < -90 {
	// 	vbio = -90
	// }
	g := kp.Gbar * *m
	kp.DM(vbio, m)
	return g
}

//gosl: end chans
