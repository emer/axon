// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"cogentcore.org/core/mat32"
)

//gosl: start chans

// KirParams control the Kir potassium inwardly rectifying current,
// based on the equations from Lindroos et al (2018).
// The conductance is highest at low membrane potentials
type KirParams struct {

	// overall strength multiplier of Kir current.
	Gbar float32 `def:"0,0.012,0.015"`

	// Q10 is the 10 degree temperature correction factor
	Q10 float32 `def:"3"`

	// Asymptotic gating factor M, offset
	MinfOff float32 `def:"-102"`

	// Asymptotic gating factor M, time constant
	MinfTau float32 `def:"13"`

	// Rise time constant as a function of voltage, offset
	RiseOff float32 `def:"-60"`

	// Rise time constant as a function of voltage, time constant factor
	RiseTau float32 `def:"14"`

	// Decay time constant as a function of voltage, offset
	DecayOff float32 `def:"-31"`

	// Decay time constant as a function of voltage, time constant factor
	DecayTau float32 `def:"23"`
}

func (kp *KirParams) Defaults() {
	kp.Gbar = 0.0
	kp.Q10 = 3
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
	dm := (minf - *m) / (mtau * kp.Q10)
	*m += dm
	return dm
}

// MRates returns minf and mtau as a function of bio voltage
func (kp *KirParams) Minf(vbio float32) float32 {
	return 1.0 / (1 + mat32.FastExp((vbio-(kp.MinfOff))/kp.MinfTau))
}

// MRates returns minf and mtau as a function of bio voltage
func (kp *KirParams) MRates(vbio float32, minf, mtau *float32) {
	*minf = kp.Minf(vbio)
	alpha := 0.1 * mat32.FastExp((vbio-(kp.RiseOff))/(-kp.RiseTau))
	beta := 0.27 / (1 + mat32.FastExp((vbio-(kp.DecayOff))/(-kp.DecayTau)))
	sum := alpha + beta
	*mtau = 1.0 / sum
	return
}

// Gkir returns the overall net Kir conductance, and updates the m gating parameter
func (kp *KirParams) Gkir(v float32, m *float32) float32 {
	vbio := VToBio(v)
	g := kp.Gbar * *m
	kp.DM(vbio, m)
	return g
}

//gosl: end chans
