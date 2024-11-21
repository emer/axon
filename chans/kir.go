// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"cogentcore.org/core/math32"
)

//gosl:start chans

// KirParams control the Kir potassium inwardly rectifying current,
// based on the equations from Lindroos et al (2018).
// The conductance is highest at low membrane potentials
type KirParams struct {

	// overall strength multiplier of Kir current.
	Gbar float32 `default:"0,0.012,0.015"`

	// Asymptotic gating factor M, offset
	MinfOff float32 `default:"-102"`

	// Asymptotic gating factor M, time constant
	MinfTau float32 `default:"13"`

	// Rise time constant as a function of voltage, offset
	RiseOff float32 `default:"-60"`

	// Rise time constant as a function of voltage, time constant factor
	RiseTau float32 `default:"14"`

	// Decay time constant as a function of voltage, offset
	DecayOff float32 `default:"-31"`

	// Decay time constant as a function of voltage, time constant factor
	DecayTau float32 `default:"23"`

	// Mrest is Minf at resting membrane potential of -70, computed from other params
	Mrest float32 `edit:"-"`
}

func (kp *KirParams) Defaults() {
	kp.Gbar = 0.0
	kp.MinfOff = -102
	kp.MinfTau = 13
	kp.RiseOff = -60
	kp.RiseTau = 14
	kp.DecayOff = -31
	kp.DecayTau = 23
	kp.Update()
}

func (kp *KirParams) Update() {
	kp.Mrest = kp.MinfRest()
}

func (kp *KirParams) ShouldDisplay(field string) bool {
	switch field {
	case "Gbar":
		return true
	default:
		return kp.Gbar > 0
	}
}

// MRates returns Minf as a function of bio voltage
func (kp *KirParams) Minf(vbio float32) float32 {
	return 1.0 / (1.0 + math32.FastExp((vbio-(kp.MinfOff))/kp.MinfTau))
}

// MinfRest returns Minf at nominal resting membrane potential of -70mV
// which serves as the initial value
func (kp *KirParams) MinfRest() float32 {
	return kp.Minf(-70.0)
}

// MTau returns mtau as a function of bio voltage
func (kp *KirParams) MTau(vbio float32) float32 {
	alpha := 0.1 * math32.FastExp((vbio-(kp.RiseOff))/(-kp.RiseTau))
	beta := 0.27 / (1.0 + math32.FastExp((vbio-(kp.DecayOff))/(-kp.DecayTau)))
	sum := alpha + beta
	return 1.0 / sum
}

// DM computes the change in M gating parameter
func (kp *KirParams) DM(vbio, m float32) float32 {
	minf := kp.Minf(vbio)
	// mtau := kp.MTau(vbio)
	// minf := float32(0.15)
	mtau := float32(4.0)
	dm := (minf - m) / (mtau * 3) // 3 = Q10
	return dm
}

// Gkir returns the overall net Kir conductance.
func (kp *KirParams) Gkir(v float32, m float32) float32 {
	return kp.Gbar * m
}

//gosl:end chans
