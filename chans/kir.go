// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"cogentcore.org/core/math32"
)

//gosl:start

// KirParams control the kIR K+ inwardly rectifying current,
// based on the equations from Lindroos et al (2018).
// The conductance is highest at low membrane potentials.
type KirParams struct {

	// Gk is the strength of Kir conductance as contribution to Gk(t) factor
	// (which is then multiplied by Gbar.K that provides pA unit scaling).
	Gk float32 `default:"0.012,0.015,0"`

	// MinfOff is the asymptotic gating factor M, offset.
	MinfOff float32 `default:"-102"`

	// MinfTau is the asymptotic gating factor M, time constant.
	MinfTau float32 `default:"13"`

	// RiseOff is the rise time constant as a function of voltage, offset.
	RiseOff float32 `default:"-60"`

	// RiseTau is the rise time constant as a function of voltage, time constant factor.
	RiseTau float32 `default:"14"`

	// DecayOff is the decay time constant as a function of voltage, offset.
	DecayOff float32 `default:"-31"`

	// DecayTau is the decay time constant as a function of voltage, time constant factor.
	DecayTau float32 `default:"23"`

	// Mrest is Minf at resting membrane potential of -70, computed from other params.
	Mrest float32 `edit:"-"`
}

func (kp *KirParams) Defaults() {
	kp.Gk = 0.0
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
	case "Gk":
		return true
	default:
		return kp.Gk > 0
	}
}

// Minf returns Minf as a function of voltage potential.
func (kp *KirParams) Minf(v float32) float32 {
	return 1.0 / (1.0 + math32.FastExp((v-kp.MinfOff)/kp.MinfTau))
}

// MinfRest returns Minf at nominal resting membrane potential of -70mV
// which serves as the initial value.
func (kp *KirParams) MinfRest() float32 {
	return kp.Minf(-70.0)
}

// MTau returns mtau as a function of voltage.
func (kp *KirParams) MTau(v float32) float32 {
	alpha := 0.1 * math32.FastExp(-(v-kp.RiseOff)/kp.RiseTau)
	beta := 0.27 / (1.0 + math32.FastExp(-(v-kp.DecayOff)/kp.DecayTau))
	return 1.0 / (alpha + beta)
}

// DM computes the change in M gating parameter.
func (kp *KirParams) DM(v, m float32) float32 {
	minf := kp.Minf(v)
	mtau := kp.MTau(v)
	dm := (minf - m) / (mtau * 3) // 3 = Q10
	return dm
}

// Gkir returns the overall net Kir conductance.
func (kp *KirParams) Gkir(v float32, m float32) float32 {
	return kp.Gk * m
}

//gosl:end
