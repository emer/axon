// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "github.com/goki/gosl/slbool"

//gosl: start chans

// KNaParams implements sodium (Na) gated potassium (K) currents
// that drive adaptation (accommodation) in neural firing.
// As neurons spike, driving an influx of Na, this activates
// the K channels, which, like leak channels, pull the membrane
// potential back down toward rest (or even below).
type KNaParams struct {
	On   slbool.Bool `desc:"if On, use this component of K-Na adaptation"`
	Rise float32     `viewif:"On" desc:"Rise rate of fast time-scale adaptation as function of Na concentration due to spiking -- directly multiplies -- 1/rise = tau for rise rate"`
	Max  float32     `viewif:"On" desc:"Maximum potential conductance of fast K channels -- divide nA biological value by 10 for the normalized units here"`
	Tau  float32     `viewif:"On" desc:"time constant in cycles for decay of adaptation, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life)"`

	Dt float32 `view:"-" desc:"1/Tau rate constant"`

	pad, pad1, pad2 int32
}

func (ka *KNaParams) Defaults() {
	ka.On.SetBool(true)
	ka.Rise = 0.01
	ka.Max = 0.1
	ka.Tau = 100
	ka.Update()
}

func (ka *KNaParams) Update() {
	ka.Dt = 1 / ka.Tau
}

// GcFmSpike updates the KNa conductance based on spike or not
func (ka *KNaParams) GcFmSpike(gKNa *float32, spike bool) {
	if ka.On.IsTrue() {
		if spike {
			*gKNa += ka.Rise * (ka.Max - *gKNa)
		} else {
			*gKNa -= ka.Dt * *gKNa
		}
	} else {
		*gKNa = 0
	}
}

// KNaMedSlow describes sodium-gated potassium channel adaptation mechanism.
// Evidence supports 2 different time constants:
// Slick (medium) and Slack (slow)
type KNaMedSlow struct {
	On        slbool.Bool `desc:"if On, apply K-Na adaptation"`
	TrialSlow slbool.Bool `desc:"engages an optional version of Slow that discretely turns on at start of new trial (NewState): nrn.GknaSlow += Slow.Max * nrn.SpkPrv -- achieves a strong form of adaptation"`

	pad, pad1 int32

	Med  KNaParams `viewif:"On" view:"inline" desc:"medium time-scale adaptation"`
	Slow KNaParams `viewif:"On" view:"inline" desc:"slow time-scale adaptation"`
}

func (ka *KNaMedSlow) Defaults() {
	ka.Med.Defaults()
	ka.Slow.Defaults()
	ka.Med.Tau = 200
	ka.Med.Rise = 0.02
	ka.Med.Max = 0.2
	ka.Slow.Tau = 1000
	ka.Slow.Rise = 0.001
	ka.Slow.Max = 0.2
	ka.Update()
}

func (ka *KNaMedSlow) Update() {
	ka.Med.Update()
	ka.Slow.Update()
}

// GcFmSpike updates med, slow time scales of KNa adaptation from spiking
func (ka *KNaMedSlow) GcFmSpike(gKNaM, gKNaS *float32, spike bool) {
	ka.Med.GcFmSpike(gKNaM, spike)
	if ka.TrialSlow.IsFalse() {
		ka.Slow.GcFmSpike(gKNaS, spike)
	}
}

//gosl: end chans
