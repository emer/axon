// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "cogentcore.org/lab/gosl/slbool"

//gosl:start

// KNaParams implements sodium (Na) gated potassium (K) currents
// that drive adaptation (accommodation) in neural firing.
// As neurons spike, driving an influx of Na, this activates
// the K channels, which, like leak channels, pull the membrane
// potential back down toward rest (or even below).
type KNaParams struct {

	// On enables this component of KNa adaptation.
	On slbool.Bool

	// Gk is the maximum potential conductance contribution to Gk(t)
	// (which is then multiplied by Gbar.K that provides pA unit scaling).
	Gk float32 `default:"0.1"`

	// Rise is the time constant in ms for increase in conductance based on Na
	// concentration due to spiking.
	Rise float32

	// Decay is the time constant in ms for decay of conductance.
	Decay float32

	// Dt = 1/Tau rate constant.
	DtRise float32 `display:"-"`

	// Dt = 1/Tau rate constant.
	DtDecay float32 `display:"-"`

	pad, pad1 int32
}

func (ka *KNaParams) Defaults() {
	ka.On.SetBool(true)
	ka.Rise = 50
	ka.Decay = 100
	ka.Gk = 0.1
	ka.Update()
}

func (ka *KNaParams) Update() {
	ka.DtRise = 1 / ka.Rise
	ka.DtDecay = 1 / ka.Decay
}

func (ka *KNaParams) ShouldDisplay(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return ka.On.IsTrue()
	}
}

// GcFromSpike updates the KNa conductance based on spike or not.
func (ka *KNaParams) GcFromSpike(gKNa *float32, spike bool) {
	if ka.On.IsTrue() {
		if spike {
			*gKNa += ka.DtRise * (ka.Gk - *gKNa)
		} else {
			*gKNa -= ka.DtDecay * *gKNa
		}
	} else {
		*gKNa = 0
	}
}

// KNaMedSlow describes sodium-gated potassium channel adaptation mechanism.
// Evidence supports 2 different time constants:
// Slick (medium) and Slack (slow)
type KNaMedSlow struct {

	// On means apply K-Na adaptation.
	On slbool.Bool

	// TrialSlow engages an optional version of Slow that discretely turns on at
	// the start of new trial (NewState): nrn.GknaSlow += Slow.Gk * nrn.CaDPrev.
	// This achieves a strong form of adaptation.
	TrialSlow slbool.Bool

	pad, pad1 int32

	// Med is medium time-scale adaptation.
	Med KNaParams `display:"inline"`

	// Slow is slow time-scale adaptation.
	Slow KNaParams `display:"inline"`
}

func (ka *KNaMedSlow) Defaults() {
	ka.Med.Defaults()
	ka.Slow.Defaults()
	ka.Med.Rise = 50
	ka.Med.Decay = 200
	ka.Med.Gk = 0.1
	ka.Slow.Rise = 1000
	ka.Slow.Decay = 1000
	ka.Slow.Gk = 0.1
	ka.Update()
}

func (ka *KNaMedSlow) Update() {
	ka.Med.Update()
	ka.Slow.Update()
}

func (ka *KNaMedSlow) ShouldDisplay(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return ka.On.IsTrue()
	}
}

// GcFromSpike updates med, slow time scales of KNa adaptation from spiking.
func (ka *KNaMedSlow) GcFromSpike(gKNaM, gKNaS *float32, spike bool) {
	ka.Med.GcFromSpike(gKNaM, spike)
	if ka.TrialSlow.IsFalse() {
		ka.Slow.GcFromSpike(gKNaS, spike)
	}
}

//gosl:end
