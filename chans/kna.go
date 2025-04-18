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

	// On enables this component of K-Na adaptation.
	On slbool.Bool

	// Rise rate of fast time-scale adaptation as function of Na
	// concentration due to spiking. Directly multiplies, 1/rise = tau
	// for rise rate.
	Rise float32

	// Max is the maximum potential conductance contribution to Gk(t)
	// (which is then multiplied by Gbar.K that provides pA unit scaling).
	Max float32

	// Tau is the time constant in cycles for decay of adaptation,
	// in ms (milliseconds) (tau is roughly how long it takes
	// for value to change significantly -- 1.4x the half-life).
	Tau float32

	// Dt = 1/Tau rate constant.
	Dt float32 `display:"-"`

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

	// On means apply K-Na adaptation.
	On slbool.Bool

	// TrialSlow engages an optional version of Slow that discretely turns on at
	// the start of new trial (NewState): nrn.GknaSlow += Slow.Max * nrn.CaDPrev.
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
