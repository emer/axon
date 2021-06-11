// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package knadapt provides code for sodium (Na) gated potassium (K) currents that drive
adaptation (accommodation) in neural firing.  As neurons spike, driving an influx of Na,
this activates the K channels, which, like leak channels, pull the membrane potential
back down toward rest (or even below).  Multiple different time constants have been
identified and this implementation supports 3:
M-type (fast), Slick (medium), and Slack (slow)

Here's a good reference:

Kaczmarek, L. K. (2013). Slack, Slick, and Sodium-Activated Potassium Channels.
ISRN Neuroscience, 2013. https://doi.org/10.1155/2013/354262

This package supports both spiking and rate-coded activations.
*/
package knadapt

// Chan describes one channel type of sodium-gated adaptation, with a specific
// set of rate constants.
type Chan struct {
	On   bool    `desc:"if On, use this component of K-Na adaptation"`
	Rise float32 `viewif:"On" desc:"Rise rate of fast time-scale adaptation as function of Na concentration -- directly multiplies -- 1/rise = tau for rise rate"`
	Max  float32 `viewif:"On" desc:"Maximum potential conductance of fast K channels -- divide nA biological value by 10 for the normalized units here"`
	Tau  float32 `viewif:"On" desc:"time constant in cycles for decay of adaptation, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life)"`
	Dt   float32 `view:"-" desc:"1/Tau rate constant"`
}

func (ka *Chan) Defaults() {
	ka.On = true
	ka.Rise = 0.01
	ka.Max = 0.1
	ka.Tau = 100
	ka.Update()
}

func (ka *Chan) Update() {
	ka.Dt = 1 / ka.Tau
}

// GcFmSpike updates the KNa conductance based on spike or not
func (ka *Chan) GcFmSpike(gKNa *float32, spike bool) {
	if ka.On {
		if spike {
			*gKNa += ka.Rise * (ka.Max - *gKNa)
		} else {
			*gKNa -= ka.Dt * *gKNa
		}
	} else {
		*gKNa = 0
	}
}

// GcFmRate updates the KNa conductance based on rate-coded activation.
// act should already have the compensatory rate multiplier prior to calling.
func (ka *Chan) GcFmRate(gKNa *float32, act float32) {
	if ka.On {
		*gKNa += act*ka.Rise*(ka.Max-*gKNa) - (ka.Dt * *gKNa)
	} else {
		*gKNa = 0
	}
}

// Params describes sodium-gated potassium channel adaptation mechanism.
// Evidence supports at least 3 different time constants:
// M-type (fast), Slick (medium), and Slack (slow)
type Params struct {
	On   bool    `desc:"if On, apply K-Na adaptation"`
	Rate float32 `viewif:"On" def:"0.8" desc:"extra multiplier for rate-coded activations on rise factors -- adjust to match discrete spiking"`
	Fast Chan    `view:"inline" desc:"fast time-scale adaptation"`
	Med  Chan    `view:"inline" desc:"medium time-scale adaptation"`
	Slow Chan    `view:"inline" desc:"slow time-scale adaptation"`
}

func (ka *Params) Defaults() {
	ka.Rate = 0.8
	ka.Fast.Defaults()
	ka.Med.Defaults()
	ka.Slow.Defaults()
	ka.Fast.Tau = 50
	ka.Fast.Rise = 0.05
	ka.Fast.Max = 0.1
	ka.Med.Tau = 200
	ka.Med.Rise = 0.02
	ka.Med.Max = 0.2
	ka.Slow.Tau = 1000
	ka.Slow.Rise = 0.001
	ka.Slow.Max = 0.2
	ka.Update()
}

func (ka *Params) Update() {
	ka.Fast.Update()
	ka.Med.Update()
	ka.Slow.Update()
}

// GcFmSpike updates all time scales of KNa adaptation from spiking
func (ka *Params) GcFmSpike(gKNaF, gKNaM, gKNaS *float32, spike bool) {
	ka.Fast.GcFmSpike(gKNaF, spike)
	ka.Med.GcFmSpike(gKNaM, spike)
	ka.Slow.GcFmSpike(gKNaS, spike)
}

// GcFmRate updates all time scales of KNa adaptation from rate code activation
func (ka *Params) GcFmRate(gKNaF, gKNaM, gKNaS *float32, act float32) {
	act *= ka.Rate
	ka.Fast.GcFmRate(gKNaF, act)
	ka.Med.GcFmRate(gKNaM, act)
	ka.Slow.GcFmRate(gKNaS, act)
}
