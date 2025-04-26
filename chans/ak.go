// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "cogentcore.org/core/math32"

//////// Simplified AK

//gosl:start

// AKsParams provides a highly simplified stateless A-type K+ channel
// that only has the voltage-gated activation (M) dynamic with a cutoff
// that ends up capturing a close approximation to the much more complex AK function.
// This is voltage gated with maximal activation around -37 mV.
// It is particularly important for counteracting the excitatory effects of
// voltage gated calcium channels which can otherwise drive runaway excitatory currents.
type AKsParams struct {

	// strength of AK conductance as contribution to g_k(t) factor
	// (which is then multiplied by gbar_k that provides pA unit scaling).
	Gk float32 `default:"0.1,0.01,2"`

	// Hf is the multiplier factor as a constant multiplier
	// on overall M factor result. Rescales M to level consistent
	// with H being present at full strength.
	Hf float32 `default:"0.076"`

	// Mf is the multiplier factor for M, determines slope of function.
	Mf float32 `default:"0.075"`

	// Voff is the voltage offset for M function.
	Voff float32 `default:"2"`

	// Vmax is the voltage level of maximum channel opening: stays flat above that.
	Vmax float32 `default:-37" desc:""`

	pad, pad1, pad2 int32
}

// Defaults sets the parameters for distal dendrites
func (ap *AKsParams) Defaults() {
	ap.Gk = 0.1
	ap.Hf = 0.076
	ap.Mf = 0.075
	ap.Voff = 2
	ap.Vmax = -37
}

func (ap *AKsParams) Update() {
}

func (ap *AKsParams) ShouldDisplay(field string) bool {
	switch field {
	case "Gk":
		return true
	default:
		return ap.Gk > 0
	}
}

// MFromV returns the M gate function from v
func (ap *AKsParams) MFromV(v float32) float32 {
	av := v
	if v > ap.Vmax {
		av = ap.Vmax
	}
	return ap.Hf / (1.0 + math32.FastExp(-ap.Mf*(av+ap.Voff)))
}

// Gak returns the conductance as a function of normalized Vm
// GBar * MFromV(v)
func (ap *AKsParams) Gak(v float32) float32 {
	return ap.Gk * ap.MFromV(v)
}

//gosl:end
