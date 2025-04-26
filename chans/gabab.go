// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"cogentcore.org/core/math32"
)

//gosl:start

// GABA-B is an inhibitory channel activated by the usual GABA inhibitory
// neurotransmitter, which is coupled to the GIRK G-protein coupled inwardly
// rectifying potassium (K) channel. It is ubiquitous in the brain, and critical
// for stability of spiking patterns over time in axon. The inward rectification
// is caused by a Mg+ ion block *from the inside* of the neuron,
// which means that these channels are most open when the neuron is hyperpolarized
// (inactive), and thus it serves to keep inactive neurons inactive.
// Based on Thomson & Destexhe (1999).
type GABABParams struct {

	// Gk is the strength of GABA-B conductance as contribution to Gk(t) factor
	// (which is then multiplied by Gbar.K that provides pA unit scaling).
	// The 0.015 default is a high value that works well in smaller networks.
	// Larger networks may benefit from lower levels (e.g., 0.012).
	// GababM activation factor can become large, so that overall GgabaB = ~50 nS.
	Gk float32 `default:"0.015,0.012,0"`

	// RiseTau is the rise time for bi-exponential time dynamics of GABA-B.
	RiseTau float32 `default:"45"`

	// DecayTau is the decay time for bi-exponential time dynamics of GABA-B.
	DecayTau float32 `default:"50"`

	// Gbase is the baseline level of GABA-B channels open independent of
	// inhibitory input (is added to spiking-produced conductance).
	Gbase float32 `default:"0.2"`

	// GiSpike is the multiplier for converting Gi to equivalent GABA spikes.
	GiSpike float32 `default:"10"`

	// MaxTime is the time offset when peak conductance occurs, in msec, computed
	// from RiseTau and DecayTau.
	MaxTime float32 `edit:"-"`

	// TauFact is the time constant factor used in integration:
	// (Decay / Rise) ^ (Rise / (Decay - Rise))
	TauFact float32 `display:"-"`

	// RiseDt = 1/Tau
	RiseDt float32 `display:"-" edit:"-"`

	// DecayDt = 1/Tau
	DecayDt float32 `display:"-" edit:"-"`

	pad, pad1, pad2 float32
}

func (gp *GABABParams) Defaults() {
	gp.Gk = 0.015
	gp.RiseTau = 45
	gp.DecayTau = 50
	gp.Gbase = 0.2
	gp.GiSpike = 10
	gp.Update()
}

func (gp *GABABParams) Update() {
	gp.TauFact = math32.Pow(gp.DecayTau/gp.RiseTau, gp.RiseTau/(gp.DecayTau-gp.RiseTau))
	gp.MaxTime = ((gp.RiseTau * gp.DecayTau) / (gp.DecayTau - gp.RiseTau)) * math32.Log(gp.DecayTau/gp.RiseTau)
	gp.RiseDt = 1.0 / gp.RiseTau
	gp.DecayDt = 1.0 / gp.DecayTau
}

func (gp *GABABParams) ShouldDisplay(field string) bool {
	switch field {
	case "Gk":
		return true
	default:
		return gp.Gk > 0
	}
}

// GFromV returns the GABA-B conductance as a function of v potential.
func (gp *GABABParams) GFromV(v float32) float32 {
	ve := max(v, -90.0)
	return (ve + 90.0) / (1.0 + math32.FastExp(0.1*((ve+90.0)+10.0)))
}

// GFromS returns the GABA-B conductance as a function of GABA spiking rate,
// based on normalized spiking factor (i.e., Gi from FFFB etc)
func (gp *GABABParams) GFromS(s float32) float32 {
	ss := s * gp.GiSpike
	if ss > 20 {
		return 1
	}
	return 1.0 / (1.0 + math32.FastExp(-(ss-7.1)/1.4))
}

// DeltaM computes the change in activation M based on the current
// activation m and the spike integration factor x.
func (gp *GABABParams) DeltaM(m, x float32) float32 {
	return (gp.TauFact*x - m) * gp.RiseDt
}

// MX updates the GABA-B / GIRK activation M and underlying X integration value
// based on current values and gi inhibitory conductance (proxy for GABA spikes)
func (gp *GABABParams) MX(gi float32, m, x *float32) {
	dM := gp.DeltaM(*m, *x)
	*x += gp.GFromS(gi) - (*x)*gp.DecayDt
	*m += dM
	return
}

// GgabaB returns the overall net GABAB / GIRK conductance including
// Gk, Gbase, and voltage-gating, as a function of activation value M.
func (gp *GABABParams) GgabaB(m, v float32) float32 {
	return gp.Gk * gp.GFromV(v) * (m + gp.Gbase)
}

//gosl:end
