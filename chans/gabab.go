// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"cogentcore.org/core/mat32"
)

//gosl: start chans

// GABABParams control the GABAB dynamics in PFC Maint neurons,
// based on Brunel & Wang (2001) parameters.
type GABABParams struct {

	// overall strength multiplier of GABA-B current. The 0.015 default is a high value that works well in smaller networks -- larger networks may benefit from lower levels (e.g., 0.012).
	Gbar float32 `default:"0,0.012,0.015"`

	// rise time for bi-exponential time dynamics of GABA-B
	RiseTau float32 `default:"45"`

	// decay time for bi-exponential time dynamics of GABA-B
	DecayTau float32 `default:"50"`

	// baseline level of GABA-B channels open independent of inhibitory input (is added to spiking-produced conductance)
	Gbase float32 `default:"0.2"`

	// multiplier for converting Gi to equivalent GABA spikes
	GiSpike float32 `default:"10"`

	// time offset when peak conductance occurs, in msec, computed from RiseTau and DecayTau
	MaxTime float32 `edit:"-"`

	// time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))
	TauFact float32 `view:"-"`

	// 1/Tau
	RiseDt float32 `view:"-" edit:"-"`

	// 1/Tau
	DecayDt float32 `view:"-" edit:"-"`

	pad, pad1, pad2 float32
}

func (gp *GABABParams) Defaults() {
	gp.Gbar = 0.015
	gp.RiseTau = 45
	gp.DecayTau = 50
	gp.Gbase = 0.2
	gp.GiSpike = 10
	gp.Update()
}

func (gp *GABABParams) Update() {
	gp.TauFact = mat32.Pow(gp.DecayTau/gp.RiseTau, gp.RiseTau/(gp.DecayTau-gp.RiseTau))
	gp.MaxTime = ((gp.RiseTau * gp.DecayTau) / (gp.DecayTau - gp.RiseTau)) * mat32.Log(gp.DecayTau/gp.RiseTau)
	gp.RiseDt = 1.0 / gp.RiseTau
	gp.DecayDt = 1.0 / gp.DecayTau
}

func (gp *GABABParams) ShouldShow(field string) bool {
	switch field {
	case "Gbar":
		return true
	default:
		return gp.Gbar > 0
	}
}

// GFromV returns the GABA-B conductance as a function of normalized membrane potential
func (gp *GABABParams) GFromV(v float32) float32 {
	vbio := VToBio(v)
	if vbio < -90 {
		vbio = -90
	}
	return (vbio + 90.0) / (1.0 + mat32.FastExp(0.1*((vbio+90.0)+10.0)))
}

// GFromS returns the GABA-B conductance as a function of GABA spiking rate,
// based on normalized spiking factor (i.e., Gi from FFFB etc)
func (gp *GABABParams) GFromS(s float32) float32 {
	ss := s * gp.GiSpike
	if ss > 20 {
		return 1
	}
	return 1.0 / (1.0 + mat32.FastExp(-(ss-7.1)/1.4))
}

// BiExp computes bi-exponential update, returns dG and dX deltas to add to g and x
func (gp *GABABParams) BiExp(g, x float32, dG, dX *float32) {
	*dG = (gp.TauFact*x - g) * gp.RiseDt
	*dX = -x * gp.DecayDt
	return
}

// GABAB returns the updated GABA-B / GIRK activation and underlying x value
// based on current values and gi inhibitory conductance (proxy for GABA spikes)
func (gp *GABABParams) GABAB(gi float32, gabaB, gabaBx *float32) {
	var dG, dX float32
	gp.BiExp(*gabaB, *gabaBx, &dG, &dX)
	*gabaBx += gp.GFromS(gi) + dX // gets new input
	*gabaB += dG
	return
}

// GgabaB returns the overall net GABAB / GIRK conductance including
// Gbar, Gbase, and voltage-gating
func (gp *GABABParams) GgabaB(gabaB, vm float32) float32 {
	return gp.Gbar * gp.GFromV(vm) * (gabaB + gp.Gbase)
}

//gosl: end chans
