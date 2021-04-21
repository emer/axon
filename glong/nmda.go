// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import "github.com/goki/mat32"

// NMDAParams control the NMDA dynamics, based on Brunel & Wang (2001)
// parameters.
type NMDAParams struct {
	GeTot float32 `desc:"how much of the NMDA is driven by total Ge synaptic input, as opposed to from projections specifically marked as NMDA-communicating type, e.g., for active maintenance, in NMDASyn"`
	Tau   float32 `def:"100" desc:"decay time constant for NMDA current -- rise time is 2 msec and not worth extra effort for biexponential"`
	Gbar  float32 `desc:"strength of NMDA current -- 0.02 is just over level sufficient to maintain in face of completely blank input"`
}

func (np *NMDAParams) Defaults() {
	np.GeTot = 1
	np.Tau = 100
	np.Gbar = 0.03
}

// GFmV returns the NMDA conductance as a function of normalized membrane potential
func (np *NMDAParams) GFmV(v float32) float32 {
	vbio := v*100 - 100
	return 1 / (1 + 0.28*mat32.FastExp(-0.062*vbio))
}

// NMDA returns the updated NMDA activation from current NMDA, GeRaw, and NMDASyn input
func (np *NMDAParams) NMDA(nmda, geraw, nmdaSyn float32) float32 {
	return nmda + np.GeTot*geraw + nmdaSyn - (nmda / np.Tau)
}

// Gnmda returns the NMDA net conductance from nmda activation and vm
func (np *NMDAParams) Gnmda(nmda, vm float32) float32 {
	return np.Gbar * np.GFmV(vm) * nmda
}
