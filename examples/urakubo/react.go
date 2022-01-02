// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// CoToN returns N based on concentration, for given volume:
// = co * vol
func CoToN(co, vol float32) float32 {
	return co * vol
}

// CoFmN returns concentration from N, for given volume:
// = co / vol
func CoFmN(n, vol float32) float32 {
	return n / vol
}

// CoFmN64 returns concentration from N, for given volume:
// = co / vol -- as float64
func CoFmN64(n, vol float32) float64 {
	return float64(n / vol)
}

const (
	CytVol = 48 // volume of cytosol, in ? units
	PSDVol = 12 // volume of PSD, in ? units
)

// React models a basic chemical reaction:
//       Kf
// A + B --> AB
//      <-- Kb
// where Kf is the forward and Kb is the backward time constant.
// Time step of integration is msec, so constants are in those units.
// Use SetSec to set in terms of seconds.
type React struct {
	Kf float32 `desc:"forward rate constant, in μM-1 msec-1"`
	Kb float32 `desc:"backward rate constant, in μM-1 msec-1"`
}

// SetSecVol sets reaction forward / backward time constants in seconds
// (converts to milliseconds), divides forward Kf by volume to compensate
// for 2 volume-based factors occurring in forward component, vs 1 in back
func (rt *React) SetSecVol(f, vol, b float32) {
	rt.Kf = CoFmN(f, vol) / 1000
	rt.Kb = b / 1000
}

// SetSec sets reaction forward / backward time constants in seconds
// (converts to milliseconds),
func (rt *React) SetSec(f, b float32) {
	rt.Kf = f / 1000
	rt.Kb = b / 1000
}

// Step computes new A, B, AB values based on current A, B, and AB values
// na, nb, nab can be nil to skip updating
func (rt *React) Step(ca, cb, cab float32, na, nb, nab *float32) {
	rt.StepK(1, ca, cb, cab, na, nb, nab)
}

// StepK computes new A, B, AB values based on current A, B, and AB values
// na, nb, nab can be nil to skip updating
// K version has special rate multiplier for K's
func (rt *React) StepK(k, ca, cb, cab float32, na, nb, nab *float32) {
	df := k*rt.Kf*ca*cb - rt.Kb*cab
	if nab != nil {
		*nab += df
		if *nab < 0 {
			*nab = 0
		}
	}
	if na != nil {
		*na -= df
		if *na < 0 {
			*na = 0
		}
	}
	if nb != nil {
		*nb -= df
		if *nb < 0 {
			*nb = 0
		}
	}
}
