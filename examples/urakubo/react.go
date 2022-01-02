// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

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

// SetSec sets reaction forward / backward time constants in seconds
// (converts to milliseconds)
func (rt *React) SetSec(f, b float32) {
	rt.Kf = f / 1000
	rt.Kb = b / 1000
}

// Step computes new A, B, AB values based on current A, B, and AB values
// na, nb, nab can be nil to skip updating
func (rt *React) Step(ca, cb, cab float32, na, nb, nab *float32) {
	rt.StepKf(1, ca, cb, cab, na, nb, nab)
}

// StepKf computes new A, B, AB values based on current A, B, and AB values
// na, nb, nab can be nil to skip updating
// Kf version has special rate multiplier for Kf
func (rt *React) StepKf(kf, ca, cb, cab float32, na, nb, nab *float32) {
	df := kf*rt.Kf*ca*cb - rt.Kb*cab
	if df > 0 && na != nil && *na < df {
		df = *na
	}
	if df > 0 && nb != nil && *nb < df {
		df = *nb
	}
	if nab != nil {
		if df < 0 && *nab < -df {
			df = -*nab
		}
		*nab += df
	}
	if na != nil {
		*na -= df
	}
	if nb != nil {
		*nb -= df
	}
}
