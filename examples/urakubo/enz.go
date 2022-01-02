// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Enz models an enzyme-catalyzed reaction based on the Michaelis-Menten kinetics
// that transforms S = substrate into P product
//       K1     K3
// S + E --> SE ---> P + E
//      <-- K2
// S = substrate, E = enzyme, P = product
// Time step of integration is msec, so constants are in those units.
// Use SetSec to set in terms of seconds.
type Enz struct {
	K1 float32 `desc:"S+E forward rate constant, in μM-1 msec-1"`
	K2 float32 `desc:"SE backward rate constant, in μM-1 msec-1"`
	K3 float32 `desc:"SE -> P + E catalyzed rate constant, in μM-1 msec-1"`
	Km float32 `inactive:"+" desc:"Michaelis constant = (K2 + K3) / K1 -- goes into the rate"`
}

func (rt *Enz) Update() {
	rt.Km = (rt.K2 + rt.K3) / rt.K1
}

// SetSec sets reaction forward / backward time constants in seconds
// (converts to milliseconds)
func (rt *Enz) SetSec(k1, k2, k3 float32) {
	rt.K1 = k1 / 1000
	rt.K2 = k2 / 1000
	rt.K3 = k3 / 1000
	rt.Update()
}

// Step computes new S, P values based on current S, E, and P values
// na, nb, nab can be nil to skip updating
func (rt *Enz) Step(cs, ce, cp float32, ns, np *float32) {
	rt.StepKf(1, cs, ce, cp, ns, np)
}

// StepKf computes new S, P values based on current S, E, and P values
// na, nb, nab can be nil to skip updating
// Kf version has special rate multiplier for Kf
func (rt *Enz) StepKf(kf, cs, ce, cp float32, ns, np *float32) {
	rate := cs * rt.K3 / (cs + (rt.Km / kf))
	if rate < 0 && np != nil && *np < -rate {
		rate = *np
	}
	if ns != nil {
		if rate > 0 && *ns < rate {
			rate = *ns
		}
		*ns -= rate
	}
	if np != nil {
		*np += rate
	}
}
