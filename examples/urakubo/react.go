// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// CoToN returns N based on concentration, for given volume: co * vol
func CoToN(co, vol float64) float64 {
	return co * vol
}

// CoFmN returns concentration from N, for given volume: co / vol
func CoFmN(n, vol float64) float64 {
	return n / vol
}

// React models a basic chemical reaction:
//       Kf
// A + B --> AB
//      <-- Kb
// where Kf is the forward and Kb is the backward time constant.
// The source Kf and Kb constants are in terms of concentrations μM-1 and sec-1
// but calculations take place using N's, and the forward direction has
// two factors while reverse only has one, so a corrective volume factor needs
// to be divided out to set the actual forward factor.
// This is different for the PSD (smaller) vs. Cyt, so PSD constants are higher
// by a factor of 4 = CytVol / PSDVol
type React struct {
	Kf float64 `desc:"forward rate constant for N / sec assuming 2 forward factors"`
	Kb float64 `desc:"backward rate constant for N / sec assuming 1 backward factor"`
}

// SetVol sets reaction forward / backward time constants in seconds,
// dividing forward Kf by volume to compensate for 2 volume-based concentrations
// occurring in forward component, vs just 1 in back
func (rt *React) SetVol(f, vol, b float64) {
	rt.Kf = CoFmN(f, vol)
	rt.Kb = b
}

// Set sets reaction forward / backward time constants in seconds
func (rt *React) Set(f, b float64) {
	rt.Kf = f
	rt.Kb = b
}

// Step computes delta A, B, AB values based on current A, B, and AB values
func (rt *React) Step(ca, cb, cab float64, da, db, dab *float64) {
	df := rt.Kf*ca*cb - rt.Kb*cab
	*dab += df
	*da -= df
	*db -= df
}

// StepK computes delta A, B, AB values based on current A, B, and AB values
// K version has additional rate multiplier for Kf
func (rt *React) StepK(kf, ca, cb, cab float64, da, db, dab *float64) {
	df := kf*rt.Kf*ca*cb - rt.Kb*cab
	*dab += df
	*da -= df
	*db -= df
}

/////////////////////////////////////////////////////////////
// Buffer

// Buffer provides a soft buffering driving deltas relative to a target N
// which can be set by concentration and volume.
type Buffer struct {
	K    float64 `desc:"rate of buffering (akin to permeability / conductance of a channel)"`
	Targ float64 `desc:"buffer target concentration -- drives delta relative to this"`
}

func (bf *Buffer) SetTargVol(targ, vol float64) {
	bf.Targ = CoToN(targ, vol)
}

// Step computes da delta for current value ca relative to target value Targ
func (bf *Buffer) Step(ca float64, da *float64) {
	*da += bf.K * (bf.Targ - ca)
}

/////////////////////////////////////////////////////////////
// Integration

// IntegrationDt is the time step of integration
// orig uses 5e-5, 2e-4 is barely stable, 5e-4 is not
// The AC1act dynamics in particular are not stable due to large ATP, AMP numbers
// todo: experiment with those
const IntegrationDt = 5e-5

// Integrate adds delta to current value with integration rate constant IntegrationDt
// new value cannot go below 0
func Integrate(c *float64, d float64) {
	// if *c > 1e-10 && d > 1e-10 { // note: exponential Euler requires separate A - B deltas
	// 	dd := math.Exp()
	// } else {
	*c += IntegrationDt * d
	if *c < 0 {
		*c = 0
	}
}
