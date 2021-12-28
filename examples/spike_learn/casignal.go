// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// CaSigVars are intracellular Ca-driven signaling variables
// located in Cytosol or PSD
type CaSigVars struct {
	PKA    float32 `desc:"PKA = protein kinase A"`
	CaMKII float32 `desc:"CaMKII = CaM kinase II"`
	PP1    float32 `desc:"PP1 = protein phosphatase 1"`
	CaN    float32 `desc:"CaN = calcineurin"`
	PP2A   float32 `desc:"PP2A = protein phosphatase 2A"`
}

// CaSigState are intracellular Ca-driven signaling states
// In two different locations and two states (inactive, active):
// Cyt = Cytosol (internal on reserve in the spine)
// PSD = Postsynaptic density (active in the synapse)
type CaSigState struct {
	CytInact CaSigVars `desc:"inactive in cytosol"`
	CytAct   CaSigVars `desc:"active in cytosol"`
	PSDInact CaSigVars `desc:"inactive in PSD"`
	PSDAct   CaSigVars `desc:"active in PSD"`
}

// React models a basic chemical reaction going from A + B <-> AB
// where Kf is the forward and Kb is the backward time constant.
type React struct {
	Kf float32
	Kb float32
}

// SetSec sets reaction forward / backward time constants in seconds
// (converts to milliseconds)
func (rt *React) SetSec(f, b float32) {
	rt.Kf = f / 1000
	rt.Kb = b / 1000
}

// Forward computes a new AB value based on current A, B, and AB values
func (rt *React) Forward(ca, cb, cab float32, nab *float32) {
	*nab += rt.Kf*a*b - rt.Kb*cab
}

// Backward computes new A and B values based on current A, B and AB values
func (rt *React) Backward(ca, cb, cab float32, na, nb *float32) {
	d := rt.Kb*cab - rt.Kf*(ca+cb)
	*na += d
	*nb += d
}

type CaSigParams struct {
	CaCaM React `desc:"Ca+CaM -> CaCaM"`
}

func (cp *CaSigParams) Defaults() {
	CaCaM.SetSec(1.0667, 200)
}
