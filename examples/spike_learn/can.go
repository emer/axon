// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// CaNVars are intracellular Ca-driven signaling variables for the
// CaN and CaM binding, at different levels of Ca binding
type CaNVars struct {
	CaN    float32 `desc:"Calcineurin"`
	CaNCaM float32 `desc:"CaN-CaM bound"`
}

func (cs *CaNVars) Init() {
	cs.CaN = 0
	cs.CaNCaM = 0
}

// CaNCaMVars are intracellular Ca-driven signaling states
// for CaN-CaM binding
type CaNCaMVars struct {
	Ca     [3]CaNVars `desc:"increasing levels of Ca binding, 0-2"`
	CaNact float32    `desc:"active CaN = Ca[2].CaNCaM"`
}

func (cs *CaNCaMVars) Init() {
	for i := range cs.Ca {
		cs.Ca[i].Init()
	}
	cs.Ca[0].CaN = 3
	cs.CaNact = 0
}

// CaNState is overall intracellular Ca-driven signaling states
// for CaN-CaM binding in Cyt and PSD
type CaNState struct {
	Cyt CaNCaMVars `desc:"in cytosol -- volume = 0.08 fl"`
	PSD CaNCaMVars `desc:"in PSD  -- volume = 0.02 fl"`
}

func (cs *CaNState) Init() {
	cs.Cyt.Init()
	cs.PSD.Init()
}

// CaNParams are the parameters governing the Ca+CaN-CaM binding
type CaNParams struct {
	CaNCaM  React `desc:"1: CaN+CaM -> CaN-CaM"`
	CaCaN01 React `desc:"2: Ca+CaM -> CaCaM"`
	CaCaN12 React `desc:"3: Ca+CaCaM -> 2CaCaM"`
}

func (cp *CaNParams) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// Cyt = 1/48 * values listed in Table SIh (0.02083333)
	cp.CaNCaM.SetSec(0.83333, 0.04) // 1: 40 μM-1, PSD = 3.3333
	cp.CaCaN01.SetSec(0.41667, 1.0) // 2: 20 μM-1, PSD = 1.6667
	cp.CaCaN12.SetSec(0.20833, 2.0) // 3: 10 μM-1, PSD = 0.83333
}

// StepCaN does the bulk of Ca + CaN + CaM binding reactions, in a given region
// kf is an additional forward multiplier, which is 1 for Cyt and 4 for PSD
// cCaM, nCaM = current, next 3CaCaM from CaMKIIVars
// cCa, nCa = current next Ca
func (cp *CaNParams) StepCaN(kf float32, c, n *CaNCaMVars, cCa, cCaM float32, nCa, nCaM *float32) {
	for i := 0; i < 3; i++ {
		cp.CaNCaM.StepKf(kf, c.Ca[i].CaN, cCaM, c.Ca[i].CaNCaM, &n.Ca[i].CaN, nCaM, &n.Ca[i].CaNCaM) // 1
	}
	cp.CaCaN01.StepKf(kf, c.Ca[0].CaN, cCa, c.Ca[1].CaN, &n.Ca[0].CaN, nCa, &n.Ca[1].CaN)             // 2
	cp.CaCaN01.StepKf(kf, c.Ca[0].CaNCaM, cCa, c.Ca[1].CaNCaM, &n.Ca[0].CaNCaM, nCa, &n.Ca[1].CaNCaM) // 2

	cp.CaCaN12.StepKf(kf, c.Ca[1].CaN, cCa, c.Ca[2].CaN, &n.Ca[1].CaN, nCa, &n.Ca[2].CaN)             // 3
	cp.CaCaN12.StepKf(kf, c.Ca[1].CaNCaM, cCa, c.Ca[2].CaNCaM, &n.Ca[1].CaNCaM, nCa, &n.Ca[2].CaNCaM) // 3

	n.CaNact = n.Ca[2].CaNCaM
}

// Step does full CaN updating, c=current, n=next
// Next has already been initialized to current
// cCa, nCa = current, next Ca
func (cp *CaNParams) Step(c, n *CaNState, cCaM, nCaM *CaMKIIState, cCa float32, nCa *float32) {
	cp.StepCaN(1, &c.Cyt, &n.Cyt, cCa, cCaM.Cyt.Ca[3].CaM, nCa, &nCaM.Cyt.Ca[3].CaM)
	cp.StepCaN(4, &c.PSD, &n.PSD, cCa, cCaM.PSD.Ca[3].CaM, nCa, &nCaM.PSD.Ca[3].CaM)
}
