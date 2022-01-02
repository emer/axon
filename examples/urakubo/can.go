// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// CaNVars are intracellular Ca-driven signaling variables for the
// CaN and CaM binding, at different levels of Ca binding
// stores N values -- Co = Concentration computed by volume as needed
type CaNVars struct {
	CaN    float32 `desc:"Calcineurin"`
	CaNCaM float32 `desc:"CaN-CaM bound"`
}

func (cs *CaNVars) Init(vol float32) {
	cs.CaN = 0
	cs.CaNCaM = 0
}

// CaNCaMVars are intracellular Ca-driven signaling states
// for CaN-CaM binding
// stores N values -- Co = Concentration computed by volume as needed
type CaNCaMVars struct {
	Ca     [3]CaNVars `desc:"increasing levels of Ca binding, 0-2"`
	CaNact float32    `desc:"active CaN = Ca[2].CaNCaM"`
}

func (cs *CaNCaMVars) Init(vol float32) {
	for i := range cs.Ca {
		cs.Ca[i].Init(vol)
	}
	cs.Ca[0].CaN = CoToN(3, vol)
	cs.CaNact = 0
}

func (cs *CaNCaMVars) Log(dt *etable.Table, vol float32, row int, pre string) {
	dt.SetCellFloat(pre+"CaNact", row, CoFmN64(cs.CaNact, vol))
}

func (cs *CaNCaMVars) ConfigLog(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "CaNact", etensor.FLOAT64, nil, nil})
}

// CaNState is overall intracellular Ca-driven signaling states
// for CaN-CaM binding in Cyt and PSD
type CaNState struct {
	Cyt CaNCaMVars `desc:"in cytosol -- volume = 0.08 fl"`
	PSD CaNCaMVars `desc:"in PSD  -- volume = 0.02 fl"`
}

func (cs *CaNState) Init() {
	cs.Cyt.Init(CytVol)
	cs.PSD.Init(PSDVol)
}

func (cs *CaNState) Log(dt *etable.Table, row int) {
	cs.Cyt.Log(dt, CytVol, row, "Cyt_")
	cs.PSD.Log(dt, PSDVol, row, "PSD_")
}

func (cs *CaNState) ConfigLog(sch *etable.Schema) {
	cs.Cyt.ConfigLog(sch, "Cyt_")
	cs.PSD.ConfigLog(sch, "PSD_")
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
	cp.CaNCaM.SetSecVol(40, CytVol, 0.04) // 1: 40 μM-1 = 0.83333, PSD = 3.3333
	cp.CaCaN01.SetSecVol(20, CytVol, 1.0) // 2: 20 μM-1 = 0.41667, PSD = 1.6667
	cp.CaCaN12.SetSecVol(10, CytVol, 2.0) // 3: 10 μM-1 = 0.20833, PSD = 0.83333
}

// StepCaN does the bulk of Ca + CaN + CaM binding reactions, in a given region
// kf is an additional forward multiplier, which is 1 for Cyt and 4 for PSD
// cCaM, nCaM = current, next 3CaCaM from CaMKIIVars
// cCa, nCa = current next Ca
func (cp *CaNParams) StepCaN(vol float32, c, n *CaNCaMVars, cCa, cCaM float32, nCa, nCaM *float32) {
	k := CytVol / vol
	for i := 0; i < 3; i++ {
		cp.CaNCaM.StepK(k, c.Ca[i].CaN, cCaM, c.Ca[i].CaNCaM, &n.Ca[i].CaN, nCaM, &n.Ca[i].CaNCaM) // 1
	}
	cp.CaCaN01.StepK(k, c.Ca[0].CaN, cCa, c.Ca[1].CaN, &n.Ca[0].CaN, nCa, &n.Ca[1].CaN)             // 2
	cp.CaCaN01.StepK(k, c.Ca[0].CaNCaM, cCa, c.Ca[1].CaNCaM, &n.Ca[0].CaNCaM, nCa, &n.Ca[1].CaNCaM) // 2

	cp.CaCaN12.StepK(k, c.Ca[1].CaN, cCa, c.Ca[2].CaN, &n.Ca[1].CaN, nCa, &n.Ca[2].CaN)             // 3
	cp.CaCaN12.StepK(k, c.Ca[1].CaNCaM, cCa, c.Ca[2].CaNCaM, &n.Ca[1].CaNCaM, nCa, &n.Ca[2].CaNCaM) // 3

	n.CaNact = n.Ca[2].CaNCaM
}

// Step does full CaN updating, c=current, n=next
// Next has already been initialized to current
// cCa, nCa = current, next Ca
func (cp *CaNParams) Step(c, n *CaNState, cCaM, nCaM *CaMKIIState, cCa, nCa *CaState) {
	cp.StepCaN(CytVol, &c.Cyt, &n.Cyt, cCa.Cyt, cCaM.Cyt.Ca[3].CaM, &nCa.Cyt, &nCaM.Cyt.Ca[3].CaM)
	cp.StepCaN(PSDVol, &c.PSD, &n.PSD, cCa.PSD, cCaM.PSD.Ca[3].CaM, &nCa.PSD, &nCaM.PSD.Ca[3].CaM)
}
