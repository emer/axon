// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// CaMVars are intracellular Ca-driven signaling variables for the
// CaMKII+CaM binding -- each can have different numbers of Ca bound
// Dupont = DupontHouartDekonnick03, has W* terms used in Genesis code
type CaMVars struct {
	CaM         float32 `desc:"CaM = Ca calmodulin, [0-3]Ca bound but unbound to CaMKII"`
	CaM_CaMKII  float32 `desc:"CaMKII-CaM bound together = WBn in Dupont"`
	CaM_CaMKIIP float32 `desc:"CaMKIIP-CaM bound together, P = phosphorylated at Thr286 = WTn in Dupont"`
}

func (cs *CaMVars) Init() {
	cs.CaM = 0
	cs.CaM_CaMKII = 0
	cs.CaM_CaMKIIP = 0
}

// CaMKIIVars are intracellular Ca-driven signaling states
// for CaMKII binding and phosphorylation with CaM + Ca
// Dupont = DupontHouartDekonnick03, has W* terms used in Genesis code
type CaMKIIVars struct {
	Ca        [4]CaMVars `desc:"increasing levels of Ca binding, 0-3"`
	CaMKII    float32    `desc:"unbound CaMKII = CaM kinase II -- WI in Dupont"`
	CaMKIIP   float32    `desc:"unbound CaMKII P = phosphorylated at Thr286 -- shown with * in Figure S13 = WA in Dupont"`
	CaMKIIact float32    `desc:"total active CaMKII: .75 * Ca[1..3]CaM_CaMKII + 1 * Ca[3]CaM_CaMKIIP + .8 * Ca[1..2]CaM_CaMKIIP"`
	CaMKIItot float32    `desc:"total CaMKII across all states"`
	ActPct    float32    `desc:"proportion of active: CaMKIIact / Total CaMKII (constant?)"`
}

func (cs *CaMKIIVars) Init() {
	for i := range cs.Ca {
		cs.Ca[i].Init()
	}
	cs.Ca[0].CaM = 80
	cs.CaMKII = 20
	cs.CaMKIIP = 0          // WA
	cs.Ca[0].CaM_CaMKII = 3 // calling this WB
	cs.Active()
}

// Active updates active total and pct
func (cs *CaMKIIVars) Active() {
	var act float32

	tot := cs.CaMKII + cs.CaMKIIP
	for i := 0; i < 4; i++ {
		tot += cs.Ca[i].CaM_CaMKII + cs.Ca[i].CaM_CaMKIIP
		if i >= 1 && i < 3 {
			act += 0.75*cs.Ca[i].CaM_CaMKII + 0.8*cs.Ca[i].CaM_CaMKII
		} else if i == 3 {
			act += 0.75*cs.Ca[i].CaM_CaMKII + cs.Ca[i].CaM_CaMKII
		}
	}
	cs.CaMKIIact = act
	cs.CaMKIItot = tot
	if tot > 0 {
		cs.ActPct = act / tot
	} else {
		cs.ActPct = 0
	}
}

func (cs *CaMKIIVars) Log(dt *etable.Table, row int, pre string) {
	dt.SetCellFloat(pre+"CaMKIIact", row, float64(cs.CaMKIIact))
}

func (cs *CaMKIIVars) ConfigLog(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "CaMKIIact", etensor.FLOAT64, nil, nil})
}

// CaMKIIState is overall intracellular Ca-driven signaling states
// for CaMKII in Cyt and PSD
type CaMKIIState struct {
	Cyt CaMKIIVars `desc:"in cytosol -- volume = 0.08 fl"`
	PSD CaMKIIVars `desc:"in PSD -- volume = 0.02 fl"`
}

func (cs *CaMKIIState) Init() {
	cs.Cyt.Init() // confirmed cyt and psd seem to start with same conc
	cs.PSD.Init()
}

func (cs *CaMKIIState) Log(dt *etable.Table, row int) {
	cs.Cyt.Log(dt, row, "Cyt_")
	cs.PSD.Log(dt, row, "PSD_")
}

func (cs *CaMKIIState) ConfigLog(sch *etable.Schema) {
	cs.Cyt.ConfigLog(sch, "Cyt_")
	cs.PSD.ConfigLog(sch, "PSD_")
}

// CaMKIIParams are the parameters governing the Ca+CaM binding
type CaMKIIParams struct {
	CaCaM01        React `desc:"1: Ca+CaM -> 1CaCaM = CaM-bind-Ca"`
	CaCaM12        React `desc:"2: Ca+1CaM -> 2CaCaM = CaMCa-bind-Ca"`
	CaCaM23        React `desc:"3: Ca+2CaM -> 3CaCaM = CaMCa2-bind-Ca"`
	CaMCaMKII      React `desc:"4: CaM+CaMKII -> CaM-CaMKII [0-2] -- kIB_kBI_[0-2] -- WI = plain CaMKII, WBn = CaM bound"`
	CaMCaMKII3     React `desc:"5: 3CaCaM+CaMKII -> 3CaCaM-CaMKII = kIB_kBI_3"`
	CaCaM23_CaMKII React `desc:"6: Ca+2CaCaM-CaMKII -> 3CaCaM-CaMKII = CaMCa2-bind-Ca"`
	CaCaM_CaMKIIP  React `desc:"8: Ca+nCaCaM-CaMKIIP -> n+1CaCaM-CaMKIIP = kTP_PT_*"`
	CaMCaMKIIP     React `desc:"9: CaM+CaMKIIP -> CaM-CaMKIIP = kAT_kTA"` // note: typo in SI3 for top PP1, PP2A
	PP1Thr286      Enz   `desc:"10: PP1 dephosphorylating CaMKIIP"`
	PP2AThr286     Enz   `desc:"11: PP2A dephosphorylating CaMKIIP"`
}

func (cp *CaMKIIParams) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// Cyt = 1/48 * values listed in Table SId (0.02083333)
	cp.CaCaM01.SetSec(1.0667, 200)     // 1: 51.202 μM-1, PSD 4.2667 = CaM-bind-Ca
	cp.CaCaM12.SetSec(2.7771, 1000)    // 2: 133.3 μM-1, PSD 11.108 = CaMCa-bind-Ca
	cp.CaCaM23.SetSec(0.53333, 400)    // 3: 25.6 μM-1, PSD 2.1333 = CaMCa2-bind-Ca
	cp.CaMCaMKII.SetSec(8.3333e-06, 1) // 4: 0.0004 μM-1, PSD 3.3333e-5 =  = kIB_kBI_[0-2]
	cp.CaMCaMKII3.SetSec(0.16667, 1)   // 5: 8 μM-1, PSD 3.3333e-5 = kIB_kBI_3

	cp.CaCaM23_CaMKII.SetSec(0.53333, 0.02) // 6: 25.6 μM-1, PSD 2.1333 = CaMCa2-bind-Ca
	cp.CaCaM_CaMKIIP.SetSec(0.020834, 1)    // 8: 1 μM-1, PSD 0.0833335 = kTP_PT_*
	cp.CaMCaMKIIP.SetSec(0.16667, 0.001)    // 9: 8 μM-1, PSD 0.66667 = kAT_kTA

	cp.PP1Thr286.SetSec(0.0031724, 1.34, 0.335)  // 10: 11 μM Km
	cp.PP2AThr286.SetSec(0.0031724, 1.34, 0.335) // 11: 11 μM Km
}

// StepCaMKIIP is the special CaMKII phosphorylation function from Dupont et al, 2003
func (cp *CaMKIIParams) StepCaMKIIP(c, t float32, n *float32) {
	fact := t * (-0.22 + 1.826*t + 0.8*t*t)
	if fact < 0 {
		return
	}
	*n += 0.00029 * fact * c
}

// StepCaMKII does the bulk of Ca + CaM + CaMKII binding reactions, in a given region
// kf is an additional forward multiplier, which is 1 for Cyt and 4 for PSD
// cCa, nCa = current next Ca
func (cp *CaMKIIParams) StepCaMKII(kf float32, c, n *CaMKIIVars, cCa, pp1, pp2a float32, nCa *float32) {
	cp.CaCaM01.StepKf(kf, c.Ca[0].CaM, cCa, c.Ca[1].CaM, &n.Ca[0].CaM, nCa, &n.Ca[1].CaM) // 1
	cp.CaCaM12.StepKf(kf, c.Ca[1].CaM, cCa, c.Ca[2].CaM, &n.Ca[1].CaM, nCa, &n.Ca[2].CaM) // 2
	cp.CaCaM23.StepKf(kf, c.Ca[2].CaM, cCa, c.Ca[3].CaM, &n.Ca[2].CaM, nCa, &n.Ca[3].CaM) // 3

	cp.CaCaM01.StepKf(kf, c.Ca[0].CaM_CaMKII, cCa, c.Ca[1].CaM_CaMKII, &n.Ca[0].CaM_CaMKII, nCa, &n.Ca[1].CaM_CaMKII)        // 1
	cp.CaCaM12.StepKf(kf, c.Ca[1].CaM_CaMKII, cCa, c.Ca[2].CaM_CaMKII, &n.Ca[1].CaM_CaMKII, nCa, &n.Ca[2].CaM_CaMKII)        // 2
	cp.CaCaM23_CaMKII.StepKf(kf, c.Ca[2].CaM_CaMKII, cCa, c.Ca[3].CaM_CaMKII, &n.Ca[2].CaM_CaMKII, nCa, &n.Ca[3].CaM_CaMKII) // 6

	for i := 0; i < 3; i++ {
		cp.CaMCaMKII.StepKf(kf, c.Ca[i].CaM, c.CaMKII, c.Ca[i].CaM_CaMKII, &n.Ca[i].CaM, &n.CaMKII, &n.Ca[i].CaM_CaMKII) // 4
	}
	cp.CaMCaMKII3.StepKf(kf, c.Ca[3].CaM, c.CaMKII, c.Ca[3].CaM_CaMKII, &n.Ca[3].CaM, &n.CaMKII, &n.Ca[3].CaM_CaMKII) // 5

	cp.CaMCaMKIIP.StepKf(kf, c.Ca[0].CaM, c.CaMKIIP, c.Ca[0].CaM_CaMKIIP, &n.Ca[0].CaM, &n.CaMKIIP, &n.Ca[0].CaM_CaMKIIP) // 9
	for i := 0; i < 3; i++ {
		cp.CaCaM_CaMKIIP.StepKf(kf, c.Ca[i].CaM_CaMKIIP, cCa, c.Ca[i+1].CaM_CaMKIIP, &n.Ca[i].CaM_CaMKIIP, nCa, &n.Ca[i+1].CaM_CaMKIIP) // 8
	}

	for i := 0; i < 4; i++ {
		cp.StepCaMKIIP(c.Ca[i].CaM_CaMKII, c.ActPct, &n.Ca[i].CaM_CaMKIIP) // 7
		cp.PP1Thr286.StepKf(kf, c.Ca[i].CaM_CaMKIIP, pp1, n.Ca[i].CaM_CaMKIIP, &n.Ca[i].CaM_CaMKII, &n.Ca[i].CaM_CaMKIIP)
		cp.PP2AThr286.StepKf(kf, c.Ca[i].CaM_CaMKIIP, pp1, n.Ca[i].CaM_CaMKIIP, &n.Ca[i].CaM_CaMKII, &n.Ca[i].CaM_CaMKIIP)
	}

	// when all done, update act
	n.Active()
}

// Step does one step of CaMKII updating, c=current, n=next
// Next has already been initialized to current
// cCa, nCa = current, next Ca
// pp2a = current cyt pp2a
func (cp *CaMKIIParams) Step(c, n *CaMKIIState, pp1 *PP1State, cCa, pp2a float32, nCa *float32) {
	cp.StepCaMKII(1, &c.Cyt, &n.Cyt, cCa, pp1.Cyt.PP1act, pp2a, nCa)
	cp.StepCaMKII(4, &c.PSD, &n.PSD, cCa, pp1.PSD.PP1act, 0, nCa)
}
