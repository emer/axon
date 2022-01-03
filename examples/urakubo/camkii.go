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
// stores N values -- Co = Concentration computed by volume as needed
type CaMVars struct {
	CaM         float64 `desc:"CaM = Ca calmodulin, [0-3]Ca bound but unbound to CaMKII"`
	CaM_CaMKII  float64 `desc:"CaMKII-CaM bound together = WBn in Dupont"`
	CaM_CaMKIIP float64 `desc:"CaMKIIP-CaM bound together, P = phosphorylated at Thr286 = WTn in Dupont"`
}

func (cs *CaMVars) Init(vol float64) {
	cs.CaM = 0
	cs.CaM_CaMKII = 0
	cs.CaM_CaMKIIP = 0
}

func (cs *CaMVars) Zero() {
	cs.CaM = 0
	cs.CaM_CaMKII = 0
	cs.CaM_CaMKIIP = 0
}

func (cs *CaMVars) Integrate(d *CaMVars) {
	Integrate(&cs.CaM, d.CaM)
	Integrate(&cs.CaM_CaMKII, d.CaM_CaMKII)
	Integrate(&cs.CaM_CaMKIIP, d.CaM_CaMKIIP)
}

// CaMKIIVars are intracellular Ca-driven signaling states
// for CaMKII binding and phosphorylation with CaM + Ca
// Dupont = DupontHouartDekonnick03, has W* terms used in Genesis code
// stores N values -- Co = Concentration computed by volume as needed
type CaMKIIVars struct {
	Ca          [4]CaMVars `desc:"increasing levels of Ca binding, 0-3"`
	CaMKII      float64    `desc:"unbound CaMKII = CaM kinase II -- WI in Dupont"`
	CaMKIIP     float64    `desc:"unbound CaMKII P = phosphorylated at Thr286 -- shown with * in Figure S13 = WA in Dupont"`
	PP1Thr286C  float64    `desc:"PP1+CaMKIIP complex for PP1Thr286 enzyme reaction"`
	PP2AThr286C float64    `desc:"PP2A+CaMKIIP complex for PP2AThr286 enzyme reaction"`

	Active float64 `inactive:"+" desc:"computed total active CaMKII"`
	Total  float64 `inactive:"+" desc:"computed total CaMKII across all states"`
	Kauto  float64 `inactive:"+" desc:"rate constant for auto-phosphorylation"`
}

func (cs *CaMKIIVars) Init(vol float64) {
	for i := range cs.Ca {
		cs.Ca[i].Init(vol)
	}
	cs.Ca[0].CaM = CoToN(80, vol)
	cs.CaMKII = CoToN(20, vol)
	cs.CaMKIIP = 0 // WA
	cs.PP1Thr286C = 0
	cs.PP2AThr286C = 0
	cs.UpdtActive()
}

func (cs *CaMKIIVars) Zero() {
	for i := range cs.Ca {
		cs.Ca[i].Zero()
	}
	cs.CaMKII = 0
	cs.CaMKIIP = 0
	cs.PP1Thr286C = 0
	cs.PP2AThr286C = 0
	cs.Active = 0
	cs.Total = 0
	cs.Kauto = 0
}

func (cs *CaMKIIVars) Integrate(d *CaMKIIVars) {
	for i := range cs.Ca {
		cs.Ca[i].Integrate(&d.Ca[i])
	}
	Integrate(&cs.CaMKII, d.CaMKII)
	Integrate(&cs.CaMKIIP, d.CaMKIIP)
	Integrate(&cs.PP1Thr286C, d.PP1Thr286C)
	Integrate(&cs.PP2AThr286C, d.PP2AThr286C)
	cs.UpdtActive()
}

// UpdtActive updates active, total, and the Kauto auto-phosphorylation rate constant
// Code is from genesis_customizing/T286Phos/T286Phos.c and would be impossible to
// reconstruct without that source (my first guess was wildy off, based only on
// the supplement)
func (cs *CaMKIIVars) UpdtActive() {
	WI := cs.CaMKII
	WA := cs.CaMKIIP

	var WB, WT float64

	for i := 0; i < 3; i++ {
		WB += cs.Ca[i].CaM_CaMKII
		WT += cs.Ca[i].CaM_CaMKIIP
	}
	WB += cs.Ca[3].CaM_CaMKII
	WP := cs.Ca[3].CaM_CaMKIIP

	TotalW := WI + WB + WP + WT + WA
	Wb := WB / TotalW
	Wp := WP / TotalW
	Wt := WT / TotalW
	Wa := WA / TotalW
	cb := 0.75
	ct := 0.8
	ca := 0.8

	T := Wb + Wp + Wt + Wa
	tmp := T * (-0.22 + 1.826*T + -0.8*T*T)
	tmp *= 0.75 * (cb*Wb + Wp + ct*Wt + ca*Wa)
	if tmp < 0 {
		tmp = 0
	}
	cs.Kauto = 0.29 * tmp
	cs.Active = cb*WB + WP + ct*WT + ca*WA
	cs.Total = T
}

func (cs *CaMKIIVars) Log(dt *etable.Table, vol float64, row int, pre string) {
	// dt.SetCellFloat(pre+"CaM", row, CoFmN(cs.Ca[0].CaM, vol))
	// dt.SetCellFloat(pre+"CaCaM", row, CoFmN(cs.Ca[1].CaM, vol))
	dt.SetCellFloat(pre+"Ca3CaM", row, CoFmN(cs.Ca[3].CaM, vol))
	// dt.SetCellFloat(pre+"Ca0CaM_CaMKII", row, CoFmN(cs.Ca[0].CaM_CaMKII, vol))
	// dt.SetCellFloat(pre+"Ca1CaM_CaMKII", row, CoFmN(cs.Ca[1].CaM_CaMKII, vol))
	// dt.SetCellFloat(pre+"Ca0CaM_CaMKIIP", row, CoFmN(cs.Ca[0].CaM_CaMKIIP, vol))
	// dt.SetCellFloat(pre+"Ca1CaM_CaMKIIP", row, CoFmN(cs.Ca[1].CaM_CaMKIIP, vol))
	// dt.SetCellFloat(pre+"CaMKII", row, CoFmN(cs.CaMKII, vol))
	// dt.SetCellFloat(pre+"CaMKIIP", row, CoFmN(cs.CaMKIIP, vol))
	dt.SetCellFloat(pre+"CaMKIIact", row, CoFmN(cs.Active, vol))
}

func (cs *CaMKIIVars) ConfigLog(sch *etable.Schema, pre string) {
	// *sch = append(*sch, etable.Column{pre + "CaM", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "CaCaM", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Ca3CaM", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca0CaM_CaMKII", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca1CaM_CaMKII", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca0CaM_CaMKIIP", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca1CaM_CaMKIIP", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "CaMKII", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "CaMKIIP", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "CaMKIIact", etensor.FLOAT64, nil, nil})
}

// CaMKIIState is overall intracellular Ca-driven signaling states
// for CaMKII in Cyt and PSD
type CaMKIIState struct {
	Cyt CaMKIIVars `desc:"in cytosol -- volume = 0.08 fl = 48"`
	PSD CaMKIIVars `desc:"in PSD -- volume = 0.02 fl = 12"`
}

func (cs *CaMKIIState) Init() {
	cs.Cyt.Init(CytVol) // confirmed cyt and psd seem to start with same conc
	cs.PSD.Init(PSDVol)
}

func (cs *CaMKIIState) Zero() {
	cs.Cyt.Zero()
	cs.PSD.Zero()
}

func (cs *CaMKIIState) Integrate(d *CaMKIIState) {
	cs.Cyt.Integrate(&d.Cyt)
	cs.PSD.Integrate(&d.PSD)
}

func (cs *CaMKIIState) Log(dt *etable.Table, row int) {
	cs.Cyt.Log(dt, CytVol, row, "Cyt_")
	cs.PSD.Log(dt, PSDVol, row, "PSD_")
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
	// See React docs for more info
	cp.CaCaM01.SetVol(51.202, CytVol, 200) // 1: 51.202 μM-1 = 1.0667, PSD 4.2667 = CaM-bind-Ca
	cp.CaCaM12.SetVol(133.3, CytVol, 1000) // 2: 133.3 μM-1 = 2.7771, PSD 11.108 = CaMCa-bind-Ca
	cp.CaCaM23.SetVol(25.6, CytVol, 400)   // 3: 25.6 μM-1 = 0.53333, PSD 2.1333 = CaMCa2-bind-Ca
	cp.CaMCaMKII.SetVol(0.0004, CytVol, 1) // 4: 0.0004 μM-1 = 8.3333e-6, PSD 3.3333e-5 = kIB_kBI_[0-2]
	cp.CaMCaMKII3.SetVol(8, CytVol, 1)     // 5: 8 μM-1 = 0.16667, PSD 3.3333e-5 = kIB_kBI_3

	cp.CaCaM23_CaMKII.SetVol(25.6, CytVol, 0.02) // 6: 25.6 μM-1 = 0.53333, PSD 2.1333 = CaMCa2-bind-Ca
	cp.CaCaM_CaMKIIP.SetVol(1, CytVol, 1)        // 8: 1 μM-1 = 0.020834, PSD 0.0833335 = kTP_PT_*
	cp.CaMCaMKIIP.SetVol(8, CytVol, 0.001)       // 9: 8 μM-1 = 0.16667, PSD 0.66667 = kAT_kTA

	cp.PP1Thr286.SetKmVol(11, CytVol, 1.34, 0.335)  // 10: 11 μM Km = 0.0031724
	cp.PP2AThr286.SetKmVol(11, CytVol, 1.34, 0.335) // 11: 11 μM Km = 0.0031724
}

// StepCaMKII does the bulk of Ca + CaM + CaMKII binding reactions, in a given region
// cCa, nCa = current next Ca
func (cp *CaMKIIParams) StepCaMKII(vol float64, c, d *CaMKIIVars, cCa, pp1, pp2a float64, dCa, dpp1, dpp2a *float64) {
	kf := CytVol / vol
	cp.CaCaM01.StepK(kf, c.Ca[0].CaM, cCa, c.Ca[1].CaM, &d.Ca[0].CaM, dCa, &d.Ca[1].CaM) // 1
	cp.CaCaM12.StepK(kf, c.Ca[1].CaM, cCa, c.Ca[2].CaM, &d.Ca[1].CaM, dCa, &d.Ca[2].CaM) // 2
	cp.CaCaM23.StepK(kf, c.Ca[2].CaM, cCa, c.Ca[3].CaM, &d.Ca[2].CaM, dCa, &d.Ca[3].CaM) // 3

	cp.CaCaM01.StepK(kf, c.Ca[0].CaM_CaMKII, cCa, c.Ca[1].CaM_CaMKII, &d.Ca[0].CaM_CaMKII, dCa, &d.Ca[1].CaM_CaMKII)        // 1
	cp.CaCaM12.StepK(kf, c.Ca[1].CaM_CaMKII, cCa, c.Ca[2].CaM_CaMKII, &d.Ca[1].CaM_CaMKII, dCa, &d.Ca[2].CaM_CaMKII)        // 2
	cp.CaCaM23_CaMKII.StepK(kf, c.Ca[2].CaM_CaMKII, cCa, c.Ca[3].CaM_CaMKII, &d.Ca[2].CaM_CaMKII, dCa, &d.Ca[3].CaM_CaMKII) // 6

	for i := 0; i < 3; i++ {
		cp.CaMCaMKII.StepK(kf, c.Ca[i].CaM, c.CaMKII, c.Ca[i].CaM_CaMKII, &d.Ca[i].CaM, &d.CaMKII, &d.Ca[i].CaM_CaMKII) // 4
	}
	cp.CaMCaMKII3.StepK(kf, c.Ca[3].CaM, c.CaMKII, c.Ca[3].CaM_CaMKII, &d.Ca[3].CaM, &d.CaMKII, &d.Ca[3].CaM_CaMKII) // 5

	cp.CaMCaMKIIP.StepK(kf, c.Ca[0].CaM, c.CaMKIIP, c.Ca[0].CaM_CaMKIIP, &d.Ca[0].CaM, &d.CaMKIIP, &d.Ca[0].CaM_CaMKIIP) // 9
	for i := 0; i < 3; i++ {
		cp.CaCaM_CaMKIIP.StepK(kf, c.Ca[i].CaM_CaMKIIP, cCa, c.Ca[i+1].CaM_CaMKIIP, &d.Ca[i].CaM_CaMKIIP, dCa, &d.Ca[i+1].CaM_CaMKIIP) // 8
	}

	// cs, ce, cc, cp -> ds, de, dc, dp
	cp.PP1Thr286.StepK(kf, c.CaMKIIP, pp1, c.PP1Thr286C, c.CaMKII, &d.CaMKIIP, dpp1, &d.PP1Thr286C, &d.CaMKII) // 10
	if dpp2a != nil {
		cp.PP2AThr286.StepK(kf, c.CaMKIIP, pp2a, c.PP2AThr286C, c.CaMKII, &d.CaMKIIP, dpp2a, &d.PP2AThr286C, &d.CaMKII) // 11
	}

	for i := 0; i < 4; i++ {
		cc := &c.Ca[i]
		dc := &d.Ca[i]
		dc.CaM_CaMKIIP += c.Kauto * cc.CaM_CaMKII // forward only autophos
		// cs, ce, cc, cp -> ds, de, dc, dp
		cp.PP1Thr286.StepK(kf, cc.CaM_CaMKIIP, pp1, c.PP1Thr286C, cc.CaM_CaMKII, &dc.CaM_CaMKIIP, dpp1, &d.PP1Thr286C, &dc.CaM_CaMKII) // 10
		if dpp2a != nil {
			cp.PP2AThr286.StepK(kf, cc.CaM_CaMKIIP, pp2a, c.PP2AThr286C, cc.CaM_CaMKII, &dc.CaM_CaMKIIP, dpp2a, &d.PP2AThr286C, &dc.CaM_CaMKII) // 11
		}
	}
}

// Step does one step of CaMKII updating, c=current, d=delta
// pp2a = current cyt pp2a
func (cp *CaMKIIParams) Step(c, d *CaMKIIState, cCa, dCa *CaState, pp1, dpp1 *PP1State, pp2a float64, dpp2a *float64) {
	cp.StepCaMKII(CytVol, &c.Cyt, &d.Cyt, cCa.Cyt, pp1.Cyt.PP1act, pp2a, &dCa.Cyt, &dpp1.Cyt.PP1act, dpp2a)
	cp.StepCaMKII(PSDVol, &c.PSD, &d.PSD, cCa.PSD, pp1.PSD.PP1act, 0, &dCa.PSD, &dpp1.PSD.PP1act, nil)
}
