// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/emergent/chem"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// CaCaMKIIVars are intracellular Ca-driven signaling variables for the
// CaMKII+CaM binding -- each can have different numbers of Ca bound
// Dupont = DupontHouartDekonnick03, has W* terms used in Genesis code
// stores N values -- Co = Concentration computed by volume as needed
type CaCaMKIIVars struct {
	CaM_CaMKII  float64 `desc:"CaMKII-CaM bound together = WBn in Dupont"`
	CaM_CaMKIIP float64 `desc:"CaMKIIP-CaM bound together, P = phosphorylated at Thr286 = WTn in Dupont"`
}

func (cs *CaCaMKIIVars) Init(vol float64) {
	cs.Zero()
}

func (cs *CaCaMKIIVars) Zero() {
	cs.CaM_CaMKII = 0
	cs.CaM_CaMKIIP = 0
}

func (cs *CaCaMKIIVars) Integrate(d *CaCaMKIIVars) {
	chem.Integrate(&cs.CaM_CaMKII, d.CaM_CaMKII)
	chem.Integrate(&cs.CaM_CaMKIIP, d.CaM_CaMKIIP)
}

// AutoPVars hold the auto-phosphorylation variables, for CaMKII and DAPK1
type AutoPVars struct {
	Act   float64 `desc:"total active CaMKII"`
	Total float64 `desc:"total CaMKII across all states"`
	K     float64 `desc:"rate constant for further autophosphorylation as function of current state"`
}

func (av *AutoPVars) Zero() {
	av.Act = 0
	av.Total = 0
	av.K = 0
}

// CaMKIIVars are intracellular Ca-driven signaling states
// for CaMKII binding and phosphorylation with CaM + Ca
// Dupont = DupontHouartDekonnick03, has W* terms used in Genesis code
// stores N values -- Co = Concentration computed by volume as needed
type CaMKIIVars struct {
	Ca          [4]CaCaMKIIVars `desc:"increasing levels of Ca binding, 0-3"`
	CaMKII      float64         `desc:"unbound CaMKII = CaM kinase II -- WI in Dupont -- this is the inactive form for NMDA GluN2B binding"`
	CaMKIIP     float64         `desc:"unbound CaMKII P = phosphorylated at Thr286 -- shown with * in Figure S13 = WA in Dupont -- this is the active form for NMDA GluN2B binding"`
	PP1Thr286C  float64         `desc:"PP1+CaMKIIP complex for PP1Thr286 enzyme reaction"`
	PP2AThr286C float64         `desc:"PP2A+CaMKIIP complex for PP2AThr286 enzyme reaction"`
	Auto        AutoPVars       `view:"inline" inactive:"+" desc:"auto-phosphorylation state"`

	// todo: add competitive GluNRB binding for CaMKII and DAPK1
}

func (cs *CaMKIIVars) Init(vol float64) {
	for i := range cs.Ca {
		cs.Ca[i].Init(vol)
	}
	cs.CaMKII = chem.CoToN(20, vol)
	cs.CaMKIIP = 0 // WA
	cs.PP1Thr286C = 0
	cs.PP2AThr286C = 0

	if InitBaseline {
		cs.CaMKII = chem.CoToN(19.28, vol) // orig: 20
	}

	cs.UpdtActive()
}

// Generate Code for Initializing
func (cs *CaMKIIVars) InitCode(vol float64, pre string) {
	for i := range cs.Ca {
		fmt.Printf("\tcs.%s.Ca[%d].CaM_CaMKII = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.Ca[i].CaM_CaMKII, vol))
		fmt.Printf("\tcs.%s.Ca[%d].CaM_CaMKIIP = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.Ca[i].CaM_CaMKIIP, vol))
	}
	fmt.Printf("\tcs.%s.CaMKII = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.CaMKII, vol))
	fmt.Printf("\tcs.%s.CaMKIIP = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.CaMKIIP, vol))
	fmt.Printf("\tcs.%s.PP1Thr286C = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.PP1Thr286C, vol))
	fmt.Printf("\tcs.%s.PP2AThr286C = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.PP2AThr286C, vol))
}

func (cs *CaMKIIVars) Zero() {
	for i := range cs.Ca {
		cs.Ca[i].Zero()
	}
	cs.CaMKII = 0
	cs.CaMKIIP = 0
	cs.PP1Thr286C = 0
	cs.PP2AThr286C = 0
	cs.Auto.Zero()
}

func (cs *CaMKIIVars) Integrate(d *CaMKIIVars) {
	for i := range cs.Ca {
		cs.Ca[i].Integrate(&d.Ca[i])
	}
	chem.Integrate(&cs.CaMKII, d.CaMKII)
	chem.Integrate(&cs.CaMKIIP, d.CaMKIIP)
	chem.Integrate(&cs.PP1Thr286C, d.PP1Thr286C)
	chem.Integrate(&cs.PP2AThr286C, d.PP2AThr286C)
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
	cs.Auto.K = 0.29 * tmp
	cs.Auto.Act = cb*WB + WP + ct*WT + ca*WA
	cs.Auto.Total = T
}

func (cs *CaMKIIVars) Log(dt *etable.Table, vol float64, row int, pre string) {
	dt.SetCellFloat(pre+"CaMKIIact", row, chem.CoFmN(cs.Auto.Act, vol))
	// dt.SetCellFloat(pre+"Ca0CaM_CaMKII", row, chem.CoFmN(cs.Ca[0].CaM_CaMKII, vol))
	// dt.SetCellFloat(pre+"Ca1CaM_CaMKII", row, chem.CoFmN(cs.Ca[1].CaM_CaMKII, vol))
	// dt.SetCellFloat(pre+"Ca0CaM_CaMKIIP", row, chem.CoFmN(cs.Ca[0].CaM_CaMKIIP, vol))
	// dt.SetCellFloat(pre+"Ca1CaM_CaMKIIP", row, chem.CoFmN(cs.Ca[1].CaM_CaMKIIP, vol))
	// dt.SetCellFloat(pre+"CaMKII", row, chem.CoFmN(cs.CaMKII, vol))
	// dt.SetCellFloat(pre+"CaMKIIP", row, chem.CoFmN(cs.CaMKIIP, vol))
	// dt.SetCellFloat(pre+"CaMKII_AutoK", row, chem.CoFmN(cs.Auto.K, vol))
}

func (cs *CaMKIIVars) ConfigLog(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "CaMKIIact", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca0CaM_CaMKII", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca1CaM_CaMKII", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca0CaM_CaMKIIP", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca1CaM_CaMKIIP", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "CaMKII", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "CaMKIIP", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "CaMKII_AutoK", etensor.FLOAT64, nil, nil})
}

// CaMKIIState is overall intracellular Ca-driven signaling states
// for CaMKII in Cyt and PSD
// 28 state vars total
type CaMKIIState struct {
	Cyt CaMKIIVars `desc:"in cytosol -- volume = 0.08 fl = 48"`
	PSD CaMKIIVars `desc:"in PSD -- volume = 0.02 fl = 12"`
}

func (cs *CaMKIIState) Init() {
	cs.Cyt.Init(CytVol)
	cs.PSD.Init(PSDVol)

	if InitBaseline {
		// All vals below from 500 sec baseline
		// Note: all CaMKIIP = 0 after baseline
		vol := float64(CytVol)
		cs.Cyt.Ca[0].CaM_CaMKII = chem.CoToN(0.2962, vol)
		cs.Cyt.Ca[1].CaM_CaMKII = chem.CoToN(0.003792, vol)
		cs.Cyt.Ca[2].CaM_CaMKII = chem.CoToN(2.53e-05, vol)
		cs.Cyt.Ca[3].CaM_CaMKII = chem.CoToN(0.002523, vol)
		cs.Cyt.CaMKII = chem.CoToN(19.28, vol) // orig: 20

		vol = PSDVol
		cs.PSD.Ca[0].CaM_CaMKII = chem.CoToN(2.257, vol)
		cs.PSD.Ca[1].CaM_CaMKII = chem.CoToN(0.02889, vol)
		cs.PSD.Ca[2].CaM_CaMKII = chem.CoToN(0.0001927, vol)
		cs.PSD.Ca[3].CaM_CaMKII = chem.CoToN(0.01968, vol)
		cs.PSD.CaMKII = chem.CoToN(19.35, vol) // orig: 20
	}
}

func (cs *CaMKIIState) InitCode() {
	fmt.Printf("\nCaMKIIState:\n")
	cs.Cyt.InitCode(CytVol, "Cyt")
	cs.PSD.InitCode(PSDVol, "PSD")
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
	CaCaM01       chem.React `desc:"1: Ca+CaM-CaMKII -> 1CaCaM-CaMKII = CaM-bind-Ca"`
	CaCaM12       chem.React `desc:"2: Ca+1CaM-CaMKII -> 2CaCaM-CaMKII = CaMCa-bind-Ca"`
	CaCaM23       chem.React `desc:"6: Ca+2CaCaM-CaMKII -> 3CaCaM-CaMKII = CaMCa2-bind-Ca"`
	CaMCaMKII     chem.React `desc:"4: CaM+CaMKII -> CaM-CaMKII [0-2] -- kIB_kBI_[0-2] -- WI = plain CaMKII, WBn = CaM bound"`
	CaMCaMKII3    chem.React `desc:"5: 3CaCaM+CaMKII -> 3CaCaM-CaMKII = kIB_kBI_3"`
	CaCaM_CaMKIIP chem.React `desc:"8: Ca+nCaCaM-CaMKIIP -> n+1CaCaM-CaMKIIP = kTP_PT_*"`
	CaMCaMKIIP    chem.React `desc:"9: CaM+CaMKIIP -> CaM-CaMKIIP = kAT_kTA"` // note: typo in SI3 for top PP1, PP2A
	PP1Thr286     chem.Enz   `desc:"10: PP1 dephosphorylating CaMKIIP"`
	PP2AThr286    chem.Enz   `desc:"11: PP2A dephosphorylating CaMKIIP"`

	// DAPK1
	CaMDAPK1      chem.React `desc:"4: CaM+DAPK1 -> CaM-DAPK1 [0-2] -- kIB_kBI_[0-2] -- WI = plain DAPK1, WBn = CaM bound"`
	CaMDAPK13     chem.React `desc:"5: 3CaCaM+DAPK1 -> 3CaCaM-DAPK1 = kIB_kBI_3"`
	CaCaM23_DAPK1 chem.React `desc:"6: Ca+2CaCaM-DAPK1 -> 3CaCaM-DAPK1 = CaMCa2-bind-Ca"`
	CaCaM_DAPK1P  chem.React `desc:"8: Ca+nCaCaM-DAPK1P -> n+1CaCaM-DAPK1P = kTP_PT_*"`
	CaMDAPK1P     chem.React `desc:"9: CaM+DAPK1P -> CaM-DAPK1P = kAT_kTA"` // note: typo in SI3 for top PP1, PP2A
	CaNS308       chem.Enz   `desc:"CaN dephosphorylating DAPK1P"`

	CaMKIIDiffuse  chem.Diffuse `desc:"CaMKII diffusion between Cyt and PSD -- symmetric, just WI"`
	CaMKIIPDiffuse chem.Diffuse `desc:"CaMKIIP diffusion between Cyt and PSD -- asymmetric, everything else"`
}

func (cp *CaMKIIParams) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// See React docs for more info
	cp.CaCaM01.SetVol(51.202, CytVol, 200) // 1: 51.202 μM-1 = 1.0667, PSD 4.2667 = CaM-bind-Ca
	cp.CaCaM12.SetVol(133.3, CytVol, 1000) // 2: 133.3 μM-1 = 2.7771, PSD 11.108 = CaMCa-bind-Ca
	cp.CaCaM23.SetVol(25.6, CytVol, 0.02)  // 6: 25.6 μM-1 = 0.53333, PSD 2.1333 = CaMCa2-bind-Ca

	cp.CaMCaMKII.SetVol(0.0004, CytVol, 1) // 4: 0.0004 μM-1 = 8.3333e-6, PSD 3.3333e-5 = kIB_kBI_[0-2]
	cp.CaMCaMKII3.SetVol(8, CytVol, 1)     // 5: 8 μM-1 = 0.16667, PSD 3.3333e-5 = kIB_kBI_3

	cp.CaCaM_CaMKIIP.SetVol(1, CytVol, 1)  // 8: 1 μM-1 = 0.020834, PSD 0.0833335 = kTP_PT_*
	cp.CaMCaMKIIP.SetVol(8, CytVol, 0.001) // 9: 8 μM-1 = 0.16667, PSD 0.66667 = kAT_kTA

	cp.PP1Thr286.SetKmVol(11, CytVol, 1.34, 0.335)  // 10: 11 μM Km = 0.0031724
	cp.PP2AThr286.SetKmVol(11, CytVol, 1.34, 0.335) // 11: 11 μM Km = 0.0031724

	cp.CaMKIIDiffuse.SetSym(6.0 / 0.0225)
	cp.CaMKIIPDiffuse.Set(6.0/0.0225, 0.6/0.0225)
}

// StepCaMKII does the bulk of Ca + CaM + CaMKII binding reactions, in a given region
// cCa, nCa = current next Ca
func (cp *CaMKIIParams) StepCaMKII(vol float64, c, d *CaMKIIVars, cm, dm *CaMVars, cCa, pp1, pp2a float64, dCa, dpp1, dpp2a *float64) {
	kf := CytVol / vol
	cp.CaCaM01.StepK(kf, c.Ca[0].CaM_CaMKII, cCa, c.Ca[1].CaM_CaMKII, &d.Ca[0].CaM_CaMKII, dCa, &d.Ca[1].CaM_CaMKII) // 1
	cp.CaCaM12.StepK(kf, c.Ca[1].CaM_CaMKII, cCa, c.Ca[2].CaM_CaMKII, &d.Ca[1].CaM_CaMKII, dCa, &d.Ca[2].CaM_CaMKII) // 2
	cp.CaCaM23.StepK(kf, c.Ca[2].CaM_CaMKII, cCa, c.Ca[3].CaM_CaMKII, &d.Ca[2].CaM_CaMKII, dCa, &d.Ca[3].CaM_CaMKII) // 6

	for i := 0; i < 3; i++ {
		cp.CaMCaMKII.StepK(kf, cm.CaM[i], c.CaMKII, c.Ca[i].CaM_CaMKII, &dm.CaM[i], &d.CaMKII, &d.Ca[i].CaM_CaMKII) // 4
	}
	cp.CaMCaMKII3.StepK(kf, cm.CaM[3], c.CaMKII, c.Ca[3].CaM_CaMKII, &dm.CaM[3], &d.CaMKII, &d.Ca[3].CaM_CaMKII) // 5

	cp.CaMCaMKIIP.StepK(kf, cm.CaM[0], c.CaMKIIP, c.Ca[0].CaM_CaMKIIP, &dm.CaM[0], &d.CaMKIIP, &d.Ca[0].CaM_CaMKIIP) // 9
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
		dak := c.Auto.K * cc.CaM_CaMKII
		dc.CaM_CaMKIIP += dak
		dc.CaM_CaMKII -= dak
		// cs, ce, cc, cp -> ds, de, dc, dp
		cp.PP1Thr286.StepK(kf, cc.CaM_CaMKIIP, pp1, c.PP1Thr286C, cc.CaM_CaMKII, &dc.CaM_CaMKIIP, dpp1, &d.PP1Thr286C, &dc.CaM_CaMKII) // 10
		if dpp2a != nil {
			cp.PP2AThr286.StepK(kf, cc.CaM_CaMKIIP, pp2a, c.PP2AThr286C, cc.CaM_CaMKII, &dc.CaM_CaMKIIP, dpp2a, &d.PP2AThr286C, &dc.CaM_CaMKII) // 11
		}
	}
}

// StepDiffuse does Cyt <-> PSD diffusion
func (cp *CaMKIIParams) StepDiffuse(c, d *CaMKIIState) {
	for i := 0; i < 4; i++ {
		cc := &c.Cyt.Ca[i]
		cd := &c.PSD.Ca[i]
		dc := &d.Cyt.Ca[i]
		dd := &d.PSD.Ca[i]
		cp.CaMKIIPDiffuse.Step(cc.CaM_CaMKII, cd.CaM_CaMKII, CytVol, PSDVol, &dc.CaM_CaMKII, &dd.CaM_CaMKII)
		cp.CaMKIIPDiffuse.Step(cc.CaM_CaMKIIP, cd.CaM_CaMKIIP, CytVol, PSDVol, &dc.CaM_CaMKIIP, &dd.CaM_CaMKIIP)
	}
	cp.CaMKIIDiffuse.Step(c.Cyt.CaMKII, c.PSD.CaMKII, CytVol, PSDVol, &d.Cyt.CaMKII, &d.PSD.CaMKII)
	cp.CaMKIIPDiffuse.Step(c.Cyt.CaMKIIP, c.PSD.CaMKIIP, CytVol, PSDVol, &d.Cyt.CaMKIIP, &d.PSD.CaMKIIP)
}

// Step does one step of CaMKII updating, c=current, d=delta
// pp2a = current cyt pp2a
func (cp *CaMKIIParams) Step(c, d *CaMKIIState, cm, dm *CaMState, cCa, dCa *CaState, pp1, dpp1 *PP1State, pp2a float64, dpp2a *float64) {
	cp.StepCaMKII(CytVol, &c.Cyt, &d.Cyt, &cm.Cyt, &dm.Cyt, cCa.Cyt, pp1.Cyt.PP1act, pp2a, &dCa.Cyt, &dpp1.Cyt.PP1act, dpp2a)
	cp.StepCaMKII(PSDVol, &c.PSD, &d.PSD, &cm.PSD, &dm.PSD, cCa.PSD, pp1.PSD.PP1act, 0, &dCa.PSD, &dpp1.PSD.PP1act, nil)
	cp.StepDiffuse(c, d)
}
