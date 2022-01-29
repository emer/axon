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

// CaDAPK1Vars are intracellular Ca-driven signaling variables for the
// DAPK1+CaM binding -- each can have different numbers of Ca bound
// Dupont = DupontHouartDekonnick03, has W* terms used in Genesis code
// stores N values -- Co = Concentration computed by volume as needed
type CaDAPK1Vars struct {
	CaM_DAPK1  float64 `desc:"DAPK1-CaM bound together, de-phosphorylated at S308 by CaN -- this is the active form for GluN2B and CaM binding -- equating to WTn in Dupont"`
	CaM_DAPK1P float64 `desc:"DAPK1-CaM bound together, P = phosphorylated at S308 -- this is the inactive form for GluN2B and CaM binding -- equating to WBn in Dupont"`
	N2B_DAPK1  float64 `desc:"DAPK1 (noP) bound to NMDA N2B (only for PSD compartment)"`
	N2B_DAPK1P float64 `desc:"DAPK1P bound to NMDA N2B (only for PSD compartment)"`
}

func (cs *CaDAPK1Vars) Init(vol float64) {
	cs.Zero()
}

func (cs *CaDAPK1Vars) Zero() {
	cs.CaM_DAPK1 = 0
	cs.CaM_DAPK1P = 0
	cs.N2B_DAPK1 = 0
}

func (cs *CaDAPK1Vars) Integrate(d *CaDAPK1Vars) {
	chem.Integrate(&cs.CaM_DAPK1, d.CaM_DAPK1)
	chem.Integrate(&cs.CaM_DAPK1P, d.CaM_DAPK1P)
	chem.Integrate(&cs.N2B_DAPK1, d.N2B_DAPK1)
}

// DAPK1Vars are intracellular Ca-driven signaling states
// for DAPK1 binding and phosphorylation with CaM + Ca
// Dupont = DupontHouartDekonnick03, has W* terms used in Genesis code
// stores N values -- Co = Concentration computed by volume as needed
type DAPK1Vars struct {
	Ca         [4]CaDAPK1Vars `desc:"increasing levels of Ca binding, 0-3"`
	DAPK1      float64        `desc:"unbound DAPK1, de-phosphorylated at S308 by CaN -- this is the active form for NMDA GluN2B and CaM binding"`
	DAPK1P     float64        `desc:"unbound DAPK1, P = phosphorylated at S308 -- this is the inactive form for NMDA GluN2B and CaM binding"`
	N2B_DAPK1  float64        `desc:"DAPK1 (noP) bound to N2B (only for PSD compartment)"`
	CaNSer308C float64        `desc:"CaN+DAPK1P complex for CaNSer308 enzyme reaction"`
	Auto       AutoPVars      `view:"inline" inactive:"+" desc:"auto-phosphorylation state"`
	// todo: add competitive GluNRB binding
}

func (cs *DAPK1Vars) Init(vol float64) {
	for i := range cs.Ca {
		cs.Ca[i].Init(vol)
	}
	cs.DAPK1 = 0                    //
	cs.DAPK1P = chem.CoToN(20, vol) // Shani says P form in baseline -- Goodell says "highly enriched"
	cs.N2B_DAPK1 = 0
	cs.CaNSer308C = 0

	if TheOpts.InitBaseline {
		cs.DAPK1P = chem.CoToN(19.09, vol) // orig: 20
		cs.DAPK1 = chem.CoToN(2.303e-6, vol)
		// todo: update N2B
		cs.CaNSer308C = chem.CoToN(0.004463, vol)

		cs.Ca[0].CaM_DAPK1 = chem.CoToN(0.1598, vol)
		cs.Ca[0].CaM_DAPK1P = chem.CoToN(0.7048, vol)
		cs.Ca[1].CaM_DAPK1 = chem.CoToN(0.01177, vol)
		cs.Ca[1].CaM_DAPK1P = chem.CoToN(0.009085, vol)
		cs.Ca[2].CaM_DAPK1 = chem.CoToN(0.003368, vol)
		cs.Ca[2].CaM_DAPK1P = chem.CoToN(6.682e-05, vol)
		cs.Ca[3].CaM_DAPK1 = chem.CoToN(0.001596, vol)
		cs.Ca[3].CaM_DAPK1P = chem.CoToN(0.01179, vol)
	}

	cs.AutoK()
}

// Generate Code for Initializing
func (cs *DAPK1Vars) InitCode(vol float64, pre string) {
	for i := range cs.Ca {
		fmt.Printf("\tcs.%s.Ca[%d].CaM_DAPK1 = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.Ca[i].CaM_DAPK1, vol))
		fmt.Printf("\tcs.%s.Ca[%d].CaM_DAPK1P = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.Ca[i].CaM_DAPK1P, vol))
		if cs.Ca[i].N2B_DAPK1 != 0 {
			fmt.Printf("\tcs.%s.Ca[%d].N2B_DAPK1 = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.Ca[i].N2B_DAPK1, vol))
		}
	}
	fmt.Printf("\tcs.%s.DAPK1 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.DAPK1, vol))
	fmt.Printf("\tcs.%s.DAPK1P = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.DAPK1P, vol))
	fmt.Printf("\tcs.%s.N2B_DAPK1 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.N2B_DAPK1, vol))
	fmt.Printf("\tcs.%s.CaNSer308C = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.CaNSer308C, vol))
}

func (cs *DAPK1Vars) Zero() {
	for i := range cs.Ca {
		cs.Ca[i].Zero()
	}
	cs.DAPK1 = 0
	cs.DAPK1P = 0
	cs.N2B_DAPK1 = 0
	cs.CaNSer308C = 0
	cs.Auto.Zero()
}

func (cs *DAPK1Vars) Integrate(d *DAPK1Vars) {
	for i := range cs.Ca {
		cs.Ca[i].Integrate(&d.Ca[i])
	}
	chem.Integrate(&cs.DAPK1, d.DAPK1)
	chem.Integrate(&cs.DAPK1P, d.DAPK1P)
	chem.Integrate(&cs.N2B_DAPK1, d.N2B_DAPK1)
	chem.Integrate(&cs.CaNSer308C, d.CaNSer308C)
	cs.AutoK()
}

// AutoK updates DAPK1 Auto.K, using the TOTAL Phos. activity,
// including N2B and not
func (cs *DAPK1Vars) AutoK() {
	WI := cs.DAPK1 + cs.N2B_DAPK1
	WA := cs.DAPK1P
	n2b := cs.N2B_DAPK1

	var WB, WT float64

	for i := 0; i < 3; i++ {
		WB += cs.Ca[i].CaM_DAPK1 + cs.Ca[i].N2B_DAPK1
		WT += cs.Ca[i].CaM_DAPK1P
		n2b += cs.Ca[i].N2B_DAPK1
	}
	WB += cs.Ca[3].CaM_DAPK1 + cs.Ca[3].N2B_DAPK1
	WP := cs.Ca[3].CaM_DAPK1P
	n2b += cs.Ca[3].N2B_DAPK1

	TotalW := WI + WB + WP + WT + WA
	Wb := WB / TotalW
	Wp := WP / TotalW
	Wt := WT / TotalW
	Wa := WA / TotalW
	cb := 0.75
	ct := 0.8
	ca := 0.8

	T := Wb + Wp + Wt + Wa
	tmp := T * (-0.22 + 1.826*T + -0.8*T*T) // baseline effect from total
	tmp *= 0.75 * (cb*Wb + Wp + ct*Wt + ca*Wa)
	if tmp < 0 {
		tmp = 0
	}
	cs.Auto.K = 0.29 * tmp
	cs.Auto.Total = T
	cs.Auto.N2B = n2b

	cs.Active()
}

// Active computes Auto.Act based on the non-P states
func (cs *DAPK1Vars) Active() {
	WA := cs.DAPK1 + cs.N2B_DAPK1

	var WB, WT float64

	for i := 0; i < 3; i++ {
		WB += cs.Ca[i].CaM_DAPK1P
		WT += cs.Ca[i].CaM_DAPK1 + cs.Ca[i].N2B_DAPK1
	}
	WB += cs.Ca[3].CaM_DAPK1P
	WP := cs.Ca[3].CaM_DAPK1 + cs.Ca[3].N2B_DAPK1

	// Note: the only thing not in here is WI = base DAPK1P
	// It is not 100% clear that this exact formula applies here
	// for example, it predicts Phos + CaM (WB) contributes..
	cs.Auto.Act = 0.75*WB + WP + 0.8*WT + 0.8*WA
}

func (cs *DAPK1Vars) Log(dt *etable.Table, vol float64, row int, pre string) {
	dt.SetCellFloat(pre+"DAPK1act", row, chem.CoFmN(cs.Auto.Act, vol))
	// dt.SetCellFloat(pre+"Ca0CaM_DAPK1", row, chem.CoFmN(cs.Ca[0].CaM_DAPK1, vol))
	// dt.SetCellFloat(pre+"Ca1CaM_DAPK1", row, chem.CoFmN(cs.Ca[1].CaM_DAPK1, vol))
	// dt.SetCellFloat(pre+"Ca0CaM_DAPK1P", row, chem.CoFmN(cs.Ca[0].CaM_DAPK1P, vol))
	// dt.SetCellFloat(pre+"Ca1CaM_DAPK1P", row, chem.CoFmN(cs.Ca[1].CaM_DAPK1P, vol))
	// dt.SetCellFloat(pre+"DAPK1", row, chem.CoFmN(cs.DAPK1, vol))
	// dt.SetCellFloat(pre+"DAPK1P", row, chem.CoFmN(cs.DAPK1P, vol))
	dt.SetCellFloat(pre+"DAPK1_AutoK", row, cs.Auto.K) // note: rates are not conc dep(?)
}

func (cs *DAPK1Vars) ConfigLog(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "DAPK1act", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca0CaM_DAPK1", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca1CaM_DAPK1", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca0CaM_DAPK1P", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca1CaM_DAPK1P", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "DAPK1", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "DAPK1P", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "DAPK1_AutoK", etensor.FLOAT64, nil, nil})
}

// DAPK1State is overall intracellular Ca-driven signaling states
// for DAPK1 in Cyt and PSD
// 28 state vars total
type DAPK1State struct {
	Cyt DAPK1Vars `desc:"in cytosol -- volume = 0.08 fl = 48"`
	PSD DAPK1Vars `desc:"in PSD -- volume = 0.02 fl = 12"`
}

func (cs *DAPK1State) Init() {
	cs.Cyt.Init(CytVol)
	cs.PSD.Init(PSDVol)

	if TheOpts.InitBaseline {
		// All vals below from 500 sec baseline
		// Note: all DAPK1P = 0 after baseline
	}
}

func (cs *DAPK1State) InitCode() {
	fmt.Printf("\nDAPK1State:\n")
	cs.Cyt.InitCode(CytVol, "Cyt")
	cs.PSD.InitCode(PSDVol, "PSD")
}

func (cs *DAPK1State) Zero() {
	cs.Cyt.Zero()
	cs.PSD.Zero()
}

func (cs *DAPK1State) Integrate(d *DAPK1State) {
	cs.Cyt.Integrate(&d.Cyt)
	cs.PSD.Integrate(&d.PSD)
}

func (cs *DAPK1State) Log(dt *etable.Table, row int) {
	cs.Cyt.Log(dt, CytVol, row, "Cyt_")
	cs.PSD.Log(dt, PSDVol, row, "PSD_")
}

func (cs *DAPK1State) ConfigLog(sch *etable.Schema) {
	cs.Cyt.ConfigLog(sch, "Cyt_")
	cs.PSD.ConfigLog(sch, "PSD_")
}

// DAPK1Params are the parameters governing the Ca+CaM binding
// We're using the same equations as CaMKII but the roles of P and non-P are reversed
// in respect to Ca-CaM binding.  Furthermore, the Auto.K is auto-de-phosphorylating!
type DAPK1Params struct {
	CaCaM01     chem.React `desc:"1: Ca+CaM-DAPK1P -> 1CaCaM-DAPK1P = CaM-bind-Ca"`
	CaCaM12     chem.React `desc:"2: Ca+1CaM-DAPK1P -> 2CaCaM-DAPK1P = CaMCa-bind-Ca"`
	CaCaM23     chem.React `desc:"6: Ca+2CaCaM-DAPK1P -> 3CaCaM-DAPK1P = CaMCa2-bind-Ca"`
	CaMDAPK1P   chem.React `desc:"4: CaM+DAPK1P -> CaM-DAPK1P [0-2] -- kIB_kBI_[0-2] -- WI = plain DAPK1P, WBn = CaM bound"`
	CaMDAPK1P3  chem.React `desc:"5: 3CaCaM+DAPK1P -> 3CaCaM-DAPK1P = kIB_kBI_3"`
	CaCaM_DAPK1 chem.React `desc:"8: Ca+nCaCaM-DAPK1 -> n+1CaCaM-DAPK1 = kTP_PT_*"`
	CaMDAPK1    chem.React `desc:"9: CaM+DAPK1 -> CaM-DAPK1 = kAT_kTA"`
	CaNSer308   chem.Enz   `desc:"CaN dephosphorylating DAPK1P"`
	AutoK       float64    `desc:"auto.K multiplier"`

	DAPK1Diffuse  chem.Diffuse `desc:"DAPK1 diffusion between Cyt and PSD -- symmetric, just WI"`
	DAPK1PDiffuse chem.Diffuse `desc:"DAPK1P diffusion between Cyt and PSD -- asymmetric, everything else"`
}

func (cp *DAPK1Params) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// See React docs for more info
	cp.CaCaM01.SetVol(51.202, CytVol, 200) // 1: 51.202 μM-1 = 1.0667, PSD 4.2667 = CaM-bind-Ca
	cp.CaCaM12.SetVol(133.3, CytVol, 1000) // 2: 133.3 μM-1 = 2.7771, PSD 11.108 = CaMCa-bind-Ca
	cp.CaCaM23.SetVol(25.6, CytVol, 0.02)  // 6: 25.6 μM-1 = 0.53333, PSD 2.1333 = CaMCa2-bind-Ca

	cp.CaMDAPK1P.SetVol(0.0004, CytVol, 1) // 4: 0.0004 μM-1 = 8.3333e-6, PSD 3.3333e-5 = kIB_kBI_[0-2]
	cp.CaMDAPK1P3.SetVol(8, CytVol, 1)     // 5: 8 μM-1 = 0.16667, PSD 3.3333e-5 = kIB_kBI_3

	cp.CaCaM_DAPK1.SetVol(1, CytVol, 1)  // 8: 1 μM-1 = 0.020834, PSD 0.0833335 = kTP_PT_*
	cp.CaMDAPK1.SetVol(8, CytVol, 0.001) // 9: 8 μM-1 = 0.16667, PSD 0.66667 = kAT_kTA

	cp.CaNSer308.SetKmVol(11, CytVol, 1.34, 0.335) // 10: 11 μM Km = 0.0031724

	if cp.AutoK == 0 {
		cp.AutoK = .3
	}

	cp.DAPK1Diffuse.Set(6.0/0.0225, 0.6/0.0225) // asymmetric for dephos which binds to NR2
	cp.DAPK1PDiffuse.SetSym(6.0 / 0.0225)       // symmetric
}

// StepDAPK1 does the bulk of Ca + CaM + DAPK1 binding reactions, in a given region
// cCa, nCa = current next Ca
func (cp *DAPK1Params) StepDAPK1(vol float64, c, d *DAPK1Vars, cm, dm *CaMVars, cnm, dnm *NMDARState, cCa, can float64, dCa, dcan *float64) {
	kf := CytVol / vol

	// NOTE: everything is just reversed for P vs. non-P -- inverse of CaMKII

	cp.CaCaM01.StepK(kf, c.Ca[0].CaM_DAPK1P, cCa, c.Ca[1].CaM_DAPK1P, &d.Ca[0].CaM_DAPK1P, dCa, &d.Ca[1].CaM_DAPK1P) // 1
	cp.CaCaM12.StepK(kf, c.Ca[1].CaM_DAPK1P, cCa, c.Ca[2].CaM_DAPK1P, &d.Ca[1].CaM_DAPK1P, dCa, &d.Ca[2].CaM_DAPK1P) // 2
	cp.CaCaM23.StepK(kf, c.Ca[2].CaM_DAPK1P, cCa, c.Ca[3].CaM_DAPK1P, &d.Ca[2].CaM_DAPK1P, dCa, &d.Ca[3].CaM_DAPK1P) // 6

	for i := 0; i < 3; i++ {
		cp.CaMDAPK1P.StepK(kf, cm.CaM[i], c.DAPK1P, c.Ca[i].CaM_DAPK1P, &dm.CaM[i], &d.DAPK1P, &d.Ca[i].CaM_DAPK1P) // 4
	}
	cp.CaMDAPK1P3.StepK(kf, cm.CaM[3], c.DAPK1P, c.Ca[3].CaM_DAPK1P, &dm.CaM[3], &d.DAPK1P, &d.Ca[3].CaM_DAPK1P) // 5

	cp.CaMDAPK1.StepK(kf, cm.CaM[0], c.DAPK1, c.Ca[0].CaM_DAPK1, &dm.CaM[0], &d.DAPK1, &d.Ca[0].CaM_DAPK1) // 9
	for i := 0; i < 3; i++ {
		cp.CaCaM_DAPK1.StepK(kf, c.Ca[i].CaM_DAPK1, cCa, c.Ca[i+1].CaM_DAPK1, &d.Ca[i].CaM_DAPK1, dCa, &d.Ca[i+1].CaM_DAPK1) // 8
	}

	// cs, ce, cc, cp -> ds, de, dc, dp
	cp.CaNSer308.StepK(kf, c.DAPK1P, can, c.CaNSer308C, c.DAPK1, &d.DAPK1P, dcan, &d.CaNSer308C, &d.DAPK1) // 10

	for i := 0; i < 4; i++ {
		cc := &c.Ca[i]
		dc := &d.Ca[i]
		dak := cp.AutoK * c.Auto.K * cc.CaM_DAPK1
		dc.CaM_DAPK1P += dak
		dc.CaM_DAPK1 -= dak
		// cs, ce, cc, cp -> ds, de, dc, dp
		cp.CaNSer308.StepK(kf, cc.CaM_DAPK1P, can, c.CaNSer308C, cc.CaM_DAPK1, &dc.CaM_DAPK1P, dcan, &d.CaNSer308C, &dc.CaM_DAPK1) // 10
	}
}

// StepDiffuse does Cyt <-> PSD diffusion
func (cp *DAPK1Params) StepDiffuse(c, d *DAPK1State) {
	for i := 0; i < 4; i++ {
		cc := &c.Cyt.Ca[i]
		cd := &c.PSD.Ca[i]
		dc := &d.Cyt.Ca[i]
		dd := &d.PSD.Ca[i]
		cp.DAPK1PDiffuse.Step(cc.CaM_DAPK1, cd.CaM_DAPK1, CytVol, PSDVol, &dc.CaM_DAPK1, &dd.CaM_DAPK1)
		cp.DAPK1PDiffuse.Step(cc.CaM_DAPK1P, cd.CaM_DAPK1P, CytVol, PSDVol, &dc.CaM_DAPK1P, &dd.CaM_DAPK1P)
	}
	cp.DAPK1Diffuse.Step(c.Cyt.DAPK1, c.PSD.DAPK1, CytVol, PSDVol, &d.Cyt.DAPK1, &d.PSD.DAPK1)
	cp.DAPK1PDiffuse.Step(c.Cyt.DAPK1P, c.PSD.DAPK1P, CytVol, PSDVol, &d.Cyt.DAPK1P, &d.PSD.DAPK1P)
}

// Step does one step of DAPK1 updating, c=current, d=delta
// pp2a = current cyt pp2a
func (cp *DAPK1Params) Step(c, d *DAPK1State, cm, dm *CaMState, cCa, dCa *CaState, can, dcan *CaNState, cnm, dnm *NMDARState) {
	cp.StepDAPK1(CytVol, &c.Cyt, &d.Cyt, &cm.Cyt, &dm.Cyt, nil, nil, cCa.Cyt, can.Cyt.CaNact, &dCa.Cyt, &dcan.Cyt.CaNact)
	cp.StepDAPK1(PSDVol, &c.PSD, &d.PSD, &cm.PSD, &dm.PSD, cnm, dnm, cCa.PSD, can.PSD.CaNact, &dCa.PSD, &dcan.PSD.CaNact)
	cp.StepDiffuse(c, d)
}
