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
	CaM_DAPK1  float64 `desc:"DAPK1-CaM bound together = WBn in Dupont"`
	CaM_DAPK1P float64 `desc:"DAPK1P-CaM bound together, P = phosphorylated at Thr286 = WTn in Dupont"`
}

func (cs *CaDAPK1Vars) Init(vol float64) {
	cs.Zero()
}

func (cs *CaDAPK1Vars) Zero() {
	cs.CaM_DAPK1 = 0
	cs.CaM_DAPK1P = 0
}

func (cs *CaDAPK1Vars) Integrate(d *CaDAPK1Vars) {
	chem.Integrate(&cs.CaM_DAPK1, d.CaM_DAPK1)
	chem.Integrate(&cs.CaM_DAPK1P, d.CaM_DAPK1P)
}

// DAPK1Vars are intracellular Ca-driven signaling states
// for DAPK1 binding and phosphorylation with CaM + Ca
// Dupont = DupontHouartDekonnick03, has W* terms used in Genesis code
// stores N values -- Co = Concentration computed by volume as needed
type DAPK1Vars struct {
	Ca          [4]CaDAPK1Vars `desc:"increasing levels of Ca binding, 0-3"`
	N2B         [4]CaDAPK1Vars `desc:"GluN2B bound, increasing levels of Ca binding, 0-3"`
	DAPK1       float64        `desc:"unbound DAPK1 = CaM kinase II -- WI in Dupont -- this is the active form for NMDA GluN2B binding"`
	DAPK1P      float64        `desc:"unbound DAPK1 P = phosphorylated at Thr286 -- shown with * in Figure S13 = WA in Dupont -- this is the inactive form for NMDA GluN2B binding"`
	N2B_DAPK1   float64        `desc:"DAPK1 bound to NMDA N2B (only for PSD compartment) -- main binding"`
	N2B_DAPK1P  float64        `desc:"DAPK1P bound to NMDA N2B (only for PSD compartment) -- only as a residual"`
	CaNSer308C  float64        `desc:"CaN+DAPK1P complex for CaNSer308 enzyme reaction"`
	PP2ASer308C float64        `desc:"PP2A+CaMKIIP complex for PP2ASer308 enzyme reaction"`
	Auto        AutoPVars      `view:"inline" inactive:"+" desc:"auto-phosphorylation state"`
}

func (cs *DAPK1Vars) Init(vol float64) {
	for i := range cs.Ca {
		cs.Ca[i].Init(vol)
	}
	for i := range cs.N2B {
		cs.N2B[i].Zero()
	}
	cs.DAPK1 = 0
	cs.DAPK1P = chem.CoToN(20, vol)
	cs.N2B_DAPK1 = 0
	cs.N2B_DAPK1P = 0
	cs.CaNSer308C = 0

	if TheOpts.InitBaseline {
		// cs.DAPK1 = chem.CoToN(19.28, vol) // orig: 20
	}

	cs.ActiveK()
}

// Generate Code for Initializing
func (cs *DAPK1Vars) InitCode(vol float64, pre string) {
	for i := range cs.Ca {
		fmt.Printf("\tcs.%s.Ca[%d].CaM_DAPK1 = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.Ca[i].CaM_DAPK1, vol))
		fmt.Printf("\tcs.%s.Ca[%d].CaM_DAPK1P = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.Ca[i].CaM_DAPK1P, vol))
	}
	for i := range cs.N2B {
		fmt.Printf("\tcs.%s.N2B[%d].CaM_DAPK1 = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.N2B[i].CaM_DAPK1, vol))
		fmt.Printf("\tcs.%s.N2B[%d].CaM_DAPK1P = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.N2B[i].CaM_DAPK1P, vol))
	}
	fmt.Printf("\tcs.%s.DAPK1 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.DAPK1, vol))
	fmt.Printf("\tcs.%s.DAPK1P = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.DAPK1P, vol))
	fmt.Printf("\tcs.%s.N2B_DAPK1 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.N2B_DAPK1, vol))
	fmt.Printf("\tcs.%s.N2B_DAPK1P = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.N2B_DAPK1P, vol))
	fmt.Printf("\tcs.%s.CaNSer308C = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.CaNSer308C, vol))
	fmt.Printf("\tcs.%s.PP2ASer308C = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.PP2ASer308C, vol))
}

func (cs *DAPK1Vars) Zero() {
	for i := range cs.Ca {
		cs.Ca[i].Zero()
	}
	for i := range cs.N2B {
		cs.N2B[i].Zero()
	}
	cs.DAPK1 = 0
	cs.DAPK1P = 0
	cs.N2B_DAPK1 = 0
	cs.N2B_DAPK1P = 0
	cs.CaNSer308C = 0
	cs.PP2ASer308C = 0
	cs.Auto.Zero()
}

func (cs *DAPK1Vars) Integrate(d *DAPK1Vars) {
	for i := range cs.Ca {
		cs.Ca[i].Integrate(&d.Ca[i])
	}
	for i := range cs.N2B {
		cs.N2B[i].Integrate(&d.N2B[i])
	}
	chem.Integrate(&cs.DAPK1, d.DAPK1)
	chem.Integrate(&cs.DAPK1P, d.DAPK1P)
	chem.Integrate(&cs.N2B_DAPK1, d.N2B_DAPK1)
	chem.Integrate(&cs.N2B_DAPK1P, d.N2B_DAPK1P)
	chem.Integrate(&cs.CaNSer308C, d.CaNSer308C)
	chem.Integrate(&cs.PP2ASer308C, d.PP2ASer308C)
	cs.ActiveK()
}

var DAPK1AutoK = 1.0

// ActiveK updates active, total, and the Auto.K auto-phosphorylation rate constant
// DAPK1 autoK is *inhibited* by Ca/CaM binding, driven ONLY by the
// unbound DAPKP level!
// Act kinase activity is proportional to Ca/CaM binding in the deP form
// this is the same as what drives GluN2B binding (vs. CaMKII where it is
// purely CaCaM driven and has no P component)
func (cs *DAPK1Vars) ActiveK() {
	WI := cs.DAPK1 + cs.N2B_DAPK1
	WA := cs.DAPK1P + cs.N2B_DAPK1P
	n2b := cs.N2B_DAPK1 + cs.N2B_DAPK1P

	var WB, WT float64
	for i := 0; i < 4; i++ {
		WB += cs.Ca[i].CaM_DAPK1 + cs.N2B[i].CaM_DAPK1
		WT += cs.Ca[i].CaM_DAPK1P + cs.N2B[i].CaM_DAPK1P
		n2b += cs.N2B[i].CaM_DAPK1 + cs.N2B[i].CaM_DAPK1P
	}
	n2b += cs.N2B[3].CaM_DAPK1 + cs.N2B[3].CaM_DAPK1P

	TotalW := WI + WB + WT + WA
	Wa := WA / TotalW

	T := Wa // this is only one that drives autoK
	tmp := T * T * (-0.22 + 1.826*T + -0.8*T*T)
	if tmp < 0 {
		tmp = 0
	}
	cs.Auto.K = DAPK1AutoK * tmp
	cs.Auto.Act = WB + 0.75*WI // non-P, CaM bound -- no CaM = weaker
	cs.Auto.Total = WI + WA + WB + WT
	cs.Auto.N2B = n2b
}

func (cs *DAPK1Vars) Log(dt *etable.Table, vol float64, row int, pre string) {
	dt.SetCellFloat(pre+"DAPK1act", row, chem.CoFmN(cs.Auto.Act, vol))
	dt.SetCellFloat(pre+"Ca0CaM_DAPK1", row, chem.CoFmN(cs.Ca[0].CaM_DAPK1, vol))
	dt.SetCellFloat(pre+"Ca1CaM_DAPK1", row, chem.CoFmN(cs.Ca[1].CaM_DAPK1, vol))
	dt.SetCellFloat(pre+"Ca0CaM_DAPK1P", row, chem.CoFmN(cs.Ca[0].CaM_DAPK1P, vol))
	dt.SetCellFloat(pre+"Ca1CaM_DAPK1P", row, chem.CoFmN(cs.Ca[1].CaM_DAPK1P, vol))
	dt.SetCellFloat(pre+"DAPK1", row, chem.CoFmN(cs.DAPK1, vol))
	dt.SetCellFloat(pre+"DAPK1P", row, chem.CoFmN(cs.DAPK1P, vol))
	dt.SetCellFloat(pre+"DAPK1_AutoK", row, chem.CoFmN(cs.Auto.K, vol))
	if pre == "PSD_" {
		dt.SetCellFloat(pre+"DAPK1n2b", row, chem.CoFmN(cs.Auto.N2B, vol))
		dt.SetCellFloat(pre+"N2B_Ca0CaM_DAPK1", row, chem.CoFmN(cs.N2B[0].CaM_DAPK1, vol))
		dt.SetCellFloat(pre+"N2B_Ca1CaM_DAPK1", row, chem.CoFmN(cs.N2B[1].CaM_DAPK1, vol))
		dt.SetCellFloat(pre+"N2B_Ca0CaM_DAPK1P", row, chem.CoFmN(cs.N2B[0].CaM_DAPK1P, vol))
		dt.SetCellFloat(pre+"N2B_Ca1CaM_DAPK1P", row, chem.CoFmN(cs.N2B[1].CaM_DAPK1P, vol))
		dt.SetCellFloat(pre+"N2B_DAPK1", row, chem.CoFmN(cs.N2B_DAPK1, vol))
		dt.SetCellFloat(pre+"N2B_DAPK1P", row, chem.CoFmN(cs.N2B_DAPK1P, vol))
	}
}

func (cs *DAPK1Vars) ConfigLog(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "DAPK1act", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Ca0CaM_DAPK1", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Ca1CaM_DAPK1", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Ca0CaM_DAPK1P", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "Ca1CaM_DAPK1P", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "DAPK1", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "DAPK1P", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "DAPK1_AutoK", etensor.FLOAT64, nil, nil})
	if pre == "PSD_" {
		*sch = append(*sch, etable.Column{pre + "DAPK1n2b", etensor.FLOAT64, nil, nil})
		*sch = append(*sch, etable.Column{pre + "N2B_Ca0CaM_DAPK1", etensor.FLOAT64, nil, nil})
		*sch = append(*sch, etable.Column{pre + "N2B_Ca1CaM_DAPK1", etensor.FLOAT64, nil, nil})
		*sch = append(*sch, etable.Column{pre + "N2B_Ca0CaM_DAPK1P", etensor.FLOAT64, nil, nil})
		*sch = append(*sch, etable.Column{pre + "N2B_Ca1CaM_DAPK1P", etensor.FLOAT64, nil, nil})
		*sch = append(*sch, etable.Column{pre + "N2B_DAPK1", etensor.FLOAT64, nil, nil})
		*sch = append(*sch, etable.Column{pre + "N2B_DAPK1P", etensor.FLOAT64, nil, nil})
	}
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
		vol := float64(CytVol)
		cs.Cyt.Ca[0].CaM_DAPK1 = chem.CoToN(0.007214, vol)
		cs.Cyt.Ca[0].CaM_DAPK1P = chem.CoToN(0.6018, vol)
		cs.Cyt.Ca[1].CaM_DAPK1 = chem.CoToN(0.0001168, vol)
		cs.Cyt.Ca[1].CaM_DAPK1P = chem.CoToN(0.03415, vol)
		cs.Cyt.Ca[2].CaM_DAPK1 = chem.CoToN(5.555e-06, vol)
		cs.Cyt.Ca[2].CaM_DAPK1P = chem.CoToN(0.004256, vol)
		cs.Cyt.Ca[3].CaM_DAPK1 = chem.CoToN(1.122e-05, vol)
		cs.Cyt.Ca[3].CaM_DAPK1P = chem.CoToN(0.001489, vol)
		cs.Cyt.N2B[0].CaM_DAPK1 = chem.CoToN(0, vol)
		cs.Cyt.N2B[0].CaM_DAPK1P = chem.CoToN(0, vol)
		cs.Cyt.N2B[1].CaM_DAPK1 = chem.CoToN(0, vol)
		cs.Cyt.N2B[1].CaM_DAPK1P = chem.CoToN(0, vol)
		cs.Cyt.N2B[2].CaM_DAPK1 = chem.CoToN(0, vol)
		cs.Cyt.N2B[2].CaM_DAPK1P = chem.CoToN(0, vol)
		cs.Cyt.N2B[3].CaM_DAPK1 = chem.CoToN(0, vol)
		cs.Cyt.N2B[3].CaM_DAPK1P = chem.CoToN(0, vol)
		cs.Cyt.DAPK1 = chem.CoToN(1.255, vol)
		cs.Cyt.DAPK1P = chem.CoToN(18.07, vol)
		cs.Cyt.N2B_DAPK1 = chem.CoToN(0, vol)
		cs.Cyt.N2B_DAPK1P = chem.CoToN(0, vol)
		cs.Cyt.CaNSer308C = chem.CoToN(0.0009476, vol)
		cs.Cyt.PP2ASer308C = chem.CoToN(0.03036, vol)
		vol = PSDVol
		cs.PSD.Ca[0].CaM_DAPK1 = chem.CoToN(0.001153, vol)
		cs.PSD.Ca[0].CaM_DAPK1P = chem.CoToN(0.604, vol)
		cs.PSD.Ca[1].CaM_DAPK1 = chem.CoToN(2.148e-05, vol)
		cs.PSD.Ca[1].CaM_DAPK1P = chem.CoToN(0.03418, vol)
		cs.PSD.Ca[2].CaM_DAPK1 = chem.CoToN(2.605e-06, vol)
		cs.PSD.Ca[2].CaM_DAPK1P = chem.CoToN(0.004257, vol)
		cs.PSD.Ca[3].CaM_DAPK1 = chem.CoToN(7.185e-06, vol)
		cs.PSD.Ca[3].CaM_DAPK1P = chem.CoToN(0.001489, vol)
		cs.PSD.N2B[0].CaM_DAPK1 = chem.CoToN(0.1105, vol)
		cs.PSD.N2B[0].CaM_DAPK1P = chem.CoToN(6.977e-05, vol)
		cs.PSD.N2B[1].CaM_DAPK1 = chem.CoToN(0.001436, vol)
		cs.PSD.N2B[1].CaM_DAPK1P = chem.CoToN(1.537e-06, vol)
		cs.PSD.N2B[2].CaM_DAPK1 = chem.CoToN(1.231e-05, vol)
		cs.PSD.N2B[2].CaM_DAPK1P = chem.CoToN(6.427e-07, vol)
		cs.PSD.N2B[3].CaM_DAPK1 = chem.CoToN(4.944e-06, vol)
		cs.PSD.N2B[3].CaM_DAPK1P = chem.CoToN(6.374e-07, vol)
		cs.PSD.DAPK1 = chem.CoToN(1.26, vol)
		cs.PSD.DAPK1P = chem.CoToN(18.07, vol)
		cs.PSD.N2B_DAPK1 = chem.CoToN(0.01117, vol)
		cs.PSD.N2B_DAPK1P = chem.CoToN(7.047e-07, vol)
		cs.PSD.CaNSer308C = chem.CoToN(0.0004738, vol)
		cs.PSD.PP2ASer308C = chem.CoToN(0.1214, vol)
	}
	cs.Cyt.ActiveK()
	cs.PSD.ActiveK()
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
type DAPK1Params struct {
	CaCaM01      chem.React `desc:"1: Ca+CaM-DAPK1 -> 1CaCaM-DAPK1 = CaM-bind-Ca"`
	CaCaM12      chem.React `desc:"2: Ca+1CaM-DAPK1 -> 2CaCaM-DAPK1 = CaMCa-bind-Ca"`
	CaCaM23      chem.React `desc:"6: Ca+2CaCaM-DAPK1 -> 3CaCaM-DAPK1 = CaMCa2-bind-Ca"`
	CaMDAPK1     chem.React `desc:"4: CaM+DAPK1 -> CaM-DAPK1 [0] -- raw CaM w/out Ca doesn't bind strongly"`
	CaMDAPK1P    chem.React `desc:"9: CaM+DAPK1P -> CaM-DAPK1P = kAT_kTA"`
	CaMDAPK13    chem.React `desc:"5: 3CaCaM+DAPK1 -> 3CaCaM-DAPK1 = kIB_kBI_3 -- active CaM binds strongly"`
	CaCaM_DAPK1P chem.React `desc:"8: Ca+nCaCaM-DAPK1P -> n+1CaCaM-DAPK1P = kTP_PT_*"`

	GluN2BNoPCaCaM chem.React `desc:"GluN2B S1303 binding for noP with Ca/CaM bound"`
	GluN2BP        chem.React `desc:"GluN2B binding for DAPK1P with any level of CaM -- no affinity, fast decay"`
	GluN2BNoP      chem.React `desc:"GluN2B binding for DAPK1 non-P, no Ca/CaM: no affinity, slow decay"`

	CaNSer308  chem.Enz `desc:"10: CaN dephosphorylating DAPK1P"`
	PP2ASer308 chem.Enz `desc:"11: PP2A dephosphorylating DAPK1P"`

	DAPK1Diffuse chem.Diffuse `desc:"DAPK1 symmetric diffusion between Cyt and PSD -- only for non-N2B bound or WI for non-N2B case"`
}

func (cp *DAPK1Params) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// See React docs for more info
	cp.CaCaM01.SetVol(51.202, CytVol, 200) // 1: 51.202 μM-1 = 1.0667, PSD 4.2667 = CaM-bind-Ca
	cp.CaCaM12.SetVol(133.3, CytVol, 1000) // 2: 133.3 μM-1 = 2.7771, PSD 11.108 = CaMCa-bind-Ca
	cp.CaCaM23.SetVol(25.6, CytVol, 400)   // 6 No N2B: 25.6 μM-1 = 0.53333, PSD 2.1333 = CaMCa2-bind-Ca

	cp.CaMDAPK1.SetVol(0.0004, CytVol, 1)  // 4: raw CaM with no Ca, binding to noP
	cp.CaMDAPK1P.SetVol(0.0004, CytVol, 1) // 9: low binding of CaM to P
	cp.CaMDAPK13.SetVol(400, CytVol, 1)    // 5: 8 μM-1 = 0.16667, PSD 3.3333e-5 = kIB_kBI_3 -- 3CaCaM is active

	cp.CaCaM_DAPK1P.SetVol(1, CytVol, 1) // 8: 1 μM-1 = 0.020834, PSD 0.0833335 = kTP_PT_*

	// GluN2B binding
	cp.GluN2BNoPCaCaM.SetVol(40, PSDVol, 1) // high affinity -- key question..
	cp.GluN2BNoP.SetVol(0, PSDVol, 10)      // no affinity, slow decay?
	cp.GluN2BP.SetVol(0, PSDVol, 1000)      // DAPK1P -- unbind if P -- unlike CaMKII does not stick around

	cp.CaNSer308.SetKmVol(11, CytVol, 1.34, 0.335)  // 10: 11 μM Km = 0.0031724
	cp.PP2ASer308.SetKmVol(11, CytVol, 1.34, 0.335) // 11: 11 μM Km = 0.0031724

	cp.DAPK1Diffuse.SetSym(6.0 / 0.0225)
}

// StepDAPK1 does the bulk of Ca + CaM + DAPK1 binding reactions, in a given region
// cCa, nCa = current next Ca
func (cp *DAPK1Params) StepDAPK1(vol float64, c, d *DAPK1Vars, cm, dm *CaMVars, cCa, can, pp2a, cGluN2B float64, dCa, dcan, dpp2a, dGluN2B *float64) {
	kf := CytVol / vol

	psd := vol == PSDVol

	// basic Ca binding to CaM
	cp.CaCaM01.StepK(kf, c.Ca[0].CaM_DAPK1, cCa, c.Ca[1].CaM_DAPK1, &d.Ca[0].CaM_DAPK1, dCa, &d.Ca[1].CaM_DAPK1) // 1
	cp.CaCaM12.StepK(kf, c.Ca[1].CaM_DAPK1, cCa, c.Ca[2].CaM_DAPK1, &d.Ca[1].CaM_DAPK1, dCa, &d.Ca[2].CaM_DAPK1) // 2
	cp.CaCaM23.StepK(kf, c.Ca[2].CaM_DAPK1, cCa, c.Ca[3].CaM_DAPK1, &d.Ca[2].CaM_DAPK1, dCa, &d.Ca[3].CaM_DAPK1) // 6 non

	for i := 0; i < 3; i++ {
		cp.CaMDAPK1.StepK(kf, cm.CaM[i], c.DAPK1, c.Ca[i].CaM_DAPK1, &dm.CaM[i], &d.DAPK1, &d.Ca[i].CaM_DAPK1)                    // 4
		cp.CaCaM_DAPK1P.StepK(kf, c.Ca[i].CaM_DAPK1P, cCa, c.Ca[i+1].CaM_DAPK1P, &d.Ca[i].CaM_DAPK1P, dCa, &d.Ca[i+1].CaM_DAPK1P) // 8
	}
	cp.CaMDAPK13.StepK(kf, cm.CaM[3], c.DAPK1, c.Ca[3].CaM_DAPK1, &dm.CaM[3], &d.DAPK1, &d.Ca[3].CaM_DAPK1) // 5 -- 3CaCaM is active form

	cp.CaMDAPK1P.StepK(kf, cm.CaM[0], c.DAPK1P, c.Ca[0].CaM_DAPK1P, &dm.CaM[0], &d.DAPK1P, &d.Ca[0].CaM_DAPK1P) // 9

	if psd {
		cp.CaCaM01.StepK(kf, c.N2B[0].CaM_DAPK1, cCa, c.N2B[1].CaM_DAPK1, &d.N2B[0].CaM_DAPK1, dCa, &d.N2B[1].CaM_DAPK1) // 1
		cp.CaCaM12.StepK(kf, c.N2B[1].CaM_DAPK1, cCa, c.N2B[2].CaM_DAPK1, &d.N2B[1].CaM_DAPK1, dCa, &d.N2B[2].CaM_DAPK1) // 2
		cp.CaCaM23.StepK(kf, c.N2B[2].CaM_DAPK1, cCa, c.N2B[3].CaM_DAPK1, &d.N2B[2].CaM_DAPK1, dCa, &d.N2B[3].CaM_DAPK1) // 6 reg

		for i := 0; i < 3; i++ {
			cp.CaMDAPK1.StepK(kf, cm.CaM[i], c.N2B_DAPK1, c.N2B[i].CaM_DAPK1, &dm.CaM[i], &d.N2B_DAPK1, &d.N2B[i].CaM_DAPK1)              // 4 reg
			cp.CaCaM_DAPK1P.StepK(kf, c.N2B[i].CaM_DAPK1P, cCa, c.N2B[i+1].CaM_DAPK1P, &d.N2B[i].CaM_DAPK1P, dCa, &d.N2B[i+1].CaM_DAPK1P) // 8 same..
		}
		cp.CaMDAPK13.StepK(kf, cm.CaM[3], c.N2B_DAPK1, c.N2B[3].CaM_DAPK1, &dm.CaM[3], &d.N2B_DAPK1, &d.N2B[3].CaM_DAPK1)     // 5 -- 3CaCaM is active form
		cp.CaMDAPK1P.StepK(kf, cm.CaM[0], c.N2B_DAPK1P, c.N2B[0].CaM_DAPK1P, &dm.CaM[0], &d.N2B_DAPK1P, &d.N2B[0].CaM_DAPK1P) // 9 same (P drives same accel CaM binding as N2B)
	}

	// cs, ce, cc, cp -> ds, de, dc, dp
	cp.CaNSer308.StepK(kf, c.DAPK1P, can, c.CaNSer308C, c.DAPK1, &d.DAPK1P, dcan, &d.CaNSer308C, &d.DAPK1) // 10

	if dpp2a != nil {
		cp.PP2ASer308.StepK(kf, c.DAPK1P, pp2a, c.PP2ASer308C, c.DAPK1, &d.DAPK1P, dpp2a, &d.PP2ASer308C, &d.DAPK1) // 11
	}

	for i := 0; i < 4; i++ {
		cc := &c.Ca[i]
		dc := &d.Ca[i]
		dak := c.Auto.K * cc.CaM_DAPK1 // no interaction with N2B here..
		dc.CaM_DAPK1P += dak
		dc.CaM_DAPK1 -= dak
		// cs, ce, cc, cp -> ds, de, dc, dp
		cp.CaNSer308.StepK(kf, cc.CaM_DAPK1P, can, c.CaNSer308C, cc.CaM_DAPK1, &dc.CaM_DAPK1P, dcan, &d.CaNSer308C, &dc.CaM_DAPK1) // 10
	}

	if psd {
		cp.CaNSer308.StepK(kf, c.N2B_DAPK1P, can, c.CaNSer308C, c.N2B_DAPK1, &d.N2B_DAPK1P, dcan, &d.CaNSer308C, &d.N2B_DAPK1) // 10 same..

		// GluN2B binding
		cp.GluN2BP.Step(c.DAPK1P, cGluN2B, c.N2B_DAPK1P, &d.DAPK1P, dGluN2B, &d.N2B_DAPK1P)
		cp.GluN2BNoP.Step(c.DAPK1, cGluN2B, c.N2B_DAPK1, &d.DAPK1, dGluN2B, &d.N2B_DAPK1)

		for i := 0; i < 4; i++ { // note: currently using same for all..
			cc := &c.N2B[i]
			dc := &d.N2B[i]
			dak := c.Auto.K * cc.CaM_DAPK1
			dc.CaM_DAPK1P += dak
			dc.CaM_DAPK1 -= dak

			// cs, ce, cc, cp -> ds, de, dc, dp
			cp.CaNSer308.StepK(kf, cc.CaM_DAPK1P, can, c.CaNSer308C, cc.CaM_DAPK1, &dc.CaM_DAPK1P, dcan, &d.CaNSer308C, &dc.CaM_DAPK1) // 10

			// GluN2B binding
			cp.GluN2BNoPCaCaM.Step(c.Ca[i].CaM_DAPK1, cGluN2B, cc.CaM_DAPK1, &d.Ca[i].CaM_DAPK1, dGluN2B, &dc.CaM_DAPK1)
			cp.GluN2BP.Step(c.Ca[i].CaM_DAPK1P, cGluN2B, cc.CaM_DAPK1P, &d.Ca[i].CaM_DAPK1P, dGluN2B, &dc.CaM_DAPK1P)
		}
	}
}

// StepDiffuse does Cyt <-> PSD diffusion
func (cp *DAPK1Params) StepDiffuse(c, d *DAPK1State) {
	// Note: N2B bound by definition does not move at all from PSD
	// and all these use the symmetric form of diffusion for reg non-N2B
	for i := 0; i < 4; i++ {
		cc := &c.Cyt.Ca[i]
		cd := &c.PSD.Ca[i]
		dc := &d.Cyt.Ca[i]
		dd := &d.PSD.Ca[i]
		cp.DAPK1Diffuse.Step(cc.CaM_DAPK1, cd.CaM_DAPK1, CytVol, PSDVol, &dc.CaM_DAPK1, &dd.CaM_DAPK1)
		cp.DAPK1Diffuse.Step(cc.CaM_DAPK1P, cd.CaM_DAPK1P, CytVol, PSDVol, &dc.CaM_DAPK1P, &dd.CaM_DAPK1P)
	}

	cp.DAPK1Diffuse.Step(c.Cyt.DAPK1, c.PSD.DAPK1, CytVol, PSDVol, &d.Cyt.DAPK1, &d.PSD.DAPK1)
	cp.DAPK1Diffuse.Step(c.Cyt.DAPK1P, c.PSD.DAPK1P, CytVol, PSDVol, &d.Cyt.DAPK1P, &d.PSD.DAPK1P)
}

// Step does one step of DAPK1 updating, c=current, d=delta
// pp2a = current cyt pp2a
func (cp *DAPK1Params) Step(c, d *DAPK1State, cm, dm *CaMState, cCa, dCa *CaState, can, dcan *CaNState, pp2a, cGluN2B float64, dpp2a, dGluN2B *float64) {
	cp.StepDAPK1(CytVol, &c.Cyt, &d.Cyt, &cm.Cyt, &dm.Cyt, cCa.Cyt, can.Cyt.CaNact, pp2a, cGluN2B, &dCa.Cyt, &dcan.Cyt.CaNact, dpp2a, dGluN2B)
	cp.StepDAPK1(PSDVol, &c.PSD, &d.PSD, &cm.PSD, &dm.PSD, cCa.PSD, can.PSD.CaNact, pp2a, cGluN2B, &dCa.PSD, &dcan.PSD.CaNact, dpp2a, dGluN2B)
	cp.StepDiffuse(c, d)
}
