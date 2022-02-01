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
	N2B   float64 `desc:"total N2B bound CaMKII"`
}

func (av *AutoPVars) Zero() {
	av.Act = 0
	av.Total = 0
	av.K = 0
	av.N2B = 0
}

// CaMKIIVars are intracellular Ca-driven signaling states
// for CaMKII binding and phosphorylation with CaM + Ca
// Dupont = DupontHouartDekonnick03, has W* terms used in Genesis code
// stores N values -- Co = Concentration computed by volume as needed
type CaMKIIVars struct {
	Ca          [4]CaCaMKIIVars `desc:"increasing levels of Ca binding, 0-3"`
	N2B         [4]CaCaMKIIVars `desc:"GluN2B bound, increasing levels of Ca binding, 0-3"`
	CaMKII      float64         `desc:"unbound CaMKII = CaM kinase II -- WI in Dupont -- this is the inactive form for NMDA GluN2B binding"`
	CaMKIIP     float64         `desc:"unbound CaMKII P = phosphorylated at Thr286 -- shown with * in Figure S13 = WA in Dupont -- this is the active form for NMDA GluN2B binding"`
	N2B_CaMKII  float64         `desc:"CaMKII bound to NMDA N2B (only for PSD compartment) -- only exists as de-P of bound P"`
	N2B_CaMKIIP float64         `desc:"CaMKIIP bound to NMDA N2B (only for PSD compartment)"`
	PP1Thr286C  float64         `desc:"PP1+CaMKIIP complex for PP1Thr286 enzyme reaction"`
	PP2AThr286C float64         `desc:"PP2A+CaMKIIP complex for PP2AThr286 enzyme reaction"`
	Auto        AutoPVars       `view:"inline" inactive:"+" desc:"auto-phosphorylation state"`
}

func (cs *CaMKIIVars) Init(vol float64) {
	for i := range cs.Ca {
		cs.Ca[i].Init(vol)
	}
	for i := range cs.N2B {
		cs.N2B[i].Zero()
	}
	cs.CaMKII = chem.CoToN(20, vol)
	cs.CaMKIIP = 0 // WA
	cs.N2B_CaMKII = 0
	cs.N2B_CaMKIIP = 0
	cs.PP1Thr286C = 0
	cs.PP2AThr286C = 0

	if TheOpts.InitBaseline {
		cs.CaMKII = chem.CoToN(19.28, vol) // orig: 20
	}

	cs.ActiveK()
}

// Generate Code for Initializing
func (cs *CaMKIIVars) InitCode(vol float64, pre string) {
	for i := range cs.Ca {
		fmt.Printf("\tcs.%s.Ca[%d].CaM_CaMKII = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.Ca[i].CaM_CaMKII, vol))
		fmt.Printf("\tcs.%s.Ca[%d].CaM_CaMKIIP = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.Ca[i].CaM_CaMKIIP, vol))
	}
	for i := range cs.N2B {
		fmt.Printf("\tcs.%s.N2B[%d].CaM_CaMKII = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.N2B[i].CaM_CaMKII, vol))
		fmt.Printf("\tcs.%s.N2B[%d].CaM_CaMKIIP = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.N2B[i].CaM_CaMKIIP, vol))
	}
	fmt.Printf("\tcs.%s.CaMKII = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.CaMKII, vol))
	fmt.Printf("\tcs.%s.CaMKIIP = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.CaMKIIP, vol))
	fmt.Printf("\tcs.%s.N2B_CaMKII = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.N2B_CaMKII, vol))
	fmt.Printf("\tcs.%s.N2B_CaMKIIP = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.N2B_CaMKIIP, vol))
	fmt.Printf("\tcs.%s.PP1Thr286C = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.PP1Thr286C, vol))
	fmt.Printf("\tcs.%s.PP2AThr286C = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(cs.PP2AThr286C, vol))
}

func (cs *CaMKIIVars) Zero() {
	for i := range cs.Ca {
		cs.Ca[i].Zero()
	}
	for i := range cs.N2B {
		cs.N2B[i].Zero()
	}
	cs.CaMKII = 0
	cs.CaMKIIP = 0
	cs.N2B_CaMKII = 0
	cs.N2B_CaMKIIP = 0
	cs.PP1Thr286C = 0
	cs.PP2AThr286C = 0
	cs.Auto.Zero()
}

func (cs *CaMKIIVars) Integrate(d *CaMKIIVars) {
	for i := range cs.Ca {
		cs.Ca[i].Integrate(&d.Ca[i])
	}
	for i := range cs.N2B {
		cs.N2B[i].Integrate(&d.N2B[i])
	}
	chem.Integrate(&cs.CaMKII, d.CaMKII)
	chem.Integrate(&cs.CaMKIIP, d.CaMKIIP)
	chem.Integrate(&cs.N2B_CaMKII, d.N2B_CaMKII)
	chem.Integrate(&cs.N2B_CaMKIIP, d.N2B_CaMKIIP)
	chem.Integrate(&cs.PP1Thr286C, d.PP1Thr286C)
	chem.Integrate(&cs.PP2AThr286C, d.PP2AThr286C)
	cs.ActiveK()
}

// ActiveK updates active, total, and the Auto.K auto-phosphorylation rate constant
// Code is from genesis_customizing/T286Phos/T286Phos.c and would be impossible to
// reconstruct without that source (my first guess was wildy off, based only on
// the supplement)
func (cs *CaMKIIVars) ActiveK() {
	WI := cs.CaMKII + cs.N2B_CaMKII
	WA := cs.CaMKIIP + cs.N2B_CaMKIIP
	n2b := cs.N2B_CaMKII + cs.N2B_CaMKIIP

	var WB, WT float64

	for i := 0; i < 3; i++ {
		WB += cs.Ca[i].CaM_CaMKII + cs.N2B[i].CaM_CaMKII
		WT += cs.Ca[i].CaM_CaMKIIP + cs.N2B[i].CaM_CaMKIIP
		n2b += cs.N2B[i].CaM_CaMKII + cs.N2B[i].CaM_CaMKIIP
	}
	WB += cs.Ca[3].CaM_CaMKII + cs.N2B[3].CaM_CaMKII
	WP := cs.Ca[3].CaM_CaMKIIP + cs.N2B[3].CaM_CaMKIIP
	n2b += cs.N2B[3].CaM_CaMKII + cs.N2B[3].CaM_CaMKIIP

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
	cs.Auto.N2B = n2b
}

func (cs *CaMKIIVars) Log(dt *etable.Table, vol float64, row int, pre string) {
	dt.SetCellFloat(pre+"CaMKIIact", row, chem.CoFmN(cs.Auto.Act, vol))
	// dt.SetCellFloat(pre+"Ca0CaM_CaMKII", row, chem.CoFmN(cs.Ca[0].CaM_CaMKII, vol))
	// dt.SetCellFloat(pre+"Ca1CaM_CaMKII", row, chem.CoFmN(cs.Ca[1].CaM_CaMKII, vol))
	// dt.SetCellFloat(pre+"Ca0CaM_CaMKIIP", row, chem.CoFmN(cs.Ca[0].CaM_CaMKIIP, vol))
	// dt.SetCellFloat(pre+"Ca1CaM_CaMKIIP", row, chem.CoFmN(cs.Ca[1].CaM_CaMKIIP, vol))
	dt.SetCellFloat(pre+"CaMKII", row, chem.CoFmN(cs.CaMKII, vol))
	dt.SetCellFloat(pre+"CaMKIIP", row, chem.CoFmN(cs.CaMKIIP, vol))
	// dt.SetCellFloat(pre+"CaMKII_AutoK", row, chem.CoFmN(cs.Auto.K, vol))
	if pre == "PSD_" {
		dt.SetCellFloat(pre+"CaMKIIn2b", row, chem.CoFmN(cs.Auto.N2B, vol))
		// dt.SetCellFloat(pre+"N2B_Ca0CaM_CaMKII", row, chem.CoFmN(cs.N2B[0].CaM_CaMKII, vol))
		// dt.SetCellFloat(pre+"N2B_Ca1CaM_CaMKII", row, chem.CoFmN(cs.N2B[1].CaM_CaMKII, vol))
		// dt.SetCellFloat(pre+"N2B_Ca0CaM_CaMKIIP", row, chem.CoFmN(cs.N2B[0].CaM_CaMKIIP, vol))
		// dt.SetCellFloat(pre+"N2B_Ca1CaM_CaMKIIP", row, chem.CoFmN(cs.N2B[1].CaM_CaMKIIP, vol))
		dt.SetCellFloat(pre+"N2B_CaMKII", row, chem.CoFmN(cs.N2B_CaMKII, vol))
		dt.SetCellFloat(pre+"N2B_CaMKIIP", row, chem.CoFmN(cs.N2B_CaMKIIP, vol))
	}
}

func (cs *CaMKIIVars) ConfigLog(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "CaMKIIact", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca0CaM_CaMKII", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca1CaM_CaMKII", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca0CaM_CaMKIIP", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca1CaM_CaMKIIP", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "CaMKII", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "CaMKIIP", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "CaMKII_AutoK", etensor.FLOAT64, nil, nil})
	if pre == "PSD_" {
		*sch = append(*sch, etable.Column{pre + "CaMKIIn2b", etensor.FLOAT64, nil, nil})
		// *sch = append(*sch, etable.Column{pre + "N2B_Ca0CaM_CaMKII", etensor.FLOAT64, nil, nil})
		// *sch = append(*sch, etable.Column{pre + "N2B_Ca1CaM_CaMKII", etensor.FLOAT64, nil, nil})
		// *sch = append(*sch, etable.Column{pre + "N2B_Ca0CaM_CaMKIIP", etensor.FLOAT64, nil, nil})
		// *sch = append(*sch, etable.Column{pre + "N2B_Ca1CaM_CaMKIIP", etensor.FLOAT64, nil, nil})
		*sch = append(*sch, etable.Column{pre + "N2B_CaMKII", etensor.FLOAT64, nil, nil})
		*sch = append(*sch, etable.Column{pre + "N2B_CaMKIIP", etensor.FLOAT64, nil, nil})
	}
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

	if TheOpts.InitBaseline {
		if TheOpts.UseDAPK1 {
			vol := float64(CytVol)
			cs.Cyt.Ca[0].CaM_CaMKII = chem.CoToN(0.1345, vol)
			cs.Cyt.Ca[0].CaM_CaMKIIP = chem.CoToN(4.264e-05, vol)
			cs.Cyt.Ca[1].CaM_CaMKII = chem.CoToN(0.001728, vol)
			cs.Cyt.Ca[1].CaM_CaMKIIP = chem.CoToN(1.139e-05, vol)
			cs.Cyt.Ca[2].CaM_CaMKII = chem.CoToN(1.277e-05, vol)
			cs.Cyt.Ca[2].CaM_CaMKIIP = chem.CoToN(1.002e-05, vol)
			cs.Cyt.Ca[3].CaM_CaMKII = chem.CoToN(3.192e-06, vol)
			cs.Cyt.Ca[3].CaM_CaMKIIP = chem.CoToN(8.221e-06, vol)
			cs.Cyt.N2B[0].CaM_CaMKII = chem.CoToN(0, vol)
			cs.Cyt.N2B[0].CaM_CaMKIIP = chem.CoToN(0, vol)
			cs.Cyt.N2B[1].CaM_CaMKII = chem.CoToN(0, vol)
			cs.Cyt.N2B[1].CaM_CaMKIIP = chem.CoToN(0, vol)
			cs.Cyt.N2B[2].CaM_CaMKII = chem.CoToN(0, vol)
			cs.Cyt.N2B[2].CaM_CaMKIIP = chem.CoToN(0, vol)
			cs.Cyt.N2B[3].CaM_CaMKII = chem.CoToN(0, vol)
			cs.Cyt.N2B[3].CaM_CaMKIIP = chem.CoToN(0, vol)
			cs.Cyt.CaMKII = chem.CoToN(18.86, vol)
			cs.Cyt.CaMKIIP = chem.CoToN(6.744e-09, vol)
			cs.Cyt.N2B_CaMKII = chem.CoToN(0, vol)
			cs.Cyt.N2B_CaMKIIP = chem.CoToN(0, vol)
			cs.Cyt.PP1Thr286C = chem.CoToN(1.005e-08, vol)
			cs.Cyt.PP2AThr286C = chem.CoToN(1.132e-11, vol)
			vol = PSDVol
			cs.PSD.Ca[0].CaM_CaMKII = chem.CoToN(0.04955, vol)
			cs.PSD.Ca[0].CaM_CaMKIIP = chem.CoToN(4.023e-05, vol)
			cs.PSD.Ca[1].CaM_CaMKII = chem.CoToN(0.0006401, vol)
			cs.PSD.Ca[1].CaM_CaMKIIP = chem.CoToN(1.136e-05, vol)
			cs.PSD.Ca[2].CaM_CaMKII = chem.CoToN(5.612e-06, vol)
			cs.PSD.Ca[2].CaM_CaMKIIP = chem.CoToN(1.033e-05, vol)
			cs.PSD.Ca[3].CaM_CaMKII = chem.CoToN(3.242e-06, vol)
			cs.PSD.Ca[3].CaM_CaMKIIP = chem.CoToN(9.615e-06, vol)
			cs.PSD.N2B[0].CaM_CaMKII = chem.CoToN(4.031, vol)
			cs.PSD.N2B[0].CaM_CaMKIIP = chem.CoToN(0.8687, vol)
			cs.PSD.N2B[1].CaM_CaMKII = chem.CoToN(0.05161, vol)
			cs.PSD.N2B[1].CaM_CaMKIIP = chem.CoToN(0.04602, vol)
			cs.PSD.N2B[2].CaM_CaMKII = chem.CoToN(0.0003437, vol)
			cs.PSD.N2B[2].CaM_CaMKIIP = chem.CoToN(0.004259, vol)
			cs.PSD.N2B[3].CaM_CaMKII = chem.CoToN(0.0007132, vol)
			cs.PSD.N2B[3].CaM_CaMKIIP = chem.CoToN(0.001191, vol)
			cs.PSD.CaMKII = chem.CoToN(18.95, vol)
			cs.PSD.CaMKIIP = chem.CoToN(7.755e-07, vol)
			cs.PSD.N2B_CaMKII = chem.CoToN(0.002476, vol)
			cs.PSD.N2B_CaMKIIP = chem.CoToN(2.156e-06, vol)
			cs.PSD.PP1Thr286C = chem.CoToN(0.000384, vol)
			cs.PSD.PP2AThr286C = chem.CoToN(5.204e-09, vol)
		} else {
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
	cs.Cyt.ActiveK()
	cs.PSD.ActiveK()
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
	CaCaM23_N2B   chem.React `desc:"6: for N2B Ca+2CaCaM-CaMKII -> 3CaCaM-CaMKII = CaMCa2-bind-Ca"`
	CaMCaMKII     chem.React `desc:"4: CaM+CaMKII -> CaM-CaMKII [0-2] -- kIB_kBI_[0-2] -- WI = plain CaMKII, WBn = CaM bound"`
	CaMCaMKII_N2B chem.React `desc:"4: for N2B: CaM+CaMKII -> CaM-CaMKII [0-2] -- kIB_kBI_[0-2] -- WI = plain CaMKII, WBn = CaM bound -- Not much N2B_CaMKII (noP) so not a big factor"`
	CaMCaMKII3    chem.React `desc:"5: 3CaCaM+CaMKII -> 3CaCaM-CaMKII = kIB_kBI_3 -- active CaM binds strongly"`
	CaMCaMKIIP    chem.React `desc:"9: CaM+CaMKIIP -> CaM-CaMKIIP = kAT_kTA -- T286P causes strong CaM binding"`
	CaCaM_CaMKIIP chem.React `desc:"8: Ca+nCaCaM-CaMKIIP -> n+1CaCaM-CaMKIIP = kTP_PT_*"`

	GluN2BCaCaM chem.React `desc:"GluN2B S1303 binding for any level of Ca/CaM, P or noP"`
	GluN2BP     chem.React `desc:"GluN2B binding for CaMKIIP without any Ca/CaM -- maintains but does not attract"`
	GluN2BNoP   chem.React `desc:"GluN2B binding for CaMKII non-P, no Ca/CaM: no affinity"`

	PP1Thr286  chem.Enz `desc:"10: PP1 dephosphorylating CaMKIIP"`
	PP2AThr286 chem.Enz `desc:"11: PP2A dephosphorylating CaMKIIP"`

	CaMKIIDiffuse  chem.Diffuse `desc:"CaMKII symmetric diffusion between Cyt and PSD -- only for non-N2B bound or WI for non-N2B case"`
	CaMKIIPDiffuse chem.Diffuse `desc:"CaMKIIP diffusion between Cyt and PSD -- asymmetric, everything else, only when not using N2B binding"`
}

func (cp *CaMKIIParams) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// See React docs for more info
	cp.CaCaM01.SetVol(51.202, CytVol, 200)    // 1: 51.202 μM-1 = 1.0667, PSD 4.2667 = CaM-bind-Ca
	cp.CaCaM12.SetVol(133.3, CytVol, 1000)    // 2: 133.3 μM-1 = 2.7771, PSD 11.108 = CaMCa-bind-Ca
	cp.CaCaM23.SetVol(25.6, CytVol, 400)      // 6 No N2B: 25.6 μM-1 = 0.53333, PSD 2.1333 = CaMCa2-bind-Ca
	cp.CaCaM23_N2B.SetVol(25.6, CytVol, 0.02) // 6 N2B: 25.6 μM-1 = 0.53333, PSD 2.1333 = CaMCa2-bind-Ca

	cp.CaMCaMKII.SetVol(0.0004, CytVol, 1) // 4: 0.0004 μM-1 = 8.3333e-6, PSD 3.3333e-5 = kIB_kBI_[0-2]
	cp.CaMCaMKII_N2B.SetVol(8, CytVol, 1)  // 4: N2B causes CaM to bind like T286P = 9 -- todo kB = 0.001 or 1?
	cp.CaMCaMKII3.SetVol(8, CytVol, 1)     // 5: 8 μM-1 = 0.16667, PSD 3.3333e-5 = kIB_kBI_3 -- 3CaCaM is active
	cp.CaMCaMKIIP.SetVol(8, CytVol, 0.001) // 9: 8 μM-1 = 0.16667, PSD 0.66667 = kAT_kTA

	cp.CaCaM_CaMKIIP.SetVol(1, CytVol, 1) // 8: 1 μM-1 = 0.020834, PSD 0.0833335 = kTP_PT_*

	// GluN2B binding
	cp.GluN2BCaCaM.SetVol(8, PSDVol, 0.001) // high affinity
	cp.GluN2BP.SetVol(0, PSDVol, 0.0001)    // CaMKIIP -- don't go away, but not attracted either
	cp.GluN2BNoP.SetVol(0, PSDVol, 1000)    // CaMKII -- skidaddle

	cp.PP1Thr286.SetKmVol(11, CytVol, 1.34, 0.335)  // 10: 11 μM Km = 0.0031724
	cp.PP2AThr286.SetKmVol(11, CytVol, 1.34, 0.335) // 11: 11 μM Km = 0.0031724

	cp.CaMKIIDiffuse.SetSym(6.0 / 0.0225)
	cp.CaMKIIPDiffuse.Set(6.0/0.0225, 0.6/0.0225)
}

////////////////////////////////////////////////////////////////////
// No N2B versions

// StepCaMKII does the bulk of Ca + CaM + CaMKII binding reactions, in a given region
// cCa, nCa = current next Ca
func (cp *CaMKIIParams) StepCaMKII(vol float64, c, d *CaMKIIVars, cm, dm *CaMVars, cCa, pp1, pp2a float64, dCa, dpp1, dpp2a *float64) {
	kf := CytVol / vol

	cp.CaCaM01.StepK(kf, c.Ca[0].CaM_CaMKII, cCa, c.Ca[1].CaM_CaMKII, &d.Ca[0].CaM_CaMKII, dCa, &d.Ca[1].CaM_CaMKII) // 1
	cp.CaCaM12.StepK(kf, c.Ca[1].CaM_CaMKII, cCa, c.Ca[2].CaM_CaMKII, &d.Ca[1].CaM_CaMKII, dCa, &d.Ca[2].CaM_CaMKII) // 2

	cp.CaCaM23_N2B.StepK(kf, c.Ca[2].CaM_CaMKII, cCa, c.Ca[3].CaM_CaMKII, &d.Ca[2].CaM_CaMKII, dCa, &d.Ca[3].CaM_CaMKII) // 6 all N2B

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
	cp.CaMKIIPDiffuse.Step(c.Cyt.CaMKIIP, c.PSD.CaMKIIP, CytVol, PSDVol, &d.Cyt.CaMKIIP, &d.PSD.CaMKIIP) // P = N2B
}

////////////////////////////////////////////////////////////////////
// N2B versions

// StepCaMKIIN2B does the bulk of Ca + CaM + CaMKII binding reactions, in a given region.
// N2B version applies non-N2B rates for non-N2B bound, N2B for bound in PSD region only.
// PSD has both so one func needs to handle Cyt and PSD cases.
// cCa, nCa = current next Ca
func (cp *CaMKIIParams) StepCaMKIIN2B(vol float64, c, d *CaMKIIVars, cm, dm *CaMVars, cCa, pp1, pp2a, cGluN2B float64, dCa, dpp1, dpp2a, dGluN2B *float64) {
	kf := CytVol / vol

	psd := vol == PSDVol

	// basic Ca binding to CaM
	cp.CaCaM01.StepK(kf, c.Ca[0].CaM_CaMKII, cCa, c.Ca[1].CaM_CaMKII, &d.Ca[0].CaM_CaMKII, dCa, &d.Ca[1].CaM_CaMKII) // 1
	cp.CaCaM12.StepK(kf, c.Ca[1].CaM_CaMKII, cCa, c.Ca[2].CaM_CaMKII, &d.Ca[1].CaM_CaMKII, dCa, &d.Ca[2].CaM_CaMKII) // 2
	cp.CaCaM23.StepK(kf, c.Ca[2].CaM_CaMKII, cCa, c.Ca[3].CaM_CaMKII, &d.Ca[2].CaM_CaMKII, dCa, &d.Ca[3].CaM_CaMKII) // 6 non

	for i := 0; i < 3; i++ {
		cp.CaMCaMKII.StepK(kf, cm.CaM[i], c.CaMKII, c.Ca[i].CaM_CaMKII, &dm.CaM[i], &d.CaMKII, &d.Ca[i].CaM_CaMKII)                    // 4
		cp.CaCaM_CaMKIIP.StepK(kf, c.Ca[i].CaM_CaMKIIP, cCa, c.Ca[i+1].CaM_CaMKIIP, &d.Ca[i].CaM_CaMKIIP, dCa, &d.Ca[i+1].CaM_CaMKIIP) // 8
	}
	cp.CaMCaMKII3.StepK(kf, cm.CaM[3], c.CaMKII, c.Ca[3].CaM_CaMKII, &dm.CaM[3], &d.CaMKII, &d.Ca[3].CaM_CaMKII)     // 5
	cp.CaMCaMKIIP.StepK(kf, cm.CaM[0], c.CaMKIIP, c.Ca[0].CaM_CaMKIIP, &dm.CaM[0], &d.CaMKIIP, &d.Ca[0].CaM_CaMKIIP) // 9

	if psd {
		cp.CaCaM01.StepK(kf, c.N2B[0].CaM_CaMKII, cCa, c.N2B[1].CaM_CaMKII, &d.N2B[0].CaM_CaMKII, dCa, &d.N2B[1].CaM_CaMKII)     // 1
		cp.CaCaM12.StepK(kf, c.N2B[1].CaM_CaMKII, cCa, c.N2B[2].CaM_CaMKII, &d.N2B[1].CaM_CaMKII, dCa, &d.N2B[2].CaM_CaMKII)     // 2
		cp.CaCaM23_N2B.StepK(kf, c.N2B[2].CaM_CaMKII, cCa, c.N2B[3].CaM_CaMKII, &d.N2B[2].CaM_CaMKII, dCa, &d.N2B[3].CaM_CaMKII) // 6 N2B

		for i := 0; i < 3; i++ {
			cp.CaMCaMKII_N2B.StepK(kf, cm.CaM[i], c.N2B_CaMKII, c.N2B[i].CaM_CaMKII, &dm.CaM[i], &d.N2B_CaMKII, &d.N2B[i].CaM_CaMKII)          // 4 N2B
			cp.CaCaM_CaMKIIP.StepK(kf, c.N2B[i].CaM_CaMKIIP, cCa, c.N2B[i+1].CaM_CaMKIIP, &d.N2B[i].CaM_CaMKIIP, dCa, &d.N2B[i+1].CaM_CaMKIIP) // 8 same..
		}
		cp.CaMCaMKII3.StepK(kf, cm.CaM[3], c.N2B_CaMKII, c.N2B[3].CaM_CaMKII, &dm.CaM[3], &d.N2B_CaMKII, &d.N2B[3].CaM_CaMKII)     // 5 same..
		cp.CaMCaMKIIP.StepK(kf, cm.CaM[0], c.N2B_CaMKIIP, c.N2B[0].CaM_CaMKIIP, &dm.CaM[0], &d.N2B_CaMKIIP, &d.N2B[0].CaM_CaMKIIP) // 9 same (P drives same accel CaM binding as N2B)
	}

	// cs, ce, cc, cp -> ds, de, dc, dp
	cp.PP1Thr286.StepK(kf, c.CaMKIIP, pp1, c.PP1Thr286C, c.CaMKII, &d.CaMKIIP, dpp1, &d.PP1Thr286C, &d.CaMKII) // 10

	if dpp2a != nil {
		cp.PP2AThr286.StepK(kf, c.CaMKIIP, pp2a, c.PP2AThr286C, c.CaMKII, &d.CaMKIIP, dpp2a, &d.PP2AThr286C, &d.CaMKII) // 11
	}

	for i := 0; i < 4; i++ {
		cc := &c.Ca[i]
		dc := &d.Ca[i]
		dak := c.Auto.K * cc.CaM_CaMKII // no interaction with N2B here..
		dc.CaM_CaMKIIP += dak
		dc.CaM_CaMKII -= dak
		// cs, ce, cc, cp -> ds, de, dc, dp
		cp.PP1Thr286.StepK(kf, cc.CaM_CaMKIIP, pp1, c.PP1Thr286C, cc.CaM_CaMKII, &dc.CaM_CaMKIIP, dpp1, &d.PP1Thr286C, &dc.CaM_CaMKII) // 10
	}

	if psd {
		cp.PP1Thr286.StepK(kf, c.N2B_CaMKIIP, pp1, c.PP1Thr286C, c.N2B_CaMKII, &d.N2B_CaMKIIP, dpp1, &d.PP1Thr286C, &d.N2B_CaMKII) // 10 same..

		// GluN2B binding
		cp.GluN2BP.Step(c.CaMKIIP, cGluN2B, c.N2B_CaMKIIP, &d.CaMKIIP, dGluN2B, &d.N2B_CaMKIIP)
		cp.GluN2BNoP.Step(c.CaMKII, cGluN2B, c.N2B_CaMKII, &d.CaMKII, dGluN2B, &d.N2B_CaMKII)

		for i := 0; i < 4; i++ { // note: currently using same for all..
			cc := &c.N2B[i]
			dc := &d.N2B[i]
			dak := c.Auto.K * cc.CaM_CaMKII
			dc.CaM_CaMKIIP += dak
			dc.CaM_CaMKII -= dak

			// cs, ce, cc, cp -> ds, de, dc, dp
			cp.PP1Thr286.StepK(kf, cc.CaM_CaMKIIP, pp1, c.PP1Thr286C, cc.CaM_CaMKII, &dc.CaM_CaMKIIP, dpp1, &d.PP1Thr286C, &dc.CaM_CaMKII) // 10

			// GluN2B binding
			cp.GluN2BCaCaM.Step(c.Ca[i].CaM_CaMKII, cGluN2B, cc.CaM_CaMKII, &d.Ca[i].CaM_CaMKII, dGluN2B, &dc.CaM_CaMKII)
			cp.GluN2BCaCaM.Step(c.Ca[i].CaM_CaMKIIP, cGluN2B, cc.CaM_CaMKIIP, &d.Ca[i].CaM_CaMKIIP, dGluN2B, &dc.CaM_CaMKIIP)
		}
	}

}

// StepDiffuseN2B does Cyt <-> PSD diffusion
func (cp *CaMKIIParams) StepDiffuseN2B(c, d *CaMKIIState) {
	// Note: N2B bound by definition does not move at all from PSD
	// and all these use the symmetric form of diffusion for reg non-N2B
	for i := 0; i < 4; i++ {
		cc := &c.Cyt.Ca[i]
		cd := &c.PSD.Ca[i]
		dc := &d.Cyt.Ca[i]
		dd := &d.PSD.Ca[i]
		cp.CaMKIIDiffuse.Step(cc.CaM_CaMKII, cd.CaM_CaMKII, CytVol, PSDVol, &dc.CaM_CaMKII, &dd.CaM_CaMKII)
		cp.CaMKIIDiffuse.Step(cc.CaM_CaMKIIP, cd.CaM_CaMKIIP, CytVol, PSDVol, &dc.CaM_CaMKIIP, &dd.CaM_CaMKIIP)
	}

	cp.CaMKIIDiffuse.Step(c.Cyt.CaMKII, c.PSD.CaMKII, CytVol, PSDVol, &d.Cyt.CaMKII, &d.PSD.CaMKII)
	cp.CaMKIIDiffuse.Step(c.Cyt.CaMKIIP, c.PSD.CaMKIIP, CytVol, PSDVol, &d.Cyt.CaMKIIP, &d.PSD.CaMKIIP)
}

// Step does one step of CaMKII updating, c=current, d=delta
// pp2a = current cyt pp2a
func (cp *CaMKIIParams) Step(c, d *CaMKIIState, cm, dm *CaMState, cCa, dCa *CaState, pp1, dpp1 *PP1State, pp2a, cGluN2B float64, dpp2a, dGluN2B *float64) {
	if TheOpts.UseN2B {
		cp.StepCaMKIIN2B(CytVol, &c.Cyt, &d.Cyt, &cm.Cyt, &dm.Cyt, cCa.Cyt, pp1.Cyt.PP1act, pp2a, cGluN2B, &dCa.Cyt, &dpp1.Cyt.PP1act, dpp2a, dGluN2B)
		cp.StepCaMKIIN2B(PSDVol, &c.PSD, &d.PSD, &cm.PSD, &dm.PSD, cCa.PSD, pp1.PSD.PP1act, pp2a, cGluN2B, &dCa.PSD, &dpp1.PSD.PP1act, dpp2a, dGluN2B)
		cp.StepDiffuseN2B(c, d)
	} else {
		cp.StepCaMKII(CytVol, &c.Cyt, &d.Cyt, &cm.Cyt, &dm.Cyt, cCa.Cyt, pp1.Cyt.PP1act, pp2a, &dCa.Cyt, &dpp1.Cyt.PP1act, dpp2a)
		cp.StepCaMKII(PSDVol, &c.PSD, &d.PSD, &cm.PSD, &dm.PSD, cCa.PSD, pp1.PSD.PP1act, 0, &dCa.PSD, &dpp1.PSD.PP1act, nil)
		cp.StepDiffuse(c, d)
	}
}
