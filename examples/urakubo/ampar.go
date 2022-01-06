// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/chem"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// AMPARVars have AMPAR Phosphorylation (Pd = phosphorylated, Dp = dephosphorylated) state.
// Two protein elements, separately Pd and Dp:
// AMPAR = AMPA receptor (GluR1), which can be Pd at Ser845 by PKA
// PDZs = PDZ domain binding proteins (e.g., SAP97 and stargazin), which bind to AMPAR
//        and are separately Pd by CaMKII -- denoted as StgP in Urakubo code
// Both can be Dp by PP1 and CaN (calcineurin)
// Variables named as P or D for the Pd or Dp state, 1st is AMPAR @ Ser845, 2nd is PDZs
type AMPARVars struct {
	DD  float64 `desc:"both dephosphorylated = Nophos"`
	PD  float64 `desc:"AMPA Ser845 phosphorylated by PKA = S845P"`
	DP  float64 `desc:"PDZs phosphorylated by CaMKII = StgP"`
	PP  float64 `desc:"both phosphorylated = S845PStgP"`
	Tot float64 `desc:"total of all phos levels"`
}

func (as *AMPARVars) Init() {
	as.Zero()
}

func (as *AMPARVars) Zero() {
	as.DD = 0
	as.PD = 0
	as.DP = 0
	as.PP = 0
	as.Tot = 0
}

func (as *AMPARVars) Total() {
	as.Tot = as.DD + as.PD + as.DP + as.PP
}

func (as *AMPARVars) Integrate(d *AMPARVars) {
	chem.Integrate(&as.DD, d.DD)
	chem.Integrate(&as.PD, d.PD)
	chem.Integrate(&as.DP, d.DP)
	chem.Integrate(&as.PP, d.PP)
	as.Total()
}

func (as *AMPARVars) Log(dt *etable.Table, vol float64, row int, pre string) {
	dt.SetCellFloat(pre+"AMPAR", row, chem.CoFmN(as.Tot, vol))
	// dt.SetCellFloat(pre+"DD", row, chem.CoFmN(as.DD, vol))
	// dt.SetCellFloat(pre+"PD", row, chem.CoFmN(as.PD, vol))
	// dt.SetCellFloat(pre+"DP", row, chem.CoFmN(as.DP, vol))
	// dt.SetCellFloat(pre+"PP", row, chem.CoFmN(as.PP, vol))
}

func (as *AMPARVars) ConfigLog(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "AMPAR", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "DD", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "PD", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "DP", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "PP", etensor.FLOAT64, nil, nil})
}

// AMPARState is AMPAR Phosphorylation and trafficking state.
// 4 Locations / states, which have their own time constants:
// Int = Cytosol, internal not integrated into membrane -- after endocyctosis
// Mem = Cytosol, integrated into the membrane -- after exocytosis, still governed by Cyl rates
// PSD = In the postsynaptic density, but not trapped by scaffold
// Trp = Trapped by scaffolding in the PSD -- solidly fixed in place and active
// Trp.Tot is the net effective AMPA conductance
// 20 state vars total
type AMPARState struct {
	Int      AMPARVars `view:"inline" desc:"cytosol internal"`
	Mem      AMPARVars `view:"inline" desc:"cytosol exocytosed into the membrane"`
	PSD      AMPARVars `view:"inline" desc:"in PSD but not trapped"`
	Trp      AMPARVars `view:"inline" desc:"in PSD and trapped in place"`
	Scaffold float64   `desc:"amount of unbound scaffold used for trapping"`
}

func (as *AMPARState) Init() {
	as.Int.Init()
	as.Mem.Init()
	as.PSD.Init()
	as.Trp.Init()

	as.Int.DD = chem.CoToN(3, CytVol) // Nophos_int
	as.Int.PD = chem.CoToN(3, CytVol) // S845P_int
	as.Int.Total()

	as.Trp.DD = chem.CoToN(1, PSDVol)
	as.Trp.PD = chem.CoToN(3, PSDVol)

	as.Scaffold = 0
	as.Trp.Total()
}

func (as *AMPARState) Zero() {
	as.Int.Zero()
	as.Mem.Zero()
	as.PSD.Zero()
	as.Trp.Zero()
	as.Scaffold = 0
}

func (as *AMPARState) Integrate(d *AMPARState) {
	as.Int.Integrate(&d.Int)
	as.Mem.Integrate(&d.Mem)
	as.PSD.Integrate(&d.PSD)
	as.Trp.Integrate(&d.Trp)
	chem.Integrate(&as.Scaffold, d.Scaffold)
}

func (as *AMPARState) Log(dt *etable.Table, row int) {
	as.Int.Log(dt, CytVol, row, "Int_")
	as.Mem.Log(dt, CytVol, row, "Mem_")
	as.PSD.Log(dt, PSDVol, row, "PSD_")
	as.Trp.Log(dt, PSDVol, row, "Trp_")
}

func (as *AMPARState) ConfigLog(sch *etable.Schema) {
	as.Int.ConfigLog(sch, "Int_")
	as.Mem.ConfigLog(sch, "Mem_")
	as.PSD.ConfigLog(sch, "PSD_")
	as.Trp.ConfigLog(sch, "Trp_")
}

// AMPAR phosphorylation and trafficking parameters
// Original kinetic rate constants are in units of (μM-1s-1),
type AMPARPhosParams struct {
	PKA       chem.SimpleEnz `desc:"rate of phosphorylation of AMPA Ser845 by PKA"`
	CaMKII    chem.SimpleEnz `desc:"rate of phosphorylation of PDZs by CaMKII"`
	PP_S845   chem.SimpleEnz `desc:"rate of dephosphorylation of AMPA Ser845 by PP1"`
	PP_PDZs   chem.SimpleEnz `desc:"rate of dephosphorylation of PDZs by PP1"`
	CaN_S845  chem.SimpleEnz `desc:"rate of dephosphorylation of AMPA Ser845 by CaN"`
	CaN_PDZs  chem.SimpleEnz `desc:"rate of dephosphorylation of PDZs by CaN"`
	PP2A_S845 chem.SimpleEnz `desc:"rate of dephosphorylation of AMPA Ser845 by PP2A"`
	PP2A_PDZs chem.SimpleEnz `desc:"rate of dephosphorylation of PDZs by PP2A"`
}

func (ap *AMPARPhosParams) Defaults() {
	ap.PKA.Kf = 20
	ap.CaMKII.Kf = 1
	ap.PP_S845.Kf = 4
	ap.PP_PDZs.Kf = 100
	ap.CaN_S845.Kf = 1.5
	ap.CaN_PDZs.Kf = 1
	ap.PP2A_S845.Kf = 4
	ap.PP2A_PDZs.Kf = 100
}

// StepP updates the phosphorylation d=delta state from c=current
// based on current kinase / pp states
func (ap *AMPARPhosParams) StepP(c, d *AMPARVars, vol, camkii, can, pka, pp1 float64) {
	ap.PKA.StepCo(c.DD, pka, vol, &d.DD, &d.PD)
	ap.PKA.StepCo(c.DP, pka, vol, &d.DP, &d.PP)
	ap.CaMKII.StepCo(c.DD, camkii, vol, &d.DD, &d.DP)
	ap.CaMKII.StepCo(c.PD, camkii, vol, &d.PD, &d.PP)

	ap.PP_S845.StepCo(c.PD, pp1, vol, &d.PD, &d.DD)
	ap.PP_S845.StepCo(c.PP, pp1, vol, &d.PP, &d.DP)
	ap.PP_PDZs.StepCo(c.DP, pp1, vol, &d.DP, &d.DD)
	ap.PP_PDZs.StepCo(c.PP, pp1, vol, &d.PP, &d.PD)

	ap.CaN_S845.StepCo(c.PD, can, vol, &d.PD, &d.DD)
	ap.CaN_S845.StepCo(c.PP, can, vol, &d.PP, &d.DP)
	ap.CaN_PDZs.StepCo(c.DP, can, vol, &d.DP, &d.DD)
	ap.CaN_PDZs.StepCo(c.PP, can, vol, &d.PP, &d.PD)
}

// StepPP2A updates the phosphorylation n=next state from c=current
// based on current pp2a
func (ap *AMPARPhosParams) StepPP2A(c, d *AMPARVars, vol, pp2a float64) {
	ap.PP2A_S845.StepCo(c.PD, pp2a, vol, &d.PD, &d.DD)
	ap.PP2A_S845.StepCo(c.PP, pp2a, vol, &d.PP, &d.DP)
	ap.PP2A_PDZs.StepCo(c.DP, pp2a, vol, &d.DP, &d.DD)
	ap.PP2A_PDZs.StepCo(c.PP, pp2a, vol, &d.PP, &d.PD)
}

// AMPAR trafficking parameters
// Original kinetic rate constants are in units of (μM-1s-1),
type AMPARTrafParams struct {
	EndoExoP chem.React   `desc:"Ser845P endocytosis, excocytosis rates -- Kf = 30min, Kb = 9min"`
	EndoD    chem.React   `desc:"Ser845D endocytosis rate -- Kf = 1sec, Kb = 0"`
	TrapP    chem.React   `desc:"PDZsP trapping in the PSD -- faster when P -- Kf is PSD + Scaffold -> Trp, Kb reverse"`
	TrapD    chem.React   `desc:"PDZsD trapping in the PSD -- slower when D -- Kf is PSD + Scaffold -> Trp, Kb reverse"`
	Diffuse  chem.Diffuse `desc:"diffusion for each category, all have the same constant"`
}

func (ap *AMPARTrafParams) Defaults() {
	ap.EndoExoP.Set(1.0/(9*60), 1.0/(30*60))
	ap.EndoD.Set(1, 0)
	ap.TrapP.Set(0.041667, 0.033333)
	ap.TrapD.Set(0.0025, 0.033333)
	ap.Diffuse.SetSym(1.6)
}

// StepT computes trafficking deltas
func (ap *AMPARTrafParams) StepT(c, d *AMPARState) {

	var dummy float64
	// Exo = Int -> Mem
	ap.EndoExoP.Step(c.Mem.PD, 1, c.Int.PD, &d.Mem.PD, &dummy, &d.Int.PD)
	ap.EndoExoP.Step(c.Mem.PP, 1, c.Int.PP, &d.Mem.PP, &dummy, &d.Int.PP)

	ap.EndoD.Step(c.Mem.DD, 1, c.Int.DD, &d.Mem.DD, &dummy, &d.Int.DD)
	ap.EndoD.Step(c.Mem.DP, 1, c.Int.DP, &d.Mem.DP, &dummy, &d.Int.DP)

	ap.TrapP.Step(c.PSD.DP, c.Scaffold, c.Trp.DP, &d.PSD.DP, &d.Scaffold, &d.Trp.DP)
	ap.TrapP.Step(c.PSD.PP, c.Scaffold, c.Trp.PP, &d.PSD.PP, &d.Scaffold, &d.Trp.PP)

	ap.TrapD.Step(c.PSD.DD, c.Scaffold, c.Trp.DD, &d.PSD.DD, &d.Scaffold, &d.Trp.DD)
	ap.TrapD.Step(c.PSD.PD, c.Scaffold, c.Trp.PD, &d.PSD.PD, &d.Scaffold, &d.Trp.PD)

	// Diffuse = Mem -> PSD
	ap.Diffuse.Step(c.Mem.DD, c.PSD.DD, CytVol, PSDVol, &d.Mem.DD, &d.PSD.DD)
	ap.Diffuse.Step(c.Mem.PD, c.PSD.PD, CytVol, PSDVol, &d.Mem.PD, &d.PSD.PD)
	ap.Diffuse.Step(c.Mem.DP, c.PSD.DP, CytVol, PSDVol, &d.Mem.DP, &d.PSD.DP)
	ap.Diffuse.Step(c.Mem.PP, c.PSD.PP, CytVol, PSDVol, &d.Mem.PP, &d.PSD.PP)
}

// AMPAR phosphorylation and trafficking parameters
// Original kinetic rate constants are in units of (μM-1s-1),
// converted to msec instead of sec
type AMPARParams struct {
	Phos    AMPARPhosParams `view:"inline" desc:"Phosphorylation parameters"`
	Traffic AMPARTrafParams `view:"inline" desc:"Trafficking parameters"`
}

func (ap *AMPARParams) Defaults() {
	ap.Phos.Defaults()
	ap.Traffic.Defaults()
}

// Step does full AMPAR updating, c=current, n=next
// based on current Ca signaling state
func (ap *AMPARParams) Step(c, d *AMPARState, cas *CaSigState, pp2a float64) {
	ap.Phos.StepP(&c.Int, &d.Int, CytVol, cas.CaMKII.Cyt.Active, cas.CaN.Cyt.CaNact, cas.PKA.Cyt.PKAact, cas.PP1.Cyt.PP1act)
	ap.Phos.StepP(&c.Mem, &d.Mem, CytVol, cas.CaMKII.Cyt.Active, cas.CaN.Cyt.CaNact, cas.PKA.Cyt.PKAact, cas.PP1.Cyt.PP1act)
	ap.Phos.StepP(&c.Trp, &d.Trp, PSDVol, cas.CaMKII.PSD.Active, cas.CaN.PSD.CaNact, cas.PKA.PSD.PKAact, cas.PP1.PSD.PP1act)
	ap.Phos.StepP(&c.PSD, &d.PSD, PSDVol, cas.CaMKII.PSD.Active, cas.CaN.PSD.CaNact, cas.PKA.PSD.PKAact, cas.PP1.PSD.PP1act)

	ap.Phos.StepPP2A(&c.Int, &d.Int, CytVol, pp2a) // Cyt only
	ap.Phos.StepPP2A(&c.Mem, &d.Mem, CytVol, pp2a) // Cyt only

	ap.Traffic.StepT(c, d)
}
