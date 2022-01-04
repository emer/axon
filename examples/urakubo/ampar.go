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
	PD  float64 `desc:"AMPA Ser845 phosphorylated = S845P"`
	DP  float64 `desc:"PDZs phosphorylated = StgP"`
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

func (as *AMPARVars) Log(dt *etable.Table, row int, pre string) {
	dt.SetCellFloat(pre+"AMPAR", row, float64(as.Tot))
}

func (as *AMPARVars) ConfigLog(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "AMPAR", etensor.FLOAT64, nil, nil})
}

// AMPARState is AMPAR Phosphorylation and trafficking state.
// 4 Locations / states, which have their own time constants:
// Cyt = Cytosol, not integrated into membrane -- after endocyctosis
// Int = Integrated into the membrane -- after exocytosis, still governed by Cyl rates
// PSD = In the postsynaptic density -- includes non-trapped and trapped
// Trp = Trapped by scaffolding in the PSD -- solidly fixed in place and active
// Trp.Tot is the net effective AMPA conductance
// 20 state vars total
type AMPARState struct {
	Cyt AMPARVars `view:"inline" desc:"in cytosol"`
	Int AMPARVars `view:"inline" desc:"in integrated state"`
	PSD AMPARVars `view:"inline" desc:"in PSD but not trapped"`
	Trp AMPARVars `view:"inline" desc:"in PSD and trapped in place"`
}

func (as *AMPARState) Init() {
	as.Cyt.Init()
	as.Int.Init()
	as.PSD.Init()
	as.Trp.Init()

	as.Int.DD = 3 // Nophos_int
	as.Int.PD = 3 // S845P_int
	as.Int.Total()

	as.Trp.DD = 1
	as.Trp.PD = 3
	as.Trp.Total()
}

func (as *AMPARState) Zero() {
	as.Cyt.Zero()
	as.Int.Zero()
	as.PSD.Zero()
	as.Trp.Zero()
}

func (as *AMPARState) Integrate(d *AMPARState) {
	as.Cyt.Integrate(&d.Cyt)
	as.Int.Integrate(&d.Int)
	as.PSD.Integrate(&d.PSD)
	as.Trp.Integrate(&d.Trp)
}

func (as *AMPARState) Log(dt *etable.Table, row int) {
	as.Cyt.Log(dt, row, "Cyt_")
	as.Int.Log(dt, row, "Int_")
	as.PSD.Log(dt, row, "PSD_")
	as.Trp.Log(dt, row, "Trp_")
}

func (as *AMPARState) ConfigLog(sch *etable.Schema) {
	as.Cyt.ConfigLog(sch, "Cyt_")
	as.Int.ConfigLog(sch, "Int_")
	as.PSD.ConfigLog(sch, "PSD_")
	as.Trp.ConfigLog(sch, "Trp_")
}

// AMPAR phosphorylation and trafficking parameters
// Original kinetic rate constants are in units of (μM-1s-1),
type AMPARPhosParams struct {
	PKA       float64 `def:"20" desc:"rate of phosphorylation of AMPA Ser845 by PKA"`
	CaMKII    float64 `def:"1" desc:"rate of phosphorylation of PDZs by CaMKII"`
	PP_S845   float64 `def:"4" desc:"rate of dephosphorylation of AMPA Ser845 by PP1"`
	PP_PDZs   float64 `def:"100" desc:"rate of dephosphorylation of PDZs by PP1"`
	CaN_S845  float64 `def:"1.5" desc:"rate of dephosphorylation of AMPA Ser845 by CaN"`
	CaN_PDZs  float64 `def:"1" desc:"rate of dephosphorylation of PDZs by CaN"`
	PP2A_S845 float64 `def:"100" desc:"rate of dephosphorylation of AMPA Ser845 by PP2A"`
	PP2A_PDZs float64 `def:"4" desc:"rate of dephosphorylation of PDZs by PP2A"`
}

func (ap *AMPARPhosParams) Defaults() {
	ap.PKA = 20
	ap.CaMKII = 1
	ap.PP_S845 = 4
	ap.PP_PDZs = 100
	ap.CaN_S845 = 1.5
	ap.CaN_PDZs = 1
	ap.PP2A_S845 = 100
	ap.PP2A_PDZs = 4
}

// StepP updates the phosphorylation d=delta state from c=current
// based on current kinase / pp states
func (ap *AMPARPhosParams) StepP(c, d *AMPARVars, camkii, pka, pp1, can float64) {
	d.PD += ap.PKA * pka * c.DD
	d.PP += ap.PKA * pka * c.DP
	d.DP += ap.CaMKII * camkii * c.DD
	d.PP += ap.CaMKII * camkii * c.PD

	d.DD += ap.PP_S845 * pp1 * c.PD
	d.DP += ap.PP_S845 * pp1 * c.PP
	d.DD += ap.PP_PDZs * pp1 * c.DP
	d.PD += ap.PP_PDZs * pp1 * c.PP

	d.DD += ap.CaN_S845 * can * c.PD
	d.DP += ap.CaN_S845 * can * c.PP
	d.DD += ap.CaN_PDZs * can * c.DP
	d.PD += ap.CaN_PDZs * can * c.PP
}

// StepPP2A updates the phosphorylation n=next state from c=current
// based on current pp2a
func (ap *AMPARPhosParams) StepPP2A(c, d *AMPARVars, pp2a float64) {
	d.DD += ap.PP2A_S845 * pp2a * c.PD
	d.DP += ap.PP2A_S845 * pp2a * c.PP
	d.DD += ap.PP2A_PDZs * pp2a * c.DP
	d.PD += ap.PP2A_PDZs * pp2a * c.PP

}

// AMPAR trafficking parameters
// Original kinetic rate constants are in units of (μM-1s-1),
// converted to msec instead of sec
type AMPARTrafParams struct {
	ExoP    float64 `def:"0.0055555" desc:"Ser845P excocytosis rate -- 30min"`
	EndoP   float64 `def:"0.0185" desc:"Ser845P endcytosis rate -- 9min"`
	EndoD   float64 `def:"1" desc:"Ser845D endcytosis rate -- 1sec"`
	Diffuse float64 `def:"3.19488" desc:"diffusion between Int and PSD"`
	OnP     float64 `def:"0.5" desc:"PDZsP trapping in the PSD -- faster when P"`
	OnD     float64 `def:"0.03" desc:"PDZsD trapping in the PSD -- slower when D"`
	Off     float64 `def:"0.0333" desc:"un-trapping in the PSD -- all off are the same"`
}

func (ap *AMPARTrafParams) Defaults() {
	ap.ExoP = 1.0 / (30 * 60)
	ap.EndoP = 1.0 / (9 * 60)
	ap.EndoD = 1.0
	ap.Diffuse = 1.0 / 0.313
	ap.OnP = 0.5
	ap.OnD = 0.03
	ap.Off = 1.0 / 30
}

// StepT computes trafficking deltas
func (ap *AMPARTrafParams) StepT(c, d *AMPARState) {
	// Exo = Cyt -> Int
	d.Int.PD += ap.ExoP * c.Cyt.PD // only Ser845P
	d.Int.PP += ap.ExoP * c.Cyt.PP // only Ser845P
	// zero the other way

	// Endo = Int -> Cyt
	d.Cyt.PD += ap.EndoP * c.Int.PD // only Ser845P
	d.Cyt.PP += ap.EndoP * c.Int.PP // only Ser845P
	d.Cyt.DD += ap.EndoD * c.Int.DD // only Ser845D
	d.Cyt.DP += ap.EndoD * c.Int.DP // only Ser845D

	// Off = Trp -> PSD
	d.PSD.DP += ap.Off * c.Trp.DP
	d.PSD.PP += ap.Off * c.Trp.PP
	d.PSD.DD += ap.Off * c.Trp.DD
	d.PSD.PD += ap.Off * c.Trp.PD

	// Diffuse = Int -> PSD
	d.PSD.DD += ap.Diffuse * c.Int.DD
	d.PSD.PD += ap.Diffuse * c.Int.PD
	d.PSD.DP += ap.Diffuse * c.Int.DP
	d.PSD.PP += ap.Diffuse * c.Int.PP

	// Diffuse = PSD -> Int
	d.Int.DD += ap.Diffuse * c.PSD.DD
	d.Int.PD += ap.Diffuse * c.PSD.PD
	d.Int.DP += ap.Diffuse * c.PSD.DP
	d.Int.PP += ap.Diffuse * c.PSD.PP

	// On = PSD -> Trp
	d.Trp.DP += ap.OnP * c.PSD.DP
	d.Trp.PP += ap.OnP * c.PSD.PP
	d.Trp.DD += ap.OnD * c.PSD.DD
	d.Trp.PD += ap.OnD * c.PSD.PD

	// Off = Trp -> PSD
	d.PSD.DP += ap.Off * c.Trp.DP
	d.PSD.PP += ap.Off * c.Trp.PP
	d.PSD.DD += ap.Off * c.Trp.DD
	d.PSD.PD += ap.Off * c.Trp.PD
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
	ap.Phos.StepP(&c.Cyt, &d.Cyt, cas.CaMKII.Cyt.Active, cas.CaN.Cyt.CaNact, cas.PKA.Cyt.PKAact, cas.PP1.Cyt.PP1act)
	ap.Phos.StepP(&c.Int, &d.Int, cas.CaMKII.Cyt.Active, cas.CaN.Cyt.CaNact, cas.PKA.Cyt.PKAact, cas.PP1.Cyt.PP1act)
	ap.Phos.StepP(&c.Trp, &d.Trp, cas.CaMKII.PSD.Active, cas.CaN.PSD.CaNact, cas.PKA.PSD.PKAact, cas.PP1.PSD.PP1act)
	ap.Phos.StepP(&c.PSD, &d.PSD, cas.CaMKII.PSD.Active, cas.CaN.PSD.CaNact, cas.PKA.PSD.PKAact, cas.PP1.PSD.PP1act)

	ap.Phos.StepPP2A(&c.Cyt, &d.Cyt, pp2a) // Cyt only

	ap.Traffic.StepT(c, d)
}
