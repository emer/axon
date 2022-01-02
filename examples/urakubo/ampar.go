// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// AMPARVars have AMPAR Phosphorylation (Pd = phosphorylated, Dp = dephosphorylated) state.
// Two protein elements, separately Pd and Dp:
// AMPAR = AMPA receptor (GluR1), which can be Pd at Ser845 by PKA
// PDZs = PDZ domain binding proteins (e.g., SAP97 and stargazin), which bind to AMPAR
//        and are separately Pd by CaMKII -- denoted as StgP in Urakubo code
// Both can be Dp by PP1 and CaN (calcineurin)
// Variables named as P or D for the Pd or Dp state, 1st is AMPAR @ Ser845, 2nd is PDZs
type AMPARVars struct {
	DD float32 `desc:"both dephosphorylated = Nophos"`
	PD float32 `desc:"AMPA Ser845 phosphorylated = S845P"`
	DP float32 `desc:"PDZs phosphorylated = StgP"`
	PP float32 `desc:"both phosphorylated = S845PStgP"`
}

func (as *AMPARVars) Init() {
	as.DD = 0
	as.PD = 0
	as.DP = 0
	as.PP = 0
}

// AMPARState is AMPAR Phosphorylation and trafficking state.
// 4 Locations / states, which have their own time constants:
// Cyt = Cytosol, not integrated into membrane -- after endocyctosis
// Int = Integrated into the membrane -- after exocytosis, still governed by Cyl rates
// PSD = In the postsynaptic density -- includes non-trapped and trapped
// Trp = Trapped by scaffolding in the PSD -- solidly fixed in place and active
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
}

// AMPAR phosphorylation and trafficking parameters
// Original kinetic rate constants are in units of (μM-1s-1),
// converted to msec instead of sec
type AMPARPhosParams struct {
	PKA       float32 `def:"0.020" desc:"rate of phosphorylation of AMPA Ser845 by PKA"`
	CaMKII    float32 `def:"0.001" desc:"rate of phosphorylation of PDZs by CaMKII"`
	PP_S845   float32 `def:"0.004" desc:"rate of dephosphorylation of AMPA Ser845 by PP1"`
	PP_PDZs   float32 `def:"0.1" desc:"rate of dephosphorylation of PDZs by PP1"`
	CaN_S845  float32 `def:"0.0015" desc:"rate of dephosphorylation of AMPA Ser845 by CaN"`
	CaN_PDZs  float32 `def:"0.001" desc:"rate of dephosphorylation of PDZs by CaN"`
	PP2A_S845 float32 `def:"0.1" desc:"rate of dephosphorylation of AMPA Ser845 by PP2A"`
	PP2A_PDZs float32 `def:"0.004" desc:"rate of dephosphorylation of PDZs by PP2A"`
}

func (ap *AMPARPhosParams) Defaults() {
	ap.PKA = 20.0 / 1000
	ap.CaMKII = 1.0 / 1000
	ap.PP_S845 = 4.0 / 1000
	ap.PP_PDZs = 100.0 / 1000
	ap.CaN_S845 = 1.5 / 1000
	ap.CaN_PDZs = 1.0 / 1000
	ap.PP2A_S845 = 100.0 / 1000
	ap.PP2A_PDZs = 4.0 / 1000
}

// StepP updates the phosphorylation n=next state from c=current
// based on current kinase / pp states
func (ap *AMPARPhosParams) StepP(c, n *AMPARVars, camkii, pka, pp1, can float32) {
	n.PD += ap.PKA * pka * c.DD
	n.PP += ap.PKA * pka * c.DP
	n.DP += ap.CaMKII * camkii * c.DD
	n.PP += ap.CaMKII * camkii * c.PD

	n.DD += ap.PP_S845 * pp1 * c.PD
	n.DP += ap.PP_S845 * pp1 * c.PP
	n.DD += ap.PP_PDZs * pp1 * c.DP
	n.PD += ap.PP_PDZs * pp1 * c.PP

	n.DD += ap.CaN_S845 * can * c.PD
	n.DP += ap.CaN_S845 * can * c.PP
	n.DD += ap.CaN_PDZs * can * c.DP
	n.PD += ap.CaN_PDZs * can * c.PP
}

// StepPP2A updates the phosphorylation n=next state from c=current
// based on current pp2a
func (ap *AMPARPhosParams) StepPP2A(c, n *AMPARVars, pp2a float32) {
	n.DD += ap.PP2A_S845 * pp2a * c.PD
	n.DP += ap.PP2A_S845 * pp2a * c.PP
	n.DD += ap.PP2A_PDZs * pp2a * c.DP
	n.PD += ap.PP2A_PDZs * pp2a * c.PP

}

// AMPAR trafficking parameters
// Original kinetic rate constants are in units of (μM-1s-1),
// converted to msec instead of sec
type AMPARTrafParams struct {
	ExoP    float32 `def:"0.0000055555" desc:"Ser845P excocytosis rate -- 30min"`
	EndoP   float32 `def:"0.0000185" desc:"Ser845P endcytosis rate -- 9min"`
	EndoD   float32 `def:"0.001" desc:"Ser845D endcytosis rate -- 1sec"`
	Diffuse float32 `def:"0.00319488" desc:"diffusion between Int and PSD"`
	OnP     float32 `def:"0.005" desc:"PDZsP trapping in the PSD -- faster when P"`
	OnD     float32 `def:"0.0003" desc:"PDZsD trapping in the PSD -- slower when D"`
	Off     float32 `def:"0.000333" desc:"un-trapping in the PSD -- all off are the same"`
}

func (ap *AMPARTrafParams) Defaults() {
	ap.ExoP = 1.0 / (30 * 60 * 1000)
	ap.EndoP = 1.0 / (9 * 60 * 1000)
	ap.EndoD = 1.0 / 1000
	ap.Diffuse = 1.0 / (0.313 * 1000)
	ap.OnP = 0.5 / 1000
	ap.OnD = 0.03 / 1000
	ap.Off = 1.0 / (30 * 1000)
}

func (ap *AMPARTrafParams) StepT(c, n *AMPARState) {
	// Exo = Cyt -> Int
	n.Int.PD += ap.ExoP * c.Cyt.PD // only Ser845P
	n.Int.PP += ap.ExoP * c.Cyt.PP // only Ser845P
	// zero the other way

	// Endo = Int -> Cyt
	n.Cyt.PD += ap.EndoP * c.Int.PD // only Ser845P
	n.Cyt.PP += ap.EndoP * c.Int.PP // only Ser845P
	n.Cyt.DD += ap.EndoD * c.Int.DD // only Ser845D
	n.Cyt.DP += ap.EndoD * c.Int.DP // only Ser845D

	// Off = Trp -> PSD
	n.PSD.DP += ap.Off * c.Trp.DP
	n.PSD.PP += ap.Off * c.Trp.PP
	n.PSD.DD += ap.Off * c.Trp.DD
	n.PSD.PD += ap.Off * c.Trp.PD

	// Diffuse = Int -> PSD
	n.PSD.DD += ap.Diffuse * c.Int.DD
	n.PSD.PD += ap.Diffuse * c.Int.PD
	n.PSD.DP += ap.Diffuse * c.Int.DP
	n.PSD.PP += ap.Diffuse * c.Int.PP

	// Diffuse = PSD -> Int
	n.Int.DD += ap.Diffuse * c.PSD.DD
	n.Int.PD += ap.Diffuse * c.PSD.PD
	n.Int.DP += ap.Diffuse * c.PSD.DP
	n.Int.PP += ap.Diffuse * c.PSD.PP

	// On = PSD -> Trp
	n.Trp.DP += ap.OnP * c.PSD.DP
	n.Trp.PP += ap.OnP * c.PSD.PP
	n.Trp.DD += ap.OnD * c.PSD.DD
	n.Trp.PD += ap.OnD * c.PSD.PD

	// Off = Trp -> PSD
	n.PSD.DP += ap.Off * c.Trp.DP
	n.PSD.PP += ap.Off * c.Trp.PP
	n.PSD.DD += ap.Off * c.Trp.DD
	n.PSD.PD += ap.Off * c.Trp.PD
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
func (ap *AMPARParams) Step(c, n *AMPARState, cas *CaSigState, pp2a float32) {
	ap.Phos.StepP(&c.Cyt, &n.Cyt, cas.CaMKII.Cyt.CaMKIIact, cas.CaN.Cyt.CaNact, cas.PKA.Cyt.PKAact, cas.PP1.Cyt.PP1act)
	ap.Phos.StepP(&c.Int, &n.Int, cas.CaMKII.Cyt.CaMKIIact, cas.CaN.Cyt.CaNact, cas.PKA.Cyt.PKAact, cas.PP1.Cyt.PP1act)
	ap.Phos.StepP(&c.Trp, &n.Trp, cas.CaMKII.PSD.CaMKIIact, cas.CaN.PSD.CaNact, cas.PKA.PSD.PKAact, cas.PP1.PSD.PP1act)
	ap.Phos.StepP(&c.PSD, &n.PSD, cas.CaMKII.PSD.CaMKIIact, cas.CaN.PSD.CaNact, cas.PKA.PSD.PKAact, cas.PP1.PSD.PP1act)

	ap.Phos.StepPP2A(&c.Cyt, &n.Cyt, pp2a) // Cyt only

	ap.Traffic.StepT(c, n)
}
