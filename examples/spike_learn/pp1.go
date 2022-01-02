// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// PP1Vars are intracellular Ca-driven signaling variables for the
// PP1 - I-1 system
type PP1Vars struct {
	I1      float32 `desc:"dephosphorylated I-1 = I1_inactive"`
	I1P     float32 `desc:"phosphorylated (active) I-1 = I1_active"`
	PP1_I1P float32 `desc:"PP1 = protein phosphatase 1 bound with I-1P"`
	PP1act  float32 `desc:"activated PP1"`
}

func (ps *PP1Vars) Init() {
	ps.I1 = 2
	ps.I1P = 0
	ps.PP1_I1P = 2
	ps.PP1act = 0
}

// PP1State is overall intracellular Ca-driven signaling states
// for PP1-I-1 in Cyt and PSD
type PP1State struct {
	Cyt PP1Vars `desc:"in cytosol -- volume = 0.08 fl"`
	PSD PP1Vars `desc:"in PSD -- volume = 0.02 fl"`
}

func (ps *PP1State) Init() {
	ps.Cyt.Init()
	ps.PSD.Init()
}

// PP1Params are the parameters governing the PP1-I-1 binding
type PP1Params struct {
	I1PP1   React `desc:"1: I-1P + PP1act -> PP1-I1P -- Table SIi constants are backward = I1-PP1"`
	PKAILP  Enz   `desc:"2: I-1P phosphorylated by PKA -- Table SIj numbers != Figure SI4"`
	CaNILP  Enz   `desc:"3: I-1P dephosphorylated by CaN -- Table SIj number"`
	PP2AILP Enz   `desc:"4: I-1P dephosphorylated by PP2A -- Table SIj number"`
}

func (cp *PP1Params) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// Cyt = 1/48 * values listed in Table SIh (0.02083333)
	cp.I1PP1.SetSec(2.0834, 1)             // Kb = 100 μM-1 -- reversed for product = PP1-I1P
	cp.PKAILP.SetSec(0.068157, 21.2, 5.3)  // Km = 8.1 μM-1
	cp.CaNILP.SetSec(0.097222, 11.2, 2.8)  // Km = 3 μM-1
	cp.PP2AILP.SetSec(0.097222, 11.2, 2.8) // Km = 3 μM-1
}

// StepPP1 does the bulk of Ca + PP1 + CaM binding reactions, in a given region
// kf is an additional forward multiplier, which is 1 for Cyt and 4 for PSD
// cCaM, nCaM = current, new 3CaCaM from CaMKIIVars
// cCa, nCa = current new Ca
func (cp *PP1Params) StepPP1(kf float32, c, n *PP1Vars, pka, can, pp2a float32) {
	cp.I1PP1.StepKf(kf, c.I1P, c.PP1act, c.PP1_I1P, &n.I1P, &n.PP1act, &n.PP1_I1P) // 1

	cp.PKAILP.StepKf(kf, c.I1, pka, c.I1P, &n.I1, &n.I1P)   // 2
	cp.CaNILP.StepKf(kf, c.I1P, can, c.I1, &n.I1P, &n.I1)   // 3
	cp.PP2AILP.StepKf(kf, c.I1P, pp2a, c.I1, &n.I1P, &n.I1) // 3
}

// Step does one step of updating
// Next has already been initialized to current
func (cp *PP1Params) Step(c, n *PP1State, pka *PKAState, can *CaNState, pp2a float32) {
	cp.StepPP1(1, &c.Cyt, &n.Cyt, pka.Cyt.PKAact, can.Cyt.CaNact, pp2a)
	cp.StepPP1(4, &c.PSD, &n.PSD, pka.PSD.PKAact, can.PSD.CaNact, 0)
}
