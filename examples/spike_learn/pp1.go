// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// PP1Vars are intracellular Ca-driven signaling variables for the
// PP1 - I-1 system
type PP1Vars struct {
	I1      float32 `desc:"dephosphorylated I-1"`
	I1P     float32 `desc:"phosphorylated I-1"`
	PP1_I1P float32 `desc:"PP1 = protein phosphatase 1 bound with I-1P"`
	PP1act  float32 `desc:"activated PP1"`
}

// PP1State is overall intracellular Ca-driven signaling states
// for PP1-I-1 in Cyt and PSD
type PP1State struct {
	Cyt PP1Vars `desc:"in cytosol"`
	PSD PP1Vars `desc:"in PSD"`
}

// PP1Params are the parameters governing the PP1-I-1 binding
type PP1Params struct {
	PP1act  React `desc:"1: I-1P + PP1act -> PP1-I1P -- Table SIi constants are backward"`
	PKAILP  Enz   `desc:"2: I-1P phosphorylated by PKA -- Table SIj numbers != Figure SI4"`
	CaNILP  Enz   `desc:"3: I-1P dephosphorylated by CaN -- Table SIj number"`
	PP2AILP Enz   `desc:"4: I-1P dephosphorylated by PP2A -- Table SIj number"`
}

func (cp *PP1Params) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// Cyt = 1/48 * values listed in Table SIh (0.02083333)
	cp.PP1act.SetSec(2.0833, 1)            // Kb = 100 μM-1 -- reversed for product = PP1-I1P
	cp.PKAILP.SetSec(0.068157, 21.2, 5.3)  // Km = 8.1 μM-1
	cp.CaNILP.SetSec(0.097222, 11.2, 2.8)  // Km = 3 μM-1
	cp.PP2AILP.SetSec(0.097222, 11.2, 2.8) // Km = 3 μM-1
}

// StepPP1 does the bulk of Ca + PP1 + CaM binding reactions, in a given region
// kf is an additional forward multiplier, which is 1 for Cyt and 4 for PSD
// cCaM, nCaM = current, new 3CaCaM from CaMKIIVars
// cCa, nCa = current new Ca
func (cp *PP1Params) StepPP1(kf float32, c, n *PP1Vars, pka, cna, pp2a float32) {
	cp.PP1act.StepKf(kf, c.I1P, c.PP1act, c.PP1_I1P, &n.I1P, &n.PP1act, &n.PP1_I1P) // 1

	cp.PKAILP.Step(c.I1, pka, c.I1P, &n.I1, &n.I1P)   // 2
	cp.CaNILP.Step(c.I1P, cna, c.I1, &n.I1P, &n.I1)   // 3
	cp.PP2AILP.Step(c.I1P, pp2a, c.I1, &n.I1P, &n.I1) // 3
}

func (cp *PP1Params) Step(c, n *PP1State, pka, cna, pp2a float32) {
	*n = *c
	cp.StepPP1(1, &c.Cyt, &n.Cyt, pka, cna, pp2a)
	cp.StepPP1(4, &c.PSD, &n.PSD, pka, cna, pp2a)
}
