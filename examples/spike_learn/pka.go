// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// PKAVars are intracellular Ca-driven signaling states
// for PKA binding and phosphorylation with cAMP
type PKAVars struct {
	AC1      float32 `desc:"AC1"`
	AC1act   float32 `desc:"active AC1"`
	PDEact   float32 `desc:"active PDE"`
	ATP      float32 `desc:"ATP power"`
	CAMP     float32 `desc:"cAMP power"`
	AMP      float32 `desc:"AMP"`
	R2C2     float32 `desc:"R2C2"`
	R2C2_B   float32 `desc:"R2C2-cAMP_B"`
	R2C2_BB  float32 `desc:"R2C2-2cAMP-B-B"`
	R2C2_AB  float32 `desc:"R2C2-2cAMP-A-B"`
	R2C2_ABB float32 `desc:"R2C2-3cAMP-A-B-B"`
	R2C2_4   float32 `desc:"R2C2-4cAMP"`
	R2C_3    float32 `desc:"R2C-3cAMP -- note Fig SI4 mislabeled as R2-3"`
	R2C_4    float32 `desc:"R2C-4cAMP"`
	R2_3     float32 `desc:"R2-3cAMP"`
	R2_4     float32 `desc:"R2-4cAMP"`
	PKAact   float32 `desc:"active PKA"`
}

// PKAtate is overall intracellular Ca-driven signaling states
// for PKA binding and phosphorylation with cAMP
type PKAState struct {
	Cyt PKAVars `desc:"in cytosol"`
	PSD PKAVars `desc:"in PSD"`
}

// PKAParams are the parameters governing the
// PKA binding and phosphorylation with cAMP
type PKAParams struct {
	CaMAC1   React `desc:"1: 3Ca-CaM + AC1 -> AC1act"`
	ATPcAMP  React `desc:"2: basal activity of ATP -> cAMP without AC1 enzyme"`
	R2C2     React `desc:"3: R2C2 + cAMP"`
	R2C2_B1  React `desc:"4: R2C2-cAMP B + cAMP -> BB"`
	R2C2_B2  React `desc:"5: R2C2-cAMP B + cAMP -> AB"`
	R2C2_BB  React `desc:"6: R2C2-cAMP BB + cAMP"`
	R2C2_AB  React `desc:"7: R2C2-cAMP BB + cAMP"`
	R2C2_ABB React `desc:"8: R2C2-cAMP ABB + cAMP"`
	R2C_34   React `desc:"9: R2C-3cAMP -> R2C-4cAMP"`
	R2_34    React `desc:"10: R2-3cAMP -> R2-4cAMP"`
	R2C_3    React `desc:"11: R2C-3cAMP + PKAact -> R2C2-3cAMP ABB (backwards) -- Fig SI4 R2-3 -> R2C-3"`
	R2C_4    React `desc:"12: R2C-4cAMP + PKAact -> R2C2-4cAMP (backwards)"`
	R2_3     React `desc:"13: R2-3cAMP + PKAact -> R2C-3cAMP (backwards)"`
	R2_4     React `desc:"14: R2-4cAMP + PKAact -> R2C-4cAMP (backwards)"`
	AC1ATP   Enz   `desc:"15: AC1act catalyzing ATP -> cAMP -- table SIg numbered 9 -> 15"`
	PDEcAMP  Enz   `desc:"16: PDE1act catalyzing cAMP -> AMP -- table SIg numbered 10 -> 16"`
}

func (cp *PKAParams) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// Cyt = 1/48 * values listed in Table SIh (0.02083333)
	cp.CaMAC1.SetSec(0.10416, 1)     // 1: 5 μM-1
	cp.ATPcAMP.SetSec(4.0e-7, 0)     // 2: raw
	cp.R2C2.SetSec(0.0041667, 0.1)   // 3: 0.2 μM-1
	cp.R2C2_B1.SetSec(0.002083, 0.2) // 4: 0.1 μM-1
	cp.R2C2_B2.SetSec(0.041667, 5)   // 5: 2 μM-1
	cp.R2C2_BB.SetSec(0.083333, 5)   // 6: 4 μM-1
	cp.R2C2_AB.SetSec(0.002083, 0.1) // 7: 0.1 μM-1
	cp.R2C2_ABB.SetSec(0.041667, 10) // 8: 2 μM-1
	cp.R2C_34.SetSec(0.41667, 1)     // 9: 20 μM-1
	cp.R2_34.SetSec(4.1667, 0.1)     // 10: 200 μM-1

	cp.R2C_3.SetSec(0.20833, 2)   // 11: 10 μM-1
	cp.R2C_4.SetSec(0.020833, 20) // 12: 1 μM-1
	cp.R2_3.SetSec(0.41667, 1)    // 13: 20 μM-1
	cp.R2_4.SetSec(0.041667, 10)  // 14: 2 μM-1

	cp.AC1ATP.SetSec(0.26042, 40, 10)  // 15: Km = 40?
	cp.PDEcAMP.SetSec(0.20834, 80, 20) // 16: Km = 10?
}

// StepPKA does the PKA + cAMP reactions, in a given region
// kf is an additional forward multiplier, which is 1 for Cyt and 4 for PSD
// cCaM, nCaM = current, new 3CaCaM from CaMKIIVars
func (cp *PKAParams) StepPKA(kf float32, c, n *PKAVars, cCaM float32, nCaM *float32) {
	cp.CaMAC1.StepKf(kf, c.AC1, cCaM, c.AC1act, &n.AC1, nCaM, &n.AC1act)                   // 1
	cp.ATPcAMP.StepKf(kf, c.ATP, 0, c.CAMP, &n.ATP, nil, &n.CAMP)                          // 2
	cp.R2C2.StepKf(kf, c.CAMP, c.R2C2, c.R2C2_B, &n.CAMP, &n.R2C2, &n.R2C2_B)              // 3
	cp.R2C2_B1.StepKf(kf, c.CAMP, c.R2C2_B, c.R2C2_BB, &n.CAMP, &n.R2C2, &n.R2C2_BB)       // 4
	cp.R2C2_B2.StepKf(kf, c.CAMP, c.R2C2_B, c.R2C2_AB, &n.CAMP, &n.R2C2, &n.R2C2_AB)       // 5
	cp.R2C2_BB.StepKf(kf, c.CAMP, c.R2C2_BB, c.R2C2_ABB, &n.CAMP, &n.R2C2_BB, &n.R2C2_ABB) // 6
	cp.R2C2_AB.StepKf(kf, c.CAMP, c.R2C2_AB, c.R2C2_ABB, &n.CAMP, &n.R2C2_AB, &n.R2C2_ABB) // 7
	cp.R2C2_ABB.StepKf(kf, c.CAMP, c.R2C2_ABB, c.R2C2_4, &n.CAMP, &n.R2C2_ABB, &n.R2C2_4)  // 8
	cp.R2C_34.StepKf(kf, c.CAMP, c.R2C_3, c.R2C_4, &n.CAMP, &n.R2C_3, &n.R2C_4)            // 9
	cp.R2_34.StepKf(kf, c.CAMP, c.R2_3, c.R2_4, &n.CAMP, &n.R2_3, &n.R2_4)                 // 10

	cp.R2C_3.StepKf(kf, c.PKAact, c.R2C_3, c.R2C2_ABB, &n.PKAact, &n.R2C_3, &n.R2C2_ABB) // 11
	cp.R2C_4.StepKf(kf, c.PKAact, c.R2C_4, c.R2C2_4, &n.PKAact, &n.R2C_4, &n.R2C2_4)     // 12
	cp.R2_3.StepKf(kf, c.PKAact, c.R2_3, c.R2C_3, &n.PKAact, &n.R2_3, &n.R2C_3)          // 13
	cp.R2_4.StepKf(kf, c.PKAact, c.R2_4, c.R2C_4, &n.PKAact, &n.R2_4, &n.R2C_4)          // 14

	cp.AC1ATP.Step(c.ATP, c.AC1act, c.CAMP, &n.AC1act, &n.CAMP)
	cp.PDEcAMP.Step(c.CAMP, c.PDEact, c.AMP, &n.CAMP, &n.PDEact)
}

func (cp *PKAParams) Step(c, n *PKAState, cCaM, nCaM *CaMKIIState) {
	*n = *c
	cp.StepPKA(1, &c.Cyt, &n.Cyt, cCaM.Cyt.Ca[3].CaM, &nCaM.Cyt.Ca[3].CaM)
	cp.StepPKA(4, &c.PSD, &n.PSD, cCaM.PSD.Ca[3].CaM, &nCaM.PSD.Ca[3].CaM)
}
