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

// PKAVars are intracellular Ca-driven signaling states
// for PKA binding and phosphorylation with cAMP
// stores N values -- Co = Concentration computed by volume as needed
type PKAVars struct {
	AC1      float64 `desc:"AC1"`
	AC1act   float64 `desc:"active AC1 = CaM-AC1"`
	PDEact   float64 `desc:"active PDE = cAMP-PDE -- buffered to 1"`
	ATP      float64 `desc:"ATP -- buffered to 10000"`
	CAMP     float64 `desc:"cAMP"`
	AMP      float64 `desc:"AMP -- buffered to 1000"`
	R2C2     float64 `desc:"R2C2"`
	R2C2_B   float64 `desc:"R2C2-cAMP_B"`
	R2C2_BB  float64 `desc:"R2C2-2cAMP-B-B"`
	R2C2_AB  float64 `desc:"R2C2-2cAMP-A-B"`
	R2C2_ABB float64 `desc:"R2C2-3cAMP-A-B-B"`
	R2C2_4   float64 `desc:"R2C2-4cAMP"`
	R2C_3    float64 `desc:"R2C-3cAMP -- note Fig SI4 mislabeled as R2-3"`
	R2C_4    float64 `desc:"R2C-4cAMP"`
	R2_3     float64 `desc:"R2-3cAMP"`
	R2_4     float64 `desc:"R2-4cAMP"`
	PKAact   float64 `desc:"active PKA"`
	AC1ATPC  float64 `desc:"AC1act+ATP complex for AC1ATP enzyme reaction -- reflects rate"`
	PDEcAMPC float64 `desc:"PDEact+cAMP complex for PDEcAMP enzyme reaction"`
}

func (ps *PKAVars) Init(vol float64) {
	ps.AC1 = chem.CoToN(2, vol)
	ps.AC1act = 0
	ps.PDEact = chem.CoToN(1, vol)  // buffered!
	ps.ATP = chem.CoToN(10000, vol) // buffered! -- note: large #'s here contribute significantly to instability
	// todo: experiment with significantly smaller #'s
	ps.CAMP = 0
	ps.AMP = chem.CoToN(1000, vol) // buffered!
	ps.R2C2 = chem.CoToN(2, vol)
	ps.R2C2_B = 0
	ps.R2C2_BB = 0
	ps.R2C2_AB = 0
	ps.R2C2_ABB = 0
	ps.R2C2_4 = 0
	ps.R2C_3 = 0
	ps.R2C_4 = 0
	ps.R2_3 = 0
	ps.R2_4 = 0
	ps.PKAact = chem.CoToN(0.05, vol)
	ps.AC1ATPC = chem.CoToN(0.00025355, vol)
	ps.PDEcAMPC = 0

	if TheOpts.InitBaseline {
		if TheOpts.UseDAPK1 {
			ps.AC1 = chem.CoToN(2, vol)
			ps.AC1act = chem.CoToN(0.0001049, vol)
			ps.CAMP = chem.CoToN(0.003709, vol)
			ps.R2C2 = chem.CoToN(1.985, vol)
			ps.R2C2_B = chem.CoToN(0.01471, vol)
			ps.R2C2_BB = chem.CoToN(2.724e-05, vol)
			ps.R2C2_AB = chem.CoToN(2.179e-05, vol)
			ps.R2C2_ABB = chem.CoToN(8.07e-08, vol)
			ps.R2C2_4 = chem.CoToN(5.973e-11, vol)
			ps.R2C_3 = chem.CoToN(3.506e-07, vol)
			ps.R2C_4 = chem.CoToN(2.596e-08, vol)
			ps.R2_3 = chem.CoToN(3.81e-07, vol)
			ps.R2_4 = chem.CoToN(2.822e-06, vol)
			ps.PKAact = chem.CoToN(0.04599, vol)
			ps.AC1ATPC = chem.CoToN(1.1e+05, vol)
			ps.PDEcAMPC = chem.CoToN(0.0003709, vol)
		} else {
			ps.AC1act = chem.CoToN(0.0004371, vol)
			ps.CAMP = chem.CoToN(0.005518, vol)
			ps.R2C2 = chem.CoToN(1.978, vol)
			ps.R2C2_B = chem.CoToN(0.02181, vol)
			ps.R2C2_BB = chem.CoToN(6.006e-05, vol)
			ps.R2C2_AB = chem.CoToN(4.814e-05, vol)
			ps.R2C2_ABB = chem.CoToN(2.635e-07, vol)
			ps.R2C2_4 = chem.CoToN(2.859e-10, vol)

			ps.R2C_3 = chem.CoToN(1.162e-06, vol)
			ps.R2C_4 = chem.CoToN(1.271e-07, vol)
			ps.R2_3 = chem.CoToN(1.295e-06, vol)
			ps.R2_4 = chem.CoToN(1.423e-05, vol)
			ps.PKAact = chem.CoToN(0.04461, vol)
			ps.AC1ATPC = chem.CoToN(3.832e+05, vol)
			ps.PDEcAMPC = chem.CoToN(0.0005518, vol)
		}
	}
}

func (ps *PKAVars) InitCode(vol float64, pre string) {
	fmt.Printf("\tps.%s.AC1 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.AC1, vol))
	fmt.Printf("\tps.%s.AC1act = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.AC1act, vol))
	fmt.Printf("\tps.%s.CAMP = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.CAMP, vol))
	fmt.Printf("\tps.%s.R2C2 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2C2, vol))
	fmt.Printf("\tps.%s.R2C2_B = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2C2_B, vol))
	fmt.Printf("\tps.%s.R2C2_BB = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2C2_BB, vol))
	fmt.Printf("\tps.%s.R2C2_AB = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2C2_AB, vol))
	fmt.Printf("\tps.%s.R2C2_ABB = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2C2_ABB, vol))
	fmt.Printf("\tps.%s.R2C2_4 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2C2_4, vol))
	fmt.Printf("\tps.%s.R2C_3 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2C_3, vol))
	fmt.Printf("\tps.%s.R2C_4 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2C_4, vol))
	fmt.Printf("\tps.%s.R2_3 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2_3, vol))
	fmt.Printf("\tps.%s.R2_4 = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.R2_4, vol))
	fmt.Printf("\tps.%s.PKAact = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.PKAact, vol))
	fmt.Printf("\tps.%s.AC1ATPC = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.AC1ATPC, vol))
	fmt.Printf("\tps.%s.PDEcAMPC = chem.CoToN(%.4g, vol)\n", pre, chem.CoFmN(ps.PDEcAMPC, vol))
}

func (ps *PKAVars) Zero() {
	ps.AC1 = 0
	ps.AC1act = 0
	ps.PDEact = 0
	ps.ATP = 0
	ps.CAMP = 0
	ps.AMP = 0
	ps.R2C2 = 0
	ps.R2C2_B = 0
	ps.R2C2_BB = 0
	ps.R2C2_AB = 0
	ps.R2C2_ABB = 0
	ps.R2C2_4 = 0
	ps.R2C_3 = 0
	ps.R2C_4 = 0
	ps.R2_3 = 0
	ps.R2_4 = 0
	ps.PKAact = 0
	ps.AC1ATPC = 0
	ps.PDEcAMPC = 0
}

func (ps *PKAVars) Integrate(d *PKAVars) {
	chem.Integrate(&ps.AC1, d.AC1)
	chem.Integrate(&ps.AC1act, d.AC1act)
	// PDEact buffered
	// ATP buffered
	chem.Integrate(&ps.CAMP, d.CAMP)
	// AMP buffered
	chem.Integrate(&ps.R2C2, d.R2C2)
	chem.Integrate(&ps.R2C2_B, d.R2C2_B)
	chem.Integrate(&ps.R2C2_BB, d.R2C2_BB)
	chem.Integrate(&ps.R2C2_AB, d.R2C2_AB)
	chem.Integrate(&ps.R2C2_ABB, d.R2C2_ABB)
	chem.Integrate(&ps.R2C2_4, d.R2C2_4)
	chem.Integrate(&ps.R2C_3, d.R2C_3)
	chem.Integrate(&ps.R2C_4, d.R2C_4)
	chem.Integrate(&ps.R2_3, d.R2_3)
	chem.Integrate(&ps.R2_4, d.R2_4)
	chem.Integrate(&ps.PKAact, d.PKAact)
	// chem.Integrate(&ps.AC1ATPC, d.AC1ATPC) // set directly
	chem.Integrate(&ps.PDEcAMPC, d.PDEcAMPC)
}

func (ps *PKAVars) Log(dt *etable.Table, vol float64, row int, pre string) {
	dt.SetCellFloat(pre+"AC1act", row, chem.CoFmN(ps.AC1act, vol))
	dt.SetCellFloat(pre+"cAMP", row, chem.CoFmN(ps.CAMP, vol))
	dt.SetCellFloat(pre+"PKAact", row, chem.CoFmN(ps.PKAact, vol))
	// dt.SetCellFloat(pre+"AC1", row, chem.CoFmN(ps.AC1, vol))
	// dt.SetCellFloat(pre+"R2C2", row, chem.CoFmN(ps.R2C2, vol))
	// dt.SetCellFloat(pre+"R2C2_B", row, chem.CoFmN(ps.R2C2_B, vol))
	// dt.SetCellFloat(pre+"R2C2_ABB", row, chem.CoFmN(ps.R2C2_ABB, vol))
}

func (ps *PKAVars) ConfigLog(sch *etable.Schema, pre string) {
	*sch = append(*sch, etable.Column{pre + "AC1act", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "cAMP", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "PKAact", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "AC1", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "R2C2", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "R2C2_B", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "R2C2_ABB", etensor.FLOAT64, nil, nil})
}

// PKAState is overall intracellular Ca-driven signaling states
// for PKA binding and phosphorylation with cAMP
// 32 state vars total
type PKAState struct {
	Cyt PKAVars `desc:"in cytosol -- volume = 0.08 fl = 48"`
	PSD PKAVars `desc:"in PSD -- volume = 0.02 fl = 12"`
}

func (ps *PKAState) Init() {
	ps.Cyt.Init(CytVol)
	ps.PSD.Init(PSDVol)
}

func (ps *PKAState) InitCode() {
	fmt.Printf("\nPKAState:\n")
	ps.Cyt.InitCode(CytVol, "Cyt")
	ps.PSD.InitCode(PSDVol, "PSD")
}

func (ps *PKAState) Zero() {
	ps.Cyt.Zero()
	ps.PSD.Zero()
}

func (ps *PKAState) Integrate(d *PKAState) {
	ps.Cyt.Integrate(&d.Cyt)
	ps.PSD.Integrate(&d.PSD)
}

func (ps *PKAState) Log(dt *etable.Table, row int) {
	ps.Cyt.Log(dt, CytVol, row, "Cyt_")
	ps.PSD.Log(dt, PSDVol, row, "PSD_")
}

func (ps *PKAState) ConfigLog(sch *etable.Schema) {
	ps.Cyt.ConfigLog(sch, "Cyt_")
	ps.PSD.ConfigLog(sch, "PSD_")
}

// PKAParams are the parameters governing the
// PKA binding and phosphorylation with cAMP
type PKAParams struct {
	CaMAC1      chem.React   `desc:"1: 3Ca-CaM + AC1 -> AC1act"`
	ATPcAMP     chem.React   `desc:"2: basal activity of ATP -> cAMP without AC1 enzyme"`
	R2C2_B      chem.React   `desc:"3: R2C2 + cAMP = cAMP-bind-site-B"`
	R2C2_B1     chem.React   `desc:"4: R2C2-cAMP B + cAMP -> BB = cAMP-bind-site-B[1]"`
	R2C2_A1     chem.React   `desc:"5: R2C2-cAMP B + cAMP -> AB = cAMP-bind-site-A[1]"`
	R2C2_A2     chem.React   `desc:"6: R2C2-cAMP BB + cAMP -> ABB = cAMP-bind-site-A[2]"`
	R2C2_B2     chem.React   `desc:"7: R2C2-cAMP AB + cAMP -> ABB = cAMP-bind-site-B[2]"`
	R2C2_A      chem.React   `desc:"8: R2C2-cAMP ABB + cAMP -> 4 = cAMP-bind-site-A"`
	R2C_A3      chem.React   `desc:"9: R2C-3cAMP -> R2C-4cAMP = cAMP-bind-site-A[3]"`
	R2_A4       chem.React   `desc:"10: R2-3cAMP -> R2-4cAMP = cAMP-bind-site-A[4]"`
	R2C_3       chem.React   `desc:"11: R2C-3cAMP + PKAact -> R2C2-3cAMP ABB (backwards) = Release-C1[1] -- Fig SI4 R2-3 -> R2C-3"`
	R2C_4       chem.React   `desc:"12: R2C-4cAMP + PKAact -> R2C2-4cAMP (backwards) = Release-C1"`
	R2_3        chem.React   `desc:"13: R2-3cAMP + PKAact -> R2C-3cAMP (backwards) = Release-C2[1]"`
	R2_4        chem.React   `desc:"14: R2-4cAMP + PKAact -> R2C-4cAMP (backwards) = Release-C2"`
	AC1ATP      chem.EnzRate `desc:"15: AC1act catalyzing ATP -> cAMP -- table SIg numbered 9 -> 15 -- note: uses EnzRate not std Enz -- does not consume AC1act"`
	PDEcAMP     chem.Enz     `desc:"16: PDE1act catalyzing cAMP -> AMP -- table SIg numbered 10 -> 16"`
	PKADiffuse  chem.Diffuse `desc:"PKA diffusion between Cyt and PSD"`
	CAMPDiffuse chem.Diffuse `desc:"cAMP diffusion between Cyt and PSD"`
}

func (cp *PKAParams) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// See React docs for more info
	cp.CaMAC1.SetVol(6, CytVol, 1)      // 1: 6 μM-1 = 0.125 -- NOTE: error in table (5) vs model 0.10416
	cp.ATPcAMP.Set(4.0e-7, 0)           // 2: called "leak"
	cp.R2C2_B.SetVol(0.2, CytVol, 0.1)  // 3: 0.2 μM-1 = 0.0041667 = cAMP-bind-site-B
	cp.R2C2_B1.SetVol(0.1, CytVol, 0.2) // 4: 0.1 μM-1 = 0.002083, cAMP-bind-site-B[1]
	cp.R2C2_A1.SetVol(2, CytVol, 5)     // 5: 2 μM-1 = 0.041667 = cAMP-bind-site-A[1]
	cp.R2C2_A2.SetVol(4, CytVol, 5)     // 6: 4 μM-1 = 0.083333 = cAMP-bind-site-A[2]
	cp.R2C2_B2.SetVol(0.1, CytVol, 0.1) // 7: 0.1 μM-1 = 0.002083 = cAMP-bind-site-B[2]
	cp.R2C2_A.SetVol(2, CytVol, 10)     // 8: 2 μM-1 = 0.041667 = cAMP-bind-site-A
	cp.R2C_A3.SetVol(20, CytVol, 1)     // 9: 20 μM-1 = 0.41667 = cAMP-bind-site-A[3]
	cp.R2_A4.SetVol(200, CytVol, 0.1)   // 10: 200 μM-1 = 4.1667 = cAMP-bind-site-A[4]

	cp.R2C_3.SetVol(10, CytVol, 2) // 11: 10 μM-1 = 0.20833 = Release-C1[1]
	cp.R2C_4.SetVol(1, CytVol, 20) // 12: 1 μM-1 = 0.020833 = Release-C1
	cp.R2_3.SetVol(20, CytVol, 1)  // 13: 20 μM-1 = 0.41667 = Release-C2[1]
	cp.R2_4.SetVol(2, CytVol, 10)  // 14: 2 μM-1 = 0.041667 = Release-C2

	cp.AC1ATP.SetKmVol(40, CytVol, 40, 10)  // 15: Km = 40 * 48 (ac) = 0.026042
	cp.PDEcAMP.SetKmVol(10, CytVol, 80, 20) // 16: Km = 10 = 0.20834

	cp.PKADiffuse.SetSym(32.0 / 0.0225)
	cp.CAMPDiffuse.SetSym(500.0 / 0.0225)
}

// StepPKA does the PKA + cAMP reactions, in a given region
// cCaM, dCaM = current, delta 3CaCaM from CaMKIIVars
func (cp *PKAParams) StepPKA(vol float64, c, d *PKAVars, cCaM float64, dCaM *float64) {
	kf := CytVol / vol
	var dummy float64
	cp.CaMAC1.StepK(kf, c.AC1, cCaM, c.AC1act, &d.AC1, dCaM, &d.AC1act)                   // 1
	cp.ATPcAMP.StepK(kf, c.ATP, 1, c.CAMP, &d.ATP, &dummy, &d.CAMP)                       // 2
	cp.R2C2_B.StepK(kf, c.CAMP, c.R2C2, c.R2C2_B, &d.CAMP, &d.R2C2, &d.R2C2_B)            // 3
	cp.R2C2_B1.StepK(kf, c.CAMP, c.R2C2_B, c.R2C2_BB, &d.CAMP, &d.R2C2, &d.R2C2_BB)       // 4
	cp.R2C2_A1.StepK(kf, c.CAMP, c.R2C2_B, c.R2C2_AB, &d.CAMP, &d.R2C2, &d.R2C2_AB)       // 5
	cp.R2C2_A2.StepK(kf, c.CAMP, c.R2C2_BB, c.R2C2_ABB, &d.CAMP, &d.R2C2_BB, &d.R2C2_ABB) // 6
	cp.R2C2_B2.StepK(kf, c.CAMP, c.R2C2_AB, c.R2C2_ABB, &d.CAMP, &d.R2C2_AB, &d.R2C2_ABB) // 7
	cp.R2C2_A.StepK(kf, c.CAMP, c.R2C2_ABB, c.R2C2_4, &d.CAMP, &d.R2C2_ABB, &d.R2C2_4)    // 8
	cp.R2C_A3.StepK(kf, c.CAMP, c.R2C_3, c.R2C_4, &d.CAMP, &d.R2C_3, &d.R2C_4)            // 9
	cp.R2_A4.StepK(kf, c.CAMP, c.R2_3, c.R2_4, &d.CAMP, &d.R2_3, &d.R2_4)                 // 10

	cp.R2C_3.StepK(kf, c.PKAact, c.R2C_3, c.R2C2_ABB, &d.PKAact, &d.R2C_3, &d.R2C2_ABB) // 11
	cp.R2C_4.StepK(kf, c.PKAact, c.R2C_4, c.R2C2_4, &d.PKAact, &d.R2C_4, &d.R2C2_4)     // 12
	cp.R2_3.StepK(kf, c.PKAact, c.R2_3, c.R2C_3, &d.PKAact, &d.R2_3, &d.R2C_3)          // 13
	cp.R2_4.StepK(kf, c.PKAact, c.R2_4, c.R2C_4, &d.PKAact, &d.R2_4, &d.R2C_4)          // 14

	// cs, ce, ds, dp, cc
	cp.AC1ATP.StepK(kf, c.ATP, c.AC1act, &d.ATP, &d.CAMP, &c.AC1ATPC)
	// cs, ce, cc, cp -> ds, de, dc, dp
	cp.PDEcAMP.StepK(kf, c.CAMP, c.PDEact, c.PDEcAMPC, c.AMP, &d.CAMP, &d.PDEact, &d.PDEcAMPC, &d.AMP)
}

// StepDiffuse does diffusion update, c=current, d=delta
func (cp *PKAParams) StepDiffuse(c, d *PKAState) {
	cp.PKADiffuse.Step(c.Cyt.R2C2, c.PSD.R2C2, CytVol, PSDVol, &d.Cyt.R2C2, &d.PSD.R2C2)
	cp.PKADiffuse.Step(c.Cyt.R2C2_B, c.PSD.R2C2_B, CytVol, PSDVol, &d.Cyt.R2C2_B, &d.PSD.R2C2_B)
	cp.PKADiffuse.Step(c.Cyt.R2C2_BB, c.PSD.R2C2_BB, CytVol, PSDVol, &d.Cyt.R2C2_BB, &d.PSD.R2C2_BB)
	cp.PKADiffuse.Step(c.Cyt.R2C2_AB, c.PSD.R2C2_AB, CytVol, PSDVol, &d.Cyt.R2C2_AB, &d.PSD.R2C2_AB)
	cp.PKADiffuse.Step(c.Cyt.R2C2_ABB, c.PSD.R2C2_ABB, CytVol, PSDVol, &d.Cyt.R2C2_ABB, &d.PSD.R2C2_ABB)
	cp.PKADiffuse.Step(c.Cyt.R2C2_4, c.PSD.R2C2_4, CytVol, PSDVol, &d.Cyt.R2C2_4, &d.PSD.R2C2_4)
	cp.PKADiffuse.Step(c.Cyt.R2C_3, c.PSD.R2C_3, CytVol, PSDVol, &d.Cyt.R2C_3, &d.PSD.R2C_3)
	cp.PKADiffuse.Step(c.Cyt.R2C_4, c.PSD.R2C_4, CytVol, PSDVol, &d.Cyt.R2C_4, &d.PSD.R2C_4)
	cp.PKADiffuse.Step(c.Cyt.R2_3, c.PSD.R2_3, CytVol, PSDVol, &d.Cyt.R2_3, &d.PSD.R2_3)
	cp.PKADiffuse.Step(c.Cyt.R2_4, c.PSD.R2_4, CytVol, PSDVol, &d.Cyt.R2_4, &d.PSD.R2_4)

	cp.CAMPDiffuse.Step(c.Cyt.CAMP, c.PSD.CAMP, CytVol, PSDVol, &d.Cyt.CAMP, &d.PSD.CAMP)
	cp.CAMPDiffuse.Step(c.Cyt.PDEact, c.PSD.PDEact, CytVol, PSDVol, &d.Cyt.PDEact, &d.PSD.PDEact)
}

// Step does full PKA updating, c=current, d=delta
func (cp *PKAParams) Step(c, d *PKAState, cCaM, dCaM *CaMState) {
	cp.StepPKA(CytVol, &c.Cyt, &d.Cyt, cCaM.Cyt.CaM[3], &dCaM.Cyt.CaM[3])
	cp.StepPKA(PSDVol, &c.PSD, &d.PSD, cCaM.PSD.CaM[3], &dCaM.PSD.CaM[3])
	cp.StepDiffuse(c, d)
}
