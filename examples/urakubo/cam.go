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

// CaMVars are intracellular Ca-driven signaling states for CaM binding with Ca
// From Urakubo: Ca2+ binding kinetics of CaM has extensively been analyzed
// (Linse et al., 1991; Holmes, 2000). CaM binds to four Ca2+ ions,
// but two or three Ca2+-binding is enough to activate CaM
// (James et al., 1995; Chin and Means, 2000).
// For simplicity, 3Ca2+⋅CaM is assumed to be an active form,
// and reactions for 4Ca2+⋅CaM are omitted.
type CaMVars struct {
	CaM [4]float64 `desc:"increasing levels of Ca binding to CaM, 0-3, [3] is active form"`
}

func (cs *CaMVars) Init(vol float64) {
	for i := range cs.CaM {
		cs.CaM[i] = 0
	}
	cs.CaM[0] = chem.CoToN(80, vol)

	if TheOpts.InitBaseline {
		cs.CaM[0] = chem.CoToN(89.2, vol)     // orig: 80
		cs.CaM[1] = chem.CoToN(1.142, vol)    // orig: 0
		cs.CaM[2] = chem.CoToN(0.007617, vol) // orig: 0
		cs.CaM[3] = chem.CoToN(3.654-05, vol) // orig: 0
	}
}

// Generate Code for Initializing
func (cs *CaMVars) InitCode(vol float64, pre string) {
	for i := range cs.CaM {
		fmt.Printf("\tcs.%s.CaM[%d] = chem.CoToN(%.4g, vol)\n", pre, i, chem.CoFmN(cs.CaM[i], vol))
	}
}

func (cs *CaMVars) Zero() {
	for i := range cs.CaM {
		cs.CaM[i] = 0
	}
}

func (cs *CaMVars) Integrate(d *CaMVars) {
	for i := range cs.CaM {
		chem.Integrate(&cs.CaM[i], d.CaM[i])
	}
}

func (cs *CaMVars) Log(dt *etable.Table, vol float64, row int, pre string) {
	// dt.SetCellFloat(pre+"CaM", row, chem.CoFmN(cs.CaM[0], vol))
	dt.SetCellFloat(pre+"CaMact", row, chem.CoFmN(cs.CaM[3], vol))
	// dt.SetCellFloat(pre+"CaCaM", row, chem.CoFmN(cs.Ca[1], vol))
	// dt.SetCellFloat(pre+"Ca2CaM", row, chem.CoFmN(cs.Ca[2], vol))
}

func (cs *CaMVars) ConfigLog(sch *etable.Schema, pre string) {
	// *sch = append(*sch, etable.Column{pre + "CaM", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{pre + "CaMact", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "CaCaM", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{pre + "Ca2CaM", etensor.FLOAT64, nil, nil})
}

// CaMState is overall intracellular Ca-driven signaling states
// for CaM in Cyt and PSD
// 32 state vars total
type CaMState struct {
	Cyt CaMVars `desc:"in cytosol -- volume = 0.08 fl = 48"`
	PSD CaMVars `desc:"in PSD -- volume = 0.02 fl = 12"`
}

func (cs *CaMState) Init() {
	cs.Cyt.Init(CytVol)
	cs.PSD.Init(PSDVol)

	if TheOpts.InitBaseline {
		if TheOpts.UseDAPK1 {
			vol := float64(CytVol)
			cs.Cyt.CaM[0] = chem.CoToN(80.2, vol)
			cs.Cyt.CaM[1] = chem.CoToN(1.027, vol)
			cs.Cyt.CaM[2] = chem.CoToN(0.006837, vol)
			cs.Cyt.CaM[3] = chem.CoToN(8.372e-06, vol)
			vol = PSDVol
			cs.PSD.CaM[0] = chem.CoToN(80.2, vol)
			cs.PSD.CaM[1] = chem.CoToN(1.027, vol)
			cs.PSD.CaM[2] = chem.CoToN(0.006837, vol)
			cs.PSD.CaM[3] = chem.CoToN(8.74e-06, vol)
		} else {

		}
		vol := float64(CytVol)
		cs.Cyt.CaM[3] = chem.CoToN(3.645e-05, vol)
		vol = PSDVol
		cs.PSD.CaM[3] = chem.CoToN(4.852e-05, vol)
	}
}

func (cs *CaMState) InitCode() {
	fmt.Printf("\nCaMState:\n")
	cs.Cyt.InitCode(CytVol, "Cyt")
	cs.PSD.InitCode(PSDVol, "PSD")
}

func (cs *CaMState) Zero() {
	cs.Cyt.Zero()
	cs.PSD.Zero()
}

func (cs *CaMState) Integrate(d *CaMState) {
	cs.Cyt.Integrate(&d.Cyt)
	cs.PSD.Integrate(&d.PSD)
}

func (cs *CaMState) Log(dt *etable.Table, row int) {
	cs.Cyt.Log(dt, CytVol, row, "Cyt_")
	cs.PSD.Log(dt, PSDVol, row, "PSD_")
}

func (cs *CaMState) ConfigLog(sch *etable.Schema) {
	cs.Cyt.ConfigLog(sch, "Cyt_")
	cs.PSD.ConfigLog(sch, "PSD_")
}

// CaMParams are the parameters governing the Ca+CaM binding
type CaMParams struct {
	CaCaM01    chem.React   `desc:"1: Ca+CaM -> 1CaCaM = CaM-bind-Ca"`
	CaCaM12    chem.React   `desc:"2: Ca+1CaM -> 2CaCaM = CaMCa-bind-Ca"`
	CaCaM23    chem.React   `desc:"3: Ca+2CaM -> 3CaCaM = CaMCa2-bind-Ca"`
	CaMDiffuse chem.Diffuse `desc:"CaM diffusion between Cyt and PSD"`
}

func (cp *CaMParams) Defaults() {
	// note: following are all in Cyt -- PSD is 4x for first values
	// See React docs for more info
	cp.CaCaM01.SetVol(51.202, CytVol, 200) // 1: 51.202 μM-1 = 1.0667, PSD 4.2667 = CaM-bind-Ca
	cp.CaCaM12.SetVol(133.3, CytVol, 1000) // 2: 133.3 μM-1 = 2.7771, PSD 11.108 = CaMCa-bind-Ca
	cp.CaCaM23.SetVol(25.6, CytVol, 400)   // 3: 25.6 μM-1 = 0.53333, PSD 2.1333 = CaMCa2-bind-Ca
	cp.CaMDiffuse.SetSym(130.0 / 0.0225)
}

// StepCaM does the bulk of Ca + CaM + CaM binding reactions, in a given region
// cCa, nCa = current next Ca
func (cp *CaMParams) StepCaM(vol float64, c, d *CaMVars, cCa float64, dCa *float64) {
	kf := CytVol / vol
	cp.CaCaM01.StepK(kf, c.CaM[0], cCa, c.CaM[1], &d.CaM[0], dCa, &d.CaM[1]) // 1
	cp.CaCaM12.StepK(kf, c.CaM[1], cCa, c.CaM[2], &d.CaM[1], dCa, &d.CaM[2]) // 2
	cp.CaCaM23.StepK(kf, c.CaM[2], cCa, c.CaM[3], &d.CaM[2], dCa, &d.CaM[3]) // 3
}

// StepDiffuse does Cyt <-> PSD diffusion
func (cp *CaMParams) StepDiffuse(c, d *CaMState) {
	for i := 0; i < 4; i++ {
		cc := c.Cyt.CaM[i]
		cd := c.PSD.CaM[i]
		dc := &d.Cyt.CaM[i]
		dd := &d.PSD.CaM[i]
		cp.CaMDiffuse.Step(cc, cd, CytVol, PSDVol, dc, dd)
	}
}

// Step does one step of CaM updating, c=current, d=delta
// pp2a = current cyt pp2a
func (cp *CaMParams) Step(c, d *CaMState, cCa, dCa *CaState) {
	cp.StepCaM(CytVol, &c.Cyt, &d.Cyt, cCa.Cyt, &dCa.Cyt)
	cp.StepCaM(PSDVol, &c.PSD, &d.PSD, cCa.PSD, &dCa.PSD)
	cp.StepDiffuse(c, d)
}
