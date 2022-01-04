// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/chem"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// CaState records the Ca levels
// 2 state vars total
type CaState struct {
	Cyt float64 `desc:"in cytosol"`
	PSD float64 `desc:"in PSD"`
}

func (cs *CaState) Init() {
	cs.Cyt = chem.CoToN(0.05, CytVol)
	cs.PSD = chem.CoToN(0.05, PSDVol)
}

func (cs *CaState) Zero() {
	cs.Cyt = 0
	cs.PSD = 0
}

func (cs *CaState) Integrate(d *CaState) {
	chem.Integrate(&cs.Cyt, d.Cyt)
	chem.Integrate(&cs.PSD, d.PSD)
}

func (cs *CaState) Log(dt *etable.Table, row int) {
	dt.SetCellFloat("Cyt_Ca", row, chem.CoFmN(cs.Cyt, CytVol))
	dt.SetCellFloat("PSD_Ca", row, chem.CoFmN(cs.PSD, PSDVol))
}

func (cs *CaState) ConfigLog(sch *etable.Schema) {
	*sch = append(*sch, etable.Column{"Cyt_Ca", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{"PSD_Ca", etensor.FLOAT64, nil, nil})
}

// CaBufParams manages soft buffering dynamics of calcium
type CaBufParams struct {
	Cyt chem.Buffer `desc:"Ca buffering in the cytosol"`
	PSD chem.Buffer `desc:"Ca buffering in the PSD"`
}

func (cp *CaBufParams) Defaults() {
	// note: verified constants from initial_routines/Ca2_efflux.g
	// using showmsg /efflux_PSD / cytosol and showfield /efflux_PSD *
	// and doing the math.. replicates corresponding behavior in model
	cp.Cyt.SetTargVol(0.05, CytVol)
	cp.Cyt.K = (1.0426e5 * 0.8) / 12
	cp.PSD.SetTargVol(0.05, PSDVol)
	cp.PSD.K = (1.7927e5 * 0.8) / 12
}

func (cp *CaBufParams) Step(c *CaState, d *CaState) {
	cp.Cyt.Step(c.Cyt, &d.Cyt)
	cp.PSD.Step(c.PSD, &d.PSD)
}
