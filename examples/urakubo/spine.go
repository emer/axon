// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/emergent/chem"
	"github.com/emer/etable/etable"
)

const (
	CytVol = 48 // volume of cytosol, in essentially arbitrary units
	PSDVol = 12 // volume of PSD
)

func init() {
	chem.IntegrationDt = 5e-6
}

// CaSigState is entire intracellular Ca-driven signaling state
// Total state vars: 2 + 32 + 14 + 32 + 14 + 1 = 95
type CaSigState struct {
	Ca     CaState     `desc:"Ca state"`
	CaMKII CaMKIIState `desc:"CaMKII state"`
	CaN    CaNState    `desc:"CaN = calcineurin state"`
	PKA    PKAState    `desc:"PKA = protein kinase A"`
	PP1    PP1State    `desc:"PP1 = protein phosphatase 1"`
	PP2A   float64     `desc:"PP2A = protein phosphatase 2A, only in Cyt"`
}

func (cs *CaSigState) Init() {
	cs.Ca.Init()
	cs.CaMKII.Init()
	cs.CaN.Init()
	cs.PKA.Init()
	cs.PP1.Init()
	cs.PP2A = chem.CoToN(0.03, CytVol)
}

func (cs *CaSigState) Zero() {
	cs.Ca.Zero()
	cs.CaMKII.Zero()
	cs.CaN.Zero()
	cs.PKA.Zero()
	cs.PP1.Zero()
	cs.PP2A = 0
}

func (cs *CaSigState) Integrate(d *CaSigState) {
	cs.Ca.Integrate(&d.Ca)
	cs.CaMKII.Integrate(&d.CaMKII)
	cs.CaN.Integrate(&d.CaN)
	cs.PKA.Integrate(&d.PKA)
	cs.PP1.Integrate(&d.PP1)
	chem.Integrate(&cs.PP2A, d.PP2A)
}

func (cs *CaSigState) Log(dt *etable.Table, row int) {
	cs.Ca.Log(dt, row)
	cs.CaMKII.Log(dt, row)
	cs.CaN.Log(dt, row)
	cs.PKA.Log(dt, row)
	cs.PP1.Log(dt, row)
}

func (cs *CaSigState) ConfigLog(sch *etable.Schema) {
	cs.Ca.ConfigLog(sch)
	cs.CaMKII.ConfigLog(sch)
	cs.CaN.ConfigLog(sch)
	cs.PKA.ConfigLog(sch)
	cs.PP1.ConfigLog(sch)
}

// SpineState is entire state of spine including Ca signaling and AMPAR
// Total state vars: 95 + 20 = 115
type SpineState struct {
	NMDAR NMDARState `desc:"NMDA receptor state"`
	CaSig CaSigState `desc:"calcium signaling systems"`
	AMPAR AMPARState `desc:"AMPA receptor state"`
	Vm    float64    `desc:"clamped Vm in spine"`
	Spike float64    `desc:"discrete spike firing"`
}

func (ss *SpineState) Init() {
	ss.NMDAR.Init()
	ss.CaSig.Init()
	ss.AMPAR.Init()
	ss.Vm = -65
	ss.Spike = 0
}

func (ss *SpineState) Zero() {
	ss.NMDAR.Zero()
	ss.CaSig.Zero()
	ss.AMPAR.Zero()
	ss.Vm = -65
	ss.Spike = 0
}

func (ss *SpineState) Integrate(d *SpineState) {
	ss.CaSig.Integrate(&d.CaSig)
	ss.AMPAR.Integrate(&d.AMPAR)
}

func (ss *SpineState) Log(dt *etable.Table, row int) {
	dt.SetCellFloat("Vm", row, ss.Vm)
	dt.SetCellFloat("Spike", row, ss.Spike)
	ss.NMDAR.Log(dt, row)
	ss.CaSig.Log(dt, row)
	ss.AMPAR.Log(dt, row)
}

func (ss *SpineState) ConfigLog(sch *etable.Schema) {
	// *sch = append(*sch, etable.Column{"Vm", etensor.FLOAT64, nil, nil})
	// *sch = append(*sch, etable.Column{"Spike", etensor.FLOAT64, nil, nil})
	ss.NMDAR.ConfigLog(sch)
	ss.CaSig.ConfigLog(sch)
	ss.AMPAR.ConfigLog(sch)
}

// Spine represents all of the state and parameters of the Spine
// involved in LTP / LTD
type Spine struct {
	NMDAR  NMDARParams  `desc:"NMDA receptors"`
	Ca     CaParams     `desc:"Ca buffering and diffusion parameters"`
	CaMKII CaMKIIParams `desc:"CaMKII parameters"`
	CaN    CaNParams    `desc:"CaN calcineurin parameters"`
	PKA    PKAParams    `desc:"PKA = protein kinase A parameters"`
	PP1    PP1Params    `desc:"PP1 = protein phosphatase 1 parameters"`
	AMPAR  AMPARParams  `desc:"AMPAR parameters"`

	States SpineState `desc:"the current spine states"`
	Deltas SpineState `desc:"the derivative changes in spine states"`
}

func (sp *Spine) Defaults() {
	sp.NMDAR.Defaults()
	sp.Ca.Defaults()
	sp.CaMKII.Defaults()
	sp.CaN.Defaults()
	sp.PKA.Defaults()
	sp.PP1.Defaults()
	sp.AMPAR.Defaults()
	fmt.Printf("Integration Dt = %g (%g steps per msec)\n", chem.IntegrationDt, 0.001/chem.IntegrationDt)
}

func (sp *Spine) Init() {
	sp.States.Init()
	sp.Deltas.Zero()
}

// Step computes the new Delta values
func (sp *Spine) Step() {
	sp.Deltas.Zero()

	vm := sp.States.Vm
	spike := sp.States.Spike > 0 // note: nmda has a 5msec delay!
	// fmt.Printf("vm: %g  spike: %g\n", vm, sp.States.Spike)

	sp.NMDAR.Step(&sp.States.NMDAR, vm, sp.States.CaSig.Ca.PSD, sp.States.CaSig.CaMKII.PSD.Ca[2].CaM, sp.States.CaSig.CaMKII.PSD.Ca[3].CaM, spike)
	sp.CaMKII.Step(&sp.States.CaSig.CaMKII, &sp.Deltas.CaSig.CaMKII, &sp.States.CaSig.Ca, &sp.Deltas.CaSig.Ca, &sp.States.CaSig.PP1, &sp.Deltas.CaSig.PP1, sp.States.CaSig.PP2A, &sp.Deltas.CaSig.PP2A)
	sp.CaN.Step(&sp.States.CaSig.CaN, &sp.Deltas.CaSig.CaN, &sp.States.CaSig.CaMKII, &sp.Deltas.CaSig.CaMKII, &sp.States.CaSig.Ca, &sp.Deltas.CaSig.Ca)
	sp.PKA.Step(&sp.States.CaSig.PKA, &sp.Deltas.CaSig.PKA, &sp.States.CaSig.CaMKII, &sp.Deltas.CaSig.CaMKII)
	sp.PP1.Step(&sp.States.CaSig.PP1, &sp.Deltas.CaSig.PP1, &sp.States.CaSig.PKA, &sp.Deltas.CaSig.PKA, &sp.States.CaSig.CaN, &sp.Deltas.CaSig.CaN, sp.States.CaSig.PP2A, &sp.Deltas.CaSig.PP2A)
	sp.Ca.Step(&sp.States.CaSig.Ca, &sp.Deltas.CaSig.Ca)
	sp.AMPAR.Step(&sp.States.AMPAR, &sp.Deltas.AMPAR, &sp.States.CaSig, sp.States.CaSig.PP2A)
}

// Integrate integrates the deltas
func (sp *Spine) Integrate() {
	sp.States.Integrate(&sp.Deltas)
}

// StepTime steps and integrates for given amount of time in secs
func (sp *Spine) StepTime(secs float64) {
	for t := 0.0; t < secs; t += chem.IntegrationDt {
		sp.Step()
		sp.Integrate()
	}
}

func (sp *Spine) Log(dt *etable.Table, row int) {
	sp.States.Log(dt, row)
}

func (sp *Spine) ConfigLog(sch *etable.Schema) {
	sp.States.ConfigLog(sch)
}
