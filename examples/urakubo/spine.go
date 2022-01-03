// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// The Stater interface defines the functions implemented for State
// structures.  This interface is not actually used, but is for
// documentation purposes.
type Stater interface {
	// Init Initializes the state to starting default values (concentrations)
	Init()

	// Zero sets all state variables to zero -- called for deltas after integration
	Zero()

	// Integrate is called with the deltas -- each state value calls Integrate()
	// to update from deltas.
	Integrate(d Stater)

	// Log records relevant state variables in given table at given row
	Log(dt *etable.Table, row int)

	// ConfigLog configures the table Schema to add column(s) for what is logged
	ConfigLog(sch *etable.Schema)
}

// The Paramer interface defines the functions implemented for Params
// structures.  This interface is not actually used, but is for
// documentation purposes.
type Paramer interface {
	// Defaults sets default parameters
	Defaults()

	// Step computes deltas d based on current values c
	Step(c, d Paramer)
}

// CaState records the Ca levels
type CaState struct {
	Cyt float64 `desc:"in cytosol"`
	PSD float64 `desc:"in PSD"`
}

func (cs *CaState) Init() {
	cs.Cyt = CoToN(0.05, CytVol)
	cs.PSD = CoToN(0.05, PSDVol)
}

func (cs *CaState) Zero() {
	cs.Cyt = 0
	cs.PSD = 0
}

func (cs *CaState) Integrate(d *CaState) {
	Integrate(&cs.Cyt, d.Cyt)
	Integrate(&cs.PSD, d.PSD)
}

func (cs *CaState) Log(dt *etable.Table, row int) {
	dt.SetCellFloat("Cyt_Ca", row, CoFmN(cs.Cyt, CytVol))
	dt.SetCellFloat("PSD_Ca", row, CoFmN(cs.PSD, PSDVol))
}

func (cs *CaState) ConfigLog(sch *etable.Schema) {
	*sch = append(*sch, etable.Column{"Cyt_Ca", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{"PSD_Ca", etensor.FLOAT64, nil, nil})
}

// CaSigState is entire intracellular Ca-driven signaling state
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
	cs.PP2A = CoToN(0.03, CytVol)
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
	Integrate(&cs.PP2A, d.PP2A)
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
type SpineState struct {
	CaSig CaSigState `desc:"calcium signaling systems"`
	AMPAR AMPARState `desc:"AMPA receptor state"`
}

func (ss *SpineState) Init() {
	ss.CaSig.Init()
	ss.AMPAR.Init()
}

func (ss *SpineState) Zero() {
	ss.CaSig.Zero()
	ss.AMPAR.Zero()
}

func (ss *SpineState) Integrate(d *SpineState) {
	ss.CaSig.Integrate(&d.CaSig)
	ss.AMPAR.Integrate(&d.AMPAR)
}

func (ss *SpineState) Log(dt *etable.Table, row int) {
	ss.CaSig.Log(dt, row)
	ss.AMPAR.Log(dt, row)
}

func (ss *SpineState) ConfigLog(sch *etable.Schema) {
	ss.CaSig.ConfigLog(sch)
	ss.AMPAR.ConfigLog(sch)
}

// Spine represents all of the state and parameters of the Spine
// involved in LTP / LTD
type Spine struct {
	CaMKII CaMKIIParams `desc:"CaMKII parameters"`
	CaN    CaNParams    `desc:"CaN calcineurin parameters"`
	PKA    PKAParams    `desc:"PKA = protein kinase A parameters"`
	PP1    PP1Params    `desc:"PP1 = protein phosphatase 1 parameters"`
	AMPAR  AMPARParams  `desc:"AMPAR parameters"`

	States SpineState `desc:"the current spine states"`
	Deltas SpineState `desc:"the derivative changes in spine states"`
}

func (sp *Spine) Defaults() {
	sp.CaMKII.Defaults()
	sp.CaN.Defaults()
	sp.PKA.Defaults()
	sp.PP1.Defaults()
	sp.AMPAR.Defaults()
	fmt.Printf("Integration Dt: %g\n", IntegrationDt)
}

func (sp *Spine) Init() {
	sp.States.Init()
	sp.Deltas.Zero()
}

// AddCa injects calcium into the deltas -- call before calling Integrate
func (sp *Spine) AddCa(cyt, psd float64) {
	sp.Deltas.CaSig.Ca.Cyt += cyt
	sp.Deltas.CaSig.Ca.PSD += psd
}

// Step computes the new Delta values
func (sp *Spine) Step() {
	sp.Deltas.Zero()
	sp.CaMKII.Step(&sp.States.CaSig.CaMKII, &sp.Deltas.CaSig.CaMKII, &sp.States.CaSig.Ca, &sp.Deltas.CaSig.Ca, &sp.States.CaSig.PP1, &sp.Deltas.CaSig.PP1, sp.States.CaSig.PP2A, &sp.Deltas.CaSig.PP2A)
	sp.CaN.Step(&sp.States.CaSig.CaN, &sp.Deltas.CaSig.CaN, &sp.States.CaSig.CaMKII, &sp.Deltas.CaSig.CaMKII, &sp.States.CaSig.Ca, &sp.Deltas.CaSig.Ca)
	sp.PKA.Step(&sp.States.CaSig.PKA, &sp.Deltas.CaSig.PKA, &sp.States.CaSig.CaMKII, &sp.Deltas.CaSig.CaMKII)
	sp.PP1.Step(&sp.States.CaSig.PP1, &sp.Deltas.CaSig.PP1, &sp.States.CaSig.PKA, &sp.Deltas.CaSig.PKA, &sp.States.CaSig.CaN, &sp.Deltas.CaSig.CaN, sp.States.CaSig.PP2A, &sp.Deltas.CaSig.PP2A)
	// sp.AMPAR.Step(&sp.States.AMPAR, &sp.Deltas.AMPAR, &sp.States.CaSig, sp.States.CaSig.PP2A)
}

// Integrate integrates the deltas
func (sp *Spine) Integrate() {
	sp.States.Integrate(&sp.Deltas)
}

// StepTime steps and integrates for given amount of time in secs
func (sp *Spine) StepTime(secs float64) {
	for t := 0.0; t < secs; t += IntegrationDt {
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
