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
	CaM    CaMState    `desc:"CaM calmodulin state"`
	CaMKII CaMKIIState `desc:"CaMKII state"`
	DAPK1  DAPK1State  `desc:"DAPK1 state"`
	CaN    CaNState    `desc:"CaN = calcineurin state"`
	PKA    PKAState    `desc:"PKA = protein kinase A"`
	PP1    PP1State    `desc:"PP1 = protein phosphatase 1"`
	PP2A   float64     `desc:"PP2A = protein phosphatase 2A, only in Cyt"`
}

func (cs *CaSigState) Init() {
	cs.Ca.Init()
	cs.CaM.Init()
	cs.CaMKII.Init()
	cs.DAPK1.Init()
	cs.CaN.Init()
	cs.PKA.Init()
	cs.PP1.Init()

	cs.PP2A = chem.CoToN(0.03, CytVol)

	if TheOpts.InitBaseline {
		if TheOpts.UseDAPK1 {
			cs.PP2A = chem.CoToN(0.01849, CytVol)
		} else {
			cs.PP2A = chem.CoToN(0.02239, CytVol)
		}
	}
}

func (cs *CaSigState) InitCode() {
	cs.CaM.InitCode()
	cs.CaMKII.InitCode()
	cs.DAPK1.InitCode()
	cs.CaN.InitCode()
	cs.PKA.InitCode()
	cs.PP1.InitCode()
	fmt.Printf("\nCaSigState:\n")
	fmt.Printf("\tcs.PP2A = chem.CoToN(%.4g, CytVol)\n", chem.CoFmN(cs.PP2A, CytVol))
}

func (cs *CaSigState) Zero() {
	cs.Ca.Zero()
	cs.CaM.Zero()
	cs.CaMKII.Zero()
	cs.DAPK1.Zero()
	cs.CaN.Zero()
	cs.PKA.Zero()
	cs.PP1.Zero()
	cs.PP2A = 0
}

func (cs *CaSigState) Integrate(d *CaSigState) {
	cs.Ca.Integrate(&d.Ca)
	cs.CaM.Integrate(&d.CaM)
	cs.CaMKII.Integrate(&d.CaMKII)
	cs.DAPK1.Integrate(&d.DAPK1)
	cs.CaN.Integrate(&d.CaN)
	cs.PKA.Integrate(&d.PKA)
	cs.PP1.Integrate(&d.PP1)
	chem.Integrate(&cs.PP2A, d.PP2A)
}

func (cs *CaSigState) Log(dt *etable.Table, row int) {
	cs.Ca.Log(dt, row)
	cs.CaM.Log(dt, row)
	cs.CaMKII.Log(dt, row)
	cs.DAPK1.Log(dt, row)
	cs.CaN.Log(dt, row)
	cs.PKA.Log(dt, row)
	cs.PP1.Log(dt, row)
}

func (cs *CaSigState) ConfigLog(sch *etable.Schema) {
	cs.Ca.ConfigLog(sch)
	cs.CaM.ConfigLog(sch)
	cs.CaMKII.ConfigLog(sch)
	cs.DAPK1.ConfigLog(sch)
	cs.CaN.ConfigLog(sch)
	cs.PKA.ConfigLog(sch)
	cs.PP1.ConfigLog(sch)
}

// SpineState is entire state of spine including Ca signaling and AMPAR
// Total state vars: 95 + 20 = 115
type SpineState struct {
	Time      float64    `desc:"internal time counter, in seconds, incremented by Dt"`
	NMDAR     NMDARState `desc:"NMDA receptor state"`
	CaSig     CaSigState `desc:"calcium signaling systems"`
	AMPAR     AMPARState `desc:"AMPA receptor state"`
	VmS       float64    `desc:"Vm in spine"`
	PreSpike  float64    `desc:"discrete spike firing -- 0 = no spike, 1 = spike"`
	PreSpikeT float64    `desc:"time of last spike firing -- needed to prevent repeated spiking from same singal"`
}

func (ss *SpineState) Init() {
	ss.Time = 0
	ss.NMDAR.Init()
	ss.CaSig.Init()
	ss.AMPAR.Init()
	ss.VmS = -65
	ss.PreSpike = 0
	ss.PreSpikeT = -1
}

func (ss *SpineState) InitCode() {
	ss.CaSig.InitCode()
	ss.AMPAR.InitCode()
}

func (ss *SpineState) Zero() {
	ss.Time = 0
	ss.NMDAR.Zero()
	ss.CaSig.Zero()
	ss.AMPAR.Zero()
	ss.VmS = 0
	ss.PreSpike = 0
	ss.PreSpikeT = 0
}

func (ss *SpineState) Integrate(d *SpineState) {
	ss.Time += chem.IntegrationDt
	ss.NMDAR.Integrate(&d.NMDAR)
	ss.CaSig.Integrate(&d.CaSig)
	ss.AMPAR.Integrate(&d.AMPAR)
}

func (ss *SpineState) Log(dt *etable.Table, row int) {
	dt.SetCellFloat("VmS", row, ss.VmS)
	dt.SetCellFloat("PreSpike", row, ss.PreSpike)
	ss.NMDAR.Log(dt, row)
	ss.CaSig.Log(dt, row)
	ss.AMPAR.Log(dt, row)
}

func (ss *SpineState) ConfigLog(sch *etable.Schema) {
	*sch = append(*sch, etable.Column{"VmS", etensor.FLOAT64, nil, nil})
	*sch = append(*sch, etable.Column{"PreSpike", etensor.FLOAT64, nil, nil})
	ss.NMDAR.ConfigLog(sch)
	ss.CaSig.ConfigLog(sch)
	ss.AMPAR.ConfigLog(sch)
}

// Spine represents all of the state and parameters of the Spine
// involved in LTP / LTD
type Spine struct {
	NMDAR  NMDARParams  `desc:"NMDA receptors"`
	Ca     CaParams     `desc:"Ca buffering and diffusion parameters"`
	CaM    CaMParams    `desc:"CaM calmodulin Ca binding parameters"`
	CaMKII CaMKIIParams `desc:"CaMKII parameters"`
	DAPK1  DAPK1Params  `desc:"DAPK1 parameters"`
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
	sp.CaM.Defaults()
	sp.CaMKII.Defaults()
	sp.DAPK1.Defaults()
	sp.CaN.Defaults()
	sp.PKA.Defaults()
	sp.PP1.Defaults()
	sp.AMPAR.Defaults()
	// fmt.Printf("Integration Dt = %g (%g steps per msec)\n", chem.IntegrationDt, 0.001/chem.IntegrationDt)
}

func (sp *Spine) Init() {
	sp.States.Init()
	sp.Deltas.Zero()
	sp.Ca.Init()                    // drivers
	sp.NMDAR.Init(&sp.States.NMDAR) // special init
}

func (sp *Spine) InitCode() {
	sp.States.InitCode()
}

// Step computes the new Delta values
func (sp *Spine) Step() {
	sp.Deltas.Zero()

	vms := sp.States.VmS
	preSpike := false
	if sp.States.PreSpike > 0 {
		if sp.States.Time-sp.States.PreSpikeT > 0.003 { // refractory period
			preSpike = true
			sp.States.PreSpikeT = sp.States.Time
		}
	}

	sp.NMDAR.Step(&sp.States.NMDAR, &sp.Deltas.NMDAR, vms, chem.CoFmN(sp.States.CaSig.Ca.PSD, PSDVol), chem.CoFmN(sp.States.CaSig.CaM.PSD.CaM[2], PSDVol), chem.CoFmN(sp.States.CaSig.CaM.PSD.CaM[3], PSDVol), preSpike, &sp.Deltas.CaSig.Ca.PSD)

	sp.CaM.Step(&sp.States.CaSig.CaM, &sp.Deltas.CaSig.CaM, &sp.States.CaSig.Ca, &sp.Deltas.CaSig.Ca)

	sp.CaMKII.Step(&sp.States.CaSig.CaMKII, &sp.Deltas.CaSig.CaMKII, &sp.States.CaSig.CaM, &sp.Deltas.CaSig.CaM, &sp.States.CaSig.Ca, &sp.Deltas.CaSig.Ca, &sp.States.CaSig.PP1, &sp.Deltas.CaSig.PP1, sp.States.CaSig.PP2A, sp.States.NMDAR.GluN2B, &sp.Deltas.CaSig.PP2A, &sp.Deltas.NMDAR.GluN2B)

	if TheOpts.UseDAPK1 {
		sp.DAPK1.Step(&sp.States.CaSig.DAPK1, &sp.Deltas.CaSig.DAPK1, &sp.States.CaSig.CaM, &sp.Deltas.CaSig.CaM, &sp.States.CaSig.Ca, &sp.Deltas.CaSig.Ca, &sp.States.CaSig.CaN, &sp.Deltas.CaSig.CaN, sp.States.CaSig.PP2A, sp.States.NMDAR.GluN2B, &sp.Deltas.CaSig.PP2A, &sp.Deltas.NMDAR.GluN2B)
	}

	sp.CaN.Step(&sp.States.CaSig.CaN, &sp.Deltas.CaSig.CaN, &sp.States.CaSig.CaM, &sp.Deltas.CaSig.CaM, &sp.States.CaSig.Ca, &sp.Deltas.CaSig.Ca)

	sp.PKA.Step(&sp.States.CaSig.PKA, &sp.Deltas.CaSig.PKA, &sp.States.CaSig.CaM, &sp.Deltas.CaSig.CaM)

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
