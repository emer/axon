// Copyright (c) 2021 The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// CaSigState is entire intracellular Ca-driven signaling state
type CaSigState struct {
	CaMKII CaMKIIState `desc:"CaMKII state"`
	CaN    CaNState    `desc:"CaN = calcineurin state"`
	PKA    PKAState    `desc:"PKA = protein kinase A"`
	PP1    PP1State    `desc:"PP1 = protein phosphatase 1"`
	PP2A   float32     `desc:"PP2A = protein phosphatase 2A, only in Cyt"`
}

func (cs *CaSigState) Init() {
	cs.CaMKII.Init()
	cs.CaN.Init()
	cs.PKA.Init()
	cs.PP1.Init()
	cs.PP2A = 0.03
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

// Spine represents all of the state and parameters of the Spine
// involved in LTP / LTD
type Spine struct {
	CaMKII CaMKIIParams `desc:"CaMKII parameters"`
	CaN    CaNParams    `desc:"CaN calcineurin parameters"`
	PKA    PKAParams    `desc:"PKA = protein kinase A parameters"`
	PP1    PP1Params    `desc:"PP1 = protein phosphatase 1 parameters"`
	AMPAR  AMPARParams  `desc:"AMPAR parameters"`

	State   [2]SpineState `desc:"the current and next spine states"`
	CurIdx  int           `desc:"index of the current state in State -- toggled"`
	NextIdx int           `desc:"idex of the next state in State -- toggled"`
}

func (sp *Spine) Cur() *SpineState {
	return &sp.State[sp.CurIdx]
}

func (sp *Spine) Next() *SpineState {
	return &sp.State[sp.NextIdx]
}

func (sp *Spine) Defaults() {
	sp.CaMKII.Defaults()
	sp.CaN.Defaults()
	sp.PKA.Defaults()
	sp.PP1.Defaults()
	sp.AMPAR.Defaults()
}

func (sp *Spine) Init() {
	sp.CurIdx = 0
	sp.NextIdx = 1
	sp.Cur().Init()
}

// Step does one step of updating, given the current and next levels of calcium
// cCa, nCa = current, new Ca -- caller has to initialize nCa
func (sp *Spine) Step(cCa float32, nCa *float32) {
	sp.CurIdx = 1 - sp.CurIdx
	sp.NextIdx = 1 - sp.NextIdx
	c := sp.Cur()
	n := sp.Next()
	*n = *c // start from current
	sp.CaMKII.Step(&c.CaSig.CaMKII, &n.CaSig.CaMKII, &c.CaSig.PP1, cCa, c.CaSig.PP2A, nCa)
	sp.CaN.Step(&c.CaSig.CaN, &n.CaSig.CaN, &c.CaSig.CaMKII, &n.CaSig.CaMKII, cCa, nCa)
	sp.PKA.Step(&c.CaSig.PKA, &n.CaSig.PKA, &c.CaSig.CaMKII, &n.CaSig.CaMKII)
	sp.PP1.Step(&c.CaSig.PP1, &n.CaSig.PP1, &c.CaSig.PKA, &c.CaSig.CaN, c.CaSig.PP2A)
	sp.AMPAR.Step(&c.AMPAR, &n.AMPAR, &c.CaSig, c.CaSig.PP2A)
}
