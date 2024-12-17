// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/patgen"
)

// PFCMaintEnv implements a simple store-maintain-recall active maintenance task
type PFCMaintEnv struct {

	// name of environment -- Train or Test
	Name string

	// training or testing env?
	Mode etime.Modes

	// sequence counter is for the outer loop of maint per item
	Sequence env.Counter `display:"inline"`

	// trial counter is for the maint step within item
	Trial env.Counter `display:"inline"`

	// number of different items to maintain
	NItems int

	// number of trials to maintain
	NTrials int

	// state rep, number of units, Y
	NUnitsY int `display:"-"`

	// state rep, number of units, X
	NUnitsX int `display:"-"`

	// total number of units
	NUnits int `display:"-"`

	// item patterns
	Pats table.Table

	// random number generator for the env -- all random calls must use this
	Rand randx.SysRand `display:"-"`

	// random seed
	RandSeed int64 `edit:"-"`

	// named states: ACCPos, ACCNeg
	States map[string]*tensor.Float32
}

func (ev *PFCMaintEnv) Label() string { return ev.Name }

func (ev *PFCMaintEnv) String() string {
	return fmt.Sprintf("%d", ev.Trial.Cur)
}

func (ev *PFCMaintEnv) Defaults() {
	ev.NItems = 10
	ev.NTrials = 10
	ev.NUnitsY = 5
	ev.NUnitsX = 5
	ev.NUnits = ev.NUnitsY * ev.NUnitsX
}

// Config configures the world
func (ev *PFCMaintEnv) Config(mode etime.Modes, rndseed int64) {
	ev.Mode = mode
	ev.RandSeed = rndseed
	ev.Rand.NewRand(ev.RandSeed)
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Item"] = tensor.NewFloat32(ev.NUnitsY, ev.NUnitsX)
	ev.States["Time"] = tensor.NewFloat32(ev.NUnitsY, ev.NTrials)
	ev.States["GPi"] = tensor.NewFloat32(ev.NUnitsY, ev.NUnitsX)
	ev.Sequence.Max = ev.NItems
	ev.Trial.Max = ev.NTrials
	ev.ConfigPats()
}

func (ev *PFCMaintEnv) ConfigPats() {
	npats := ev.NItems
	ev.Pats.Init()
	ev.Pats.DeleteAll()
	ev.Pats.AddStringColumn("Name")
	ev.Pats.AddFloat32Column("Item", ev.NUnitsY, ev.NUnitsX)
	ev.Pats.SetNumRows(npats)

	pctAct := float32(0.2)
	minPctDiff := float32(0.5)

	nUn := ev.NUnitsY * ev.NUnitsX
	nOn := patgen.NFromPct(pctAct, nUn)
	minDiff := patgen.NFromPct(minPctDiff, nOn)

	patgen.PermutedBinaryMinDiff(ev.Pats.Columns.At("Item").(*tensor.Float32), nOn, 1, 0, minDiff)
}

func (ev *PFCMaintEnv) Init(run int) {
	ev.Sequence.Init()
	ev.Trial.Init()
}

func (ev *PFCMaintEnv) State(el string) tensor.Values {
	return ev.States[el]
}

// RenderLocalist renders localist * NUnitsPer
func (ev *PFCMaintEnv) RenderLocalist(name string, idx int) {
	av := ev.States[name]
	av.SetZeros()
	for yi := 0; yi < ev.NUnitsY; yi++ {
		av.Set(1, yi, idx)
	}
}

// RenderState renders the given condition, trial
func (ev *PFCMaintEnv) RenderState(item, trial int) {
	st := ev.States["Item"]
	st.CopyFrom(ev.Pats.Column("Item").RowTensor(item))
	ev.RenderLocalist("Time", trial)
	st = ev.States["GPi"]
	st.CopyFrom(ev.Pats.Column("Item").RowTensor(item))
	if trial == 0 {
		st.SetZeros()
	}
}

// Step does one step -- must set Trial.Cur first if doing testing
func (ev *PFCMaintEnv) Step() bool {
	ev.RenderState(ev.Sequence.Cur, ev.Trial.Cur)
	ev.Sequence.Same()
	if ev.Trial.Incr() {
		ev.Sequence.Incr()
	}
	return true
}

func (ev *PFCMaintEnv) Action(action string, nop tensor.Values) {
}
