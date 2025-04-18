// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pfcmaint

import (
	"fmt"
	"log/slog"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/patterns"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
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

	// Di is the data parallel index.
	Di int

	// ndata is number of data parallel total.
	NData int

	// number of different items to maintain.
	NItems int

	// StartItem is item we start on, based on Di, NData.
	StartItem int

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
func (ev *PFCMaintEnv) Config(mode etime.Modes, di, ndata int, rndseed int64) {
	ev.Mode = mode
	ev.Di = di
	ev.NData = ndata
	ev.RandSeed = rndseed
	ev.Rand.NewRand(ev.RandSeed)
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Item"] = tensor.NewFloat32(ev.NUnitsY, ev.NUnitsX)
	ev.States["Time"] = tensor.NewFloat32(ev.NUnitsY, ev.NTrials)
	ev.States["GPi"] = tensor.NewFloat32(ev.NUnitsY, ev.NUnitsX)
	if ev.NItems%ndata != 0 {
		slog.Error("PFCMaintEnv: Number of items must be evenly divisible by NData", "NItems:", ev.NItems, "NData:", ndata)
	}
	nper := ev.NItems / ndata
	ev.Sequence.Max = nper
	ev.StartItem = ev.Di * nper
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
	nOn := patterns.NFromPct(float64(pctAct), nUn)
	minDiff := patterns.NFromPct(float64(minPctDiff), nOn)

	patterns.PermutedBinaryMinDiff(ev.Pats.Columns.At("Item"), nOn, 1, 0, minDiff)
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
	item := ev.StartItem + ev.Sequence.Cur
	ev.RenderState(item, ev.Trial.Cur)
	ev.Sequence.Same()
	if ev.Trial.Incr() {
		ev.Sequence.Incr()
	}
	return true
}

func (ev *PFCMaintEnv) Action(action string, nop tensor.Values) {
}
