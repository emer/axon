// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
)

// PFCMaintEnv implements a simple store-maintain-recall active maintenance task
type PFCMaintEnv struct {

	// name of environment -- Train or Test
	Nm string

	// training or testing env?
	Mode etime.Modes

	// sequence counter is for the outer loop of maint per item
	Sequence env.Ctr `view:"inline"`

	// trial counter is for the maint step within item
	Trial env.Ctr `view:"inline"`

	// number of different items to maintain
	NItems int

	// number of trials to maintain
	NTrials int

	// state rep, number of units, Y
	NUnitsY int `view:"-"`

	// state rep, number of units, X
	NUnitsX int `view:"-"`

	// total number of units
	NUnits int `view:"-"`

	// item patterns
	Pats etable.Table

	// pattern vocab
	PatVocab patgen.Vocab

	// random number generator for the env -- all random calls must use this
	Rand erand.SysRand `view:"-"`

	// random seed
	RndSeed int64 `edit:"-"`

	// named states: ACCPos, ACCNeg
	States map[string]*etensor.Float32
}

func (ev *PFCMaintEnv) Name() string {
	return ev.Nm
}

func (ev *PFCMaintEnv) Desc() string {
	return "PFCMaintEnv"
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
	ev.RndSeed = rndseed
	ev.Rand.NewRand(ev.RndSeed)
	ev.States = make(map[string]*etensor.Float32)
	ev.States["Item"] = etensor.NewFloat32([]int{ev.NUnitsY, ev.NUnitsX}, nil, []string{"Y", "X"})
	ev.States["Time"] = etensor.NewFloat32([]int{ev.NUnitsY, ev.NTrials}, nil, []string{"Y", "Time"})
	ev.States["GPi"] = etensor.NewFloat32([]int{ev.NUnitsY, ev.NUnitsX}, nil, []string{"Y", "X"})
	ev.Sequence.Max = ev.NItems
	ev.Trial.Max = ev.NTrials
	ev.ConfigPats()
}

func (ev *PFCMaintEnv) ConfigPats() {
	ev.PatVocab = patgen.Vocab{}

	npats := ev.NItems
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Item", etensor.FLOAT32, []int{ev.NUnitsY, ev.NUnitsX}, []string{"Y", "X"}},
	}
	ev.Pats.SetFromSchema(sch, npats)

	pctAct := float32(0.2)
	minPctDiff := float32(0.5)

	nUn := ev.NUnitsY * ev.NUnitsX
	nOn := patgen.NFmPct(pctAct, nUn)
	minDiff := patgen.NFmPct(minPctDiff, nOn)

	patgen.PermutedBinaryMinDiff(ev.Pats.ColByName("Item").(*etensor.Float32), nOn, 1, 0, minDiff)
}

func (ev *PFCMaintEnv) Validate() error {
	return nil
}

func (ev *PFCMaintEnv) Init(run int) {
	ev.Sequence.Init()
	ev.Trial.Init()
}

func (ev *PFCMaintEnv) Counter(scale env.TimeScales) (cur, prv int, changed bool) {
	switch scale {
	case env.Sequence:
		return ev.Sequence.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return 0, 0, false
}

func (ev *PFCMaintEnv) State(el string) etensor.Tensor {
	return ev.States[el]
}

// RenderLocalist renders localist * NUnitsPer
func (ev *PFCMaintEnv) RenderLocalist(name string, idx int) {
	av := ev.States[name]
	av.SetZeros()
	for yi := 0; yi < ev.NUnitsY; yi++ {
		av.Set([]int{yi, idx}, 1)
	}
}

// RenderState renders the given condition, trial
func (ev *PFCMaintEnv) RenderState(item, trial int) {
	st := ev.States["Item"]
	st.CopyFrom(ev.Pats.CellTensor("Item", item))
	ev.RenderLocalist("Time", trial)
	st = ev.States["GPi"]
	st.CopyFrom(ev.Pats.CellTensor("Item", item))
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

func (ev *PFCMaintEnv) Action(action string, nop etensor.Tensor) {
}

func (ev *PFCMaintEnv) ComputeDA(rew float32) {
}
