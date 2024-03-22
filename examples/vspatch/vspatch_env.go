// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
)

// VSPatchEnv implements simple Go vs. NoGo input patterns to test BG learning.
type VSPatchEnv struct {

	// name of environment -- Train or Test
	Nm string

	// training or testing env?
	Mode etime.Modes

	// sequence counter is for the condition
	Sequence env.Ctr `view:"inline"`

	// trial counter is for the step within condition
	Trial env.Ctr `view:"inline"`

	// number of conditions, each of which can have a different reward value
	NConds int

	// number of trials
	NTrials int

	// condition current values
	CondVals []float32

	// state rep, number of units, Y
	NUnitsY int `view:"-"`

	// state rep, number of units, X
	NUnitsX int `view:"-"`

	// total number of units
	NUnits int `view:"-"`

	// condition, time-step patterns
	Pats etable.Table

	// pattern vocab
	PatVocab patgen.Vocab

	// random number generator for the env -- all random calls must use this
	Rand erand.SysRand `view:"-"`

	// random seed
	RndSeed int64 `edit:"-"`

	// named states: ACCPos, ACCNeg
	States map[string]*etensor.Float32

	// current reward value
	Rew float32 `edit:"-"`

	// reward prediction from model
	RewPred float32 `edit:"-"`

	// DA = reward prediction error: Rew - RewPred
	DA float32 `edit:"-"`
}

func (ev *VSPatchEnv) Name() string {
	return ev.Nm
}

func (ev *VSPatchEnv) Desc() string {
	return "VSPatchEnv"
}

func (ev *VSPatchEnv) Defaults() {
	ev.NConds = 4
	ev.NTrials = 3
	ev.NUnitsY = 5
	ev.NUnitsX = 5
	ev.NUnits = ev.NUnitsY * ev.NUnitsX
}

// SetRandCondVals sets values of each condition to random numbers
func (ev *VSPatchEnv) SetRandCondVals() {
	for i := 0; i < ev.NConds; i++ {
		ev.CondVals[i] = ev.Rand.Float32(-1)
	}
}

// Config configures the world
func (ev *VSPatchEnv) Config(mode etime.Modes, rndseed int64) {
	ev.Mode = mode
	ev.RndSeed = rndseed
	ev.Rand.NewRand(ev.RndSeed)
	ev.States = make(map[string]*etensor.Float32)
	ev.States["State"] = etensor.NewFloat32([]int{ev.NUnitsY, ev.NUnitsX}, nil, []string{"Y", "X"})
	ev.CondVals = make([]float32, ev.NConds)
	ev.Sequence.Max = ev.NConds
	ev.Trial.Max = ev.NTrials
	ev.SetRandCondVals()
	ev.ConfigPats()
}

func (ev *VSPatchEnv) ConfigPats() {
	ev.PatVocab = patgen.Vocab{}

	pctAct := float32(0.2)
	minDiff := float32(0.5)
	flipPct := float32(0.2)
	nUn := ev.NUnitsY * ev.NUnitsX
	nOn := patgen.NFmPct(pctAct, nUn)
	flipBits := patgen.NFmPct(flipPct, nOn)
	patgen.AddVocabPermutedBinary(ev.PatVocab, "Protos", ev.NConds, ev.NUnitsY, ev.NUnitsX, pctAct, minDiff)

	npats := ev.NConds * ev.NTrials
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{ev.NUnitsY, ev.NUnitsX}, []string{"Y", "X"}},
	}
	ev.Pats.SetFromSchema(sch, npats)

	idx := 0
	for i := 0; i < ev.NConds; i++ {
		condNm := fmt.Sprintf("cond%d", i)
		tsr, _ := patgen.AddVocabRepeat(ev.PatVocab, condNm, ev.NTrials, "Protos", i)
		patgen.FlipBitsRows(tsr, flipBits, flipBits, 1, 0)
		for j := 0; j < ev.NTrials; j++ {
			ev.Pats.SetCellTensor("Input", idx+j, tsr.SubSpace([]int{j}))
			ev.Pats.SetCellString("Name", idx+j, fmt.Sprintf("Cond%d_Trial%d", i, j))
		}
		idx += ev.NTrials
	}
}

func (ev *VSPatchEnv) Validate() error {
	return nil
}

func (ev *VSPatchEnv) Init(run int) {
	ev.Sequence.Init()
	ev.Trial.Init()
}

func (ev *VSPatchEnv) Counter(scale env.TimeScales) (cur, prv int, changed bool) {
	switch scale {
	case env.Sequence:
		return ev.Sequence.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return 0, 0, false
}

func (ev *VSPatchEnv) State(el string) etensor.Tensor {
	return ev.States[el]
}

// RenderState renders the given condition, trial
func (ev *VSPatchEnv) RenderState(cond, trial int) {
	st := ev.States["State"]
	idx := cond*ev.NTrials + trial
	st.CopyFrom(ev.Pats.CellTensor("Input", idx))
}

// Step does one step -- must set Trial.Cur first if doing testing
func (ev *VSPatchEnv) Step() bool {
	ev.RenderState(ev.Sequence.Cur, ev.Trial.Cur)
	ev.Rew = 0
	if ev.Trial.Cur == ev.NTrials-1 {
		ev.Rew = ev.CondVals[ev.Sequence.Cur]
	}
	ev.Sequence.Same()
	if ev.Trial.Incr() {
		ev.Sequence.Incr()
	}
	return true
}

func (ev *VSPatchEnv) Action(action string, nop etensor.Tensor) {
}

func (ev *VSPatchEnv) ComputeDA(rew float32) {
}
