// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/metric"
	"cogentcore.org/core/tensor/stats/simat"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/patgen"
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

	// current condition index
	Cond int

	// current condition target reward
	CondRew float32

	// if true, reward value is a probability of getting a 1 reward
	Probs bool

	// number of conditions, each of which can have a different reward value
	NConds int

	// number of trials
	NTrials int

	// condition current values
	CondValues []float32

	// state rep, number of units, Y
	NUnitsY int `view:"-"`

	// state rep, number of units, X
	NUnitsX int `view:"-"`

	// total number of units
	NUnits int `view:"-"`

	// condition, time-step patterns
	Pats *table.Table

	// pattern vocab
	PatVocab patgen.Vocab

	// pattern similarity matrix
	PatSimMat simat.SimMat

	// random number generator for the env -- all random calls must use this
	Rand randx.SysRand `view:"-"`

	// random seed
	RandSeed int64 `edit:"-"`

	// named states: ACCPos, ACCNeg
	States map[string]*tensor.Float32

	// current reward value -- is 0 until final trial
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
	ev.Probs = true
	ev.NConds = 4
	ev.NTrials = 3
	ev.NUnitsY = 5
	ev.NUnitsX = 5
	ev.NUnits = ev.NUnitsY * ev.NUnitsX
}

// SetCondValues sets values of each condition incrementing upward
func (ev *VSPatchEnv) SetCondValues() {
	pc := float32(1) / (float32(ev.NConds) + 1)
	for i := 0; i < ev.NConds; i++ {
		ev.CondValues[i] = float32(1+i) * pc
	}
}

// SetCondValuesPermute sets permuted order of values
func (ev *VSPatchEnv) SetCondValuesPermute(ord []int) {
	pc := float32(1) / (float32(ev.NConds) + 1)
	for i := 0; i < ev.NConds; i++ {
		ev.CondValues[ord[i]] = float32(1+i) * pc
	}
}

// Config configures the world
func (ev *VSPatchEnv) Config(mode etime.Modes, rndseed int64) {
	ev.Mode = mode
	ev.RandSeed = rndseed
	ev.Rand.NewRand(ev.RandSeed)
	ev.States = make(map[string]*tensor.Float32)
	ev.States["State"] = tensor.NewFloat32([]int{ev.NUnitsY, ev.NUnitsX}, "Y", "X")
	ev.CondValues = make([]float32, ev.NConds)
	ev.Sequence.Max = ev.NConds
	ev.Trial.Max = ev.NTrials
	ev.SetCondValues()
}

// ConfigPats configures patterns -- only done on the first env
func (ev *VSPatchEnv) ConfigPats() {
	ev.PatVocab = patgen.Vocab{}
	pctAct := float32(0.2)
	minDiff := float32(0.5)
	flipPct := float32(0.2)
	nUn := ev.NUnitsY * ev.NUnitsX
	nOn := patgen.NFromPct(pctAct, nUn)
	flipBits := patgen.NFromPct(flipPct, nOn)
	patgen.AddVocabPermutedBinary(ev.PatVocab, "Protos", ev.NConds, ev.NUnitsY, ev.NUnitsX, pctAct, minDiff)

	npats := ev.NConds * ev.NTrials
	ev.Pats = table.NewTable(npats)
	ev.Pats.AddStringColumn("Name")
	ev.Pats.AddFloat32TensorColumn("Input", []int{ev.NUnitsY, ev.NUnitsX}, "Y", "X")

	idx := 0
	for i := 0; i < ev.NConds; i++ {
		condNm := fmt.Sprintf("cond%d", i)
		tsr, _ := patgen.AddVocabRepeat(ev.PatVocab, condNm, ev.NTrials, "Protos", i)
		patgen.FlipBitsRows(tsr, flipBits, flipBits, 1, 0)
		for j := 0; j < ev.NTrials; j++ {
			ev.Pats.SetTensor("Input", idx+j, tsr.SubSpace([]int{j}))
			ev.Pats.SetString("Name", idx+j, fmt.Sprintf("Cond%d_Trial%d", i, j))
		}
		idx += ev.NTrials
	}

	ev.PatSimMat.TableCol(table.NewIndexView(ev.Pats), "Input", "Name", true, metric.Correlation64)
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

func (ev *VSPatchEnv) State(el string) tensor.Tensor {
	return ev.States[el]
}

// RenderState renders the given condition, trial
func (ev *VSPatchEnv) RenderState(cond, trial int) {
	st := ev.States["State"]
	idx := cond*ev.NTrials + trial
	st.CopyFrom(ev.Pats.Tensor("Input", idx))
}

// Step does one step -- must set Trial.Cur first if doing testing
func (ev *VSPatchEnv) Step() bool {
	ev.RenderState(ev.Sequence.Cur, ev.Trial.Cur)
	ev.Rew = 0
	rv := ev.CondValues[ev.Sequence.Cur]
	ev.Cond = ev.Sequence.Cur // todo: randomize
	ev.CondRew = rv
	if ev.Trial.Cur == ev.NTrials-1 {
		if ev.Probs {
			if randx.BoolP32(rv, &ev.Rand) {
				ev.Rew = 1
			} else {
				ev.Rew = 0.001
			}
		} else {
			ev.Rew = rv
		}
	}
	ev.Sequence.Same()
	if ev.Trial.Incr() {
		ev.Sequence.Incr()
	}
	return true
}

func (ev *VSPatchEnv) Action(action string, nop tensor.Tensor) {
}

func (ev *VSPatchEnv) ComputeDA(rew float32) {
}
