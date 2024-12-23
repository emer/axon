// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/patgen"
)

// VSPatchEnv implements a simple training environment for VSPatch reward
// prediction learning, with a given number of theta steps within
// an overall trial, that lead up to a reward outcome delivery at the end.
// There is a fixed progression of patterns for each theta step, generated from
// a prototypical pattern for each condition. The different trial steps
// iterate through different conditions, each of which has a different reward level.
// Rewards can be graded or probabilistic.
type VSPatchEnv struct {

	// name of environment -- Train or Test
	Name string

	// training or testing env?
	Mode etime.Modes

	// Di is the data parallel index for this env, which determines
	// the starting offset for the condition so that it matches
	// what would happen sequentially.
	Di int

	// trial counter is outer loop over thetas, iterates over Conds up to NCond
	Trial env.Counter `display:"inline"`

	// theta counter is for the step within trial
	Theta env.Counter `display:"inline"`

	// current condition index
	Cond int

	// current condition target reward
	CondRew float32

	// if true, reward value is a probability of getting a 1 reward
	Probs bool

	// number of conditions, each of which can have a different reward value
	NConds int

	// number of trials
	Thetas int

	// condition current values
	CondValues []float32

	// state rep, number of units, Y
	NUnitsY int `display:"-"`

	// state rep, number of units, X
	NUnitsX int `display:"-"`

	// total number of units
	NUnits int `display:"-"`

	// condition, time-step patterns
	Pats *table.Table

	// pattern vocab
	PatVocab patgen.Vocab

	// pattern similarity matrix
	// PatSimMat simat.SimMat

	// random number generator for the env -- all random calls must use this
	Rand randx.SysRand `display:"-"`

	// random seed
	RandSeed int64 `edit:"-"`

	// named states: ACCPos, ACCNeg
	States map[string]*tensor.Float32

	// current reward value -- is 0 until final theta step.
	Rew float32 `edit:"-"`

	// reward prediction from model for current theta step.
	RewPred float32 `edit:"-"`

	// DA = reward prediction error on current theta step: Rew - RewPred
	DA float32 `edit:"-"`

	// non-reward prediction from model, from theta step just before reward.
	RewPred_NR float32 `edit:"-"`

	// DA = non-reward prediction error: Rew - RewPred_NR
	DA_NR float32 `edit:"-"`
}

func (ev *VSPatchEnv) Label() string { return ev.Name }

func (ev *VSPatchEnv) Defaults() {
	ev.Probs = true
	ev.NConds = 4
	ev.Thetas = 3
	ev.NUnitsY = 5
	ev.NUnitsX = 5
	ev.NUnits = ev.NUnitsY * ev.NUnitsX
}

func (ev *VSPatchEnv) String() string {
	return fmt.Sprintf("Cond_%d_Rew_%g", ev.Cond, ev.CondRew)
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
func (ev *VSPatchEnv) Config(mode etime.Modes, di int, rndseed int64) {
	ev.Mode = mode
	ev.Di = di
	ev.RandSeed = rndseed
	ev.Rand.NewRand(ev.RandSeed)
	ev.States = make(map[string]*tensor.Float32)
	ev.States["State"] = tensor.NewFloat32(ev.NUnitsY, ev.NUnitsX)
	ev.CondValues = make([]float32, ev.NConds)
	ev.Trial.Max = ev.NConds
	ev.Theta.Max = ev.Thetas
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

	npats := ev.NConds * ev.Thetas
	ev.Pats = table.New()
	ev.Pats.AddStringColumn("Name")
	ev.Pats.AddFloat32Column("Input", ev.NUnitsY, ev.NUnitsX)
	ev.Pats.SetNumRows(npats)

	idx := 0
	for i := 0; i < ev.NConds; i++ {
		condNm := fmt.Sprintf("cond%d", i)
		tsr, _ := patgen.AddVocabRepeat(ev.PatVocab, condNm, ev.Thetas, "Protos", i)
		patgen.FlipBitsRows(tsr, flipBits, flipBits, 1, 0)
		for j := 0; j < ev.Thetas; j++ {
			ev.Pats.Column("Input").SetRowTensor(tsr.SubSpace(j), idx+j)
			ev.Pats.Column("Name").SetStringRow(fmt.Sprintf("Cond%d_Theta%d", i, j), idx+j, 0)
		}
		idx += ev.Thetas
	}

	// ev.PatSimMat.TableColumn(table.NewIndexView(ev.Pats), "Input", "Name", true, metric.Correlation64)
}

func (ev *VSPatchEnv) Init(run int) {
	ev.Trial.Init()
	ev.Theta.Init()
}

func (ev *VSPatchEnv) State(el string) tensor.Values {
	return ev.States[el]
}

// RenderState renders the given condition, trial
func (ev *VSPatchEnv) RenderState(cond, trial int) {
	st := ev.States["State"]
	idx := cond*ev.Thetas + trial
	st.CopyFrom(ev.Pats.Column("Input").RowTensor(idx))
}

// Step does one step -- must set Theta.Cur first if doing testing
func (ev *VSPatchEnv) Step() bool {
	cond := (ev.Di + ev.Trial.Cur) % ev.NConds
	ev.Cond = cond
	ev.RenderState(cond, ev.Theta.Cur)
	ev.Rew = 0
	rv := ev.CondValues[cond]
	ev.CondRew = rv
	if ev.Theta.Cur == ev.Thetas-1 {
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
	ev.Trial.Same()
	if ev.Theta.Incr() {
		ev.Trial.Incr()
	}
	return true
}

func (ev *VSPatchEnv) Action(action string, nop tensor.Values) {
}
