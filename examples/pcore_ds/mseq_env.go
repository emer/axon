// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strconv"

	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/etable/v2/etensor"
)

// MotorSeqEnv implements simple motor sequencing patterns to test DS BG learning.
// Simplest mode has sequential State inputs which require distinct motor actions.
// Also implements simple reward prediction error for dopamine.
// The first trial is blank, and the last trial has the reward.
type MotorSeqEnv struct {

	// name of environment -- Train or Test
	Nm string

	// training or testing env?
	Mode etime.Modes

	// trial counter for index into sequence
	Trial env.Ctr

	// current length of sequence to generate
	SeqLen int

	// max sequence length that can be represented
	MaxSeqLen int

	// learning rate for reward prediction
	RewPredLRate float32

	// minimum rewpred value
	RewPredMin float32

	//	give reward in proportion to number of correct actions in sequence,
	// otherwise only for perfect sequence
	PartialCredit bool

	// sequence map from sequential input state to target motor action
	SeqMap map[int]int

	// current target correct action according to the sequence
	Target int `edit:"-"`

	// current action taken by network
	CurAction int `edit:"-"`

	// previous action taken by network
	PrevAction int `edit:"-"`

	// is current action correct
	Correct bool `edit:"-"`

	// number of correct actions taken this sequence
	NCorrect int `edit:"-"`

	// previous number of correct actions taken, when reward is computed (NCorrect is reset)
	PrevNCorrect int `edit:"-"`

	// raw reward based on action sequence, computed at end of seq
	Rew float32 `edit:"-"`

	// reward prediction based on incremental learning: RewPredLRate * (Rew - RewPred)
	RewPred float32 `edit:"-"`

	// reward prediction error: Rew - RewPred
	RPE float32 `edit:"-"`

	// number of units per localist representation, in Y axis
	NUnitsPer int `view:"-"`

	// total number of units: MaxSeqLen * NUnitsPer
	NUnits int `view:"-"`

	// random number generator for the env -- all random calls must use this
	Rand erand.SysRand `view:"-"`

	// random seed
	RndSeed int64 `edit:"-"`

	// named states: State, Target, PrevAction, Action
	States map[string]*etensor.Float32
}

func (ev *MotorSeqEnv) Name() string {
	return ev.Nm
}

func (ev *MotorSeqEnv) Desc() string {
	return "MotorSeqEnv"
}

func (ev *MotorSeqEnv) Defaults() {
	ev.SeqLen = 2
	ev.RewPredLRate = 0.002
	ev.RewPredMin = 0.1
	ev.MaxSeqLen = 5
	ev.NUnitsPer = 5
	ev.NUnits = ev.NUnitsPer * ev.MaxSeqLen
}

// Config configures the world
func (ev *MotorSeqEnv) Config(mode etime.Modes, rndseed int64) {
	ev.Mode = mode
	ev.RndSeed = rndseed
	ev.Rand.NewRand(ev.RndSeed)
	ev.States = make(map[string]*etensor.Float32)
	ev.States["State"] = etensor.NewFloat32([]int{ev.NUnitsPer, ev.MaxSeqLen}, nil, []string{"Y", "X"})
	ev.States["Target"] = etensor.NewFloat32([]int{ev.NUnitsPer, ev.MaxSeqLen}, nil, []string{"Y", "X"})
	ev.States["Action"] = etensor.NewFloat32([]int{ev.NUnitsPer, ev.MaxSeqLen}, nil, []string{"Y", "X"})
	ev.States["PrevAction"] = etensor.NewFloat32([]int{ev.NUnitsPer, ev.MaxSeqLen}, nil, []string{"Y", "X"})
	ev.States["Rew"] = etensor.NewFloat32([]int{1, 1}, nil, nil)
	ev.States["SNc"] = etensor.NewFloat32([]int{1, 1}, nil, nil)
}

func (ev *MotorSeqEnv) Validate() error {
	return nil
}

func (ev *MotorSeqEnv) Init(run int) {
	ev.Trial.Max = ev.SeqLen + 1 // rew
	ev.Trial.Init()
	ev.Trial.Cur = -1 // first step goes to 0
	ev.NCorrect, ev.Rew, ev.RPE = 0, 0, 0
	ev.RewPred = ev.RewPredMin
}

func (ev *MotorSeqEnv) Counter(scale env.TimeScales) (cur, prv int, changed bool) {
	if scale == env.Trial {
		return ev.Trial.Query()
	}
	return 0, 0, false
}

func (ev *MotorSeqEnv) State(el string) etensor.Tensor {
	return ev.States[el]
}

// RenderBlank renders blank
func (ev *MotorSeqEnv) RenderBlank(name string) {
	av := ev.States[name]
	av.SetZeros()
}

// RenderLocalist renders localist * NUnitsPer
func (ev *MotorSeqEnv) RenderLocalist(name string, idx int) {
	av := ev.States[name]
	av.SetZeros()
	for yi := 0; yi < ev.NUnitsPer; yi++ {
		av.Set([]int{yi, idx}, 1)
	}
}

func (ev *MotorSeqEnv) IsRewTrial() bool {
	return ev.Trial.Cur == ev.Trial.Max-1
}

// RenderState renders the current state
func (ev *MotorSeqEnv) RenderState() {
	trl := ev.Trial.Cur
	ev.RenderBlank("Action")
	if ev.IsRewTrial() {
		ev.PrevAction = 0
		ev.RenderBlank("State")
		ev.RenderBlank("Target")
		ev.RenderBlank("PrevAction")
	} else {
		st := trl
		ev.Target = st // todo: starting with simple 1-to-1
		ev.RenderLocalist("State", st)
		ev.RenderLocalist("Target", ev.Target)
		if trl > 0 {
			ev.RenderLocalist("PrevAction", 1+ev.PrevAction)
		} else {
			ev.RenderLocalist("PrevAction", 0)
		}
	}
}

// Step does one step, advancing the Trial counter, rendering states
func (ev *MotorSeqEnv) Step() bool {
	ev.Trial.Incr()
	ev.RenderState()
	return true
}

// Action records the current action taken by model, at end of minus phase
// Computes Rew* at end of sequence
func (ev *MotorSeqEnv) Action(action string, nop etensor.Tensor) {
	ev.CurAction, _ = strconv.Atoi(action)
	if ev.CurAction == ev.Target {
		ev.Correct = true
		ev.NCorrect++
	} else {
		ev.Correct = false
	}
	ev.RenderLocalist("Action", ev.CurAction)
	if ev.Trial.Cur == ev.Trial.Max-2 { // trial before reward trial
		ev.ComputeReward()
	}
}

func (ev *MotorSeqEnv) ComputeReward() {
	ev.Rew = 0
	if ev.PartialCredit {
		ev.Rew = float32(ev.NCorrect) / float32(ev.SeqLen)
	} else {
		if ev.NCorrect == ev.SeqLen {
			ev.Rew = 1
		}
	}
	ev.RPE = ev.Rew - ev.RewPred
	ev.RewPred += ev.RewPredLRate * (ev.Rew - ev.RewPred)
	if ev.RewPred < ev.RewPredMin {
		ev.RewPred = ev.RewPredMin
	}
	ev.PrevNCorrect = ev.NCorrect
	ev.NCorrect = 0
}

func (ev *MotorSeqEnv) DecodeAct(vt *etensor.Float32) int {
	mxi := ev.DecodeLocalist(vt)
	return mxi
}

func (ev *MotorSeqEnv) DecodeLocalist(vt *etensor.Float32) int {
	dx := vt.Dim(1)
	var max float32
	var mxi int
	for i := 0; i < dx; i++ {
		var sum float32
		for j := 0; j < ev.NUnitsPer; j++ {
			sum += vt.Value([]int{j, i})
		}
		if sum > max {
			max = sum
			mxi = i
		}
	}
	return mxi
}
