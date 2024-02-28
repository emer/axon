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

	// sequence length.
	SeqLen int

	// number of distinct actions represented: determines the difficulty
	// of learning in terms of the size of the space that must be searched.
	// effective size = NActions ^ SeqLen
	// 4 ^ 3 = 64 or 7 ^2 = 49 are reliably solved
	NActions int

	// learning rate for reward prediction
	RewPredLRate float32

	// minimum rewpred value
	RewPredMin float32

	//	give reward with probability in proportion to number of
	// correct actions in sequence, above given threshold.  If 0, don't use
	PartialCreditAt int

	//	if doing partial credit, also make the reward value graded (weaker for fewer)
	PartialGraded bool

	// sequence map from sequence index to target motor action
	SeqMap []int

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

	// total number of units: NActions * NUnitsPer
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
	ev.SeqLen = 3           // 2x5 is easily solved, 3x4 is 100% with 49u
	ev.NActions = 4         // 2x6 is 100%; 2x7 100% with 25u
	ev.PartialCreditAt = 1  // 1 default: critical for seq len = 3
	ev.PartialGraded = true // key for seq 3
	ev.RewPredLRate = 0.01  // GPU 16 0.01 > 0.02 >> 0.05 > 0.1, 0.2 for partial, seq3
	ev.RewPredMin = 0.1     // 0.1 > 0.05 > 0.2
	ev.NUnitsPer = 5
	ev.NUnits = ev.NUnitsPer * ev.NActions
}

// Config configures the world
func (ev *MotorSeqEnv) Config(mode etime.Modes, rndseed int64) {
	ev.Mode = mode
	ev.RndSeed = rndseed
	ev.Rand.NewRand(ev.RndSeed)
	ev.States = make(map[string]*etensor.Float32)
	ev.States["State"] = etensor.NewFloat32([]int{ev.NUnitsPer, ev.NActions}, nil, []string{"Y", "X"})
	ev.States["Target"] = etensor.NewFloat32([]int{ev.NUnitsPer, ev.NActions}, nil, []string{"Y", "X"})
	ev.States["Action"] = etensor.NewFloat32([]int{ev.NUnitsPer, ev.NActions}, nil, []string{"Y", "X"})
	ev.States["PrevAction"] = etensor.NewFloat32([]int{ev.NUnitsPer, ev.NActions}, nil, []string{"Y", "X"})
	ev.States["Rew"] = etensor.NewFloat32([]int{1, 1}, nil, nil)
	ev.States["SNc"] = etensor.NewFloat32([]int{1, 1}, nil, nil)
}

func (ev *MotorSeqEnv) Validate() error {
	return nil
}

func (ev *MotorSeqEnv) InitSeqMap() {
	// pord := ev.Rand.Perm(ev.NActions, -1)
	ev.SeqMap = make([]int, ev.SeqLen)
	for i := 0; i < ev.SeqLen; i++ {
		ev.SeqMap[i] = i // no randomness!  otherwise doesn't work on gpu!
	}
	// ev.SeqMap[0] = 4 // todo: cheating -- 4 is initial bias; 0 also learns quickly
	// ev.SeqMap[0] = 3 // 3, 2 good test cases -- can learn but not initial bias -- 3 esp hard
}

func (ev *MotorSeqEnv) Init(run int) {
	ev.Trial.Max = ev.SeqLen + 1 // rew
	ev.Trial.Init()
	ev.Trial.Cur = 0
	ev.InitSeqMap()
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

func (ev *MotorSeqEnv) IsRewTrialPostStep() bool {
	return ev.Trial.Cur == 0
}

// RenderState renders the current state
func (ev *MotorSeqEnv) RenderState() {
	trl := ev.Trial.Cur
	ev.RenderBlank("Action")
	ev.States["SNc"].Set1D(0, ev.RPE)
	ev.States["Rew"].Set1D(0, ev.Rew)
	if ev.IsRewTrial() {
		ev.PrevAction = 0
		ev.RenderBlank("State")
		ev.RenderBlank("Target")
		ev.RenderBlank("PrevAction")
	} else {
		st := ev.SeqMap[trl]
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
	// fmt.Println("\nstep:", ev.Trial.Cur)
	ev.RenderState()
	ev.Trial.Incr()
	return true
}

// Action records the current action taken by model, at end of minus phase
// Computes Rew* at end of sequence
func (ev *MotorSeqEnv) Action(action string, nop etensor.Tensor) {
	ev.CurAction, _ = strconv.Atoi(action)
	// fmt.Println("act:", ev.Trial.Cur, action, ev.CurAction, ev.Target, ev.NCorrect)
	if ev.CurAction == ev.Target {
		ev.Correct = true
		ev.NCorrect++
		// fmt.Println("correct:", ev.NCorrect)
	} else {
		ev.Correct = false
		// fmt.Println("incorrect:")
	}
	ev.RenderLocalist("Action", ev.CurAction)
	if ev.Trial.Cur == ev.Trial.Max-1 { // trial before reward trial
		ev.ComputeReward()
	}
}

func (ev *MotorSeqEnv) ComputeReward() {
	ev.Rew = 0
	// fmt.Println("rew, ncor:", ev.NCorrect, ev.SeqLen)
	if ev.PartialCreditAt > 0 {
		prew := float32(ev.NCorrect) / float32(ev.SeqLen)
		doRew := erand.BoolP32(prew, -1, &ev.Rand)
		if doRew {
			if ev.PartialGraded {
				ev.Rew = prew
			} else {
				ev.Rew = 1
			}
		}
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
	var mx float32
	var mxi int
	for i := 0; i < dx; i++ {
		var sum float32
		for j := 0; j < ev.NUnitsPer; j++ {
			sum += vt.Value([]int{j, i})
		}
		if sum > mx {
			mx = sum
			mxi = i
		}
	}
	return mxi
}
