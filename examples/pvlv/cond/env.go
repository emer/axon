// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

//go:generate goki generate -add-types

import (
	"fmt"

	"github.com/emer/emergent/v2/env"
	"goki.dev/etable/v2/etensor"
)

// CondEnv provides a flexible implementation of standard Pavlovian
// conditioning experiments involving CS -> US sequences (trials).
// Has a large database of standard conditioning paradigms
// parameterized in a controlled manner.
//
// Time hierarchy:
// * Run = sequence of 1 or more Conditions
// * Condition = specific mix of trial types, generated at start of Condition
// * Block = one full pass through all trial types generated for condition (like Epoch)
// * Trial = one behavioral trial consisting of CS -> US presentation over time steps (Ticks)
// * Tick = discrete time steps within behavioral Trial, typically one Network update (Alpha / Theta cycle)
type CondEnv struct {

	// name of this environment
	Nm string

	// description of this environment
	Dsc string

	// number of Y repetitions for localist reps
	NYReps int

	// current run name
	RunName string

	// description of current run
	RunDesc string

	// name of current condition
	CondName string

	// description of current condition
	CondDesc string

	// counter over runs
	Run env.Ctr `inactive:"+" view:"inline"`

	// counter over Condition within a run -- Max depends on number of conditions specified in given Run
	Condition env.Ctr `inactive:"+" view:"inline"`

	// counter over full blocks of all trial types within a Condition -- like an Epoch
	Block env.Ctr `inactive:"+" view:"inline"`

	// counter of behavioral trials within a Block
	Trial env.Ctr `inactive:"+" view:"inline"`

	// counter of discrete steps within a behavioral trial -- typically maps onto Alpha / Theta cycle in network
	Tick env.Ctr `inactive:"+" view:"inline"`

	// name of current trial step
	TrialName string `inactive:"+"`

	// type of current trial step
	TrialType string `inactive:"+"`

	// decoded value of USTimeIn
	USTimeInStr string `inactive:"+"`

	// current generated set of trials per Block
	Trials []*Trial

	// copy of current run parameters
	CurRun Run

	// copy of info for current trial
	CurTrial Trial

	// current rendered state tensors -- extensible map
	CurStates map[string]*etensor.Float32
}

func (ev *CondEnv) Name() string { return ev.Nm }
func (ev *CondEnv) Desc() string { return ev.Dsc }

func (ev *CondEnv) Config(rmax int, rnm string) {
	ev.RunName = rnm
	ev.Run.Max = rmax
	ev.NYReps = 4
	ev.Run.Scale = env.Run
	ev.Condition.Scale = env.Condition
	ev.Block.Scale = env.Block
	ev.Trial.Scale = env.Trial
	ev.Tick.Scale = env.Tick

	ev.CurStates = make(map[string]*etensor.Float32)

	stsh := []int{StimShape[0], StimShape[1], ev.NYReps, 1}
	ev.CurStates["CS"] = etensor.NewFloat32(stsh, nil, nil)
	ctsh := []int{ContextShape[0], ContextShape[1], ev.NYReps, 1}
	ev.CurStates["ContextIn"] = etensor.NewFloat32(ctsh, nil, nil)
	ustsh := make([]int, 4)
	copy(ustsh, USTimeShape)
	ustsh[2] = ev.NYReps
	ev.CurStates["USTimeIn"] = etensor.NewFloat32(ustsh, nil, nil)
	ev.CurStates["Time"] = etensor.NewFloat32([]int{1, MaxTime, ev.NYReps, 1}, nil, nil)
	ussh := []int{USShape[0], USShape[1], ev.NYReps, 1}
	ev.CurStates["USpos"] = etensor.NewFloat32(ussh, nil, nil)
	ev.CurStates["USneg"] = etensor.NewFloat32(ussh, nil, nil)
}

func (ev *CondEnv) Validate() error {
	return nil
}

// Init sets current run index and max
func (ev *CondEnv) Init(ridx int) {
	run := AllRuns[ev.RunName]
	ev.CurRun = *run
	ev.RunDesc = run.Desc
	ev.Run.Set(ridx)
	ev.Condition.Init()
	ev.Condition.Max = run.NConds()
	ev.InitCond()
	ev.Tick.Cur = -1
}

// InitCond initializes for current condition index
func (ev *CondEnv) InitCond() {
	if ev.RunName == "" {
		ev.RunName = "PosAcq_A100B50"
	}
	run := AllRuns[ev.RunName]
	run.Name = ev.RunName
	cnm, cond := run.Cond(ev.Condition.Cur)
	ev.CondName = cnm
	ev.CondDesc = cond.Desc
	ev.Block.Init()
	ev.Block.Max = cond.NBlocks
	ev.Trial.Init()
	ev.Trial.Max = cond.NTrials
	ev.Trials = GenerateTrials(cnm)
	ev.Tick.Init()
	trl := ev.Trials[0]
	ev.Tick.Max = trl.NTicks
}

func (ev *CondEnv) State(element string) etensor.Tensor {
	return ev.CurStates[element]
}

func (ev *CondEnv) Step() bool {
	ev.Condition.Same()
	ev.Block.Same()
	ev.Trial.Same()
	if ev.Tick.Incr() {
		if ev.Trial.Incr() {
			if ev.Block.Incr() {
				if ev.Condition.Incr() {
					if ev.Run.Incr() {
						return false
					}
				}
				ev.InitCond()
			}
		}
		trl := ev.Trials[ev.Trial.Cur]
		ev.Tick.Max = trl.NTicks
	}
	ev.RenderTrial(ev.Trial.Cur, ev.Tick.Cur)
	return true
}

func (ev *CondEnv) Action(_ string, _ etensor.Tensor) {
	// nop
}

func (ev *CondEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Condition:
		return ev.Condition.Query()
	case env.Block:
		return ev.Block.Query()
	case env.Trial:
		return ev.Trial.Query()
	case env.Tick:
		return ev.Tick.Query()
	}
	return -1, -1, false
}

func (ev *CondEnv) RenderTrial(trli, tick int) {
	for _, tsr := range ev.CurStates {
		tsr.SetZeros()
	}
	trl := ev.Trials[trli]
	ev.CurTrial = *trl

	ev.TrialName = fmt.Sprintf("%s_%d", trl.CS, tick)
	ev.TrialType = ev.CurTrial.Name

	stim := ev.CurStates["CS"]
	ctxt := ev.CurStates["ContextIn"]
	ustime := ev.CurStates["USTimeIn"]
	time := ev.CurStates["Time"]
	SetTime(time, ev.NYReps, tick)
	if tick >= trl.CSStart && tick <= trl.CSEnd {
		ev.CurTrial.CSOn = true
		cs := trl.CS[0:1]
		stidx := SetStim(stim, ev.NYReps, cs)
		SetUSTime(ustime, ev.NYReps, stidx, tick, trl.CSStart, trl.CSEnd)
	}
	if (len(trl.CS) > 1) && (tick >= trl.CS2Start) && (tick <= trl.CS2End) {
		ev.CurTrial.CSOn = true
		cs := trl.CS[1:2]
		stidx := SetStim(stim, ev.NYReps, cs)
		SetUSTime(ustime, ev.NYReps, stidx, tick, trl.CSStart, trl.CSEnd)
	}
	minStart := trl.CSStart
	if trl.CS2Start > 0 {
		minStart = min(minStart, trl.CS2Start)
	}
	maxEnd := max(trl.CSEnd, trl.CS2End)

	if tick >= minStart && tick <= maxEnd {
		SetContext(ctxt, ev.NYReps, trl.Context)
	}

	if tick == maxEnd+1 {
		// use last stimulus for US off signal
		SetUSTime(ustime, ev.NYReps, NStims-1, MaxTime, 0, MaxTime)
	}

	ev.CurTrial.USOn = false
	if trl.USOn && (tick >= trl.USStart) && (tick <= trl.USEnd) {
		ev.CurTrial.USOn = true
		if trl.Valence == Pos {
			SetUS(ev.CurStates["USpos"], ev.NYReps, trl.US, trl.USMag)
			ev.TrialName += fmt.Sprintf("_Pos%d", trl.US)
		}
		if trl.Valence == Neg || trl.MixedUS {
			SetUS(ev.CurStates["USneg"], ev.NYReps, trl.US, trl.USMag)
			ev.TrialName += fmt.Sprintf("_Neg%d", trl.US)
		}
	}
}
