// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

//go:generate core generate -add-types

import (
	"fmt"

	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
)

// CondEnv provides a flexible implementation of standard Pavlovian
// conditioning experiments involving CS -> US sequences.
// Has a large database of standard conditioning paradigms
// parameterized in a controlled manner.
//
// Time hierarchy:
// * Run = sequence of 1 or more Conditions
// * Condition = specific mix of sequence types, generated at start of Condition
// * Block = one full pass through all sequence types generated for condition (like Epoch)
// * Sequence = one behavioral trial consisting of CS -> US presentation over time steps (Ticks)
// * Tick = discrete time steps within behavioral Sequence, typically one Network update (Alpha / Theta cycle)
type CondEnv struct {

	// name of this environment
	Name string

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
	Run env.Counter `edit:"-" display:"inline"`

	// counter over Condition within a run -- Max depends on number of conditions specified in given Run
	Condition env.Counter `edit:"-" display:"inline"`

	// counter over full blocks of all sequence types within a Condition -- like an Epoch
	Block env.Counter `edit:"-" display:"inline"`

	// counter of behavioral sequences within a Block
	Sequence env.Counter `edit:"-" display:"inline"`

	// counter of discrete steps within a sequence -- typically maps onto Alpha / Theta cycle in network
	Tick env.Counter `edit:"-" display:"inline"`

	// name of current sequence step
	SequenceName string `edit:"-"`

	// type of current sequence step
	SequenceType string `edit:"-"`

	// decoded value of USTimeIn
	USTimeInStr string `edit:"-"`

	// current generated set of sequences per Block
	Sequences []*Sequence

	// copy of current run parameters
	CurRun Run

	// the current rendered tick
	CurTick Sequence

	// current rendered state tensors -- extensible map
	CurStates map[string]*tensor.Float32
}

func (ev *CondEnv) Config(rmax int, rnm string) {
	ev.RunName = rnm
	ev.Run.Max = rmax
	ev.NYReps = 4
	ev.Run.Scale = etime.Run
	ev.Condition.Scale = etime.Condition
	ev.Block.Scale = etime.Block
	ev.Sequence.Scale = etime.Sequence
	ev.Tick.Scale = etime.Tick

	ev.CurStates = make(map[string]*tensor.Float32)

	stsh := []int{StimShape[0], StimShape[1], ev.NYReps, 1}
	ev.CurStates["CS"] = tensor.NewFloat32(stsh)
	ctsh := []int{ContextShape[0], ContextShape[1], ev.NYReps, 1}
	ev.CurStates["ContextIn"] = tensor.NewFloat32(ctsh)
	ustsh := make([]int, 4)
	copy(ustsh, USTimeShape)
	ustsh[2] = ev.NYReps
	ev.CurStates["USTimeIn"] = tensor.NewFloat32(ustsh)
	ev.CurStates["Time"] = tensor.NewFloat32([]int{1, MaxTime, ev.NYReps, 1})
	ussh := []int{USShape[0], USShape[1], ev.NYReps, 1}
	ev.CurStates["USpos"] = tensor.NewFloat32(ussh)
	ev.CurStates["USneg"] = tensor.NewFloat32(ussh)
}

func (ev *CondEnv) Label() string { return ev.Name }

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
	ev.Sequence.Init()
	ev.Sequence.Max = cond.NSequences
	ev.Sequences = SequenceReps(cnm)
	ev.Tick.Init()
	trl := ev.Sequences[0]
	ev.Tick.Max = trl.NTicks
}

func (ev *CondEnv) State(element string) tensor.Tensor {
	return ev.CurStates[element]
}

func (ev *CondEnv) Step() bool {
	ev.Condition.Same()
	ev.Block.Same()
	ev.Sequence.Same()
	if ev.Tick.Incr() {
		if ev.Sequence.Incr() {
			if ev.Block.Incr() {
				if ev.Condition.Incr() {
					if ev.Run.Incr() {
						return false
					}
				}
				ev.InitCond()
			}
		}
		trl := ev.Sequences[ev.Sequence.Cur]
		ev.Tick.Max = trl.NTicks
	}
	ev.RenderSequence(ev.Sequence.Cur, ev.Tick.Cur)
	return true
}

func (ev *CondEnv) Action(_ string, _ tensor.Tensor) {
	// nop
}

func (ev *CondEnv) RenderSequence(trli, tick int) {
	for _, tsr := range ev.CurStates {
		tsr.SetZeros()
	}
	trl := ev.Sequences[trli]
	ev.CurTick = *trl

	ev.SequenceName = fmt.Sprintf("%s_%d", trl.CS, tick)
	ev.SequenceType = ev.CurTick.Name
	ev.CurTick.Type = Pre

	stim := ev.CurStates["CS"]
	ctxt := ev.CurStates["ContextIn"]
	ustime := ev.CurStates["USTimeIn"]
	time := ev.CurStates["Time"]
	SetTime(time, ev.NYReps, tick)
	if tick >= trl.CSStart && tick <= trl.CSEnd {
		ev.CurTick.CSOn = true
		if tick == trl.CSStart {
			ev.CurTick.Type = CS
		} else {
			ev.CurTick.Type = Maint
		}
		cs := trl.CS[0:1]
		stidx := SetStim(stim, ev.NYReps, cs)
		SetUSTime(ustime, ev.NYReps, stidx, tick, trl.CSStart, trl.CSEnd)
	}
	if (len(trl.CS) > 1) && (tick >= trl.CS2Start) && (tick <= trl.CS2End) {
		ev.CurTick.CSOn = true
		if tick == trl.CS2Start {
			ev.CurTick.Type = CS
		} else {
			ev.CurTick.Type = Maint
		}
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

	ev.CurTick.USOn = false
	if trl.USOn && (tick >= trl.USStart) && (tick <= trl.USEnd) {
		ev.CurTick.USOn = true
		if trl.Valence == Pos {
			SetUS(ev.CurStates["USpos"], ev.NYReps, trl.US, trl.USMag)
			ev.SequenceName += fmt.Sprintf("_Pos%d", trl.US)
		}
		if trl.Valence == Neg || trl.MixedUS {
			SetUS(ev.CurStates["USneg"], ev.NYReps, trl.US, trl.USMag)
			ev.SequenceName += fmt.Sprintf("_Neg%d", trl.US)
		}
	}
	if (tick >= trl.USStart) && (tick <= trl.USEnd) {
		ev.CurTick.Type = US // even if not on, this is the type
	}
	if tick > trl.USEnd {
		ev.CurTick.Type = Post
	}
}
