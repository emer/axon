// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import (
	"github.com/emer/axon/v2/axon"
)

// TraceStates is a list of mutually exclusive states
// for tracing the behavior and internal state of Emery
type TraceStates int32 //enums:enum

const (
	// Searching is not yet goal engaged, looking for a goal
	TrSearching TraceStates = iota

	// Deciding is having some partial gating but not in time for action
	TrDeciding

	// JustEngaged means just decided to engage in a goal
	TrJustEngaged

	// Approaching is goal engaged, approaching the goal
	TrApproaching

	// Consuming is consuming the US, first step (prior to getting reward, step1)
	TrConsuming

	// Rewarded is just received reward from a US
	TrRewarded

	// GiveUp is when goal is abandoned
	TrGiveUp

	// Bumping is bumping into a wall
	TrBumping
)

// TraceRec holds record of info for tracing behavior, state
type TraceRec struct {

	// absolute time
	Time float32

	// trial counter
	Trial int

	// current arm
	Arm int

	// position in arm
	Pos int

	// behavioral / internal state summary
	State TraceStates

	// NDrives current drive state level
	Drives []float32
}

// StateTrace holds trace records
type StateTrace []*TraceRec

// AddRec adds a record with data from given sources
func (tr *StateTrace) AddRec(ctx *axon.Context, di uint32, ev *Env, net *axon.Network, state TraceStates) *TraceRec {
	rec := &TraceRec{Arm: ev.Arm, Pos: ev.Pos, State: state}
	rec.Drives = make([]float32, ev.Config.NDrives)
	if ctx != nil {
		rec.Time = ctx.Time
		rec.Trial = int(ctx.TrialsTotal)
		for i := 0; i < ev.Config.NDrives; i++ {
			rec.Drives[i] = axon.GlbUSposV(ctx, di, axon.GvDrives, uint32(1+i))
		}
	}

	*tr = append(*tr, rec)
	return rec
}
