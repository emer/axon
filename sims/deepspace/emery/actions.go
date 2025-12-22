// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import "fmt"

// Actions are motor actions as abstracted coordinated plans
// that unfold over time, at a level above individual muscles.
type Actions int64 //enums:bitflag

const (
	Forward Actions = iota
	Rotate
)

// Simple sinusoidal force curve:
// f(t) = f sin(t)
// Total duration = 2pi
// d =

// Duration computes the duration for a given singular action,
// with given value. Larger actions take longer, etc.
func (a Actions) Duration(val float32) int {
	maxVel := float32(2)
}

// Action represents an action state.
type Action struct {
	// Time is the timestamp (Cycles, ms) for this state.
	Time int

	// Actions has the bits of active actions.
	Actions Actions

	// Values are the action parameters for each action
	// (e.g., rotation degrees).
	Values []float32

	// Durations are the number of cycles for the actions (from start),
	// computed from the Value and the type of action.
	Durations []int
}

// Init initializes state for given timestamp (ms).
func (a *Action) Init(ms int) {
	a.Time = ms
	a.Actions = 0
	a.Time = 0
	a.Values = make([]float32, ActionsN)
	a.Durations = make([]int, ActionsN)
}

func (a *Action) String() string {
	s := fmt.Sprintf("t:%d", a.Time)
	for ai := range a.Actions {
		if a.Actions.HasFlag(ai) {
			s += fmt.Sprintf(" %s_%g", ai.String(), a.Values[ai])
		}
	}
	return s
}

// Start starts a set of action(s).
// acts = bitmask of actions, vals = ordinal
// list of parameters in order of actions.
func (a *Action) Start(acts Actions, vals ...float32) {
	a.Actions = acts
	vi := 0
	for ai := range a.Actions {
		if acts.Actions.HasFlag(ai) {
			a.Values[ai] = vals[vi]

			vi++
		}
	}
}

// ActionBuffer is a ring buffer for actions.
type ActionBuffer struct {
	// States are the action states.
	States []Action

	// WriteIndex is the current write index where a new item will be
	// written (within range of States). Add post-increments.
	WriteIndex int
}

func (ab *ActionBuffer) Init(n int) {
	ab.States = make([]Action, n)
	ab.WriteIndex = 0
	for i := range n {
		a := &ab.States[i]
		a.Init(0)
	}
}

// Add adds a new action at given time stamp (ms).
// Returns pointer to an existing state per ring buffer logic.
func (ab *ActionBuffer) Add(ms int) *Action {
	a := &ab.States[ab.WriteIndex]
	a.Init(ms)
	ab.WriteIndex++
	if ab.WriteIndex >= len(ab.States) {
		ab.WriteIndex = 0
	}
	return a
}

// Prior returns action state that is given number of items
// prior to the last-added item. 0 = last-added item.
func (ab *ActionBuffer) Prior(nPrior int) *Action {
	n := len(ab.States)
	ix := (ab.WriteIndex - 1) - nPrior
	for ix < 0 {
		ix += n
	}
	return &ab.States[ix]
}
