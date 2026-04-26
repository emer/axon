// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import (
	"cogentcore.org/core/base/num"
	"cogentcore.org/lab/base/randx"
)

// Actions are motor actions as abstracted coordinated plans
// that unfold over time, at a level above individual muscles.
// They are recorded in data continuously, with 0 meaning no
// action being taken, and non-zero indicating strength of action.
type Actions int32 //enums:enum

const (
	// Rotate is overall body rotation.
	// Positive or negative for left vs. right.
	Rotate Actions = iota

	// Forward moving forward or backward.
	Forward

	// EyeH horizontal eye rotation: adds to target angle.
	EyeH

	// VORInhib is the meta control action to inhibit the VOR reflex
	VORInhib
)

// ActionMaxValues are expected max sensory value, for normalizing.
var ActionMaxValues = [ActionsN]float32{3, 3, 1, 1}

// NextAction specifies the next value for given action, for given data parallel agent.
// This simulates the sequence of planning a new action followed by that action
// actually being executed. The planning state is critical for predictive learning.
// Multiple calls can be made per step, for as many actions as need updating.
// Call RenderNextActions when done specifying NextAction's, so they are presented
// to the sim.
func (ev *EmeryEnv) NextAction(di int, act Actions, val float32) {
	es := ev.EmeryState(di)
	switch act {
	case VORInhib:
		vorInhib := randx.BoolP32(ev.Params.VORInhibP, ev.Rand)
		val = num.FromBool[float32](vorInhib)
	}
	es.NextActions[act] = val
}

// TakeNextActions starts the action performance process by copying
// prepared NextActions to CurActions, and writing the ActionData that
// will be consumed with the Params.ActDelay to implement motor delays.
func (ev *EmeryEnv) TakeNextActions() {
	for di := range ev.NData {
		es := ev.EmeryState(di)
		es.InitMax()
		for act := range ActionsN {
			val := es.NextActions[act]
			es.CurActions[act] = val
			ev.WriteData(ev.ActionData, di, act.String(), val) // goes in current = 0
		}
	}
	ev.RenderCurActions() // efferent copy of action. also called in Step()
}

// TakeAction specifies the value for a current action,
// for given data parallel agent, for actions that are updated online,
// as from network state.
func (ev *EmeryEnv) TakeAction(di int, act Actions, val float32) {
	es := ev.EmeryState(di)
	switch act {
	case VORInhib:
		vorInhib := randx.BoolP32(ev.Params.VORInhibP, ev.Rand)
		val = num.FromBool[float32](vorInhib)
	}
	es.CurActions[act] = val
	ev.WriteData(ev.ActionData, di, act.String(), val) // goes in current = 0
}

// DoActions actually performs current actions in physics.
func (ev *EmeryEnv) DoActions() {
	for di := range ev.NData {
		for act := range ActionsN {
			val := ev.ReadData(ev.ActionData, di, act.String(), ev.Params.ActDelay) // 0 = last written
			ev.DoAction(di, act, val)                                               // in emery.go
		}
	}
}

// ZeroActions writes zero action values after WriteIndex has been incremented.
// Thus, each action requires new WriteData to implement.
func (ev *EmeryEnv) ZeroActions() {
	for di := range ev.NData {
		for act := range ActionsN {
			ev.WriteData(ev.ActionData, di, act.String(), 0)
		}
	}
}

//////// Rendering

// RenderCurActions renders efferent copy states for current actions.
// Called in Step()
func (ev *EmeryEnv) RenderCurActions() {
	for act := range ActionsN {
		for di := range ev.NData {
			ev.renderAction(di, act, false)
		}
	}
}

// RenderNextActions renders efferent copy states for next actions.
// note: not currently used.
func (ev *EmeryEnv) RenderNextActions() {
	for act := range ActionsN {
		for di := range ev.NData {
			ev.renderAction(di, act, true)
		}
	}
}

// renderAction renders given action state, from CurActions values
// or NextActions if next = true (state name adds "Next" suffix).
func (ev *EmeryEnv) renderAction(di int, act Actions, next bool) {
	es := ev.EmeryState(di)
	val := es.CurActions[act]
	name := act.String()
	if next {
		val = es.NextActions[act]
		name += "Next"
	}
	switch act {
	case VORInhib:
		ev.RenderControl(di, name, val)
	default:
		val /= ActionMaxValues[act]
		ev.RenderValue(di, name, val)
	}
}
