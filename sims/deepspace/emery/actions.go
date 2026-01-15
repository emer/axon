// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

// Actions are motor actions as abstracted coordinated plans
// that unfold over time, at a level above individual muscles.
// They are recorded in data continuously, with 0 meaning no
// action being taken, and non-zero indicating strength of action.
type Actions int32 //enums:enum

const (
	Rotate Actions = iota
	Forward
)

// ActionMaxValues are expected max sensory value, for normalizing.
var ActionMaxValues = [ActionsN]float32{3, 3}

// NextAction specifies the next value for given action, for given data parallel agent.
// This simulates the sequence of planning a new action followed by that action
// actually being executed. The planning state is critical for predictive learning.
// Multiple calls can be made per step, for as many actions as need updating.
// Call RenderNextActions when done specifying NextAction's, so they are presented
// to the sim.
func (ev *EmeryEnv) NextAction(di int, act Actions, val float32) {
	es := ev.EmeryState(di)
	es.NextActions[act] = val
}

// TakeNextActions actually starts performing in the physics model the
// actions specified by prior NextAction calls, copying NextActions
// to CurActions and activating them in the model.
// This calls RenderCurAction so the current action is shown to the sim.
func (ev *EmeryEnv) TakeNextActions() {
	for di := range ev.NData {
		es := ev.EmeryState(di)
		for act := range ActionsN {
			val := es.NextActions[act]
			es.CurActions[act] = val
			ev.WriteData(ev.ActionData, di, act.String(), val)
		}
	}
	ev.RenderCurActions()
}

// TakeActions applies current actions to physics.
func (ev *EmeryEnv) TakeActions() {
	for di := range ev.NData {
		for act := range ActionsN {
			val := ev.ReadData(ev.ActionData, di, act.String(), 10) // 0 = last written
			ev.TakeAction(di, act, val)
		}
	}
}

// ZeroActions zero action values after WriteIndex has been incremented.
func (ev *EmeryEnv) ZeroActions() {
	for di := range ev.NData {
		for act := range ActionsN {
			ev.WriteData(ev.ActionData, di, act.String(), 0)
		}
	}
}

//////// Rendering

// RenderNextActions renders the action values specified in NextAction calls.
func (ev *EmeryEnv) RenderNextActions() {
	ev.renderActions(false)
}

// RenderCurActions renders the current action values, from TakeNextActions.
func (ev *EmeryEnv) RenderCurActions() {
	ev.renderActions(true)
}

// renderActions renders sensory states for current sensory values.
func (ev *EmeryEnv) renderActions(cur bool) {
	for act := range Forward { // only render below Forward for now
		for di := range ev.NData {
			es := ev.EmeryState(di)
			val := es.NextActions[act]
			if cur {
				val = es.CurActions[act]
			}
			val /= ActionMaxValues[act]
			ev.RenderValue(di, act.String(), val)
		}
	}
}
