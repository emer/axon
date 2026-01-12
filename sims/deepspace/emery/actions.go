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
	Forward Actions = iota
	Rotate
)

// Action specifies the value for given action, for given data parallel agent.
// Actions persist at prior value until updated.
// Multiple calls can be made per step, for as many actions as need updating.
func (ev *EmeryEnv) Action(di int, act Actions, val float32) {
	ev.WriteData(ev.ActionData, di, act.String(), val)
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

// PersistActions copies action values from prior row, after WriteIndex
// has been incremented.
func (ev *EmeryEnv) PersistActions() {
	for di := range ev.NData {
		for act := range ActionsN {
			val := ev.ReadData(ev.ActionData, di, act.String(), 1) // 1 prior
			ev.WriteData(ev.ActionData, di, act.String(), val)
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

// TakeAction performs given action in Emery
func (ev *EmeryEnv) TakeAction(di int, act Actions, val float32) {
	// fmt.Println("Action:", di, act, val)
	jd := ev.Physics.Builder.ReplicaJoint(ev.Emery.XZ, di)
	switch act {
	case Forward:
		// todo:
	case Rotate:
		jd.AddTargetAngle(2, val, ev.ActionStiff)
	}
}
