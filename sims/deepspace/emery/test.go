// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

// Test is the data for one test.
type Test struct {
	// Name of test
	Name string

	// Action to perform
	Act Actions

	// Action value
	ActVal float32

	// Second action to perform
	Act2 Actions

	// Action value
	Act2Val float32

	// Sensory gain factor
	SenseGain float32

	// Expect an error?
	ExpectErr bool
}

// Tests are a list of test conditions to run in the testing
// version of the environment.
var Tests = []Test{
	{"RotL_VOR_On", Rotate, -3, VORCtrl, 0, 1, false}, // first one is just to prime
	{"RotL_VOR_On", Rotate, -3, VORCtrl, 0, 1, false},
	{"RotL_VOR_Inh", Rotate, -3, VORCtrl, 1, 1, false},
	{"RotL_VOR_Inh", Rotate, -3, VORCtrl, 1, 1, false},
	{"RotL_VOR0_Err10", Rotate, -0.3, VORCtrl, 0, 10, true},
}

// NextTest configures the next test condition, to be called
// at same point as NextAction.
func (ev *EmeryEnv) NextTest() {
	ev.TestTrial.Incr()
	tst := Tests[ev.TestTrial.Cur]
	for di := range ev.NData {
		ev.NextAction(di, tst.Act, tst.ActVal)
		if tst.Act2 < ActionsN {
			ev.NextAction(di, tst.Act2, tst.Act2Val)
		}
	}
}

// TestSetSenseGain sets the SenseGain parameter for testing.
// Called in TakeNextActions.
func (ev *EmeryEnv) TestSetSenseGain() {
	tst := Tests[ev.TestTrial.Cur]
	ev.SenseGain = tst.SenseGain
}
