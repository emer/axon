// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

// Run is a sequence of Conditions to run in order
type Run struct {

	// Name of the run
	Name string `desc:"Name of the run"`

	// Description
	Desc string `desc:"Description"`

	// name of condition for weights file to load prior to starting -- allows faster testing but weights may be out of date
	Weights string `desc:"name of condition for weights file to load prior to starting -- allows faster testing but weights may be out of date"`

	// name of condition 1
	Cond1 string `desc:"name of condition 1"`

	// name of condition 2
	Cond2 string `desc:"name of condition 2"`

	// name of condition 3
	Cond3 string `desc:"name of condition 3"`

	// name of condition 4
	Cond4 string `desc:"name of condition 4"`

	// name of condition 5
	Cond5 string `desc:"name of condition 5"`
}

// NConds returns the number of conditions in this Run
func (rn *Run) NConds() int {
	switch {
	case rn.Cond5 != "":
		return 5
	case rn.Cond4 != "":
		return 4
	case rn.Cond3 != "":
		return 3
	case rn.Cond2 != "":
		return 2
	default:
		return 1
	}
}

// Cond returns the condition name and Condition at the given index
func (rn *Run) Cond(cidx int) (string, *Condition) {
	cnm := ""
	switch cidx {
	case 0:
		cnm = rn.Cond1
	case 1:
		cnm = rn.Cond2
	case 2:
		cnm = rn.Cond3
	case 3:
		cnm = rn.Cond4
	case 4:
		cnm = rn.Cond5
	}
	return cnm, AllConditions[cnm]
}
