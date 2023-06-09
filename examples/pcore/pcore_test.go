package main

import (
	"testing"
)

func TestPCore(t *testing.T) {
	sim := &Sim{}

	sim.New()
	sim.Config()

	sim.Args.SetBool("epclog", true)    // set to true to debug runs
	sim.Args.SetBool("tstepclog", true) // set to true to debug runs
	sim.Args.SetBool("runlog", false)
	sim.Args.SetBool("gpu", true) // todo: false for real test

	sim.RunNoGUI()

	// expectedVals := []expectedVal{
	// 	{"ActMatch", 0.95},
	// 	{"GateUS", 1.0},
	// 	{"GateCS", 1.0},
	// 	{"MaintEarly", 0.0},
	// 	{"WrongCSGate", 0.0},
	// 	{"Rew", 0.6},
	// 	{"RewPred", 0.001},
	// }
	// epochTable := sim.Logs.Table(etime.Train, etime.Epoch)
	// for _, expected := range expectedVals {
	// 	val := epochTable.CellFloat(expected.name, epochTable.Rows-1)
	// 	assert.False(t, math.IsNaN(val), "%s is NaN", expected.name)
	// 	if expected.val == 1.0 || expected.val == 0 {
	// 		assert.Equal(t, expected.val, val, "%s: %f, want %f", expected.name, val, expected.val)
	// 	} else {
	// 		assert.True(t, val >= expected.val, "%s: %f, want >= %f", expected.name, val, expected.val)
	// 	}
	// }

}
