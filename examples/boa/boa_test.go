package main

import (
	"math"
	"testing"

	"github.com/alecthomas/assert/v2"
	"github.com/emer/emergent/etime"
)

type expectedVal struct {
	name string
	val  float64
}

// TestBoa runs the boa sim for enough epochs to check it's basically working.
func TestBoa(t *testing.T) {
	sim := &Sim{}
	sim.New()
	sim.Config()

	assert.NoError(t, sim.Net.Threads.Set(16, 16, 16))

	sim.Args.SetInt("runs", 1)
	sim.Args.SetInt("epochs", 3)
	sim.Args.SetInt("seqs", 25)
	sim.Args.SetBool("epclog", false)
	sim.Args.SetBool("runlog", false)

	sim.RunNoGUI()

	expectedVals := []expectedVal{
		{"ActMatch", 1.0},
		{"GateUS", 1.0},
		{"GateCS", 1.0},
		{"MaintEarly", 0.0},
		{"WrongCSGate", 0.0},
		{"Rew", 0.8},
		{"RewPred", 0.17488},
	}
	epochTable := sim.Logs.Table(etime.Train, etime.Epoch)
	for _, expected := range expectedVals {
		val := epochTable.CellFloat(expected.name, epochTable.Rows-1)
		assert.False(t, math.IsNaN(val), "%s is NaN", expected.name)
		if expected.val == 1.0 || expected.val == 0 {
			assert.Equal(t, expected.val, val, "%s: %f, want %f", expected.name, val, expected.val)
		} else {
			assert.True(t, val >= expected.val, "%s: %f, want >= %f", expected.name, val, expected.val)
		}
	}
}
