//go:build race

package main

import (
	"testing"

	"github.com/alecthomas/assert/v2"
)

// TestRace runs the boa sim for a little bit to check for race conditions.
func TestRace(t *testing.T) {
	sim := &Sim{}
	sim.New()
	sim.Config()

	assert.NoError(t, sim.Net.Threads.Set(16, 16, 16))

	sim.Args.SetInt("runs", 1)
	sim.Args.SetInt("epochs", 1)
	sim.Args.SetInt("seqs", 1)
	sim.Args.SetBool("epclog", false)
	sim.Args.SetBool("runlog", false)

	sim.RunNoGUI()
}
