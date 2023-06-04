//go:build race

package main

import (
	"testing"
)

// TestRace runs the boa sim for a little bit to check for race conditions.
func TestRace(t *testing.T) {
	sim := &Sim{}

	sim.New()
	sim.Sim.NTrials = 2

	sim.Config()

	sim.Net.SetNThreads(16)

	sim.Args.SetInt("runs", 1)
	sim.Args.SetInt("epochs", 1)
	sim.Args.SetBool("epclog", false)
	sim.Args.SetBool("runlog", false)

	sim.RunNoGUI()
}
