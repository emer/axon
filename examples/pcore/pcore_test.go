package main

import (
	"os"
	"testing"

	"github.com/emer/etable/tsragg"
)

func TestPCore(t *testing.T) {
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}
	sim := &Sim{}

	sim.New()
	sim.Config()

	sim.Args.SetBool("epclog", false)    // set to true to debug runs
	sim.Args.SetBool("tstepclog", false) // set to true to debug runs
	sim.Args.SetBool("runlog", false)
	sim.Args.SetBool("gpu", false) // set to false for CI testing -- set to true for interactive testing

	sim.RunNoGUI()

	tstdt := sim.Logs.MiscTable("TestTrialStats")
	matchAvg := tsragg.Mean(tstdt.ColByName("Match"))
	// fmt.Printf("matchAvg: %g\n", matchAvg)
	if matchAvg < .8 {
		t.Errorf("PCore test match: %g is below threshold of .8\n", matchAvg)
	}
	// todo: could test that PFCVM_RT is longer for more conflicted cases
}
