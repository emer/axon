package main

import (
	"testing"
)

// TestBoaRACE runs the boa model for one epoch, but doesn't do anything with the results
// We use this to check for race conditions on CI
func TestBoaRACE(t *testing.T) {
	TheSim.New()
	TheSim.Config()

	err := TheSim.Net.Threads.Set(16, 16, 16)
	if err != nil {
		t.Fatal(err)
	}
	TheSim.Args.SetInt("runs", 1)
	TheSim.Args.SetInt("epochs", 1)

	TheSim.CmdArgs()
}
