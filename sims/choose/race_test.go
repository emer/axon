//go:build race

package choose

import (
	"testing"
)

// TestRace runs the boa sim for a little bit to check for race conditions.
func TestRace(t *testing.T) {
	sim := &Sim{}

	sim.New()

	sim.Config.GUI = false
	sim.Config..GPU = false
	sim.Config.Run.NData = 2
	sim.Config.Run.NRuns = 1
	sim.Config.Run.NEpochs = 1
	sim.Config.Run.NTrials = 2
	sim.Config.Run.NThreads = 8
	sim.Config.Log.Run = false
	sim.Config.Log.Epoch = false

	sim.ConfigAll()

	sim.RunNoGUI()
}
