package main

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/etime"
	"testing"
)

// This doesn't actually test anything, but it can be used for profiling.
func TestSupervised(t *testing.T) {
	var sim Sim
	sim.WorldEnv = sim.ConfigEnv()
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()
	sim.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = 1

	userInterface := egui.UserInterface{
		Looper:                    sim.Loops,
		Network:                   sim.Net.EmerNet,
		AddNetworkLoggingCallback: axon.AddCommonLogItemsForOutputLayers,
	}
	userInterface.AddDefaultLogging()
	userInterface.RunWithoutGui()
}
