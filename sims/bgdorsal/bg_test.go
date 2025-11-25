//go:build not

package bgdorsal

import (
	"os"
	"testing"

	"cogentcore.org/lab/stats/stats"
)

func TestPCore(t *testing.T) {
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}
	sim := &Sim{}

	sim.New()

	sim.Config.GUI = false
	sim.Config..GPU = false // for CI
	sim.Config.Run.NRuns = 1
	sim.Config.Log.Run = false
	sim.Config.Log.Epoch = false
	sim.Config.Log.Run = false
	sim.Config.Log.TestEpoch = false

	sim.ConfigAll()
	sim.RunNoGUI()

	tstdt := sim.Logs.MiscTable("TestTrialStats")
	matchAvg := stats.MeanTensor(tstdt.ColumnByName("Match"))
	// fmt.Printf("matchAvg: %g\n", matchAvg)
	if matchAvg < .8 {
		t.Errorf("PCore test match: %g is below threshold of .8\n", matchAvg)
	}
	// todo: could test that PFCVM_RT is longer for more conflicted cases
}
