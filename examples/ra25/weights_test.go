//go:build multinet

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/emer/emergent/v2/etime"
	"goki.dev/gi"
)

func TestWeightsSave(t *testing.T) {
	t.Skip("now")
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}
	sim := &Sim{}

	sim.New()

	sim.Config.Log.Epoch = false
	sim.Config.Log.Run = false
	sim.Config.Run.GPU = false
	sim.Config.Run.NRuns = 1

	sim.ConfigAll()
	sim.RunNoGUI()

	sim.Net.SaveWtsJSON(gi.FileName("wtstest.wts.gz"))
}

func TestWeightsLoad(t *testing.T) {
	t.Skip("now")
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}
	sim := &Sim{}

	sim.New()
	sim.ConfigAll()

	sim.Init()

	sim.Net.OpenWtsJSON(gi.FileName("wtstest.wts.gz"))
	sim.Loops.Run(etime.Test)

	epcTable := sim.Logs.Table(etime.Test, etime.Epoch)
	// epcTable.SaveCSV("wtstest.tst.tsv", etable.Tab, etable.Headers)
	pctCor := epcTable.CellFloat("PctCor", 0)
	fmt.Printf("pctCor: %g\n", pctCor)
	if pctCor < 1 {
		t.Errorf("loaded weights pct cor < 1: %g\n", pctCor)
	}
}

func TestWeightsLoadGPU(t *testing.T) {
	t.Skip() // note: crashing for some reason due to multi-tests..
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}
	sim := &Sim{}

	sim.New()
	sim.ConfigAll()

	sim.Init()

	// gpu version:
	sim.Net.ConfigGPUnoGUI(&sim.Context) // must happen after gui or no gui

	sim.Net.OpenWtsJSON(gi.FileName("wtstest.wts.gz"))
	sim.Loops.Run(etime.Test)

	epcTable := sim.Logs.Table(etime.Test, etime.Epoch)
	// epcTable.SaveCSV("wtstest.tst.tsv", etable.Tab, etable.Headers)
	pctCor := epcTable.CellFloat("PctCor", 0)
	fmt.Printf("pctCor: %g\n", pctCor)
	if pctCor < 1 {
		t.Errorf("loaded weights pct cor < 1: %g\n", pctCor)
	}
}

func TestWeightsTrain(t *testing.T) {
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}
	sim := &Sim{}

	sim.New()

	sim.Config.GUI = false
	sim.Config.Log.Epoch = false
	sim.Config.Log.Run = false
	sim.Config.Run.GPU = false
	// note: gets a few errors and takes longer on GPU vs. CPU..  hmm..
	sim.Config.Run.NRuns = 1
	sim.Config.Run.StartWts = "wtstest.wts.gz"

	sim.ConfigAll()

	sim.RunNoGUI()

	epcTable := sim.Logs.Table(etime.Train, etime.Epoch)
	nrows := epcTable.Rows
	fmt.Printf("nrows: %d\n", nrows)
	if nrows > 10 {
		t.Errorf("more than 10 epochs to learn after loading weights\n")
	}
}
