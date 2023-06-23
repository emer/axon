//go:build multinet

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/emer/emergent/etime"
	"github.com/goki/gi/gi"
)

func TestWeightsSave(t *testing.T) {
	t.Skip("now")
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
	sim.Args.SetInt("runs", 1)

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
	sim.Config()

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
	sim.Config()

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
	sim.Config()

	sim.Args.SetBool("epclog", true)     // set to true to debug runs
	sim.Args.SetBool("tstepclog", false) // set to true to debug runs
	sim.Args.SetBool("runlog", false)
	sim.Args.SetBool("gpu", false) // set to false for CI testing -- set to true for interactive testing
	// note: gets a few errors and takes longer on GPU vs. CPU..  hmm..
	sim.Args.SetInt("runs", 1)
	sim.Args.SetString("startWts", "wtstest.wts.gz")

	sim.RunNoGUI()

	epcTable := sim.Logs.Table(etime.Train, etime.Epoch)
	nrows := epcTable.Rows
	fmt.Printf("nrows: %d\n", nrows)
	if nrows > 3 {
		t.Errorf("more than 3 epochs to learn after loading weights\n")
	}
}
