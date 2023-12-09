package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/emer/emergent/v2/etime"
)

// basic pos acq, ext
func TestPosAcqExtA100_A0(t *testing.T) {
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}

	expects := []Expect{
		{"C0_CS_DA", 0, .9, GT, "A CS Acq DA"},
		{"C0_CS_DA", 1, .2, LT, "A CS Ext DA"},
		{"C0_US_DA", 0, .2, LT, "A US Acq DA"},
		{"C0_US_DA", 1, .1, LT, "A US Ext DA"},
		{"C0_US_VSPatch", 0, .5, GT, "A US Acq VSPatch"},
		{"C0_US_VSPatch", 1, .1, LT, "A US Ext VSPatch"},
	}

	RunTest(t, "PosAcqExt_A100_A0", expects)
}

// basic neg acq
func TestNegAcqD100(t *testing.T) {
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}

	expects := []Expect{
		{"C0_CS_DA", 0, -.8, LT, "A CS Acq DA"},
		{"C0_US_DA", 0, -.5, LT, "A US Acq DA"},
	}

	RunTest(t, "NegAcq_D100", expects)
}

// A100, B50
func TestPosAcqExtA100B50_A0B0(t *testing.T) {
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}

	expects := []Expect{
		{"C0_CS_DA", 0, .7, GT, "A CS Acq DA"},
		{"C1_CS_DA", 0, .6, GT, "B CS Acq DA"}, // note: should be lower than A
		{"C0_CS_DA", 1, .1, LT, "A CS Ext DA"},
		{"C1_CS_DA", 1, .1, LT, "B CS Ext DA"},
		{"C0_US_VSPatch", 0, .3, GT, "A US Acq VSPatch"},
		{"C1_US_VSPatch", 0, 0.0, GT, "B US Acq VSPatch"}, // pretty variable
		{"C0_US_VSPatch", 1, .1, LT, "A US Ext VSPatch"},
		{"C1_US_VSPatch", 1, .1, LT, "B US Ext VSPatch"},
	}

	RunTest(t, "PosAcqExt_A100B50_A0B0", expects)
}

// A100, 50, 0 cycling
func TestPosAcq_ACycle100_50_0(t *testing.T) {
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}

	expects := []Expect{
		{"C0_CS_DA", 0, .8, GT, "100% CS DA"},
		{"C0_CS_DA", 1, .5, GT, "50%  CS DA"},
		{"C0_CS_DA", 2, .2, LT, "0%   CS DA"},
		{"C0_CS_DA", 3, .4, GT, "50%  CS DA"},
		{"C0_CS_DA", 4, .8, GT, "100% CS DA"},

		{"C0_US_VSPatch", 0, .4, GT, "100% US VSPatch"},
		{"C0_US_VSPatch", 1, .1, GT, "50%  US VSPatch"},
		{"C0_US_VSPatch", 2, .1, LT, "0%   US VSPatch"},
		{"C0_US_VSPatch", 3, 0, GT, "50%  US VSPatch"},
		{"C0_US_VSPatch", 4, .3, GT, "100% US VSPatch"},
	}

	// RunTest(t, "PosAcq_ACycle100_50_0_Blk10", expects)
	RunTest(t, "PosAcq_ACycle100_50_0_Blk20", expects) // more reliable with 20
}

/////////////////////////////////////////////////////////////
// Testing infra:

const (
	LT = false
	GT = true
)

func LTGTString(gt bool) string {
	if gt {
		return ">"
	}
	return "<"
}

type Expect struct {
	Cell  string
	Row   int
	Val   float64
	Rel   bool // LT or GT
	Label string
}

func CheckVal(t *testing.T, val, trg float64, gt bool, msg string) {
	cmp := LTGTString(gt)
	fmt.Printf("%s  val: %7.2g %s trg: %g\n", msg, val, cmp, trg)
	if gt {
		if val > trg {
			return
		}
	} else {
		if val < trg {
			return
		}
	}
	t.Errorf("%s  val: %7.2g %s %g  failed!\n", msg, val, cmp, trg)
}

func RunTest(t *testing.T, runName string, expect []Expect) {
	sim := &Sim{}

	sim.New()

	sim.Config.GUI = false
	sim.Config.Run.GPU = false // for CI
	sim.Config.Run.NRuns = 1
	sim.Config.Log.Block = true // false
	sim.Config.Log.Cond = true  // false
	sim.Config.Log.Trial = false
	sim.Config.Env.RunName = runName

	sim.ConfigAll()

	sim.RunNoGUI()

	dt := sim.Logs.Table(etime.Train, etime.Condition)
	// dt.OpenCSV("testdata/PVLV_Base_cnd.tsv", etable.Tab) // for tuning tests from saved data

	for _, ex := range expect {
		cell := dt.CellFloat(ex.Cell, ex.Row)
		CheckVal(t, cell, ex.Val, ex.Rel, ex.Label)
	}

}
