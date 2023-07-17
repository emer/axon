package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/emer/emergent/etime"
)

const (
	LT = false
	GT = true
)

func CheckVal(t *testing.T, val, trg float64, gt bool, msg string) {
	fmt.Printf("%s  val: %g  trg: %g\n", msg, val, trg)
	cmp := ">"
	if gt {
		if val > trg {
			return
		}
	} else {
		cmp = "<"
		if val < trg {
			return
		}
	}
	t.Errorf("%s  val: %g %s %g  failed!\n", msg, val, cmp, trg)
}

func TestPVLVAcqExt(t *testing.T) {
	if os.Getenv("TEST_LONG") != "true" {
		t.Skip("Set TEST_LONG=true env var to run longer-running tests")
	}
	sim := &Sim{}

	sim.New()

	sim.Config.GUI = false
	sim.Config.Run.GPU = false // for CI
	sim.Config.Run.NRuns = 1
	sim.Config.Log.Block = false
	sim.Config.Log.Cond = false
	sim.Config.Log.Trial = false
	sim.Config.Env.RunName = "PosAcqExt_A100B50_A0B0" // acquire then extinguish

	sim.ConfigAll()

	sim.RunNoGUI()

	dt := sim.Logs.Table(etime.Train, etime.Condition)
	// dt.OpenCSV("testdata/PVLV_Base_cnd.tsv", etable.Tab) // for tuning tests from saved data

	a_acq_cs_da := dt.CellFloat("C0_CS_DA", 0)
	b_acq_cs_da := dt.CellFloat("C1_CS_DA", 0)

	a_ext_cs_da := dt.CellFloat("C0_CS_DA", 1)
	b_ext_cs_da := dt.CellFloat("C1_CS_DA", 1)

	a_acq_us_vs := dt.CellFloat("C0_US_VSPatch", 0)
	b_acq_us_vs := dt.CellFloat("C1_US_VSPatch", 0)

	a_ext_us_vs := dt.CellFloat("C0_US_VSPatch", 1)
	b_ext_us_vs := dt.CellFloat("C1_CS_VSPatch", 1)

	CheckVal(t, a_acq_cs_da, .9, GT, "A CS Acq DA")
	CheckVal(t, a_acq_cs_da, b_acq_cs_da, GT, "A CS Acq DA > B")
	CheckVal(t, a_ext_cs_da, .2, LT, "A CS Ext DA")
	CheckVal(t, b_ext_cs_da, .2, LT, "B CS Ext DA")

	CheckVal(t, a_acq_us_vs, .8, GT, "A US Acq VSPatch")
	CheckVal(t, a_acq_us_vs, b_acq_us_vs, GT, "A US Acq VSPatch > B")
	CheckVal(t, a_ext_us_vs, .2, LT, "A CS Ext VSPatch")
	CheckVal(t, b_ext_us_vs, .2, LT, "B CS Ext VSPatch")
}
