//go:build !race

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/alecthomas/assert/v2"
	"github.com/emer/emergent/etime"
	"github.com/goki/mat32"
	"golang.org/x/exp/maps"
)

var (
	// set to true to save a new standard for test data
	SaveStandard    = false
	StandardFile    = "testdata/boa_test_std.json"
	LayTolerance    = float32(0.2) // there is horrible variability on vspatch
	GlobalTolerance = float32(0.05)
	NotMaintTol     = float32(0.02)
	PTTol           = float32(0.08) // note: nmda diverges in maint on GPU, even with fastexp on GPU
)

// ReportValDiffs
func ReportValDiffs(t *testing.T, va, vb map[string]float32, aLabel, bLabel string, exclude []string) {
	keys := maps.Keys(va)
	sort.Strings(keys)
	nerrs := 0
	for _, k := range keys {
		hasEx := false
		for _, ex := range exclude {
			if strings.Contains(k, ex) {
				hasEx = true
				break
			}
		}
		if hasEx {
			continue
		}
		av := va[k]
		bv := vb[k]
		dif := mat32.Abs(av - bv)

		isLay := strings.Contains(k, "Lay:")
		tol := LayTolerance
		if isLay {
			if strings.Contains(k, "PT\t") {
				tol = PTTol
			}
		} else {
			if strings.Contains(k, "GvNotMaint") {
				tol = NotMaintTol
			} else {
				tol = GlobalTolerance
			}
		}
		if dif > tol {
			if nerrs == 0 {
				t.Errorf("Diffs found between two runs (10 max): A = %s  B = %s\n", aLabel, bLabel)
			}
			fmt.Printf("%s\tA: %g\tB: %g\tDiff: %g\n", k, av, bv, dif)
			nerrs++
			if nerrs > 100 {
				fmt.Printf("Max diffs exceeded, increase for more\n")
				break
			}
		}
	}
}

type expectedVal struct {
	name string
	val  float64
}

// RunPerfTest runs the boa sim for enough epochs to check it's basically working.
func RunPerfTest(t *testing.T, gpu bool, ndata int) {
	sim := &Sim{}

	sim.New()
	sim.Sim.NData = ndata
	sim.Config()

	sim.Args.SetInt("runs", 1)
	sim.Args.SetInt("epochs", 4)
	sim.Args.SetBool("epclog", false) // set to true to debug runs
	sim.Args.SetBool("runlog", false)
	sim.Args.SetBool("gpu", gpu)

	sim.RunNoGUI()

	expectedVals := []expectedVal{
		{"ActMatch", 0.95},
		{"GateUS", 1.0},
		{"GateCS", 1.0},
		{"MaintEarly", 0.0},
		{"WrongCSGate", 0.0},
		{"Rew", 0.6},
		{"RewPred", 0.001},
	}
	epochTable := sim.Logs.Table(etime.Train, etime.Epoch)
	for _, expected := range expectedVals {
		val := epochTable.CellFloat(expected.name, epochTable.Rows-1)
		assert.False(t, math.IsNaN(val), "%s is NaN", expected.name)
		if expected.val == 1.0 || expected.val == 0 {
			assert.Equal(t, expected.val, val, "%s: %f, want %f", expected.name, val, expected.val)
		} else {
			assert.True(t, val >= expected.val, "%s: %f, want >= %f", expected.name, val, expected.val)
		}
	}
}

func TestPerfCPUnData1(t *testing.T) {
	if os.Getenv("TEST_DEBUG") != "true" {
		t.Skip("Set TEST_DEBUG env var to run debug tests that are informative but long / fail")
	}
	RunPerfTest(t, false, 1)
}

func TestPerfCPUnData2(t *testing.T) {
	if os.Getenv("TEST_DEBUG") != "true" {
		t.Skip("Set TEST_DEBUG env var to run debug tests that are informative but long / fail")
	}
	RunPerfTest(t, false, 2)
}

func TestPerfGPUnData1(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	if os.Getenv("TEST_DEBUG") != "true" {
		t.Skip("Set TEST_DEBUG env var to run debug tests that are informative but long / fail")
	}
	RunPerfTest(t, true, 1)
}

func TestPerfGPUnData2(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	if os.Getenv("TEST_DEBUG") != "true" {
		t.Skip("Set TEST_DEBUG env var to run debug tests that are informative but long / fail")
	}
	RunPerfTest(t, true, 2)
}

// RunStdTest runs the boa sim and compares against standard results
func RunStdTest(t *testing.T, gpu, excludeLays bool, ndata int) {
	sim := &Sim{}

	GPU = gpu

	sim.New()
	sim.Sim.NData = ndata
	sim.Sim.NTrials = 9

	sim.Args.SetBool("test", true)

	sim.Config()

	sim.Args.SetInt("runs", 1)
	sim.Args.SetInt("epochs", 1)
	sim.Args.SetBool("gpu", gpu)
	sim.Args.SetBool("epclog", false)
	sim.Args.SetBool("runlog", false)

	sim.RunNoGUI()

	tdata := sim.TestData
	if SaveStandard {
		fp, err := os.Create(StandardFile)
		defer fp.Close()
		if err != nil {
			t.Fatal(err)
		}
		bw := bufio.NewWriter(fp)
		enc := json.NewEncoder(bw)
		enc.SetIndent("", " ")
		err = enc.Encode(tdata)
		if err != nil {
			t.Fatal(err)
		}
		bw.Flush()
	} else {
		std := make(map[string]float32)
		fp, err := os.Open(StandardFile)
		defer fp.Close()
		if err != nil {
			t.Fatal(err)
		}
		br := bufio.NewReader(fp)
		dec := json.NewDecoder(br)
		err = dec.Decode(&std)
		if err != nil {
			t.Fatal(err)
		}
		ex := []string{}
		if excludeLays {
			ex = []string{"Lay:", "GvNotMaint"}
		}
		ReportValDiffs(t, std, tdata, "Std", "Test", ex)
	}
}

func TestStdCPUnData1(t *testing.T) {
	RunStdTest(t, false, true, 1)
}

func TestStdGPUnData1(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	RunStdTest(t, true, true, 1)
}

// this test reports layer-level differences which diverge in GPU due to
// the nmda computation -- see issue #
func TestStdGPUnData1Debug(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	if os.Getenv("TEST_DEBUG") != "true" {
		t.Skip("Set TEST_DEBUG env var to run debug tests that are informative but long / fail")
	}
	RunStdTest(t, true, false, 1)
}

//////////////////////////////////////////////////////
//   NData Tests

// ReportNDataDiffs
func ReportNDataDiffs(t *testing.T, vals map[string]float32) {
	keys := maps.Keys(vals)
	d0k := make([]string, 0, len(vals))
	for _, k := range keys {
		if strings.Contains(k, "Di: 0") && !strings.Contains(k, "GvEffortCurMax") {
			d0k = append(d0k, k)
		}
	}
	sort.Strings(d0k)
	nerrs := 0
	for _, k := range d0k {
		d1k := strings.ReplaceAll(k, "Di: 0", "Di: 1")
		d0v := vals[k]
		d1v := vals[d1k]
		dif := mat32.Abs(d0v - d1v)

		tol := float32(0.000001)
		if dif > tol {
			if nerrs == 0 {
				t.Errorf("Diffs found between two runs (10 max): A = %s  B = %s\n", "Di=0", "Di=1")
			}
			fmt.Printf("%s\tA: %g\tB: %g\tDiff: %g\n", k, d0v, d1v, dif)
			nerrs++
			if nerrs > 100 {
				fmt.Printf("Max diffs exceeded, increase for more\n")
				break
			}
		}
	}
}

// RunNDataTest runs the boa sim for testing ndata results all the same
func RunNDataTest(t *testing.T, gpu bool) {
	sim := &Sim{}

	sim.New()
	sim.Sim.NData = 2
	sim.Sim.EnvSameSeed = true
	sim.Sim.NTrials = 50

	sim.Args.SetBool("test", true)

	sim.Config()

	sim.Args.SetInt("runs", 1)
	sim.Args.SetInt("epochs", 1)
	sim.Args.SetBool("gpu", gpu)
	sim.Args.SetBool("epclog", false)
	sim.Args.SetBool("runlog", false)

	sim.RunNoGUI()

	ReportNDataDiffs(t, sim.TestData)
}

func TestNDataCPU(t *testing.T) {
	RunNDataTest(t, false)
}

func TestNDataGPU(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	RunNDataTest(t, true)
}
