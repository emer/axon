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
	StandardFile    = "boa_test_std.json"
	LayTolerance    = float32(0.1)
	GlobalTolerance = float32(0.01)
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
		inRange := false
		if isLay {
			inRange = dif < LayTolerance
		} else {
			inRange = dif < GlobalTolerance
		}
		if !inRange {
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

	GPU = gpu
	sim.New()
	sim.Sim.NData = ndata
	sim.Config()

	sim.Args.SetInt("runs", 1)
	sim.Args.SetInt("epochs", 4)
	sim.Args.SetBool("epclog", true)
	sim.Args.SetBool("runlog", false)

	sim.RunNoGUI()

	expectedVals := []expectedVal{
		{"ActMatch", 0.9},
		{"GateUS", 1.0},
		{"GateCS", 1.0},
		{"MaintEarly", 0.0},
		{"WrongCSGate", 0.0},
		{"Rew", 0.6},
		{"RewPred", 0.01},
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

func TestPerfCpuNdata1(t *testing.T) {
	RunPerfTest(t, false, 1)
}

// RunStdTest runs the boa sim and compares against standard results
func RunStdTest(t *testing.T, gpu bool, ndata int) {
	sim := &Sim{}

	GPU = gpu

	sim.New()
	sim.Sim.NData = ndata

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
		if gpu {
			ex = []string{"Lay:"}
		}
		ReportValDiffs(t, std, tdata, "Std", "Test", ex)
	}
}

func TestStdCpuNdata1(t *testing.T) {
	RunStdTest(t, false, 1)
}

func TestStdGpuNdata1(t *testing.T) {
	RunStdTest(t, true, 1)
}
