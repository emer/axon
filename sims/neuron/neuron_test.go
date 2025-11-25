//go:build not

package neuron

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"cogentcore.org/core/math32"
	"golang.org/x/exp/maps"
)

// todo: set params so GabaB.Gk and NMDA.Ge are 0
// run basic tests that way too.

// tolerance levels -- different tests pass at different levels
var (
	Tol3 = float32(1.0e-3)
	Tol4 = float32(1.0e-4)
	Tol5 = float32(1.0e-5)
	Tol6 = float32(1.0e-6)
	Tol7 = float32(1.0e-7)
	Tol8 = float32(1.0e-8)
)

// ReportValDiffs -- reports diffs between a, b values at given tolerance
func ReportValDiffs(t *testing.T, tolerance float32, va, vb map[string]float32, aLabel, bLabel string, exclude []string) {
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
		dif := math32.Abs(av - bv)
		if dif > tolerance { // allow for small numerical diffs
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

func RunNeuron(t *testing.T, gpu bool) map[string]float32 {
	sim := &Sim{}

	sim.New()

	sim.Config.GUI = false
	sim.Config..GPU = false // for CI
	sim.Config.Run.NRuns = 1
	sim.Config.Log.Cycle = false

	sim.Config.Params.Sheet = "Testing"
	sim.GababGk = 0
	sim.NmdaGe = 0

	sim.ConfigAll()
	sim.RunNoGUI()

	return sim.ValMap
}

func TestNeuronGPU(t *testing.T) {
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU=true env var to run GPU tests")
	}
	cpuValues := RunNeuron(t, false)
	gpuValues := RunNeuron(t, true)
	// On Mac, Tol7 works for most, Tol6 leaves only 2..
	ReportValDiffs(t, Tol5, cpuValues, gpuValues, "CPU", "GPU", []string{"CaPMax", "CaPMaxCa", "GABAB", "NmdaCa"})
}
