// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

package axon

import (
	"testing"

	"goki.dev/mat32/v2"
)

// TOLERANCE is the numerical difference tolerance for comparing vs. target values
const TOLERANCE = float32(1.0e-8)

func TestActUpdate(t *testing.T) {
	geinc := []float32{.01, .02, .03, .04, .05, .1, .2, .3, .2}
	correctGe := []float32{0.01, 0.038, 0.090399995, 0.17232, 0.28785598, 0.48028478, 0.8342278, 1.4173822, 2.0839057}
	ge := make([]float32, len(geinc))
	correctInet := []float32{0.006738626, 0.024824237, 0.056748517, 0.10130837, 0.15287264, 0.22038713, 0.42749277, 3.0841203, -1.1801764}
	inet := make([]float32, len(geinc))
	correctVm := []float32{0.30244464, 0.31147462, 0.33222097, 0.36955467, 0.4265467, 0.51036406, 0.67449206, 1, 0.5800084}
	vm := make([]float32, len(geinc))
	correctSpike := []float32{0, 0, 0, 0, 0, 0, 0, 1, 0}
	spike := make([]float32, len(geinc))
	correctAct := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0}
	act := make([]float32, len(geinc))

	ac := ActParams{}
	ac.Defaults()
	ac.Gbar.L = 0.2 // correct default

	neuron := &Neuron{}
	ac.InitActs(neuron)

	for i := range geinc {
		neuron.GeRaw += geinc[i]
		ac.GeFmRaw(neuron, neuron.GeRaw, 0)
		ac.GiFmRaw(neuron, neuron.GiRaw)
		ac.VmFmG(neuron)
		ac.SpikeFmVm(neuron)
		ge[i] = neuron.Ge
		inet[i] = neuron.Inet
		vm[i] = neuron.Vm
		spike[i] = neuron.Spike
		act[i] = neuron.Act
		difge := mat32.Abs(ge[i] - correctGe[i])
		if difge > TOLERANCE { // allow for small numerical diffs
			t.Errorf("ge err: idx: %v, geinc: %v, ge: %v, corge: %v, dif: %v\n", i, geinc[i], ge[i], correctGe[i], difge)
		}
		difinet := mat32.Abs(inet[i] - correctInet[i])
		if difinet > TOLERANCE { // allow for small numerical diffs
			t.Errorf("Inet err: idx: %v, geinc: %v, inet: %v, corinet: %v, dif: %v\n", i, geinc[i], inet[i], correctInet[i], difinet)
		}
		difvm := mat32.Abs(vm[i] - correctVm[i])
		if difvm > TOLERANCE { // allow for small numerical diffs
			t.Errorf("Vm err: idx: %v, geinc: %v, vm: %v, corvm: %v, dif: %v\n", i, geinc[i], vm[i], correctVm[i], difvm)
		}
		difspk := mat32.Abs(spike[i] - correctSpike[i])
		if difspk > TOLERANCE { // allow for small numerical diffs
			t.Errorf("Spk err: idx: %v, geinc: %v, spk: %v, corspk: %v, dif: %v\n", i, geinc[i], spike[i], correctSpike[i], difspk)
		}
		difact := mat32.Abs(act[i] - correctAct[i])
		if difact > TOLERANCE { // allow for small numerical diffs
			t.Errorf("Act err: idx: %v, geinc: %v, act: %v, coract: %v, dif: %v\n", i, geinc[i], act[i], correctAct[i], difact)
		}
	}
	// fmt.Printf("ge vals: %v\n", ge)
	// fmt.Printf("Inet vals: %v\n", inet)
	// fmt.Printf("vm vals: %v\n", vm)
	// fmt.Printf("act vals: %v\n", act)
}
