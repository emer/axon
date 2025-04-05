// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgventral

import (
	"fmt"
	"math/rand"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/popcode"
)

// GoNoEnv implements simple Go vs. NoGo input patterns to test BG learning.
type GoNoEnv struct {

	// name of environment -- Train or Test
	Name string

	// training or testing env?
	Mode Modes

	// trial counter -- set by caller for testing
	Trial env.Counter

	// if true, ACCPos and Neg are set manually for testing specific cases;
	// do not generate random vals for training or auto-increment ACCPos / Neg values during test
	ManualValues bool

	// activation of ACC positive valence -- drives go
	ACCPos float32

	// activation of ACC neg valence -- drives nogo
	ACCNeg float32

	// threshold on diff between ACCPos - ACCNeg for counting as a Go trial
	PosNegThr float32

	// learning rate for reward prediction
	RewPredLRate float32

	// minimum rewpred value
	RewPredMin float32

	// reward value for case where it gated and it should have:
	// nominally 1 but can lead to over-learning, RPE would decrease over time
	GatedShould float32

	// reward value for case where it did not gate and it should have:
	// in real case, would not get anything for this, but 1 is a cheat to improve perf
	NoGatedShould float32

	// reward value for case where it gated and it should not have.  should be -1
	GatedShouldnt float32

	// reward value for case where it did not gate and it should not have:
	// should be 0
	NoGatedShouldnt float32

	// increment in testing activation for test all
	TestInc float32

	// number of repetitions per testing level
	TestReps int

	// number of units, Y
	NUnitsY int `display:"-"`

	// number of units, X
	NUnitsX int `display:"-"`

	// total number of units
	NUnits int `display:"-"`

	// pop code the values in ACCPos and Neg
	PopCode popcode.OneD

	// random number generator for the env -- all random calls must use this
	Rand randx.SysRand `display:"-"`

	// random seed
	RandSeed int64 `edit:"-"`

	// named states: ACCPos, ACCNeg
	States map[string]*tensor.Float32

	// true if Pos - Neg > Thr
	Should bool `edit:"-"`

	// true if model gated on this trial
	Gated bool `edit:"-"`

	// true if gated == should
	Match bool `edit:"-"`

	// reward based on match between Should vs. Gated
	Rew float32 `edit:"-"`

	// reward prediction based on incremental learning: RewPredLRate * (Rew - RewPred)
	RewPred float32 `edit:"-"`

	// reward prediction error: Rew - RewPred
	RPE float32 `edit:"-"`
}

func (ev *GoNoEnv) Label() string { return ev.Name }

func (ev *GoNoEnv) Defaults() {
	ev.TestInc = 0.1
	ev.TestReps = 32
	ev.NUnitsY = 5
	ev.NUnitsX = 5
	ev.NUnits = ev.NUnitsY * ev.NUnitsX
	ev.PosNegThr = 0
	ev.RewPredLRate = 0.01 // GPU 16 0.01 > 0.02 >> 0.05 > 0.1, 0.2 for partial, seq3
	ev.RewPredMin = 0.1    // 0.1 > 0.05 > 0.2
	ev.GatedShould = 1     // note: works; BurstGain = 0.1 helps prevent overlearning
	ev.NoGatedShould = 0   // note: works fine here -- much more realistic
	ev.GatedShouldnt = -1
	ev.NoGatedShouldnt = 0
	ev.PopCode.Defaults()
	ev.PopCode.SetRange(-0.2, 1.2, 0.1)
}

// Config configures the world
func (ev *GoNoEnv) Config(mode Modes, rndseed int64) {
	ev.Mode = mode
	ev.RandSeed = rndseed
	ev.Rand.NewRand(ev.RandSeed)
	ev.States = make(map[string]*tensor.Float32)
	ev.States["ACCPos"] = tensor.NewFloat32(ev.NUnitsY, ev.NUnitsX)
	ev.States["ACCNeg"] = tensor.NewFloat32(ev.NUnitsY, ev.NUnitsX)
	ev.States["Rew"] = tensor.NewFloat32(1, 1)
	ev.States["SNc"] = tensor.NewFloat32(1, 1)
}

func (ev *GoNoEnv) Init(run int) {
	ev.Trial.Init()
}

func (ev *GoNoEnv) State(el string) tensor.Values {
	return ev.States[el]
}

func (ev *GoNoEnv) String() string {
	return fmt.Sprintf("%4f_%4f", ev.ACCPos, ev.ACCNeg)
}

// RenderACC renders the given value in ACC popcode
func (ev *GoNoEnv) RenderACC(name string, val float32) {
	st := ev.States[name]
	ev.PopCode.Encode(&st.Values, val, ev.NUnits, false)
}

// RenderLayer renders a whole-layer popcode value
func (ev *GoNoEnv) RenderLayer(name string, val float32) {
	st := ev.States[name]
	ev.PopCode.Encode(&st.Values, val, ev.NUnits, false)
}

// RenderState renders the current state
func (ev *GoNoEnv) RenderState() {
	ev.RenderACC("ACCPos", ev.ACCPos)
	ev.RenderACC("ACCNeg", ev.ACCNeg)
}

// Step does one step -- must set Trial.Cur first if doing testing
func (ev *GoNoEnv) Step() bool {
	nTestInc := int(1.0/ev.TestInc) + 1
	if !ev.ManualValues {
		if ev.Mode == Test {
			repn := ev.Trial.Cur / ev.TestReps
			pos := repn / nTestInc
			neg := repn % nTestInc
			ev.ACCPos = float32(pos) * ev.TestInc
			ev.ACCNeg = float32(neg) * ev.TestInc
			// fmt.Printf("idx: %d  di: %d  repn: %d  pos: %d  neg: %d\n", idx, di, repn, pos, neg)
		} else {
			ev.ACCPos = rand.Float32()
			ev.ACCNeg = rand.Float32()
		}
	}
	ev.RenderState()
	return true
}

func (ev *GoNoEnv) Action(action string, nop tensor.Values) {
	if action == "Gated" {
		ev.Gated = true
	} else {
		ev.Gated = false
	}
	pndiff := (ev.ACCPos - ev.ACCNeg) - ev.PosNegThr
	should := pndiff > 0
	didGate := ev.Gated
	match := false
	var rew float32
	switch {
	case should && didGate:
		rew = ev.GatedShould
		match = true
	case should && !didGate:
		rew = ev.NoGatedShould
	case !should && didGate:
		rew = ev.GatedShouldnt
	case !should && !didGate:
		rew = ev.NoGatedShouldnt
		match = true
	}
	ev.Should = should
	ev.Match = match
	ev.Rew = rew
	ev.ComputeDA(rew)
}

func (ev *GoNoEnv) ComputeDA(rew float32) {
	ev.RPE = rew - ev.RewPred
	ev.RewPred += ev.RewPredLRate * (rew - ev.RewPred)
	if ev.RewPred < ev.RewPredMin {
		ev.RewPred = ev.RewPredMin
	}
}
