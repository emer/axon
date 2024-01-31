// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"

	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/popcode"
	"github.com/emer/etable/v2/etensor"
)

// GoNoEnv implements simple Go vs. NoGo input patterns to test BG learning.
type GoNoEnv struct {

	// name of environment -- Train or Test
	Nm string

	// training or testing env?
	Mode etime.Modes

	// trial counter -- set by caller for testing
	Trial env.Ctr

	// activation of ACC positive valence -- drives go
	ACCPos float32

	// activation of ACC neg valence -- drives nogo
	ACCNeg float32

	// threshold on diff between ACCPos - ACCNeg for counting as a Go trial
	PosNegThr float32

	// ACCPos and Neg are set manually -- do not generate random vals for training or auto-increment ACCPos / Neg values during test
	ManualVals bool

	// increment in testing activation for test all
	TestInc float32

	// number of repetitions per testing level
	TestReps int

	// for case with multiple pools evaluated in parallel (not currently used), this is the across-pools multiplier in activation of ACC positive valence -- e.g., .9 daecrements subsequent units by 10%
	ACCPosInc float32

	// for case with multiple pools evaluated in parallel (not currently used), this is the across-pools multiplier in activation of ACC neg valence, e.g., 1.1 increments subsequent units by 10%
	ACCNegInc float32

	// number of units within each pool, Y
	NUnitsY int `view:"-"`

	// number of units within each pool, X
	NUnitsX int `view:"-"`

	// total number of units within each pool
	NUnits int `view:"-"`

	// pop code the values in ACCPos and Neg
	PopCode popcode.OneD

	// random number generator for the env -- all random calls must use this
	Rand erand.SysRand `view:"-"`

	// random seed
	RndSeed int64 `edit:"-"`

	// named states: ACCPos, ACCNeg
	States map[string]*etensor.Float32

	// true if Pos - Neg > Thr
	Should bool `edit:"-"`

	// true if model gated on this trial
	Gated bool `edit:"-"`

	// true if gated == should
	Match bool `edit:"-"`

	// reward based on match between Should vs. Gated
	Rew float32 `edit:"-"`
}

func (ev *GoNoEnv) Name() string {
	return ev.Nm
}

func (ev *GoNoEnv) Desc() string {
	return "GoNoEnv"
}

func (ev *GoNoEnv) Defaults() {
	ev.TestInc = 0.1
	ev.TestReps = 32
	ev.NUnitsY = 5
	ev.NUnitsX = 5
	ev.NUnits = ev.NUnitsY * ev.NUnitsX
	ev.ACCPosInc = 0.8
	ev.ACCNegInc = 1.1
	ev.PosNegThr = 0 // todo: was 0.1?
	ev.PopCode.Defaults()
	ev.PopCode.SetRange(-0.2, 1.2, 0.1)
}

// Config configures the world
func (ev *GoNoEnv) Config(mode etime.Modes, rndseed int64) {
	ev.Mode = mode
	ev.RndSeed = rndseed
	ev.Rand.NewRand(ev.RndSeed)
	ev.States = make(map[string]*etensor.Float32)
	ev.States["ACCPos"] = etensor.NewFloat32([]int{ev.NUnitsY, ev.NUnitsX}, nil, []string{"Y", "X"})
	ev.States["ACCNeg"] = etensor.NewFloat32([]int{ev.NUnitsY, ev.NUnitsX}, nil, []string{"Y", "X"})
	ev.States["Rew"] = etensor.NewFloat32([]int{1, 1}, nil, nil)
	ev.States["SNc"] = etensor.NewFloat32([]int{1, 1}, nil, nil)
}

func (ev *GoNoEnv) Validate() error {
	return nil
}

func (ev *GoNoEnv) Init(run int) {
	ev.Trial.Init()
}

func (ev *GoNoEnv) Counter(scale env.TimeScales) (cur, prv int, changed bool) {
	if scale == env.Trial {
		return ev.Trial.Query()
	}
	return 0, 0, false
}

func (ev *GoNoEnv) State(el string) etensor.Tensor {
	return ev.States[el]
}

// RenderACC renders the given value in ACC popcode, across pools
func (ev *GoNoEnv) RenderACC(name string, val, inc float32) {
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
	ev.RenderACC("ACCPos", ev.ACCPos, ev.ACCPosInc)
	ev.RenderACC("ACCNeg", ev.ACCNeg, ev.ACCNegInc)
}

// Step does one step -- must set Trial.Cur first if doing testing
func (ev *GoNoEnv) Step() bool {
	nTestInc := int(1.0/ev.TestInc) + 1
	if !ev.ManualVals {
		if ev.Mode == etime.Test {
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

func (ev *GoNoEnv) Action(action string, nop etensor.Tensor) {
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
		rew = 1
		match = true
	case should && !didGate:
		rew = -1
	case !should && didGate:
		rew = -1
	case !should && !didGate:
		rew = 1
		match = true
	}
	ev.Should = should
	ev.Match = match
	ev.Rew = rew
}
