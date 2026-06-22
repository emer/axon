// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// consatenv implements constraint satisfaction environments.
package consatenv

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"slices"
	"strconv"
	"sync"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/popcode"
)

// todo: not working due to simmer issue: go:embed *.json
// var embedfs embed.FS

var defaultConstr = `[[{"R":"Greater","A":4,"B":1},{"R":"Greater","A":4,"B":2},{"R":"Equal","A":3,"B":3}],[{"R":"Less","A":4,"B":1},{"R":"Greater","A":1,"B":4},{"R":"Greater","A":1,"B":4}],[{"R":"Less","A":1,"B":2},{"R":"Less","A":4,"B":2},{"R":"Less","A":1,"B":4}],[{"R":"Greater","A":0,"B":1},{"R":"Equal","A":4,"B":1},{"R":"Greater","A":0,"B":1}]]`

var (
	// constraints is a shared list of constraints
	constraints Constraints

	// states is a list of all possible input states
	states *tensor.Int

	// total number of states (NAry^NVariables)
	statesN int

	// items is a permuted list of all states, to use
	// as a master list for selecting items for each Di.
	items []int

	globalLock sync.Mutex
)

// ConSatEnv implements constraint satisfaction environment,
// with multiple sets of constraints, each of which specifies
// a set of required relationships among the variables.
// Each constraint is internally consistent, and incompatible
// with the other constraints, so they form a mutually exclusive set.
// The task is to activate the best-fitting constraint for each input.
type ConSatEnv struct {
	// name of environment -- Train or Test.
	Name string

	// number of mutually-exclusive states per element
	NAry int `default:"5"`

	// number of variables
	NVars int `default:"5"`

	// number of constraints,
	NConstraints int `default:"5"`

	// number of relationships per constraint
	RelationsPer int

	// NItems is number of items to select from full items list.
	// Is enforced to be < selUnsatN / NData.
	NItems int

	// number of units per localist representation, as one axis in pool in 4D space,
	// such that total is NUnitsPer^2
	NUnitsPer int

	// PopCodeUnits is the number of units to use for population code.
	PopCodeUnits int

	// PopCodeSigma is the variance of the popcode rep -- broader for low nary
	PopCodeSigma float32

	// population code for variables.
	PopCode popcode.OneD

	// data-parallel n
	NData int

	// data-parallel index
	Di int

	// Trial counts up NItems for indexing
	Trial env.Counter

	// named states
	States map[string]*tensor.Float32

	// Items is the index list into allcnfs, for items specific to this env.
	// uses global selPermList to coordinate non-overlapping items across envs.
	Items []int

	// Copy of Items that is permuted after every pass through list.
	Order []int

	// Current input variables
	Vars []int

	// Current output value: 0 = no match, 1..nvars
	Value int

	// Rand is the random number generator for the env.
	// Created in Init if not already there.
	Rand randx.Rand `display:"-"`

	// RunRandSeed is the random seed multiplier for run counter.
	// It is set to 173 if 0 at start for consistent results by default.
	RunRandSeed int64 `edit:"-"`
}

func (ev *ConSatEnv) Label() string { return ev.Name }

func (ev *ConSatEnv) Defaults() {
	ev.NAry = 5
	ev.NVars = 5
	ev.NConstraints = 4
	ev.RelationsPer = 3
	ev.NUnitsPer = 2
	ev.NItems = 1500
	ev.PopCodeUnits = 12
	ev.PopCodeSigma = 0.3
	ev.PopCode.Defaults()
}

// Config configures the world
func (ev *ConSatEnv) Config(ndata, di int, rndseed int64) {
	n := ev.NVars
	nu := ev.NUnitsPer
	np := ev.PopCodeUnits
	nc := ev.NConstraints + 1
	ev.NData = ndata
	ev.Di = di
	ev.RunRandSeed = rndseed
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Input"] = tensor.NewFloat32(n, 1, 1, np)
	ev.States["Output"] = tensor.NewFloat32(1, 1, nu, nu*nc)
	ev.Vars = make([]int, n)
	ev.PopCode.SetRange(-0.2, 1.2, ev.PopCodeSigma)
}

func (ev *ConSatEnv) Init(run int) {
	if ev.RunRandSeed == 0 {
		ev.RunRandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(run)+1))
	ev.InitOpen()
	np := len(items)
	neven := np / (2 * ev.NData) // testing and training
	ev.NItems = min(ev.NItems, neven)
	ni := ev.NItems
	ev.Trial.Max = ni
	ev.Trial.Init()
	st := ni * ev.Di
	ev.Items = make([]int, ni)
	copy(ev.Items, items[st:st+ni])
	ev.Order = slices.Clone(ev.Items)
	randx.PermuteInts(ev.Order, ev.Rand)
}

func (ev *ConSatEnv) State(el string) tensor.Values {
	return ev.States[el]
}

func (ev *ConSatEnv) String() string {
	vals := ""
	for _, v := range ev.Vars {
		vals += strconv.Itoa(v) + " "
	}
	return fmt.Sprintf("%d %d: %s", ev.Trial.Cur, ev.Value, vals)
}

func (ev *ConSatEnv) Render(item int) {
	in := ev.States["Input"]
	out := ev.States["Output"]
	nvars := ev.NVars
	nu := ev.NUnitsPer
	in.SetZeros()
	out.SetZeros()

	ev.StateVars(item, ev.Vars)
	ev.Value, _ = constraints.Eval(ev.Vars)
	ev.Value++ // 0 = no value

	for k := range nvars {
		val := float32(ev.Vars[k]) / float32(nvars-1)
		sv := in.SubSpace(k, 0, 0).(*tensor.Float32)
		ev.PopCode.Encode(&sv.Values, val, ev.PopCodeUnits, popcode.Set)
	}
	for uy := range nu {
		for ux := range nu {
			out.Set(1, 0, 0, uy, ev.Value*nu+ux)
		}
	}
}

func (ev *ConSatEnv) OutErr(tsr *tensor.Float64) float64 {
	nary := ev.NAry
	nu := ev.NUnitsPer
	maxi := 0
	maxv := 0.0
	for o := range nary {
		sum := float64(0)
		for uy := range nu {
			for ux := range nu {
				sum += tsr.Value(0, 0, uy, o*nu+ux)
			}
		}
		if sum > maxv {
			maxv = sum
			maxi = o
		}
	}
	if maxi == ev.Value {
		return 0.0
	}
	return 1.0
}

// Step does one step.
func (ev *ConSatEnv) Step() bool {
	if ev.Trial.Incr() {
		randx.PermuteInts(ev.Order, ev.Rand)
	}
	ev.Render(ev.Order[ev.Trial.Cur])
	return true
}
