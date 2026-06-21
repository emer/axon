// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// consatenv implements constraint satisfaction environments.
package consatenv

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"slices"
	"sync"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
)

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

	// SatThr is the threshold for considering something to be satisfied.
	// strictest setting is NAry-1 (full truth). Doesn't make a difference
	// it turns out, at least for n = 8 with nary = 5 or 4.
	SatThr int

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
	ev.SatThr = 3
}

// Config configures the world
func (ev *ConSatEnv) Config(ndata, di int, rndseed int64) {
	n := ev.NVars
	nu := ev.NUnitsPer
	nary := ev.NAry
	nc := ev.NConstraints + 1
	ev.NData = ndata
	ev.Di = di
	ev.RunRandSeed = rndseed
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Input"] = tensor.NewFloat32(n, 1, nu, nu*nary)
	ev.States["Output"] = tensor.NewFloat32(1, 1, nu, nu*nc)
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
	return fmt.Sprintf("%d", ev.Trial.Cur)
}

func (ev *ConSatEnv) Render(item int) {
	in := ev.States["Input"]
	out := ev.States["Output"]
	nvars := ev.NVars
	nu := ev.NUnitsPer
	in.SetZeros()
	out.SetZeros()

	vars := make([]int, nvars)
	ev.StateVars(item, vars)
	val, _ := constraints.Eval(vars, nil)

	for k := range nvars {
		for uy := range nu {
			for ux := range nu {
				in.Set(1, k, 0, uy, vars[k]*nu+ux)
			}
		}
	}
	for uy := range nu {
		for ux := range nu {
			out.Set(1, 0, 0, uy, (val+1)*nu+ux)
		}
	}
}

func (ev *ConSatEnv) OutErr(tsr *tensor.Float64) float64 {
	item := ev.Order[ev.Trial.Cur]
	nvars := ev.NVars
	nary := ev.NAry
	nu := ev.NUnitsPer
	vars := make([]int, nvars)
	ev.StateVars(item, vars)
	val, _ := constraints.Eval(vars, nil)
	val++
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
	if maxi == val {
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
