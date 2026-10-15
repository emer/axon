// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// consatenv implements constraint satisfaction environments.
package consatenv

//go:generate core generate -add-types -add-funcs -gosl

import (
	"embed"
	"fmt"
	"sync"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/popcode"
)

//go:embed *.json
var embedfs embed.FS

var (
	Debug = false

	// cats is the current sensory input category feature values to use.
	cats [][][]int

	// plans is the current plan feature values to use.
	plans [][][]int

	// items is an ordered list of all input items as generated.
	items []Item

	// order is a permuted list of item indexes, to use
	// as a master list for selecting items for each Di.
	order []int

	globalLock sync.Mutex
)

// ConSatEnv implements a constraint satisfaction environment.
// The low-level _sensory_ input is organized into channels,
// each of which has a set of features, with specific values.
// The per-channel features are organized into categories, that define
// a set of values across features, some of which can be ambiguous
// and the rest are specific. By changing the number of categories, the
// ambiguity of any given feature value for determining a category can be
// adjusted, requiring more constraint satisfaction among the subset of
// values presented on any given input to determine what the category is.
//
// The high-level _plan_ features are designed to be orthogonal to
// the sensory features, and to match _across_ channels.
// Each plan looks for _one_ feature value in each channel,
// which would be consistent with a subset of categories in that channel
// (else if multiple, it would need to be an OR relationship).
// As in the sensory case, values are ambiguous for plans, so
// constraint satisfaction at the plan level is needed to resolve the category,
// and thus all of the associated sensory-level feature values to look for.
//
// The key phenomenon is when a high-level feature value is activated
// for low-level input that is not present, but is inferred from low-level
// constraint satisfaction, where the high-level category is also ambiguous
// and must be inferred.
//
// For example, plan needs: water + conspecifics; water is inferred from bank + green
// plants, conspecifics inferred from dung smell + fresh tracks.
// Don't actually have either water or conspecifics, but inference supports it.
// bank or green plants alone doesn't uniquely imply water, etc.
type ConSatEnv struct {
	// name of environment -- Train or Test.
	Name string

	// Configuration options.
	Configs Config

	// NItems is the number of items to use
	NItems int

	// data-parallel n
	NData int

	// data-parallel index
	Di int

	// Trial counts up NItems for indexing
	Trial env.Counter

	// named states
	States map[string]*tensor.Float32

	// Items is the index list into global items, for items specific to this env.
	// uses global order to coordinate non-overlapping items across envs.
	Items []int

	// Copy of Items that is permuted after every pass through list.
	Order []int

	// Current input variables
	Vars []int

	// Current output plan
	Plan int

	// Rand is the random number generator for the env.
	// Created in Init if not already there.
	Rand randx.Rand `display:"-"`

	// RunRandSeed is the random seed multiplier for run counter.
	// It is set to 173 if 0 at start for consistent results by default.
	RunRandSeed int64 `edit:"-"`
}

func (ev *ConSatEnv) Label() string { return ev.Name }

func (ev *ConSatEnv) Defaults() {
	ev.Configs.Defaults()
}

// Config configures the world
func (ev *ConSatEnv) Config(ndata, di int, rndseed int64) {
	cfg := &ev.Configs
	nc := cfg.NChannels
	nf := cfg.NFeatures
	npop := cfg.PopCodeUnits
	ev.NData = ndata
	ev.Di = di
	ev.RunRandSeed = rndseed
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Input"] = tensor.NewFloat32(nf, nc, 1, npop)
	ev.States["Plan"] = tensor.NewFloat32(1, cfg.NPlans, cfg.UnitsPer, 1)
	cfg.PopCode.SetRange(-0.1, 1.1, cfg.PopCodeSigma())
}

func (ev *ConSatEnv) Init(run int) {
	if ev.RunRandSeed == 0 {
		ev.RunRandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(run)+1))
	ev.InitOpen()
	np := len(order)
	neven := np / (2 * ev.NData) // testing and training
	if ev.NItems > 0 {
		ev.NItems = min(ev.NItems, neven)
	} else {
		ev.NItems = neven
	}
	ni := ev.NItems
	ev.Trial.Max = ni
	ev.Trial.Init()
	st := ni * ev.Di
	ev.Items = order[st : st+ni]
	ev.Order = ev.Rand.Perm(len(ev.Items))
}

func (ev *ConSatEnv) State(el string) tensor.Values {
	if el == "Recon" {
		el = "Input"
	}
	return ev.States[el]
}

func (ev *ConSatEnv) String() string {
	return fmt.Sprintf("%d %d", ev.Trial.Cur, ev.Plan)
}

func (ev *ConSatEnv) Render(item int) {
	cfg := &ev.Configs
	in := ev.States["Input"]
	in.SetZeros()
	pln := ev.States["Plan"]
	pln.SetZeros()
	npu := cfg.UnitsPer

	it := &items[item]
	for ch := range it.Values {
		for ft, v := range it.Values[ch] {
			if v < 0 {
				continue
			}
			sv := in.SubSpace(ft, ch).(*tensor.Float32)
			fv := ev.RandVal(v)
			cfg.PopCode.Encode(&sv.Values, fv, cfg.PopCodeUnits, popcode.Set)
		}
	}
	ev.Plan = it.Plan
	for ui := range npu {
		pln.Set(1.0, 0, ev.Plan, ui, 0)
	}
}

func (ev *ConSatEnv) OutErr(tsr *tensor.Float64) float64 {
	cfg := &ev.Configs
	nu := cfg.UnitsPer
	maxi := 0
	maxv := 0.0
	for p := range cfg.NPlans {
		sum := float64(0)
		for ui := range nu {
			sum += tsr.Value(0, p, ui, 0)
		}
		if sum > maxv {
			maxv = sum
			maxi = p
		}
	}
	if maxi == ev.Plan {
		return 0.0
	}
	return 1.0
}

// Step does one step.
func (ev *ConSatEnv) Step() bool {
	if ev.Trial.Incr() {
		randx.PermuteInts(ev.Order, ev.Rand)
	}
	ev.Render(ev.Items[ev.Order[ev.Trial.Cur]])
	return true
}
