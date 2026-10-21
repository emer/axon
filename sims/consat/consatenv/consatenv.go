// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// consatenv implements constraint satisfaction environments.
package consatenv

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"sync"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
)

// //go:embed *.json
// var embedfs embed.FS

var (
	Debug = false

	// lines are the full set of lines
	lines []Line

	// order is a permuted list of lines, to use
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

	// data-parallel n
	NData int

	// data-parallel index
	Di int

	// Trial counts for one epoch
	Trial env.Counter

	// ItemCtr counts over items to present
	ItemCtr env.Counter

	// named states
	States map[string]*tensor.Float32

	// Items is the index list into global items, for items specific to this env.
	// uses global order to coordinate non-overlapping items across envs.
	Items []int

	// Copy of Items that is permuted after every pass through list.
	Order []int

	// Item is the current input item index.
	Item int

	// Current output plan number.
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
func (ev *ConSatEnv) Config(trials, ndata, di int, rndseed int64) {
	cfg := &ev.Configs
	np := cfg.NPixels
	nl := cfg.NLines
	ev.Trial.Max = trials
	ev.NData = ndata
	ev.Di = di
	ev.RunRandSeed = rndseed
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Input"] = tensor.NewFloat32(nl, nl, np, np)
	ev.States["Full"] = tensor.NewFloat32(nl, nl, np, np)
}

func (ev *ConSatEnv) Init(run int) {
	ev.InitLines()
	if ev.RunRandSeed == 0 {
		ev.RunRandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(run)+1))
	np := len(order)
	// fmt.Println("np:", np)
	// neven := np / (2 * ev.NData) // testing and training
	neven := np / (ev.NData) // training and testing get all
	ev.Trial.Init()
	ev.ItemCtr.Max = neven
	ev.ItemCtr.Init()
	st := (neven * ev.Di) % np
	// fmt.Println(ev.Di, neven, np)
	ev.Items = order[st : st+neven]
	ev.Order = ev.Rand.Perm(len(ev.Items))
}

func (ev *ConSatEnv) State(el string) tensor.Values {
	return ev.States[el]
}

func (ev *ConSatEnv) String() string {
	return fmt.Sprintf("%d %d", ev.Trial.Cur, ev.Plan)
}

// Step does one step.
func (ev *ConSatEnv) Step() bool {
	ev.Trial.Incr()
	if ev.ItemCtr.Incr() {
		randx.PermuteInts(ev.Order, ev.Rand)
	}
	ev.Item = ev.Items[ev.Order[ev.ItemCtr.Cur]]
	ev.RenderLine(lines[ev.Item])
	return true
}
