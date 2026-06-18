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

	"cogentcore.org/core/base/num"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
)

var (
	// states3 is a list of all the permuted NAry states
	// [8][3] with 0,1 in each
	states3 *tensor.Int

	// total number of states (NAry^3)
	statesN int

	// allcnfs is the full list of all problems
	// represented as [N][NClauses] negation index orderings
	allcnfs *tensor.Int32

	// allcnfsN is the total number of allcnfs
	allcnfsN int

	// outputs is output of basic min-max logic operation
	outputs *tensor.Int32

	// bools3 is a list of all the permuted negation possibilities
	// for brute-force searching. [8][3] with 0,1 in each
	bools3 *tensor.Int

	// allnegs is the full list of all problems
	// represented as [N][NClauses] negation index orderings
	allnegs *tensor.Int32

	// allnegsN is the total number of allnegs
	allnegsN int

	// sats is the corresponding list of satisfiability results for allnegs
	sats *tensor.Int32

	// selected is a selected list of indexes into allcnfs
	// for the subset of problems to train on. first selUnsatN
	// are all unsatisfied ones, remaining is random sample of others,
	// of same count.
	selected []int

	// selUnsatN is number of unsatisfiable problems, first half
	// of selected list.
	selUnsatN int

	// selPermList is a permuted list of size selUnsatN, to be used
	// for selecting subsets of items for actual usage.
	selPermList []int

	globalLock sync.Mutex
)

func states3At(idx int) (int, int, int) {
	return states3.Value(idx, 0), states3.Value(idx, 1), states3.Value(idx, 2)
}

func bools3AtInts(idx int) (int, int, int) {
	return bools3.Value(idx, 0), bools3.Value(idx, 1), bools3.Value(idx, 2)
}

func bools3At(idx int) (bool, bool, bool) {
	i, j, k := bools3AtInts(idx)
	return num.ToBool(i), num.ToBool(j), num.ToBool(k)
}

func printProb(i, n int) string {
	ss := ""
	for k := range n {
		negIdx := int(allcnfs.Value(i, k))
		ss += "["
		for j := range 3 {
			ss += fmt.Sprintf("%d ", states3.Value(negIdx, j))
		}
		ss += "] "
	}
	ss += fmt.Sprintf("= %v", sats.Value(i))
	return ss
}

// ConSatEnv implements constraint satisfaction environments.
type ConSatEnv struct {
	// name of environment -- Train or Test.
	Name string

	// number of mutually-exclusive states per element (boolean = 2)
	NAry int `default:"5"`

	// number of clauses
	NClauses int `default:"2"`

	// NItems is number of items to select from full selected list.
	// Must be < selUnsatN / NData.
	NItems int

	// number of units per localist representation, as one axis in pool in 4D space,
	// such that total is NUnitsPer^2
	NUnitsPer int

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
	ev.NClauses = 2
	ev.NUnitsPer = 2
	ev.NItems = 1500
}

// Config configures the world
func (ev *ConSatEnv) Config(ndata, di int, rndseed int64) {
	n := ev.NClauses
	nu := ev.NUnitsPer
	nary := ev.NAry
	ev.NData = ndata
	ev.Di = di
	ev.RunRandSeed = rndseed
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Input"] = tensor.NewFloat32(3, n, nu, nu*nary) // 3 elements per clause
	ev.States["Output"] = tensor.NewFloat32(1, 1, nu, nu*nary)
}

func (ev *ConSatEnv) Init(run int) {
	if ev.RunRandSeed == 0 {
		ev.RunRandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(run)+1))
	ev.MakeProblems() // ensure
	np := len(selected)
	neven := np / ev.NData
	ev.NItems = min(ev.NItems, neven)
	ev.Trial.Max = ev.NItems
	ev.Trial.Init()
	hi := ev.NItems // / 2
	st := hi * ev.Di
	ev.Items = make([]int, ev.NItems)
	for i := range hi {
		ev.Items[i] = selected[selPermList[st+i]]
	}
	// for i := range hi {
	// 	ev.Items[hi+i] = selected[selUnsatN+selPermList[st+i]]
	// }
	ev.Order = slices.Clone(ev.Items)
	randx.PermuteInts(ev.Order, ev.Rand)
}

func (ev *ConSatEnv) State(el string) tensor.Values {
	return ev.States[el]
}

func (ev *ConSatEnv) String() string {
	return fmt.Sprintf("%d", ev.Trial.Cur)
}

func pow(v, n int) int {
	r := 1
	for range n {
		r *= v
	}
	return r
}

func (ev *ConSatEnv) MakeProblems() {
	globalLock.Lock()
	defer globalLock.Unlock()

	if states3 != nil {
		return
	}
	nv := 3 // 3sat
	nary := ev.NAry
	ns := pow(nary, nv)

	states3 = tensor.NewInt(ns, nv)
	idx := 0
	for i := range nary {
		for j := range nary {
			for k := range nary {
				states3.SetInt(i, idx, 0)
				states3.SetInt(j, idx, 1)
				states3.SetInt(k, idx, 2)
				idx++
			}
		}
	}

	n := ev.NClauses

	pn := pow(ns, n)
	allcnfsN = pn
	allcnfs = tensor.NewInt32(pn, n)
	for i := range pn {
		pp := i
		for k := range n {
			j := pp % ns
			pp /= ns
			allcnfs.Set(int32(j), i, k)
		}
	}

	// basic CNF computation on states
	outputs = tensor.NewInt32(pn)
	for i := range pn {
		and := nary
		for k := range n {
			si := int(allcnfs.Value(i, k))
			t0, t1, t2 := states3At(si)
			or := max(t0, t1, t2)
			and = min(and, or)
		}
		outputs.Set(int32(and), i)
	}
	fmt.Println("n:", n, "ns:", ns, "pn:", pn)

	// now brute-force search solutions
	if false { // todo: get bools3 again, use as negation: nary-val etc
		sats = tensor.NewInt32(pn)
		selected = make([]int, 0, 2*40320)
		tsat := 0
		for i := range pn {
			sat := false
			for tIdx := range ns { // truth values across 3 variables
				csat := true
				for k := range n {
					t0, t1, t2 := bools3At(tIdx)
					negIdx := int(allnegs.Value(i, k))
					n0, n1, n2 := bools3At(negIdx)
					if n0 {
						t0 = !t0
					}
					if n1 {
						t1 = !t1
					}
					if n2 {
						t2 = !t2
					}
					if !(t0 || t1 || t2) {
						csat = false
						break
					}
					// fmt.Println(i, k, "true:", t0, t1, t2)
				}
				if csat {
					// fmt.Println(i, "sat:", tIdx)
					sat = true
					break
				}
			}
			sati := num.FromBool[int32](sat)
			sats.Set(sati, i)
			if sat {
				tsat++
			}
			if !sat {
				selected = append(selected, i)
				// fmt.Println(i, printProb(i, n))
			}
		}
		// fmt.Println("n:", n, "pn:", pn, "tsat:", tsat, "nsat:", pn-tsat, "pct:", float32(tsat)/float32(pn))
	}

	selected = ev.Rand.Perm(pn)
	selUnsatN = len(selected)

	// // select others at random
	// plist := ev.Rand.Perm(pn)
	// count := 0
	// for _, pi := range plist {
	// 	i := plist[pi]
	// 	if sats.Value(i) == 0 {
	// 		continue
	// 	}
	// 	selected = append(selected, i)
	// 	count++
	// 	if count >= selUnsatN {
	// 		break
	// 	}
	// }

	selPermList = ev.Rand.Perm(selUnsatN) // use for selecting items
}

// results for different numbers of clauses -- only get non-sat at 8 and above:
// n: 6 pn: 262144 tsat: 262144 pct: 1
// ok  	github.com/emer/axon/v2/sims/consat/consatenv	0.059s
// n: 7 pn: 2097152 tsat: 2097152 pct: 1
// ok  	github.com/emer/axon/v2/sims/consat/consatenv	0.527s
// n: 8 pn: 16777216 tsat: 16736896 nsat: 40_320, pct: 0.99759674
// ok  	github.com/emer/axon/v2/sims/consat/consatenv	5.010s
// n: 9 pn: 134217728 tsat: 132766208 nsat: 1_451_520 pct: 0.98918533
// ok  	github.com/emer/axon/v2/sims/consat/consatenv	47.034s
// n: 10 pn: 1073741824 tsat: 1043501824 nsat: 30_240_000 pct: 0.9718368
// ok  	github.com/emer/axon/v2/sims/consat/consatenv	434.655s

func (ev *ConSatEnv) Render(item int) {
	in := ev.States["Input"]
	out := ev.States["Output"]
	n := ev.NClauses
	nu := ev.NUnitsPer
	in.SetZeros()
	out.SetZeros()

	for k := range n {
		n0, n1, n2 := states3At(int(allcnfs.Value(item, k)))
		for uy := range nu {
			for ux := range nu {
				in.Set(1, 0, k, uy, n0*nu+ux)
				in.Set(1, 1, k, uy, n1*nu+ux)
				in.Set(1, 2, k, uy, n2*nu+ux)
			}
		}
	}
	for uy := range nu {
		for ux := range nu {
			ov := int(outputs.Value(item))
			out.Set(1, 0, 0, uy, ov*nu+ux)
		}
	}
}

func (ev *ConSatEnv) OutErr(tsr *tensor.Float64) float64 {
	item := ev.Order[ev.Trial.Cur]
	nary := ev.NAry
	nu := ev.NUnitsPer
	ov := int(outputs.Value(item))
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
	if maxi == ov {
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
