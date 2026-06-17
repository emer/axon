// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// consatenv implements constraint satisfaction testing environments.
package consatenv

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"sync"

	"cogentcore.org/core/base/num"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
)

var (
	// bools3 is a list of all the permuted negation possibilities
	// for brute-force searching. [8][3] with 0,1 in each
	bools3 *tensor.Int32

	// allprobs is the full list of all problems
	// represented as [N][NClauses] negation index orderings
	allprobs *tensor.Int32

	// allprobsN is the total number of allprobs
	allprobsN int

	// sats is the corresponding list of satisfiability results for allprobs
	sats *tensor.Int32

	// selected is a selected list of indexes into allprobs
	// for the subset of problems to train on. first selectedN
	// are unsatisfied ones, remaining is random sample of others.
	selected *tensor.Int

	selectedN int

	globalLock sync.Mutex
)

func bools3At(idx int) (bool, bool, bool) {
	return num.ToBool(bools3.Value(idx, 0)), num.ToBool(bools3.Value(idx, 1)), num.ToBool(bools3.Value(idx, 2))
}

func printProb(i, n int) string {
	ss := ""
	for k := range n {
		negIdx := int(allprobs.Value(i, k))
		ss += "["
		for j := range 3 {
			ss += fmt.Sprintf("%d ", bools3.Value(negIdx, j))
		}
		ss += "] "
	}
	ss += fmt.Sprintf("= %v", sats.Value(i))
	return ss
}

// ConSatEnv implements constraint satisfaction testing environments.
// Specifically: 3SAT with fixed set of three variables, so that the
// only  constraint is in the combination of negations.
type ConSatEnv struct {
	// name of environment -- Train or Test -- always Test!
	Name string

	// number of clauses
	NClauses int

	// number of units per localist representation, as one axis in pool in 4D space,
	// such that total is NUnitsPer^2
	NUnitsPer int

	// Trial is used for tracking Sequential
	Trial env.Counter

	// named states
	States map[string]*tensor.Float32

	// Rand is the random number generator for the env.
	// Created in Init if not already there.
	Rand randx.Rand `display:"-"`

	// RunRandSeed is the random seed multiplier for run counter.
	// It is set to 173 if 0 at start for consistent results by default.
	RunRandSeed int64 `edit:"-"`
}

func (ev *ConSatEnv) Label() string { return ev.Name }

func (ev *ConSatEnv) Defaults() {
	ev.NClauses = 8
	ev.NUnitsPer = 2
}

// Config configures the world
func (ev *ConSatEnv) Config(rndseed int64) {
	n := ev.NClauses
	nu := ev.NUnitsPer
	ev.RunRandSeed = rndseed
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Input"] = tensor.NewFloat32(2, n, nu, nu)
	ev.States["Output"] = tensor.NewFloat32(1, 2, nu, nu)
	ev.States["Positions"] = tensor.NewFloat32(n, 2) // X,Y coordinates
	ev.States["Distances"] = tensor.NewFloat32(n, n)
	ev.States["Result"] = tensor.NewFloat32(n) // model result order, city index
	ev.States["Pos"] = tensor.NewFloat32(1, n, nu, nu)
}

func (ev *ConSatEnv) Init(run int) {
	if ev.RunRandSeed == 0 {
		ev.RunRandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(run)+1))
	ev.Trial.Init()
	ev.MakeProblems() // ensure
}

func (ev *ConSatEnv) State(el string) tensor.Values {
	return ev.States[el]
}

func (ev *ConSatEnv) String() string {
	return ""
	// return fmt.Sprintf("%4f_%4f", ev.ACCPos, ev.ACCNeg)
}

func (ev *ConSatEnv) Make() {
	// ps := ev.States["Positions"]
	// ds := ev.States["Distances"]
	// n := ev.NCities
	// tol := 2.0 * ev.GridSpacing
	// minIdx := 0
	// minLen := float32(10)
	// for a := range n {
	// 	var ap math32.Vector2
	// 	for {
	// 		ap.Set(ev.Rand.Float32(), ev.Rand.Float32())
	// 		redo := false
	// 		for b := 0; b < a; b++ {
	// 			bp := math32.Vec2(ps.Value(b, int(math32.X)), ps.Value(b, int(math32.Y)))
	// 			d := ap.DistanceTo(bp)
	// 			if d < tol {
	// 				redo = true
	// 				break
	// 			}
	// 		}
	// 		if !redo {
	// 			break
	// 		}
	// 	}
	// 	ps.Set(ap.X, a, int(math32.X))
	// 	ps.Set(ap.Y, a, int(math32.Y))
	// 	d := ap.Length()
	// 	if d < minLen {
	// 		minLen = d
	// 		minIdx = a
	// 	}
	// }
	// ev.StartCity = minIdx
	// for a := range n {
	// 	var ap math32.Vector2
	// 	ap.Set(ps.Value(a, int(math32.X)), ps.Value(a, int(math32.Y)))
	// 	for b := range n {
	// 		var bp math32.Vector2
	// 		bp.Set(ps.Value(b, int(math32.X)), ps.Value(b, int(math32.Y)))
	// 		d := ap.DistanceTo(bp)
	// 		ds.Set(d, a, b)
	// 	}
	// }
}

func (ev *ConSatEnv) MakeProblems() {
	globalLock.Lock()
	defer globalLock.Unlock()

	if bools3 != nil {
		return
	}
	nv := 3 // 3sat

	bools3 = tensor.NewInt32(8, nv)

	idx := 0
	for i := range 2 {
		for j := range 2 {
			for k := range 2 {
				bools3.SetInt(i, idx, 0)
				bools3.SetInt(j, idx, 1)
				bools3.SetInt(k, idx, 2)
				idx++
			}
		}
	}

	n := ev.NClauses

	pow := func(v, n int) int {
		r := 1
		for range n {
			r *= v
		}
		return r
	}

	pn := pow(8, n)
	allprobsN = pn
	allprobs = tensor.NewInt32(pn, n)
	for i := range pn {
		pp := i
		for k := range n {
			j := pp % 8
			pp /= 8
			allprobs.Set(int32(j), i, k)
		}
	}

	// now brute-force search solutions
	sats = tensor.NewInt32(pn)
	selected = tensor.NewInt(0)
	tsat := 0
	for i := range pn {
		sat := false
		for tIdx := range 8 { // truth values across 3 variables
			csat := true
			for k := range n {
				t0, t1, t2 := bools3At(tIdx)
				negIdx := int(allprobs.Value(i, k))
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
			selected.AppendRowInt(i)
			// fmt.Println(i, printProb(i, n))
		}
	}
	fmt.Println("n:", n, "pn:", pn, "tsat:", tsat, "nsat:", pn-tsat, "pct:", float32(tsat)/float32(pn))

	selectedN = selected.Len()
	// select others at random
	plist := ev.Rand.Perm(selectedN)
	for _, pi := range plist {
		i := plist[pi]
		if sats.Value(i) == 0 {
			continue
		}
		selected.AppendRowInt(i)
	}
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

func (ev *ConSatEnv) RenderGrid() {
	// optx := ev.States["OptimalX"]
	// opty := ev.States["OptimalY"]
	// in := ev.States["Input"]
	// pos := ev.States["Pos"]
	// out := ev.States["Output"]
	// n := ev.NCities
	// n = 1
	// // ng := ev.NGrids
	// nu := ev.NUnitsPer
	// gs := ev.GridSpacing
	//
	// in.SetZeros()
	// out.SetZeros()
	//
	// for p := range n {
	// 	x := optx.Value(p)
	// 	y := opty.Value(p)
	// 	xi := int(math32.Round(x / gs))
	// 	yi := int(math32.Round(y / gs))
	// 	if !ev.Sequential {
	// 		out.Set(1, 0, p, yi, xi)
	// 	}
	// 	for uy := range nu {
	// 		for ux := range nu {
	// 			in.Set(1, yi, xi, uy, ux)
	// 			if ev.Sequential && p == ev.Trial.Cur {
	// 				out.Set(1, yi, xi, uy, ux)
	// 			}
	// 		}
	// 	}
	// }
	// if ev.Sequential {
	// 	for uy := range nu {
	// 		for ux := range nu {
	// 			pos.Set(1, 0, ev.Trial.Cur, uy, ux)
	// 		}
	// 	}
	// }
}

// Step does one step.
func (ev *ConSatEnv) Step() bool {
	// ev.MakeCities()
	// ev.BruteForce()
	// ev.RenderGrid()
	return true
}
