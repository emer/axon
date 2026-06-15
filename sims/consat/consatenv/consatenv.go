// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// consatenv implements constraint satisfaction testing environments
// e.g., the Traveling Salesman Problem (TSP).
package consatenv

//go:generate core generate -add-types -add-funcs -gosl

import (
	"sync"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/plot/plots"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
)

var (
	// orders is a list of all the permuted orders of cities,
	// for brute-force searching. Shared among all instances.
	orders *tensor.Int32

	// ordersN is the n! for the orders list -- number of permutations.
	ordersN int

	// ordersNCities is the NCities for the orders list
	ordersNCities int

	ordersLock sync.Mutex
)

// ConSatEnv implements constraint satisfaction testing environments
// e.g., the Traveling Salesman Problem (TSP).
// Generates an input display of cities, target is shortest order,
// as a simultaneous map of grids, ordered relative to start city,
// which is in the lower-left corner (shortest pos length).
type ConSatEnv struct {
	// name of environment -- Train or Test -- always Test!
	Name string

	// number of states (e.g., number of cities)
	NCities int

	// grid spacing in 1x1 unit square for plotting inputs, and min dist
	GridSpacing float32

	// number of grid squares: 1 / .1 + 1 = 11
	NGrids int

	// number of units per localist representation, as one axis in pool in 4D space,
	// such that total is NUnitsPer^2
	NUnitsPer int

	// Present order prompts one at a time
	Sequential bool

	// NPrompts is the number of Sequential prompts to provide (<= NCities)
	// only used if > 0 -- otherwise NCities
	NPrompts int

	// Trial is used for tracking Sequential
	Trial env.Counter

	// StartCity is the index of starting city, with shortest position Length
	StartCity int `edit:"-"`

	// named states
	States map[string]*tensor.Float32

	// Plot of cities
	Plot *plot.Plot

	// Rand is the random number generator for the env.
	// Created in Init if not already there.
	Rand randx.Rand `display:"-"`

	// RunRandSeed is the random seed multiplier for run counter.
	// It is set to 173 if 0 at start for consistent results by default.
	RunRandSeed int64 `edit:"-"`
}

func (ev *ConSatEnv) Label() string { return ev.Name }

func (ev *ConSatEnv) Defaults() {
	ev.GridSpacing = 0.1
	ev.NCities = 4
	ev.NUnitsPer = 2
	ev.Update()
}

func (ev *ConSatEnv) Update() {
	ev.NGrids = int(1.0/ev.GridSpacing) + 1
}

// Config configures the world
func (ev *ConSatEnv) Config(rndseed int64) {
	n := ev.NCities
	ng := ev.NGrids
	nu := ev.NUnitsPer
	ev.RunRandSeed = rndseed
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Positions"] = tensor.NewFloat32(n, 2) // X,Y coordinates
	ev.States["Distances"] = tensor.NewFloat32(n, n)
	ev.States["Result"] = tensor.NewFloat32(n) // model result order, city index
	ev.States["Input"] = tensor.NewFloat32(ng, ng, nu, nu)
	ev.States["Pos"] = tensor.NewFloat32(1, n, nu, nu)
	if ev.Sequential {
		ev.States["Output"] = tensor.NewFloat32(ng, ng, nu, nu)
		if ev.NPrompts > 0 {
			ev.Trial.Max = ev.NPrompts
		} else {
			ev.Trial.Max = ev.NCities
		}
	} else {
		ev.States["Output"] = tensor.NewFloat32(1, n, ng, ng)
		ev.Trial.Max = 1
	}
	ev.States["Optimal"] = tensor.NewFloat32(n)      // optimal order, city index
	ev.States["OptimalX"] = tensor.NewFloat32(n + 1) // optimal order, x coord
	ev.States["OptimalY"] = tensor.NewFloat32(n + 1) // optimal order, y coord
	ev.MakeOrders()                                  // ensure
}

func (ev *ConSatEnv) Init(run int) {
	if ev.RunRandSeed == 0 {
		ev.RunRandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(run)+1))
	ev.Trial.Init()
}

func (ev *ConSatEnv) State(el string) tensor.Values {
	return ev.States[el]
}

func (ev *ConSatEnv) String() string {
	return ""
	// return fmt.Sprintf("%4f_%4f", ev.ACCPos, ev.ACCNeg)
}

// DistWeight returns the weight for the distance between the two
// given cities, using exp(-DistExp * dist)
func (ev *ConSatEnv) DistWeight(a, b int) float32 {
	ds := ev.States["NormWeights"]
	d := ds.Value(a, b)
	return d
}

func (ev *ConSatEnv) MakeCities() {
	ps := ev.States["Positions"]
	ds := ev.States["Distances"]
	n := ev.NCities
	tol := 2.0 * ev.GridSpacing
	minIdx := 0
	minLen := float32(10)
	for a := range n {
		var ap math32.Vector2
		for {
			ap.Set(ev.Rand.Float32(), ev.Rand.Float32())
			redo := false
			for b := 0; b < a; b++ {
				bp := math32.Vec2(ps.Value(b, int(math32.X)), ps.Value(b, int(math32.Y)))
				d := ap.DistanceTo(bp)
				if d < tol {
					redo = true
					break
				}
			}
			if !redo {
				break
			}
		}
		ps.Set(ap.X, a, int(math32.X))
		ps.Set(ap.Y, a, int(math32.Y))
		d := ap.Length()
		if d < minLen {
			minLen = d
			minIdx = a
		}
	}
	ev.StartCity = minIdx
	for a := range n {
		var ap math32.Vector2
		ap.Set(ps.Value(a, int(math32.X)), ps.Value(a, int(math32.Y)))
		for b := range n {
			var bp math32.Vector2
			bp.Set(ps.Value(b, int(math32.X)), ps.Value(b, int(math32.Y)))
			d := ap.DistanceTo(bp)
			ds.Set(d, a, b)
		}
	}
}

func (ev *ConSatEnv) MakeOrders() {
	ordersLock.Lock()
	defer ordersLock.Unlock()

	n := ev.NCities
	if orders != nil && ordersNCities == n {
		return
	}
	ordersNCities = n
	no := 1
	in := make([]int32, n)
	for i := range n {
		in[i] = int32(i)
		no *= i + 1
	}
	// fmt.Println("n:", n, "no:", no)

	orders = tensor.NewInt32(no, n)
	ordersN = no

	idx := 0
	var permute func(xs []int32, low int)
	permute = func(xs []int32, low int) {
		if low+1 >= len(xs) {
			for c := range n {
				orders.Set(xs[c], idx, c)
			}
			idx++
			return
		}
		permute(xs, low+1)
		for i := low + 1; i < len(xs); i++ {
			xs[low], xs[i] = xs[i], xs[low]
			permute(xs, low+1)
			xs[low], xs[i] = xs[i], xs[low]
		}
	}
	permute(in, 0)
	// fmt.Println(orders)
}

func (ev *ConSatEnv) BruteForce() {
	ds := ev.States["Distances"]
	n := ev.NCities
	no := ordersN

	mind := float32(1000000)
	mini := -1
	for o := range no {
		td := float32(0)
		for p := 1; p < n+1; p++ {
			pi := orders.Value(o, p-1)
			ci := orders.Value(o, p%n)
			d := ds.Value(int(pi), int(ci))
			td += d
		}
		if td < mind {
			mind = td
			mini = o
		}
	}
	// fmt.Println("min dist:", mind, "index:", mini)
	opt := ev.States["Optimal"]
	pos := ev.States["Positions"]
	optx := ev.States["OptimalX"]
	opty := ev.States["OptimalY"]
	start := 0
	for p := range n {
		op := int(orders.Value(mini, p))
		if op == ev.StartCity {
			start = p
			break
		}
	}
	// find direction to go based on largest Y coord
	dir := 1
	p1y := pos.Value((start+1)%n, int(math32.Y))
	m1y := pos.Value((start+n-1)%n, int(math32.Y))
	if p1y < m1y {
		dir = -1
	}
	for p := range n + 1 {
		pi := (start + n + dir*p) % n
		op := int(orders.Value(mini, pi))
		if p < n {
			opt.Set(float32(op), p)
		}
		optx.Set(pos.Value(op, int(math32.X)), p)
		opty.Set(pos.Value(op, int(math32.Y)), p)
	}
	// fmt.Println("optimal:", opt, optx, opty)
}

func (ev *ConSatEnv) MakePlot() {
	if ev.Plot == nil {
		return
	}
	optx := ev.States["OptimalX"]
	opty := ev.States["OptimalY"]
	plot.Styler(opty, func(s *plot.Style) {
		s.Plot.Title = "Optimal Route"
		s.Plot.SetPointsOn(plot.On)
		s.Point.SetOn(plot.On)
		s.Range.SetMin(0).SetMax(1)
		s.Plot.XAxis.Range.SetMin(0).SetMax(1)
		s.Line.NegativeX = true
	})
	plots.NewLine(ev.Plot, plot.Data{plot.X: optx, plot.Y: opty})
}

func (ev *ConSatEnv) UpdatePlot() {
	if ev.Plot == nil {
		return
	}
	optx := ev.States["OptimalX"]
	opty := ev.States["OptimalY"]
	errors.Log(ev.Plot.Plotters[0].SetData(plot.Data{plot.X: optx, plot.Y: opty}))
}

func (ev *ConSatEnv) RenderGrid() {
	optx := ev.States["OptimalX"]
	opty := ev.States["OptimalY"]
	in := ev.States["Input"]
	pos := ev.States["Pos"]
	out := ev.States["Output"]
	n := ev.NCities
	n = 1
	// ng := ev.NGrids
	nu := ev.NUnitsPer
	gs := ev.GridSpacing

	in.SetZeros()
	out.SetZeros()

	for p := range n {
		x := optx.Value(p)
		y := opty.Value(p)
		xi := int(math32.Round(x / gs))
		yi := int(math32.Round(y / gs))
		if !ev.Sequential {
			out.Set(1, 0, p, yi, xi)
		}
		for uy := range nu {
			for ux := range nu {
				in.Set(1, yi, xi, uy, ux)
				if ev.Sequential && p == ev.Trial.Cur {
					out.Set(1, yi, xi, uy, ux)
				}
			}
		}
	}
	if ev.Sequential {
		for uy := range nu {
			for ux := range nu {
				pos.Set(1, 0, ev.Trial.Cur, uy, ux)
			}
		}
	}
}

// Step does one step.
func (ev *ConSatEnv) Step() bool {
	mkNew := true
	if ev.Sequential {
		if !ev.Trial.Incr() {
			mkNew = false
		}
	}
	if mkNew {
		ev.MakeCities()
		ev.BruteForce()
	}
	ev.RenderGrid()
	return true
}
