// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consatenv

import (
	"fmt"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/jsonx"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
)

func pow(v, n int) int {
	r := 1
	for range n {
		r *= v
	}
	return r
}

// MakeStates makes the global shared states list -- call under global lock.
func (ev *ConSatEnv) MakeStates() {
	if states != nil {
		return
	}
	nv := ev.NVars
	nary := ev.NAry
	ns := pow(nary, nv)
	statesN = ns

	// note: if this gets very big, could just generate like this on the fly!
	states = tensor.NewInt(ns, nv)
	for i := range ns {
		pp := i
		for k := range nv {
			j := pp % nary
			pp /= nary
			states.Set(j, i, nary-1-k)
		}
	}
	// fmt.Println(states)
}

// Relations are the possible relationships among variables
type Relations int32 //enums:enum
const (
	Equal Relations = iota
	Greater
	Less
)

func (r Relations) ShortString() string {
	switch r {
	case Equal:
		return "="
	case Greater:
		return ">"
	case Less:
		return "<"
	}
	return "?"
}

func RandRelation(rnd randx.Rand) Relations {
	return Relations(rnd.Intn(int(RelationsN)))
}

// Relation specifies one relationship between two variables
type Relation struct {
	R Relations
	A int
	B int
}

func (r *Relation) String() string {
	return fmt.Sprintf("%d %s %d", r.A, r.R.ShortString(), r.B)
}

// Eval evaluates the relationship on list of variable values
func (r *Relation) Eval(vars []int) bool {
	switch r.R {
	case Equal:
		return vars[r.A] == vars[r.B]
	case Greater:
		return vars[r.A] > vars[r.B]
	case Less:
		return vars[r.A] < vars[r.B]
	}
	return false
}

func (r *Relation) Clone() *Relation {
	nr := &Relation{}
	*nr = *r
	return nr
}

func (r *Relation) RandomTweak(nvar int, rnd randx.Rand) {
	pi := rnd.Intn(3)
	switch pi {
	case 0:
		for {
			nr := RandRelation(rnd)
			if nr != r.R {
				r.R = nr
				break
			}
		}
	case 1:
		for {
			na := rnd.Intn(nvar)
			if na != r.A {
				r.A = na
				break
			}
		}
	case 2:
		for {
			nb := rnd.Intn(nvar)
			if nb != r.B {
				r.B = nb
				break
			}
		}
	}
}

// Constraint is one set of relations
type Constraint []*Relation

func (c Constraint) String() string {
	s := ""
	for _, r := range c {
		s += r.String() + "\n"
	}
	return s
}

// Evaluate given constraint relations on given variable values.
func (c Constraint) Eval(vars []int) bool {
	all := true
	for _, r := range c {
		if !r.Eval(vars) {
			all = false
			break
		}
	}
	return all
}

func (c Constraint) Clone() Constraint {
	cc := make(Constraint, len(c))
	for i := range c {
		cc[i] = c[i].Clone()
	}
	return cc
}

func (c Constraint) RandomTweak(nvar int, rnd randx.Rand) {
	ri := rnd.Intn(len(c))
	r := c[ri]
	r.RandomTweak(nvar, rnd)
}

// Constraints is a set Constraints
type Constraints []Constraint

func (cs Constraints) Clone() Constraints {
	cc := make(Constraints, len(cs))
	for i := range cc {
		cc[i] = cs[i].Clone()
	}
	return cc
}

func (cs Constraints) RandomTweak(nvar int, rnd randx.Rand) {
	ri := rnd.Intn(len(cs))
	c := cs[ri]
	c.RandomTweak(nvar, rnd)
}

// Evaluate given constraints on given variable values.
// val is the (last) one that matches, -1 if none
// nmatch is number of matching: if > 1 then there are conflicts.
// fills in the results from each constraint in all if present,
// which should be len(cs).
func (cs Constraints) Eval(vars []int, all []bool) (val, nmatch int) {
	if all == nil {
		all = make([]bool, len(cs))
	}
	val = -1
	for i, c := range cs {
		ev := c.Eval(vars)
		all[i] = ev
		if ev {
			val = i
			nmatch++
		}
	}
	return
}

// StateVars gets given state variables for given index, as []int
func (ev *ConSatEnv) StateVars(idx int, v []int) {
	for i := range v {
		v[i] = states.Value(idx, i)
	}
}

// RandomVars generates one list of variable values at random.
func (ev *ConSatEnv) RandomVars(v []int) {
	nary := ev.NAry
	for i := range v {
		v[i] = ev.Rand.Intn(nary)
	}
}

// MakeConstraints returns a set of Constraints, where each
// constraint specifies a set of relationships among variables.
// Constrainted to only one relationship per each pair of variables.
func (ev *ConSatEnv) MakeConstraints() Constraints {
	ev.MakeStates()

	nvar := ev.NVars
	nc := ev.NConstraints
	cons := make(Constraints, nc)
	for c := range nc {
		cons[c] = ev.MakeConstraint()
	}

	fmt.Println("\n#### Optimize Across")

	// evaluate overlap on random test items, tweak accordingly
	niter := 1000 // 100 seems sufficient
	maxpure := 0
	maxany := 0
	var maxPureC, maxAnyC Constraints
	_ = maxPureC
	_ = maxAnyC
	maxPureC = cons.Clone()
	neval := statesN
	vars := make([]int, nvar)
	all := make([]bool, nc)
	for i := range niter {
		if i > 0 {
			cons = maxPureC.Clone() // start from current best
			cons.RandomTweak(nvar, ev.Rand)
		}
		npure := 0
		nany := 0
		for e := range neval {
			ev.StateVars(e, vars)
			_, nmatch := cons.Eval(vars, all)
			if nmatch == 1 {
				npure++
			}
			if nmatch > 0 {
				nany++
			}
		}
		if npure > maxpure {
			maxpure = npure
			maxPureC = cons.Clone()
			fmt.Println(i, "npure:", npure, "pct:", float32(npure)/float32(neval), "nany:", nany, "pct:", float32(nany)/float32(neval))
		}
		if nany > maxany {
			maxany = nany
			maxAnyC = cons.Clone()
			fmt.Println(i, "nany:", nany, "pct:", float32(nany)/float32(neval))
		}
	}

	ev.TestConstraints(maxPureC)

	errors.Log(jsonx.Save(maxPureC, ev.Filename()))

	return maxPureC
}

func (ev *ConSatEnv) TestConstraints(cons Constraints) {
	neval := statesN
	nc := ev.NConstraints
	vars := make([]int, ev.NVars)
	stats := make([]int, nc+1)
	all := make([]bool, nc)
	for e := range neval {
		ev.StateVars(e, vars)
		val, nmatch := cons.Eval(vars, all)
		if nmatch == 0 {
			stats[0]++
		} else {
			stats[1+val]++
		}
	}
	sts := ""
	for i := range stats {
		sts += fmt.Sprintf("%g\t", float32(stats[i])/float32(neval))
	}
	fmt.Println(sts)
}

func (ev *ConSatEnv) Filename() string {
	return fmt.Sprintf("cons_nc%d_nv%d_nr%d_nary%d.json", ev.NConstraints, ev.NVars, ev.RelationsPer, ev.NAry)
}

func (ev *ConSatEnv) OpenConstraints() Constraints {
	var cons Constraints
	errors.Log(jsonx.Open(&cons, ev.Filename()))
	ev.TestConstraints(cons)
	return cons
}

// load shared global constraints
func (ev *ConSatEnv) InitOpen() {
	globalLock.Lock()
	defer globalLock.Unlock()
	if constraints != nil {
		return
	}
	ev.MakeStates()
	constraints = ev.OpenConstraints()
	items = ev.Rand.Perm(statesN)
}

// MakeConstraint makes one set of relations as a constraint,
// Enforcing that it is not mutually contradictory, via
// random sampling.
func (ev *ConSatEnv) MakeConstraint() Constraint {
	nvar := ev.NVars
	nr := ev.RelationsPer

	vlist := ev.Rand.Perm(nvar)

	c := make(Constraint, nr)
	for r := range nr {
		rel := Relations(ev.Rand.Intn(int(RelationsN)))
		var a, b int
		for {
			randx.PermuteInts(vlist, ev.Rand)
			a = vlist[0]
			b = vlist[1]
			good := true
			for rc := 0; rc < r; rc++ {
				or := c[rc]
				if or.A == a && or.B == b || or.A == b && or.B == a {
					// either contradictory or redundant
					good = false
					break
				}
			}
			if good {
				break
			}
		}
		c[r] = &Relation{R: rel, A: a, B: b}
	}

	fmt.Println(c)

	// now optimize
	niter := 100 // 100 seems sufficient
	maxtrue := 0
	maxc := c.Clone()
	neval := statesN
	vars := make([]int, nvar)
	for i := range niter {
		if i > 0 {
			c = maxc.Clone() // start from current best
			c.RandomTweak(nvar, ev.Rand)
		}
		ntrue := 0
		for e := range neval {
			ev.StateVars(e, vars)
			tv := c.Eval(vars)
			if tv {
				ntrue++
			}
		}
		if ntrue > maxtrue {
			maxtrue = ntrue
			maxc = c.Clone()
			fmt.Println(i, "ntrue:", ntrue, "pct:", float32(ntrue)/float32(neval))
		}
	}

	return maxc
}
