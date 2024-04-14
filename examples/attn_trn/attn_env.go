// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

package main

import (
	"fmt"

	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/efuns"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/evec"
	"github.com/emer/etable/v2/etensor"
)

// Stim describes a single stimulus
type Stim struct {

	// position in normalized coordintes
	Pos math32.Vec2

	// feature number: 0-3 for V1 input, -1 for LIP attn
	Feat int

	// normalized width
	Width float32

	// normalized contrast level
	Contrast float32
}

// PosXY returns XY position projected into size of grid
func (st *Stim) PosXY(size evec.Vec2i) math32.Vec2 {
	return math32.V2(st.Pos.X*float32(size.X-1), st.Pos.Y*float32(size.Y-1))
}

// StimSet is a set of stimuli to be presented together
type StimSet struct {

	// description of set
	Name string

	// stims to present
	Stims []Stim
}

// Stims is a list of a set of stimuli to present
type Stims []StimSet

// AttnEnv is an example environment, that sets a single input point in a 2D
// input state and two output states as the X and Y coordinates of point.
// It can be used as a starting point for writing your own Env, without
// having much existing code to rewrite.
type AttnEnv struct {

	// name of this environment
	Nm string

	// description of this environment
	Dsc string

	// multiplier on contrast function
	ContrastMult float32

	// gain on contrast function inside exponential
	ContrastGain float32

	// offset on contrast function
	ContrastOff float32

	// use gaussian for LIP -- otherwise fixed circle
	LIPGauss bool

	// a list of stimuli to present
	Stims Stims

	// current stimuli presented
	CurStim *StimSet `edit:"-"`

	// activation level (midpoint) -- feature is incremented, rest decremented relative to this
	Act float32

	// size of V1 Pools
	V1Pools evec.Vec2i

	// size of V1 features per pool
	V1Feats evec.Vec2i

	// V1 rendered input state, 4D Size x Size
	V1 etensor.Float32

	// LIP top-down attention
	LIP etensor.Float32

	// current run of model as provided during Init
	Run env.Ctr `view:"inline"`

	// number of times through Seq.Max number of sequences
	Epoch env.Ctr `view:"inline"`

	// trial increments over input states -- could add Event as a lower level
	Trial env.Ctr `view:"inline"`
}

func (ev *AttnEnv) Name() string { return ev.Nm }
func (ev *AttnEnv) Desc() string { return ev.Dsc }

func (ev *AttnEnv) Defaults() {
	ev.V1Pools.Set(16, 16)
	ev.V1Feats.Set(2, 4)
	ev.Act = 0.5
	ev.ContrastMult = 1.5
	ev.ContrastGain = 0.3
	ev.ContrastOff = 0.05 // 0.01
}

// Config configures according to current settings
func (ev *AttnEnv) Config() {
	ev.Trial.Max = len(ev.Stims)
	ev.V1.SetShape([]int{ev.V1Pools.Y, ev.V1Pools.X, ev.V1Feats.Y, ev.V1Feats.X}, nil, []string{"PY", "PX", "FY", "FX"})
	ev.LIP.SetShape([]int{ev.V1Pools.Y, ev.V1Pools.X, ev.V1Feats.Y, ev.V1Feats.X}, nil, []string{"PY", "PX", "FY", "FX"})
}

func (ev *AttnEnv) Validate() error {
	if ev.V1Pools.IsNil() {
		return fmt.Errorf("AttnEnv: %v has size == 0 -- need to Config", ev.Nm)
	}
	return nil
}

func (ev *AttnEnv) State(element string) etensor.Tensor {
	switch element {
	case "V1":
		return &ev.V1
	case "LIP":
		return &ev.LIP
	}
	return nil
}

// String returns the current state as a string
func (ev *AttnEnv) String() string {
	if ev.CurStim != nil {
		return ev.CurStim.Name
	}
	return "none"
}

// Init is called to restart environment
func (ev *AttnEnv) Init(run int) {
	ev.Trial.Max = len(ev.Stims)
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

func (ev *AttnEnv) ContrastAct(act, contrast float32) float32 {
	cact := ev.ContrastMult * act * (math32.FastExp(ev.ContrastGain*contrast+ev.ContrastOff) - 1)
	// fmt.Printf("ctrst: %g  cact: %g\n", contrast, cact)
	return cact
}

// RenderV1
func (ev *AttnEnv) RenderV1(stm *Stim, tsr *etensor.Float32) {
	x := stm.Pos.X * float32(ev.V1Pools.X-1)
	y := stm.Pos.Y * float32(ev.V1Pools.Y-1)
	sig := stm.Width * float32(ev.V1Pools.X)
	cact := ev.ContrastAct(ev.Act, stm.Contrast)
	for yp := 0; yp < ev.V1Pools.Y; yp++ {
		for xp := 0; xp < ev.V1Pools.X; xp++ {
			d := math32.Hypot(float32(xp)-x, float32(yp)-y)
			gauss := efuns.Gauss1DNoNorm(d, sig)
			fi := 0
			for yf := 0; yf < ev.V1Feats.Y; yf++ {
				for xf := 0; xf < ev.V1Feats.X; xf++ {
					var v float32
					if fi == stm.Feat {
						v = gauss * (ev.Act + cact)
					} else {
						v = gauss * (ev.Act - cact)
					}
					idx := []int{yp, xp, yf, xf}
					cv := tsr.Value(idx)
					cv += v
					tsr.Set(idx, cv)
					fi++
				}
			}
		}
	}
}

// RenderLIP
func (ev *AttnEnv) RenderLIP(stm *Stim, tsr *etensor.Float32) {
	ps := stm.PosXY(ev.V1Pools)
	sig := stm.Width * float32(ev.V1Pools.X)
	wd := math32.Round(sig)
	for yp := 0; yp < ev.V1Pools.Y; yp++ {
		for xp := 0; xp < ev.V1Pools.X; xp++ {
			d := math32.Hypot(float32(xp)-ps.X, float32(yp)-ps.Y)
			var gauss float32
			if math32.Round(d) <= wd {
				if ev.LIPGauss {
					gauss = efuns.Gauss1DNoNorm(d, sig)
				} else {
					gauss = 1
				}
			} else {
				if ev.LIPGauss {
					gauss = efuns.Gauss1DNoNorm(d, sig)
				}
			}
			idx := []int{yp, xp, 0, 0}
			cv := tsr.Value(idx)
			cv += gauss
			tsr.Set(idx, cv)
		}
	}
}

// RenderStim
func (ev *AttnEnv) RenderStim(stm *Stim) {
	if stm.Feat < 0 {
		ev.RenderLIP(stm, &ev.LIP)
	} else {
		ev.RenderV1(stm, &ev.V1)
	}
}

// RenderTrial
func (ev *AttnEnv) RenderTrial(trl int) {
	ev.V1.SetZeros()
	ev.LIP.SetZeros()

	if trl < 0 {
		trl = 0
	}
	nstm := len(ev.Stims)
	if nstm == 0 {
		return
	}
	idx := trl % nstm
	ev.CurStim = &ev.Stims[idx]
	for si := range ev.CurStim.Stims {
		stm := &ev.CurStim.Stims[si]
		ev.RenderStim(stm)
	}
}

// Step is called to advance the environment state
func (ev *AttnEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	inc := ev.Trial.Incr()
	ev.RenderTrial(ev.Trial.Cur)
	if inc { // true if wraps around Max back to 0
		ev.Epoch.Incr()
	}
	return true
}

func (ev *AttnEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *AttnEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*AttnEnv)(nil)
