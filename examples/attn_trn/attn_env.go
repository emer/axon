// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/emergent/efuns"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/evec"
	"github.com/emer/etable/etensor"
	"github.com/goki/mat32"
)

// Stim describes a single stimulus
type Stim struct {
	Pos      mat32.Vec2 `desc:"position in normalized coordintes"`
	Feat     int        `desc:"feature number: 0-3 for V1 input, -1 for LIP attn"`
	Width    float32    `desc:"normalized width"`
	Contrast float32    `desc:"normalized contrast level"`
}

// PosXY returns XY position projected into size of grid
func (st *Stim) PosXY(size evec.Vec2i) mat32.Vec2 {
	return mat32.Vec2{st.Pos.X * float32(size.X-1), st.Pos.Y * float32(size.Y-1)}
}

// StimSet is a set of stimuli to be presented together
type StimSet struct {
	Name  string `desc:"description of set"`
	Stims []Stim `desc:"stims to present"`
}

// Stims is a list of a set of stimuli to present
type Stims []StimSet

// AttnEnv is an example environment, that sets a single input point in a 2D
// input state and two output states as the X and Y coordinates of point.
// It can be used as a starting point for writing your own Env, without
// having much existing code to rewrite.
type AttnEnv struct {
	Nm           string     `desc:"name of this environment"`
	Dsc          string     `desc:"description of this environment"`
	ContrastMult float32    `desc:"multiplier on contrast function"`
	ContrastGain float32    `desc:"gain on contrast function inside exponential"`
	ContrastOff  float32    `desc:"offset on contrast function"`
	LIPGauss     bool       `desc:"use gaussian for LIP -- otherwise fixed circle"`
	Stims        Stims      `desc:"a list of stimuli to present"`
	CurStim      *StimSet   `inactive:"+" desc:"current stimuli presented"`
	Act          float32    `desc:"activation level (midpoint) -- feature is incremented, rest decremented relative to this"`
	V1Pools      evec.Vec2i `desc:"size of V1 Pools"`
	V1Feats      evec.Vec2i `desc:"size of V1 features per pool"`

	V1    etensor.Float32 `desc:"V1 rendered input state, 4D Size x Size"`
	LIP   etensor.Float32 `desc:"LIP top-down attention"`
	Run   env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial env.Ctr         `view:"inline" desc:"trial increments over input states -- could add Event as a lower level"`
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
	cact := ev.ContrastMult * act * (mat32.FastExp(ev.ContrastGain*contrast+ev.ContrastOff) - 1)
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
			d := mat32.Hypot(float32(xp)-x, float32(yp)-y)
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
	wd := mat32.Round(sig)
	for yp := 0; yp < ev.V1Pools.Y; yp++ {
		for xp := 0; xp < ev.V1Pools.X; xp++ {
			d := mat32.Hypot(float32(xp)-ps.X, float32(yp)-ps.Y)
			var gauss float32
			if mat32.Round(d) <= wd {
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
