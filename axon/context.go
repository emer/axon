// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/etime"
	"github.com/goki/gosl/slbool"
	"github.com/goki/gosl/slrand"
)

//gosl: hlsl context
// #include "slrand.hlsl"
// #include "etime.hlsl"
// #include "neuromod.hlsl"
//gosl: end context

//gosl: start context

// Context contains all of the global context state info
// that is shared across every step of the computation.
// It is passed around to all relevant computational functions,
// and is updated on the CPU and synced to the GPU after every cycle.
// It is the *only* mechanism for communication from CPU to GPU.
// It contains timing, Testing vs. Training mode, random number context,
// global neuromodulation, etc.
type Context struct {
	Mode        etime.Modes `desc:"current evaluation mode, e.g., Train, Test, etc"`
	Phase       int32       `desc:"phase counter: typicaly 0-1 for minus-plus but can be more phases for other algorithms"`
	PlusPhase   slbool.Bool `desc:"true if this is the plus phase, when the outcome / bursting is occurring, driving positive learning -- else minus phase"`
	PhaseCycle  int32       `desc:"cycle within current phase -- minus or plus"`
	Cycle       int32       `desc:"cycle counter: number of iterations of activation updating (settling) on the current state -- this counts time sequentially until reset with NewState"`
	CycleTot    uint32      `desc:"total cycle count -- this increments continuously from whenever it was last reset -- typically this is number of milliseconds in simulation time"`
	Time        float32     `desc:"accumulated amount of time the network has been running, in simulation-time (not real world time), in seconds"`
	Testing     slbool.Bool `desc:"if true, the model is being run in a testing mode, so no weight changes or other associated computations are needed.  this flag should only affect learning-related behavior"`
	TimePerCyc  float32     `def:"0.001" desc:"amount of time to increment per cycle"`
	RandsPerCyc uint32      `def:"2" desc:"maximum number of random numbers used per cycle, if noise is active -- we always just increase the random counter by this amount to maintain consistency"`

	pad, pad1 int32

	RandCtr  slrand.Counter `desc:"random counter -- incremented by maximum number of possible random numbers generated per cycle, regardless of how many are actually used -- this is shared across all layers so must encompass all possible param settings."`
	NeuroMod NeuroModVals   `view:"inline" desc:"neuromodulatory state values -- these are computed separately on the CPU in CyclePost -- values are not cleared during running and remain until updated by a responsible layer type."`
}

// Defaults sets default values
func (tm *Context) Defaults() {
	tm.TimePerCyc = 0.001
	tm.RandsPerCyc = 2
}

// NewState resets counters at start of new state (trial) of processing.
// Pass the evaluation model associated with this new state --
// if !Train then testing will be set to true.
func (tm *Context) NewState(mode etime.Modes) {
	tm.Phase = 0
	tm.PlusPhase.SetBool(false)
	tm.PhaseCycle = 0
	tm.Cycle = 0
	tm.Mode = mode
	tm.Testing.SetBool(mode != etime.Train)
	tm.NeuroMod.NewState()
}

// NewPhase resets PhaseCycle = 0 and sets the plus phase as specified
func (tm *Context) NewPhase(plusPhase bool) {
	tm.PhaseCycle = 0
	tm.PlusPhase.SetBool(plusPhase)
}

// CycleInc increments at the cycle level
func (tm *Context) CycleInc() {
	tm.PhaseCycle++
	tm.Cycle++
	tm.CycleTot++
	tm.Time += tm.TimePerCyc
	tm.RandCtr.Add(tm.RandsPerCyc)
}

//gosl: end time

// Reset resets the counters all back to zero
func (tm *Context) Reset() {
	tm.Phase = 0
	tm.PlusPhase.SetBool(false)
	tm.PhaseCycle = 0
	tm.Cycle = 0
	tm.CycleTot = 0
	tm.Time = 0
	tm.Testing.SetBool(false)
	if tm.TimePerCyc == 0 {
		tm.Defaults()
	}
	tm.RandCtr.Reset()
	tm.NeuroMod.Reset()
}

// NewContext returns a new Time struct with default parameters
func NewContext() *Context {
	tm := &Context{}
	tm.Defaults()
	return tm
}
