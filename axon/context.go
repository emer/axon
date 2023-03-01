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
// #include "etime.hlsl"
// #include "axonrand.hlsl"
// #include "neuromod.hlsl"
// #include "drivepvlv.hlsl"
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
	ThetaCycles int32       `def:"200" desc:"length of the theta cycle in terms of 1 msec Cycles -- some network update steps depend on doing something at the end of the theta cycle (e.g., CTCtxtPrjn)."`
	CycleTot    int32       `desc:"total cycle count -- this increments continuously from whenever it was last reset -- typically this is number of milliseconds in simulation time -- is int32 and not uint32 b/c used with Synapse CaUpT which needs to have a -1 case for expired update time"`
	Time        float32     `desc:"accumulated amount of time the network has been running, in simulation-time (not real world time), in seconds"`
	Testing     slbool.Bool `desc:"if true, the model is being run in a testing mode, so no weight changes or other associated computations are needed.  this flag should only affect learning-related behavior"`
	TimePerCyc  float32     `def:"0.001" desc:"amount of time to increment per cycle"`
	NLayers     int32       `view:"-" desc:"number of layers in the network -- needed for GPU mode"`
	NSpiked     int32       `inactive:"+" desc:"number of neurons that spiked"`

	RandCtr   slrand.Counter `desc:"random counter -- incremented by maximum number of possible random numbers generated per cycle, regardless of how many are actually used -- this is shared across all layers so must encompass all possible param settings."`
	NeuroMod  NeuroModVals   `view:"inline" desc:"neuromodulatory state values -- these are computed separately on the CPU in CyclePost -- values are not cleared during running and remain until updated by a responsible layer type."`
	DrivePVLV DrivePVLV      `desc:"internal drives and PVLV dopamine mechanisms -- maintained centrally -- requires VSPatch and PPTg inputs, drives DA"`
}

// Defaults sets default values
func (ctx *Context) Defaults() {
	ctx.TimePerCyc = 0.001
	ctx.ThetaCycles = 200
	ctx.Mode = etime.Train
}

// NewState resets counters at start of new state (trial) of processing.
// Pass the evaluation model associated with this new state --
// if !Train then testing will be set to true.
func (ctx *Context) NewState(mode etime.Modes) {
	ctx.Phase = 0
	ctx.PlusPhase.SetBool(false)
	ctx.PhaseCycle = 0
	ctx.Cycle = 0
	ctx.NSpiked = 0
	ctx.Mode = mode
	ctx.Testing.SetBool(mode != etime.Train)
	ctx.NeuroMod.NewState()
}

// NewPhase resets PhaseCycle = 0 and sets the plus phase as specified
func (ctx *Context) NewPhase(plusPhase bool) {
	ctx.PhaseCycle = 0
	ctx.PlusPhase.SetBool(plusPhase)
}

// CycleInc increments at the cycle level
func (ctx *Context) CycleInc() {
	ctx.PhaseCycle++
	ctx.Cycle++
	ctx.CycleTot++
	ctx.Time += ctx.TimePerCyc
	ctx.RandCtr.Add(uint32(RandFunIdxN))
	ctx.NSpiked = 0
}

//gosl: end time

// Reset resets the counters all back to zero
func (ctx *Context) Reset() {
	ctx.Phase = 0
	ctx.PlusPhase.SetBool(false)
	ctx.PhaseCycle = 0
	ctx.Cycle = 0
	ctx.CycleTot = 0
	ctx.Time = 0
	ctx.Testing.SetBool(false)
	ctx.NSpiked = 0
	if ctx.TimePerCyc == 0 {
		ctx.Defaults()
	}
	ctx.RandCtr.Reset()
	ctx.NeuroMod.Init()
}

// NewContext returns a new Time struct with default parameters
func NewContext() *Context {
	ctx := &Context{}
	ctx.Defaults()
	return ctx
}
