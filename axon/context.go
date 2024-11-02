// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/goal/gosl/slbool"
	"github.com/emer/emergent/v2/etime"
)

//gosl:start
//gosl:import "github.com/emer/emergent/v2/etime"

// Context contains all of the global context state info
// that is shared across every step of the computation.
// It is passed around to all relevant computational functions,
// and is updated on the CPU and synced to the GPU after every cycle.
// It contains timing, Testing vs. Training mode, random number context, etc.
type Context struct {

	// number of data parallel items to process currently.
	NData uint32 `min:"1"`

	// current evaluation mode, e.g., Train, Test, etc
	Mode etime.Modes

	// if true, the model is being run in a testing mode, so no weight changes or other associated computations are needed.  this flag should only affect learning-related behavior.  Is automatically updated based on Mode != Train
	Testing slbool.Bool `edit:"-"`

	// phase counter: typicaly 0-1 for minus-plus but can be more phases for other algorithms
	Phase int32

	// true if this is the plus phase, when the outcome / bursting is occurring, driving positive learning -- else minus phase
	PlusPhase slbool.Bool

	// cycle within current phase -- minus or plus
	PhaseCycle int32

	// cycle counter: number of iterations of activation updating (settling) on the current state -- this counts time sequentially until reset with NewState
	Cycle int32

	// length of the theta cycle in terms of 1 msec Cycles -- some network update steps depend on doing something at the end of the theta cycle (e.g., CTCtxtPath).
	ThetaCycles int32 `default:"200"`

	// total cycle count -- increments continuously from whenever it was last reset -- typically this is number of milliseconds in simulation time -- is int32 and not uint32 b/c used with Synapse CaUpT which needs to have a -1 case for expired update time
	CyclesTotal int32

	// accumulated amount of time the network has been running, in simulation-time (not real world time), in seconds
	Time float32

	// total trial count -- increments continuously in NewState call *only in Train mode* from whenever it was last reset -- can be used for synchronizing weight updates across nodes
	TrialsTotal int32

	// amount of time to increment per cycle
	TimePerCycle float32 `default:"0.001"`

	// how frequently to perform slow adaptive processes such as synaptic scaling, inhibition adaptation, associated in the brain with sleep, in the SlowAdapt method.  This should be long enough for meaningful changes to accumulate -- 100 is default but could easily be longer in larger models.  Because SlowCtr is incremented by NData, high NData cases (e.g. 16) likely need to increase this value -- e.g., 400 seems to produce overall consistent results in various models.
	SlowInterval int32 `default:"100"`

	// counter for how long it has been since last SlowAdapt step.  Note that this is incremented by NData to maintain consistency across different values of this parameter.
	SlowCtr int32 `edit:"-"`

	// synaptic calcium counter, which drives the CaUpT synaptic value to optimize updating of this computationally expensive factor. It is incremented by 1 for each cycle, and reset at the SlowInterval, at which point the synaptic calcium values are all reset.
	SynCaCtr float32 `edit:"-"`

	// RandCtr is the random counter, incremented by maximum number of
	// possible random numbers generated per cycle, regardless of how
	// many are actually used. This is shared across all layers so must
	// encompass all possible param settings.
	RandCtr uint64

	pad, pad1 float32
}

// Defaults sets default values
func (ctx *Context) Defaults() {
	ctx.TimePerCycle = 0.001
	ctx.ThetaCycles = 200
	ctx.SlowInterval = 100
	ctx.Mode = etime.Train
}

// ItemIndex returns the main item index from an overall index over NItems * NData.
// (items = layers, neurons, synapses)
func (ctx *Context) ItemIndex(idx uint32) uint32 {
	return idx / ctx.NData
}

// DataIndex returns the data index from an overall index over NItems * NData.
func (ctx *Context) DataIndex(idx uint32) uint32 {
	return idx % ctx.NData
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
	ctx.CyclesTotal++
	ctx.Time += ctx.TimePerCycle
	ctx.SynCaCtr += 1
	// ctx.RandCtr.Add(uint32(RandFunIndexN))  TODO: gosl
}

// SlowInc increments the Slow counter and returns true if time
// to perform SlowAdapt functions (associated with sleep).
func (ctx *Context) SlowInc() bool {
	ctx.SlowCtr += int32(ctx.NData)
	if ctx.SlowCtr < ctx.SlowInterval {
		return false
	}
	ctx.SlowCtr = 0
	ctx.SynCaCtr = 0
	return true
}

//gosl:end

// NewState resets counters at start of new state (trial) of processing.
// Pass the evaluation model associated with this new state --
// if !Train then testing will be set to true.
func (ctx *Context) NewState(mode etime.Modes) {
	ctx.Phase = 0
	ctx.PlusPhase.SetBool(false)
	ctx.PhaseCycle = 0
	ctx.Cycle = 0
	ctx.Mode = mode
	ctx.Testing.SetBool(mode != etime.Train)
	if mode == etime.Train {
		ctx.TrialsTotal++
	}
}

// Reset resets the counters all back to zero
func (ctx *Context) Reset() {
	ctx.Phase = 0
	ctx.PlusPhase.SetBool(false)
	ctx.PhaseCycle = 0
	ctx.Cycle = 0
	ctx.CyclesTotal = 0
	ctx.Time = 0
	ctx.TrialsTotal = 0
	ctx.SlowCtr = 0
	ctx.SynCaCtr = 0
	ctx.Testing.SetBool(false)
	if ctx.TimePerCycle == 0 {
		ctx.Defaults()
	}
	// ctx.RandCtr.Reset()
	GlobalsReset(ctx)
}

// NewContext returns a new Time struct with default parameters
func NewContext() *Context {
	ctx := &Context{}
	ctx.Defaults()
	return ctx
}
