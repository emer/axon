// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/enums"
	"cogentcore.org/lab/gosl/slbool"
	"cogentcore.org/lab/gosl/slrand"
	"cogentcore.org/lab/gosl/sltype"
)

//gosl:start

// Context contains all of the global context state info
// that is shared across every step of the computation.
// It is passed around to all relevant computational functions,
// and is updated on the CPU and synced to the GPU after every cycle.
// It contains timing, Testing vs. Training mode, random number context, etc.
// There is one canonical instance on the network as Ctx, always get it from
// the network.Context() method.
type Context struct { //types:add -setters

	// number of data parallel items to process currently.
	NData uint32 `min:"1"`

	// current running mode, using sim-defined enum, e.g., Train, Test, etc.
	Mode int32

	// Testing is true if the model is being run in a testing mode,
	// so no weight changes or other associated computations should be done.
	// This flag should only affect learning-related behavior.
	Testing slbool.Bool `edit:"-"`

	// Phase counter: typicaly 0-1 for minus-plus.
	Phase int32

	// PlusPhase is true if this is the plus phase, when the outcome / bursting
	// is occurring, driving positive learning; else minus phase.
	PlusPhase slbool.Bool

	// Cycle within current phase, minus or plus.
	PhaseCycle int32

	// Cycle within Trial: number of iterations of activation updating (settling)
	// on the current state. This is reset at NewState.
	Cycle int32

	// ThetaCycles is the length of the theta cycle (i.e., Trial), in terms of 1 msec Cycles.
	// Some network update steps depend on doing something at the end of the
	// theta cycle (e.g., CTCtxtPath).
	ThetaCycles int32 `default:"200"`

	// PlusCycles is the number of cycles in the plus phase. Typically 50,
	// but may be set longer if ThetaCycles is above default of 200.
	PlusCycles int32 `default:"50"`

	// CaBinCycles is the number of cycles for neuron [CaBins] values used in
	// computing synaptic calcium values. Total number of bins = ThetaCycles / CaBinCycles.
	// This is fixed at 10.
	CaBinCycles int32 `default:"10"`

	// CyclesTotal is the accumulated cycle count, which increments continuously
	// from whenever it was last reset. Typically this is the number of milliseconds
	// in simulation time.
	CyclesTotal int32

	// Time is the accumulated amount of time the network has been running,
	// in simulation-time (not real world time), in seconds.
	Time float32

	// TrialsTotal is the total trial count, which increments continuously in NewState
	// _only in Train mode_ from whenever it was last reset. Can be used for synchronizing
	// weight updates across nodes.
	TrialsTotal int32

	// TimePerCycle is the amount of Time to increment per cycle.
	TimePerCycle float32 `default:"0.001"`

	// SlowInterval is how frequently in Trials to perform slow adaptive processes
	// such as synaptic scaling, associated in the brain with sleep,
	// via the SlowAdapt method.  This should be long enough for meaningful changes
	// to accumulate. 100 is default but could easily be longer in larger models.
	// Because SlowCounter is incremented by NData, high NData cases (e.g. 16) likely need to
	// increase this value, e.g., 400 seems to produce overall consistent results in various models.
	SlowInterval int32 `default:"100"`

	// SlowCounter increments for each training trial, to trigger SlowAdapt at SlowInterval.
	// This is incremented by NData to maintain consistency across different values of this parameter.
	SlowCounter int32 `edit:"-"`

	// AdaptGiInterval is how frequently in Trials to perform inhibition adaptation,
	// which needs to be even slower than the SlowInterval.
	AdaptGiInterval int32 `default:"1000"`

	// AdaptGiCounter increments for each training trial, to trigger AdaptGi at AdaptGiInterval.
	// This is incremented by NData to maintain consistency across different values of this parameter.
	AdaptGiCounter int32 `edit:"-"`

	pad, pad1 int32

	// RandCounter is the random counter, incremented by maximum number of
	// possible random numbers generated per cycle, regardless of how
	// many are actually used. This is shared across all layers so must
	// encompass all possible param settings.
	RandCounter slrand.Counter
}

// Defaults sets default values
func (ctx *Context) Defaults() {
	ctx.NData = 1
	ctx.TimePerCycle = 0.001
	ctx.ThetaCycles = 200
	ctx.PlusCycles = 50
	ctx.CaBinCycles = 10
	ctx.SlowInterval = 100
	ctx.AdaptGiInterval = 1000
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

// CycleInc increments at the cycle level. This is the one time when
// Context is used on GPU in read-write mode, vs. read-only.
//
//gosl:pointer-receiver
func (ctx *Context) CycleInc() {
	ctx.PhaseCycle++
	ctx.Cycle++
	ctx.CyclesTotal++
	ctx.Time += ctx.TimePerCycle
	// ctx.RandCounter.Add(uint32(RandFunIndexN)):
	ctx.RandCounter.Counter = sltype.Uint64Add32(ctx.RandCounter.Counter, uint32(RandFunIndexN))
	// note: cannot call writing methods on sub-fields, so have to do it manually.
}

// SlowInc increments the Slow and AdaptGi counters and returns true if it is
// time to perform SlowAdapt or AdaptGi functions.
func (ctx *Context) SlowInc() (slow bool, adaptgi bool) {
	ctx.SlowCounter += int32(ctx.NData)
	ctx.AdaptGiCounter += int32(ctx.NData)
	if ctx.SlowCounter >= ctx.SlowInterval {
		slow = true
		ctx.SlowCounter = 0
	}
	if ctx.AdaptGiCounter >= ctx.AdaptGiInterval {
		adaptgi = true
		ctx.AdaptGiCounter = 0
	}
	return
}

// PlusPhaseStart resets PhaseCycle = 0 and sets the plus phase to true.
func (ctx *Context) PlusPhaseStart() {
	ctx.PhaseCycle = 0
	ctx.PlusPhase.SetBool(true)
}

// NCaBins returns ThetaCycles / CaBinCycles
func (ctx *Context) NCaBins() int32 {
	return ctx.ThetaCycles / ctx.CaBinCycles
}

//gosl:end

// NewState resets counters at start of new state (trial) of processing.
// Pass the evaluation mode associated with this new state and testing bool.
func (ctx *Context) NewState(mode enums.Enum, testing bool) {
	ctx.Phase = 0
	ctx.PlusPhase.SetBool(false)
	ctx.PhaseCycle = 0
	ctx.Cycle = 0
	ctx.Mode = int32(mode.Int64())
	ctx.Testing.SetBool(testing)
	if !testing {
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
	ctx.SlowCounter = 0
	ctx.Testing.SetBool(false)
	if ctx.TimePerCycle == 0 {
		ctx.Defaults()
	}
	ctx.RandCounter.Reset()
	GlobalsReset()
}

// NewContext returns a new Time struct with default parameters
func NewContext() *Context {
	ctx := &Context{}
	ctx.Defaults()
	return ctx
}
