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

// CaBinCycles is the number of cycles per CaBin for integrating
// calcium-based activity values ([CaSyn]) that are used for computing
// a synaptic-level (pre * post) credit assignment factor for learning.
// This is a constant because other pre-computed factors depend on it.
const CaBinCycles = 10

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

	// MinusPhase is true if this is the minus phase, when a stimulus is present
	// and learning is occuring. Could also be in a non-learning phase when
	// no stimulus is present. This affects accumulation of CaBins values only.
	MinusPhase slbool.Bool

	// PlusPhase is true if this is the plus phase, when the outcome / bursting
	// is occurring, driving positive learning; else minus or non-learning phase.
	PlusPhase slbool.Bool

	// Cycle within current phase, minus or plus.
	PhaseCycle int32

	// Cycle within Trial: number of iterations of activation updating (settling)
	// on the current state. This is reset at NewState.
	Cycle int32

	// ThetaCycles is the length of the theta cycle (i.e., Trial),
	// in terms of 1 msec Cycles. Some network update steps depend on doing something
	// at the end of the theta cycle (e.g., CTCtxtPath).
	// Should be ISICycles + MinusCycles + PlusCycles
	ThetaCycles int32 `default:"200"`

	// ISICycles is the number of inter-stimulus-interval cycles,
	// which happen prior to the minus phase (i.e., after the last plus phase).
	ISICycles int32

	// MinusCycles is the number of cycles in the minus phase. Typically 150,
	// but may be set longer if ThetaCycles is above default of 200.
	MinusCycles int32 `default:"150"`

	// PlusCycles is the number of cycles in the plus phase. Typically 50,
	// but may be set longer if ThetaCycles is above default of 200.
	PlusCycles int32 `default:"50"`

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

	pad int32

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
	ctx.ISICycles = 0
	ctx.MinusCycles = 150
	ctx.PlusCycles = 50
	ctx.SlowInterval = 100
	ctx.AdaptGiInterval = 1000
	ctx.Update()
}

func (ctx *Context) Update() {
	ctx.ThetaCycles = ctx.ISICycles + ctx.MinusCycles + ctx.PlusCycles
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

// MinusPhaseStart resets PhaseCycle = 0 and sets the minus phase to true,
// and plus phase to false.
func (ctx *Context) MinusPhaseStart() {
	ctx.PhaseCycle = 0
	ctx.MinusPhase.SetBool(true)
	ctx.PlusPhase.SetBool(false)
}

// PlusPhaseStart resets PhaseCycle = 0 and sets the plus phase to true,
// and minus phase to false.
func (ctx *Context) PlusPhaseStart() {
	ctx.PhaseCycle = 0
	ctx.MinusPhase.SetBool(false)
	ctx.PlusPhase.SetBool(true)
}

// NCaBins returns 2 * ThetaCycles / CaBinCycles: stored in NetworkIxs.NCaBins.
func (ctx *Context) NCaBins() int32 {
	return 2 * (ctx.ThetaCycles / CaBinCycles)
}

// NCaWeights returns (MinusCycles + PlusCycles) / CaBinCycles:
// number of weights set for SynCa weighted computation of SynCaP, SynCaD.
// Weights are stored in [GlobalScalars]
func (ctx *Context) NCaWeights() int32 {
	return (ctx.MinusCycles + ctx.PlusCycles) / CaBinCycles
}

// CaBinForCycle returns the [CaBins] bin number for given CyclesTotal
// cycle index. Two ThetaCycles worth of data are stored at a CaBinCycles
// resolution, allowing learning to use any subset of data within that window.
func CaBinForCycle(cycle int32) int32 {
	return (cycle / CaBinCycles) % NetworkIxs[0].NCaBins
}

// CaBinIncrement writes given increment to the [CaBins] for given absolute cycle
// (CyclesTotal), initializing with value if it is the first one, and adding otherwise.
// Given value is divided by CaBinCycles to keep it normalized as an average across the
// CaBinCycles window.
func CaBinIncrement(incr float32, cycle int32, ni, di uint32) {
	bin := CaBinForCycle(cycle)
	incn := incr / float32(CaBinCycles)
	if (cycle % CaBinCycles) == 0 {
		Neurons.Set(incn, int(ni), int(di), int(CaBins+NeuronVars(bin)))
	} else {
		Neurons.SetAdd(incn, int(ni), int(di), int(CaBins+NeuronVars(bin)))
	}
}

//gosl:end

// ThetaCycleStart resets counters at start of new theta cycle of processing.
// Pass the evaluation mode associated with this theta cycle and testing bool.
// Resets the Minus and Plus phase states, and sets Cycle = 0.
func (ctx *Context) ThetaCycleStart(mode enums.Enum, testing bool) {
	ctx.MinusPhase.SetBool(false)
	ctx.PlusPhase.SetBool(false)
	ctx.Cycle = 0
	ctx.Mode = int32(mode.Int64())
	ctx.Testing.SetBool(testing)
	if !testing {
		ctx.TrialsTotal++
	}
}

// Reset resets the counters all back to zero
func (ctx *Context) Reset() {
	ctx.MinusPhase.SetBool(false)
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
