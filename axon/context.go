// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/goal/gosl/slbool"
	"github.com/emer/emergent/v2/etime"
)

// CopyNetStridesFrom copies strides and NetIndexes for accessing
// variables on a Network -- these must be set properly for
// the Network in question (from its Ctx field) before calling
// any compute methods with the context.  See SetCtxStrides on Network.
func (ctx *Context) CopyNetStridesFrom(srcCtx *Context) {
	ctx.NetIndexes = srcCtx.NetIndexes
}

//gosl:start

// NetIndexes are indexes and sizes for processing network
type NetIndexes struct {

	// number of data parallel items to process currently
	NData uint32 `min:"1"`

	// network index in global Networks list of networks
	NetIndex uint32 `edit:"-"`

	// maximum amount of data parallel
	MaxData uint32 `edit:"-"`

	// number of layers in the network
	NLayers uint32 `edit:"-"`

	// total number of neurons
	NNeurons uint32 `edit:"-"`

	// total number of pools excluding * MaxData factor
	NPools uint32 `edit:"-"`

	// total number of synapses
	NSyns uint32 `edit:"-"`

	// maximum size in float32 (4 bytes) of a GPU buffer -- needed for GPU access
	GPUMaxBuffFloats uint32 `edit:"-"`

	// total number of SynCa banks of GPUMaxBufferBytes arrays in GPU
	GPUSynCaBanks uint32 `edit:"-"`

	// total number of .Rubicon Drives / positive USs
	RubiconNPosUSs uint32 `edit:"-"`

	// total number of .Rubicon Costs
	RubiconNCosts uint32 `edit:"-"`

	// total number of .Rubicon Negative USs
	RubiconNNegUSs uint32 `edit:"-"`
}

// ValuesIndex returns the global network index for LayerValues
// with given layer index and data parallel index.
func (ctx *NetIndexes) ValuesIndex(li, di uint32) uint32 {
	return li*ctx.MaxData + di
}

// ItemIndex returns the main item index from an overall index over NItems * MaxData
// (items = layers, neurons, synapeses)
func (ctx *NetIndexes) ItemIndex(idx uint32) uint32 {
	return idx / ctx.MaxData
}

// DataIndex returns the data index from an overall index over N * MaxData
func (ctx *NetIndexes) DataIndex(idx uint32) uint32 {
	return idx % ctx.MaxData
}

// DataIndexIsValid returns true if the data index is valid (< NData)
func (ctx *NetIndexes) DataIndexIsValid(li uint32) bool {
	return (li < ctx.NData)
}

// LayerIndexIsValid returns true if the layer index is valid (< NLayers)
func (ctx *NetIndexes) LayerIndexIsValid(li uint32) bool {
	return (li < ctx.NLayers)
}

// NeurIndexIsValid returns true if the neuron index is valid (< NNeurons)
func (ctx *NetIndexes) NeurIndexIsValid(ni uint32) bool {
	return (ni < ctx.NNeurons)
}

// PoolIndexIsValid returns true if the pool index is valid (< NPools)
func (ctx *NetIndexes) PoolIndexIsValid(pi uint32) bool {
	return (pi < ctx.NPools)
}

// PoolDataIndexIsValid returns true if the pool*data index is valid (< NPools*MaxData)
func (ctx *NetIndexes) PoolDataIndexIsValid(pi uint32) bool {
	return (pi < ctx.NPools*ctx.MaxData)
}

// SynIndexIsValid returns true if the synapse index is valid (< NSyns)
func (ctx *NetIndexes) SynIndexIsValid(si uint32) bool {
	return (si < ctx.NSyns)
}

// Context contains all of the global context state info
// that is shared across every step of the computation.
// It is passed around to all relevant computational functions,
// and is updated on the CPU and synced to the GPU after every cycle.
// It is the *only* mechanism for communication from CPU to GPU.
// It contains timing, Testing vs. Training mode, random number context,
// global neuromodulation, etc.
type Context struct {

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

	// indexes and sizes of current network
	NetIndexes NetIndexes `display:"inline"`
}

// Defaults sets default values
func (ctx *Context) Defaults() {
	ctx.NetIndexes.NData = 1
	ctx.TimePerCycle = 0.001
	ctx.ThetaCycles = 200
	ctx.SlowInterval = 100
	ctx.Mode = etime.Train
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
	ctx.SlowCtr += int32(ctx.NetIndexes.NData)
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
