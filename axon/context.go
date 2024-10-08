// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"math"

	"cogentcore.org/core/base/num"
	"cogentcore.org/core/goal/gosl/slbool"
	"cogentcore.org/core/goal/gosl/slrand"
	"github.com/emer/emergent/v2/etime"
)

var (
	// TheNetwork is the one current network in use, needed for GPU shader kernel
	// compatible variable access in CPU mode, for !multinet build tags case.
	// Typically there is just one and it is faster to access directly.
	// This is set in Network.InitName.
	TheNetwork *Network

	// Networks is a global list of networks, needed for GPU shader kernel
	// compatible variable access in CPU mode, for multinet build tags case.
	// This is updated in Network.InitName, which sets NetIndex.
	Networks []*Network
)

func NrnHasFlag(ctx *Context, ni, di uint32, flag NeuronFlags) bool {
	return (NeuronFlags(math.Float32bits(Neurons[NrnFlags, ni, di])) & flag) > 0 // weird: != 0 does NOT work on GPU
}

func NrnSetFlag(ctx *Context, ni, di uint32, flag NeuronFlags) {
	Neurons[NrnFlags, ni, di] =  math.Float32frombits(math.Float32bits(Neurons[NrnFlags, ni, di])|uint32(flag)))
}

func NrnClearFlag(ctx *Context, ni, di uint32, flag NeuronFlags) {
	Neurons[NrnFlags, ni, di] =  math.Float32frombits(math.Float32bits(Neurons[NrnFlags, ni, di])&^uint32(flag)))
}

// NrnIsOff returns true if the neuron has been turned off (lesioned)
// Only checks the first data item -- all should be consistent.
func NrnIsOff(ctx *Context, ni uint32) bool {
	return NrnHasFlag(ctx, ni, 0, NeuronOff)
}

// NeuronAvgVars

// NrnAvgV is the CPU version of the neuron variable accessor
func NrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars) float32 {
	return GlobalNetwork(ctx).NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)]
}

// SetNrnAvgV is the CPU version of the neuron variable settor
func SetNrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars, val float32) {
	GlobalNetwork(ctx).NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)] = val
}

// AddNrnAvgV is the CPU version of the neuron variable addor
func AddNrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars, val float32) {
	GlobalNetwork(ctx).NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)] += val
}

// MulNrnAvgV is the CPU version of the neuron variable multiplier
func MulNrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars, val float32) {
	GlobalNetwork(ctx).NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)] *= val
}

// NeuronIndexes

// NrnI is the CPU version of the neuron idx accessor
func NrnI(ctx *Context, ni uint32, idx NeuronIndexes) uint32 {
	return GlobalNetwork(ctx).NeuronIxs[ctx.NeuronIndexes.Index(ni, idx)]
}

// SetNrnI is the CPU version of the neuron idx settor
func SetNrnI(ctx *Context, ni uint32, idx NeuronIndexes, val uint32) {
	GlobalNetwork(ctx).NeuronIxs[ctx.NeuronIndexes.Index(ni, idx)] = val
}

// SynapseVars

// SynV is the CPU version of the synapse variable accessor
func SynV(ctx *Context, syni uint32, svar SynapseVars) float32 {
	return GlobalNetwork(ctx).Synapses[ctx.SynapseVars.Index(syni, svar)]
}

// SetSynV is the CPU version of the synapse variable settor
func SetSynV(ctx *Context, syni uint32, svar SynapseVars, val float32) {
	GlobalNetwork(ctx).Synapses[ctx.SynapseVars.Index(syni, svar)] = val
}

// AddSynV is the CPU version of the synapse variable addor
func AddSynV(ctx *Context, syni uint32, svar SynapseVars, val float32) {
	GlobalNetwork(ctx).Synapses[ctx.SynapseVars.Index(syni, svar)] += val
}

// MulSynV is the CPU version of the synapse variable multiplier
func MulSynV(ctx *Context, syni uint32, svar SynapseVars, val float32) {
	GlobalNetwork(ctx).Synapses[ctx.SynapseVars.Index(syni, svar)] *= val
}

// SynapseCaVars

// SynCaV is the CPU version of the synapse variable accessor
func SynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars) float32 {
	return GlobalNetwork(ctx).SynapseCas[ctx.SynapseCaVars.Index(syni, di, svar)]
}

// SetSynCaV is the CPU version of the synapse variable settor
func SetSynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars, val float32) {
	GlobalNetwork(ctx).SynapseCas[ctx.SynapseCaVars.Index(syni, di, svar)] = val
}

// AddSynCaV is the CPU version of the synapse variable addor
func AddSynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars, val float32) {
	GlobalNetwork(ctx).SynapseCas[ctx.SynapseCaVars.Index(syni, di, svar)] += val
}

// MulSynCaV is the CPU version of the synapse variable multiplier
func MulSynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars, val float32) {
	GlobalNetwork(ctx).SynapseCas[ctx.SynapseCaVars.Index(syni, di, svar)] *= val
}

// SynapseIndexes

// SynI is the CPU version of the synapse idx accessor
func SynI(ctx *Context, syni uint32, idx SynapseIndexes) uint32 {
	return GlobalNetwork(ctx).SynapseIxs[ctx.SynapseIndexes.Index(syni, idx)]
}

// SetSynI is the CPU version of the synapse idx settor
func SetSynI(ctx *Context, syni uint32, idx SynapseIndexes, val uint32) {
	GlobalNetwork(ctx).SynapseIxs[ctx.SynapseIndexes.Index(syni, idx)] = val
}

/////////////////////////////////
//  Global Vars

// GlbV is the CPU version of the global variable accessor
func GlbV(ctx *Context, di uint32, gvar GlobalVars) float32 {
	return GlobalNetwork(ctx).Globals[ctx.GlobalIndex(di, gvar)]
}

// SetGlbV is the CPU version of the global variable settor
func SetGlbV(ctx *Context, di uint32, gvar GlobalVars, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalIndex(di, gvar)] = val
}

// AddGlbV is the CPU version of the global variable addor
func AddGlbV(ctx *Context, di uint32, gvar GlobalVars, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalIndex(di, gvar)] += val
}

// GlbCost is the CPU version of the global Cost variable accessor
func GlbCostV(ctx *Context, di uint32, gvar GlobalVars, negIndex uint32) float32 {
	return GlobalNetwork(ctx).Globals[ctx.GlobalCostIndex(di, gvar, negIndex)]
}

// SetGlbCost is the CPU version of the global Cost variable settor
func SetGlbCostV(ctx *Context, di uint32, gvar GlobalVars, negIndex uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalCostIndex(di, gvar, negIndex)] = val
}

// AddGlbCost is the CPU version of the global Cost variable addor
func AddGlbCostV(ctx *Context, di uint32, gvar GlobalVars, negIndex uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalCostIndex(di, gvar, negIndex)] += val
}

// GlbUSneg is the CPU version of the global USneg variable accessor
func GlbUSnegV(ctx *Context, di uint32, gvar GlobalVars, negIndex uint32) float32 {
	return GlobalNetwork(ctx).Globals[ctx.GlobalUSnegIndex(di, gvar, negIndex)]
}

// SetGlbUSneg is the CPU version of the global USneg variable settor
func SetGlbUSnegV(ctx *Context, di uint32, gvar GlobalVars, negIndex uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalUSnegIndex(di, gvar, negIndex)] = val
}

// AddGlbUSneg is the CPU version of the global USneg variable addor
func AddGlbUSnegV(ctx *Context, di uint32, gvar GlobalVars, negIndex uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalUSnegIndex(di, gvar, negIndex)] += val
}

// GlbUSposV is the CPU version of the global Drive, USpos variable accessor
func GlbUSposV(ctx *Context, di uint32, gvar GlobalVars, posIndex uint32) float32 {
	return GlobalNetwork(ctx).Globals[ctx.GlobalUSposIndex(di, gvar, posIndex)]
}

// SetGlbUSposV is the CPU version of the global Drive, USpos variable settor
func SetGlbUSposV(ctx *Context, di uint32, gvar GlobalVars, posIndex uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalUSposIndex(di, gvar, posIndex)] = val
}

// AddGlbUSposV is the CPU version of the global Drive, USpos variable adder
func AddGlbUSposV(ctx *Context, di uint32, gvar GlobalVars, posIndex uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalUSposIndex(di, gvar, posIndex)] += val
}

// CopyNetStridesFrom copies strides and NetIndexes for accessing
// variables on a Network -- these must be set properly for
// the Network in question (from its Ctx field) before calling
// any compute methods with the context.  See SetCtxStrides on Network.
func (ctx *Context) CopyNetStridesFrom(srcCtx *Context) {
	ctx.NetIndexes = srcCtx.NetIndexes
	ctx.NeuronVars = srcCtx.NeuronVars
	ctx.NeuronAvgVars = srcCtx.NeuronAvgVars
	ctx.NeuronIndexes = srcCtx.NeuronIndexes
	ctx.SynapseVars = srcCtx.SynapseVars
	ctx.SynapseCaVars = srcCtx.SynapseCaVars
	ctx.SynapseIndexes = srcCtx.SynapseIndexes
}

//gosl:end

//gosl:wgsl context
// #include "etime.wgsl"
// #include "axonrand.wgsl"
// #include "neuron.wgsl"
// #include "synapse.wgsl"
// #include "globals.wgsl"
// #include "neuromod.wgsl"
//gosl:endwgsl context

//gosl:start context

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

	// offset into GlobalVars for Cost values
	GvCostOff uint32 `edit:"-"`

	// stride into GlobalVars for Cost values
	GvCostStride uint32 `edit:"-"`

	// offset into GlobalVars for USneg values
	GvUSnegOff uint32 `edit:"-"`

	// stride into GlobalVars for USneg values
	GvUSnegStride uint32 `edit:"-"`

	// offset into GlobalVars for USpos, Drive, VSPatch values values
	GvUSposOff uint32 `edit:"-"`

	// stride into GlobalVars for USpos, Drive, VSPatch values
	GvUSposStride uint32 `edit:"-"`

	pad, pad2 uint32
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

	pad, pad1 float32

	// indexes and sizes of current network
	NetIndexes NetIndexes `display:"inline"`

	// stride offsets for accessing neuron variables
	NeuronVars NeuronVarStrides `display:"-"`

	// stride offsets for accessing neuron average variables
	NeuronAvgVars NeuronAvgVarStrides `display:"-"`

	// stride offsets for accessing neuron indexes
	NeuronIndexes NeuronIndexStrides `display:"-"`

	// stride offsets for accessing synapse variables
	SynapseVars SynapseVarStrides `display:"-"`

	// stride offsets for accessing synapse Ca variables
	SynapseCaVars SynapseCaStrides `display:"-"`

	// stride offsets for accessing synapse indexes
	SynapseIndexes SynapseIndexStrides `display:"-"`

	// random counter -- incremented by maximum number of possible random numbers generated per cycle, regardless of how many are actually used -- this is shared across all layers so must encompass all possible param settings.
	RandCtr slrand.Counter
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

// SetGlobalStrides sets global variable access offsets and strides
func (ctx *Context) SetGlobalStrides() {
	ctx.NetIndexes.GvCostOff = ctx.GlobalIndex(0, GvCost)
	ctx.NetIndexes.GvCostStride = uint32(ctx.NetIndexes.RubiconNCosts) * ctx.NetIndexes.MaxData
	ctx.NetIndexes.GvUSnegOff = ctx.GlobalCostIndex(0, GvCostRaw, ctx.NetIndexes.RubiconNCosts)
	ctx.NetIndexes.GvUSnegStride = uint32(ctx.NetIndexes.RubiconNNegUSs) * ctx.NetIndexes.MaxData
	ctx.NetIndexes.GvUSposOff = ctx.GlobalUSnegIndex(0, GvUSnegRaw, ctx.NetIndexes.RubiconNNegUSs)
	ctx.NetIndexes.GvUSposStride = uint32(ctx.NetIndexes.RubiconNPosUSs) * ctx.NetIndexes.MaxData
}

// GlobalIndex returns index into main global variables,
// before GvVtaDA
func (ctx *Context) GlobalIndex(di uint32, gvar GlobalVars) uint32 {
	return ctx.NetIndexes.MaxData*uint32(gvar) + di
}

// GlobalCostIndex returns index into Cost global variables
func (ctx *Context) GlobalCostIndex(di uint32, gvar GlobalVars, negIndex uint32) uint32 {
	return ctx.NetIndexes.GvCostOff + uint32(gvar-GvCost)*ctx.NetIndexes.GvCostStride + negIndex*ctx.NetIndexes.MaxData + di
}

// GlobalUSnegIndex returns index into USneg global variables
func (ctx *Context) GlobalUSnegIndex(di uint32, gvar GlobalVars, negIndex uint32) uint32 {
	return ctx.NetIndexes.GvUSnegOff + uint32(gvar-GvUSneg)*ctx.NetIndexes.GvUSnegStride + negIndex*ctx.NetIndexes.MaxData + di
}

// GlobalUSposIndex returns index into USpos, Drive, VSPatch global variables
func (ctx *Context) GlobalUSposIndex(di uint32, gvar GlobalVars, posIndex uint32) uint32 {
	return ctx.NetIndexes.GvUSposOff + uint32(gvar-GvDrives)*ctx.NetIndexes.GvUSposStride + posIndex*ctx.NetIndexes.MaxData + di
}

// GlobalVNFloats number of floats to allocate for Globals
func (ctx *Context) GlobalVNFloats() uint32 {
	return ctx.GlobalUSposIndex(0, GlobalVarsN, 0)
}

//gosl:end

// note: following is real code, uncommented by gosl

/*
bool NrnHasFlag(in Context ctx, uint ni, uint di, NeuronFlags flag) {
	return (NeuronFlags(asuint(Neurons[NrnFlags, ni, di])) & flag) > 0; // weird: != 0 does NOT work on GPU
}

void NrnSetFlag(in Context ctx, uint ni, uint di, NeuronFlags flag) {
	Neurons[NrnFlags, ni, di] =  asfloat(asuint(Neurons[NrnFlags, ni, di])|uint(flag)));
}

void NrnClearFlag(in Context ctx, uint ni, uint di, NeuronFlags flag) {
	Neurons[NrnFlags, ni, di] =  asfloat(asuint(Neurons[NrnFlags, ni, di])& ~uint(flag)));
}

bool NrnIsOff(in Context ctx, uint ni) {
	return NrnHasFlag(ctx, ni, 0, NeuronOff);
}

// // NeuronAvgVars

float NrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar) {
   return NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)];
}

void SetNrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar, float val) {
 	NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)] = val;
}

void AddNrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar, float val) {
 	NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)] += val;
}

void MulNrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar, float val) {
 	NeuronAvgs[ctx.NeuronAvgVars.Index(ni, nvar)] *= val;
}

// // NeuronIndexes

uint NrnI(in Context ctx, uint ni, NeuronIndexes idx) {
	return NeuronIxs[ctx.NeuronIndexes.Index(ni, idx)];
}

// // note: no SetNrnI in GPU mode -- all init done in CPU

// // SynapseVars

float SynV(in Context ctx, uint syni, SynapseVars svar) {
	return Synapses[ctx.SynapseVars.Index(syni, svar)];
}

void SetSynV(in Context ctx, uint syni, SynapseVars svar, float val) {
 	Synapses[ctx.SynapseVars.Index(syni, svar)] = val;
}

void AddSynV(in Context ctx, uint syni, SynapseVars svar, float val) {
 	Synapses[ctx.SynapseVars.Index(syni, svar)] += val;
}

void MulSynV(in Context ctx, uint syni, SynapseVars svar, float val) {
 	Synapses[ctx.SynapseVars.Index(syni, svar)] *= val;
}

// // SynapseCaVars

// // note: with NData repetition, SynCa can easily exceed the nominal 2^31 capacity
// // for buffer access.  Also, if else is significantly faster than switch case here.

float SynCaV(in Context ctx, uint syni, uint di, SynapseCaVars svar) {
	uint64 ix = ctx.SynapseCaVars.Index(syni, di, svar);
	uint bank = uint(ix / uint64(ctx.NetIndexes.GPUMaxBuffFloats));
	uint res = uint(ix % uint64(ctx.NetIndexes.GPUMaxBuffFloats));
	if (bank == 0) {
		return SynapseCas0[res];
	} else if (bank == 1) {
		return SynapseCas1[res];
	} else if (bank == 2) {
		return SynapseCas2[res];
	} else if (bank == 3) {
		return SynapseCas3[res];
	} else if (bank == 4) {
		return SynapseCas4[res];
	} else if (bank == 5) {
		return SynapseCas5[res];
	} else if (bank == 6) {
		return SynapseCas6[res];
	} else if (bank == 7) {
		return SynapseCas7[res];
	}
	return 0;
}

void SetSynCaV(in Context ctx, uint syni, uint di, SynapseCaVars svar, float val) {
	uint64 ix = ctx.SynapseCaVars.Index(syni, di, svar);
	uint bank = uint(ix / uint64(ctx.NetIndexes.GPUMaxBuffFloats));
	uint res = uint(ix % uint64(ctx.NetIndexes.GPUMaxBuffFloats));
	if (bank == 0) {
		SynapseCas0[res] = val;
	} else if (bank == 1) {
		SynapseCas1[res] = val;
	} else if (bank == 2) {
		SynapseCas2[res] = val;
	} else if (bank == 3) {
		SynapseCas3[res] = val;
	} else if (bank == 4) {
		SynapseCas4[res] = val;
	} else if (bank == 5) {
		SynapseCas5[res] = val;
	} else if (bank == 6) {
		SynapseCas6[res] = val;
	} else if (bank == 7) {
		SynapseCas7[res] = val;
	}
}

void AddSynCaV(in Context ctx, uint syni, uint di, SynapseCaVars svar, float val) {
	uint64 ix = ctx.SynapseCaVars.Index(syni, di, svar);
	uint bank = uint(ix / uint64(ctx.NetIndexes.GPUMaxBuffFloats));
	uint res = uint(ix % uint64(ctx.NetIndexes.GPUMaxBuffFloats));
	if (bank == 0) {
		SynapseCas0[res] += val;
	} else if (bank == 1) {
		SynapseCas1[res] += val;
	} else if (bank == 2) {
		SynapseCas2[res] += val;
	} else if (bank == 3) {
		SynapseCas3[res] += val;
	} else if (bank == 4) {
		SynapseCas4[res] += val;
	} else if (bank == 5) {
		SynapseCas5[res] += val;
	} else if (bank == 6) {
		SynapseCas6[res] += val;
	} else if (bank == 7) {
		SynapseCas7[res] += val;
	}
}

void MulSynCaV(in Context ctx, uint syni, uint di, SynapseCaVars svar, float val) {
	uint64 ix = ctx.SynapseCaVars.Index(syni, di, svar);
	uint bank = uint(ix / uint64(ctx.NetIndexes.GPUMaxBuffFloats));
	uint res = uint(ix % uint64(ctx.NetIndexes.GPUMaxBuffFloats));
	if (bank == 0) {
		SynapseCas0[res] *= val;
	} else if (bank == 1) {
		SynapseCas1[res] *= val;
	} else if (bank == 2) {
		SynapseCas2[res] *= val;
	} else if (bank == 3) {
		SynapseCas3[res] *= val;
	} else if (bank == 4) {
		SynapseCas4[res] *= val;
	} else if (bank == 5) {
		SynapseCas5[res] *= val;
	} else if (bank == 6) {
		SynapseCas6[res] *= val;
	} else if (bank == 7) {
		SynapseCas7[res] *= val;
	}
}

// // SynapseIndexes

uint SynI(in Context ctx, uint syni, SynapseIndexes idx) {
	return SynapseIxs[ctx.SynapseIndexes.Index(syni, idx)];
}

// // note: no SetSynI in GPU mode -- all init done in CPU

// /////////////////////////////////
// //  Global Vars

float GlbV(in Context ctx, uint di, GlobalVars gvar) {
	return Globals[ctx.GlobalIndex(di, gvar)];
}

void SetGlbV(in Context ctx, uint di, GlobalVars gvar, float val) {
	Globals[ctx.GlobalIndex(di, gvar)] = val;
}

void AddGlbV(in Context ctx, uint di, GlobalVars gvar, float val) {
	Globals[ctx.GlobalIndex(di, gvar)] += val;
}

float GlbCostV(in Context ctx, uint di, GlobalVars gvar, uint negIndex) {
	return Globals[ctx.GlobalCostIndex(di, gvar, negIndex)];
}

void SetGlbCostV(in Context ctx, uint di, GlobalVars gvar, uint negIndex, float val) {
	Globals[ctx.GlobalCostIndex(di, gvar, negIndex)] = val;
}

void AddGlbCostV(in Context ctx, uint di, GlobalVars gvar, uint negIndex, float val) {
	Globals[ctx.GlobalCostIndex(di, gvar, negIndex)] += val;
}

float GlbUSnegV(in Context ctx, uint di, GlobalVars gvar, uint negIndex) {
	return Globals[ctx.GlobalUSnegIndex(di, gvar, negIndex)];
}

void SetGlbUSnegV(in Context ctx, uint di, GlobalVars gvar, uint negIndex, float val) {
	Globals[ctx.GlobalUSnegIndex(di, gvar, negIndex)] = val;
}

void AddGlbUSnegV(in Context ctx, uint di, GlobalVars gvar, uint negIndex, float val) {
	Globals[ctx.GlobalUSnegIndex(di, gvar, negIndex)] += val;
}

float GlbUSposV(in Context ctx, uint di, GlobalVars gvar, uint posIndex) {
	return Globals[ctx.GlobalUSposIndex(di, gvar, posIndex)];
}

void SetGlbUSposV(in Context ctx, uint di, GlobalVars gvar, uint posIndex, float val) {
	Globals[ctx.GlobalUSposIndex(di, gvar, posIndex)] = val;
}

void AddGlbUSposV(in Context ctx, uint di, GlobalVars gvar, uint posIndex, float val) {
	Globals[ctx.GlobalUSposIndex(di, gvar, posIndex)] += val;
}

*/

//gosl:end context

//gosl:start context

// GlobalsReset resets all global values to 0, for all NData
func GlobalsReset(ctx *Context) {
	for di := uint32(0); di < ctx.NetIndexes.MaxData; di++ {
		for vg := GvRew; vg < GvCost; vg++ {
			SetGlbV(ctx, di, vg, 0)
		}
		for vn := GvCost; vn <= GvCostRaw; vn++ {
			for ui := uint32(0); ui < ctx.NetIndexes.RubiconNCosts; ui++ {
				SetGlbCostV(ctx, di, vn, ui, 0)
			}
		}
		for vn := GvUSneg; vn <= GvUSnegRaw; vn++ {
			for ui := uint32(0); ui < ctx.NetIndexes.RubiconNNegUSs; ui++ {
				SetGlbUSnegV(ctx, di, vn, ui, 0)
			}
		}
		for vp := GvDrives; vp < GlobalVarsN; vp++ {
			for ui := uint32(0); ui < ctx.NetIndexes.RubiconNPosUSs; ui++ {
				SetGlbUSposV(ctx, di, vp, ui, 0)
			}
		}
	}
}

// GlobalSetRew is a convenience function for setting the external reward
// state in Globals variables
func GlobalSetRew(ctx *Context, di uint32, rew float32, hasRew bool) {
	SetGlbV(ctx, di, GvHasRew, num.FromBool[float32](hasRew))
	if hasRew {
		SetGlbV(ctx, di, GvRew, rew)
	} else {
		SetGlbV(ctx, di, GvRew, 0)
	}
}

// .RubiconUSStimVal returns stimulus value for US at given index
// and valence (includes Cost).  If US > 0.01, a full 1 US activation is returned.
func RubiconUSStimValue(ctx *Context, di uint32, usIndex uint32, valence ValenceTypes) float32 {
	us := float32(0)
	switch valence {
	case Positive:
		if usIndex < ctx.NetIndexes.RubiconNPosUSs {
			us = GlbUSposV(ctx, di, GvUSpos, usIndex)
		}
	case Negative:
		if usIndex < ctx.NetIndexes.RubiconNNegUSs {
			us = GlbUSnegV(ctx, di, GvUSneg, usIndex)
		}
	case Cost:
		if usIndex < ctx.NetIndexes.RubiconNCosts {
			us = GlbCostV(ctx, di, GvCost, usIndex)
		}
	}
	return us
}

//gosl:end context

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
	ctx.RandCtr.Reset()
	GlobalsReset(ctx)
}

// NewContext returns a new Time struct with default parameters
func NewContext() *Context {
	ctx := &Context{}
	ctx.Defaults()
	return ctx
}
