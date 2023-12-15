// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"math"

	"github.com/emer/emergent/v2/etime"
	"goki.dev/glop/num"
	"goki.dev/gosl/v2/slbool"
	"goki.dev/gosl/v2/slrand"
)

var (
	// TheNetwork is the one current network in use, needed for GPU shader kernel
	// compatible variable access in CPU mode, for !multinet build tags case.
	// Typically there is just one and it is faster to access directly.
	// This is set in Network.InitName.
	TheNetwork *Network

	// Networks is a global list of networks, needed for GPU shader kernel
	// compatible variable access in CPU mode, for multinet build tags case.
	// This is updated in Network.InitName, which sets NetIdx.
	Networks []*Network
)

// note: the following nohlsl is included for the Go type inference processing
// but is then excluded from the final .hlsl file.
// this is key for cases where there are alternative versions of functions
// in GPU vs. CPU.

//gosl: nohlsl context

// NeuronVars

// note: see network_single.go and network_multi.go for GlobalNetwork function
// depending on multinet build tag.

// NrnV is the CPU version of the neuron variable accessor
func NrnV(ctx *Context, ni, di uint32, nvar NeuronVars) float32 {
	return GlobalNetwork(ctx).Neurons[ctx.NeuronVars.Idx(ni, di, nvar)]
}

// SetNrnV is the CPU version of the neuron variable settor
func SetNrnV(ctx *Context, ni, di uint32, nvar NeuronVars, val float32) {
	GlobalNetwork(ctx).Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] = val
}

// AddNrnV is the CPU version of the neuron variable addor
func AddNrnV(ctx *Context, ni, di uint32, nvar NeuronVars, val float32) {
	GlobalNetwork(ctx).Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] += val
}

// MulNrnV is the CPU version of the neuron variable multiplier
func MulNrnV(ctx *Context, ni, di uint32, nvar NeuronVars, val float32) {
	GlobalNetwork(ctx).Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] *= val
}

func NrnHasFlag(ctx *Context, ni, di uint32, flag NeuronFlags) bool {
	return (NeuronFlags(math.Float32bits(NrnV(ctx, ni, di, NrnFlags))) & flag) > 0 // weird: != 0 does NOT work on GPU
}

func NrnSetFlag(ctx *Context, ni, di uint32, flag NeuronFlags) {
	SetNrnV(ctx, ni, di, NrnFlags, math.Float32frombits(math.Float32bits(NrnV(ctx, ni, di, NrnFlags))|uint32(flag)))
}

func NrnClearFlag(ctx *Context, ni, di uint32, flag NeuronFlags) {
	SetNrnV(ctx, ni, di, NrnFlags, math.Float32frombits(math.Float32bits(NrnV(ctx, ni, di, NrnFlags))&^uint32(flag)))
}

// NrnIsOff returns true if the neuron has been turned off (lesioned)
// Only checks the first data item -- all should be consistent.
func NrnIsOff(ctx *Context, ni uint32) bool {
	return NrnHasFlag(ctx, ni, 0, NeuronOff)
}

// NeuronAvgVars

// NrnAvgV is the CPU version of the neuron variable accessor
func NrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars) float32 {
	return GlobalNetwork(ctx).NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)]
}

// SetNrnAvgV is the CPU version of the neuron variable settor
func SetNrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars, val float32) {
	GlobalNetwork(ctx).NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] = val
}

// AddNrnAvgV is the CPU version of the neuron variable addor
func AddNrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars, val float32) {
	GlobalNetwork(ctx).NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] += val
}

// MulNrnAvgV is the CPU version of the neuron variable multiplier
func MulNrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars, val float32) {
	GlobalNetwork(ctx).NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] *= val
}

// NeuronIdxs

// NrnI is the CPU version of the neuron idx accessor
func NrnI(ctx *Context, ni uint32, idx NeuronIdxs) uint32 {
	return GlobalNetwork(ctx).NeuronIxs[ctx.NeuronIdxs.Idx(ni, idx)]
}

// SetNrnI is the CPU version of the neuron idx settor
func SetNrnI(ctx *Context, ni uint32, idx NeuronIdxs, val uint32) {
	GlobalNetwork(ctx).NeuronIxs[ctx.NeuronIdxs.Idx(ni, idx)] = val
}

// SynapseVars

// SynV is the CPU version of the synapse variable accessor
func SynV(ctx *Context, syni uint32, svar SynapseVars) float32 {
	return GlobalNetwork(ctx).Synapses[ctx.SynapseVars.Idx(syni, svar)]
}

// SetSynV is the CPU version of the synapse variable settor
func SetSynV(ctx *Context, syni uint32, svar SynapseVars, val float32) {
	GlobalNetwork(ctx).Synapses[ctx.SynapseVars.Idx(syni, svar)] = val
}

// AddSynV is the CPU version of the synapse variable addor
func AddSynV(ctx *Context, syni uint32, svar SynapseVars, val float32) {
	GlobalNetwork(ctx).Synapses[ctx.SynapseVars.Idx(syni, svar)] += val
}

// MulSynV is the CPU version of the synapse variable multiplier
func MulSynV(ctx *Context, syni uint32, svar SynapseVars, val float32) {
	GlobalNetwork(ctx).Synapses[ctx.SynapseVars.Idx(syni, svar)] *= val
}

// SynapseCaVars

// SynCaV is the CPU version of the synapse variable accessor
func SynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars) float32 {
	return GlobalNetwork(ctx).SynapseCas[ctx.SynapseCaVars.Idx(syni, di, svar)]
}

// SetSynCaV is the CPU version of the synapse variable settor
func SetSynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars, val float32) {
	GlobalNetwork(ctx).SynapseCas[ctx.SynapseCaVars.Idx(syni, di, svar)] = val
}

// AddSynCaV is the CPU version of the synapse variable addor
func AddSynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars, val float32) {
	GlobalNetwork(ctx).SynapseCas[ctx.SynapseCaVars.Idx(syni, di, svar)] += val
}

// MulSynCaV is the CPU version of the synapse variable multiplier
func MulSynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars, val float32) {
	GlobalNetwork(ctx).SynapseCas[ctx.SynapseCaVars.Idx(syni, di, svar)] *= val
}

// SynapseIdxs

// SynI is the CPU version of the synapse idx accessor
func SynI(ctx *Context, syni uint32, idx SynapseIdxs) uint32 {
	return GlobalNetwork(ctx).SynapseIxs[ctx.SynapseIdxs.Idx(syni, idx)]
}

// SetSynI is the CPU version of the synapse idx settor
func SetSynI(ctx *Context, syni uint32, idx SynapseIdxs, val uint32) {
	GlobalNetwork(ctx).SynapseIxs[ctx.SynapseIdxs.Idx(syni, idx)] = val
}

/////////////////////////////////
//  Global Vars

// GlbV is the CPU version of the global variable accessor
func GlbV(ctx *Context, di uint32, gvar GlobalVars) float32 {
	return GlobalNetwork(ctx).Globals[ctx.GlobalIdx(di, gvar)]
}

// SetGlbV is the CPU version of the global variable settor
func SetGlbV(ctx *Context, di uint32, gvar GlobalVars, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalIdx(di, gvar)] = val
}

// AddGlbV is the CPU version of the global variable addor
func AddGlbV(ctx *Context, di uint32, gvar GlobalVars, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalIdx(di, gvar)] += val
}

// GlbUSneg is the CPU version of the global USneg variable accessor
func GlbUSneg(ctx *Context, di uint32, gvar GlobalVars, negIdx uint32) float32 {
	return GlobalNetwork(ctx).Globals[ctx.GlobalUSnegIdx(di, gvar, negIdx)]
}

// SetGlbUSneg is the CPU version of the global USneg variable settor
func SetGlbUSneg(ctx *Context, di uint32, gvar GlobalVars, negIdx uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalUSnegIdx(di, gvar, negIdx)] = val
}

// AddGlbUSneg is the CPU version of the global USneg variable addor
func AddGlbUSneg(ctx *Context, di uint32, gvar GlobalVars, negIdx uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalUSnegIdx(di, gvar, negIdx)] += val
}

// GlbUSposV is the CPU version of the global Drive, USpos variable accessor
func GlbUSposV(ctx *Context, di uint32, gvar GlobalVars, posIdx uint32) float32 {
	return GlobalNetwork(ctx).Globals[ctx.GlobalUSposIdx(di, gvar, posIdx)]
}

// SetGlbUSposV is the CPU version of the global Drive, USpos variable settor
func SetGlbUSposV(ctx *Context, di uint32, gvar GlobalVars, posIdx uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalUSposIdx(di, gvar, posIdx)] = val
}

// AddGlbUSposV is the CPU version of the global Drive, USpos variable adder
func AddGlbUSposV(ctx *Context, di uint32, gvar GlobalVars, posIdx uint32, val float32) {
	GlobalNetwork(ctx).Globals[ctx.GlobalUSposIdx(di, gvar, posIdx)] += val
}

// CopyNetStridesFrom copies strides and NetIdxs for accessing
// variables on a Network -- these must be set properly for
// the Network in question (from its Ctx field) before calling
// any compute methods with the context.  See SetCtxStrides on Network.
func (ctx *Context) CopyNetStridesFrom(srcCtx *Context) {
	ctx.NetIdxs = srcCtx.NetIdxs
	ctx.NeuronVars = srcCtx.NeuronVars
	ctx.NeuronAvgVars = srcCtx.NeuronAvgVars
	ctx.NeuronIdxs = srcCtx.NeuronIdxs
	ctx.SynapseVars = srcCtx.SynapseVars
	ctx.SynapseCaVars = srcCtx.SynapseCaVars
	ctx.SynapseIdxs = srcCtx.SynapseIdxs
}

//gosl: end context

//gosl: hlsl context
// #include "etime.hlsl"
// #include "axonrand.hlsl"
// #include "neuron.hlsl"
// #include "synapse.hlsl"
// #include "globals.hlsl"
// #include "neuromod.hlsl"
//gosl: end context

//gosl: start context

// NetIdxs are indexes and sizes for processing network
type NetIdxs struct {

	// number of data parallel items to process currently
	NData uint32 `min:"1"`

	// network index in global Networks list of networks -- needed for GPU shader kernel compatible network variable access functions (e.g., NrnV, SynV etc) in CPU mode
	NetIdx uint32 `inactive:"+"`

	// maximum amount of data parallel
	MaxData uint32 `inactive:"+"`

	// number of layers in the network
	NLayers uint32 `inactive:"+"`

	// total number of neurons
	NNeurons uint32 `inactive:"+"`

	// total number of pools excluding * MaxData factor
	NPools uint32 `inactive:"+"`

	// total number of synapses
	NSyns uint32 `inactive:"+"`

	// maximum size in float32 (4 bytes) of a GPU buffer -- needed for GPU access
	GPUMaxBuffFloats uint32 `inactive:"+"`

	// total number of SynCa banks of GPUMaxBufferBytes arrays in GPU
	GPUSynCaBanks uint32 `inactive:"+"`

	// total number of PVLV Drives / positive USs
	PVLVNPosUSs uint32 `inactive:"+"`

	// total number of PVLV Negative USs
	PVLVNNegUSs uint32 `inactive:"+"`

	// offset into GlobalVars for USneg values
	GvUSnegOff uint32 `inactive:"+"`

	// stride into GlobalVars for USneg values
	GvUSnegStride uint32 `inactive:"+"`

	// offset into GlobalVars for USpos, Drive, VSPatch values values
	GvUSposOff uint32 `inactive:"+"`

	// stride into GlobalVars for USpos, Drive, VSPatch values
	GvUSposStride uint32 `inactive:"+"`

	pad uint32
}

// ValsIdx returns the global network index for LayerVals
// with given layer index and data parallel index.
func (ctx *NetIdxs) ValsIdx(li, di uint32) uint32 {
	return li*ctx.MaxData + di
}

// ItemIdx returns the main item index from an overall index over NItems * MaxData
// (items = layers, neurons, synapeses)
func (ctx *NetIdxs) ItemIdx(idx uint32) uint32 {
	return idx / ctx.MaxData
}

// DataIdx returns the data index from an overall index over N * MaxData
func (ctx *NetIdxs) DataIdx(idx uint32) uint32 {
	return idx % ctx.MaxData
}

// DataIdxIsValid returns true if the data index is valid (< NData)
func (ctx *NetIdxs) DataIdxIsValid(li uint32) bool {
	return (li < ctx.NData)
}

// LayerIdxIsValid returns true if the layer index is valid (< NLayers)
func (ctx *NetIdxs) LayerIdxIsValid(li uint32) bool {
	return (li < ctx.NLayers)
}

// NeurIdxIsValid returns true if the neuron index is valid (< NNeurons)
func (ctx *NetIdxs) NeurIdxIsValid(ni uint32) bool {
	return (ni < ctx.NNeurons)
}

// PoolIdxIsValid returns true if the pool index is valid (< NPools)
func (ctx *NetIdxs) PoolIdxIsValid(pi uint32) bool {
	return (pi < ctx.NPools)
}

// PoolDataIdxIsValid returns true if the pool*data index is valid (< NPools*MaxData)
func (ctx *NetIdxs) PoolDataIdxIsValid(pi uint32) bool {
	return (pi < ctx.NPools*ctx.MaxData)
}

// SynIdxIsValid returns true if the synapse index is valid (< NSyns)
func (ctx *NetIdxs) SynIdxIsValid(si uint32) bool {
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
	Testing slbool.Bool `inactive:"+"`

	// phase counter: typicaly 0-1 for minus-plus but can be more phases for other algorithms
	Phase int32

	// true if this is the plus phase, when the outcome / bursting is occurring, driving positive learning -- else minus phase
	PlusPhase slbool.Bool

	// cycle within current phase -- minus or plus
	PhaseCycle int32

	// cycle counter: number of iterations of activation updating (settling) on the current state -- this counts time sequentially until reset with NewState
	Cycle int32

	// length of the theta cycle in terms of 1 msec Cycles -- some network update steps depend on doing something at the end of the theta cycle (e.g., CTCtxtPrjn).
	ThetaCycles int32 `def:"200"`

	// total cycle count -- increments continuously from whenever it was last reset -- typically this is number of milliseconds in simulation time -- is int32 and not uint32 b/c used with Synapse CaUpT which needs to have a -1 case for expired update time
	CyclesTotal int32

	// accumulated amount of time the network has been running, in simulation-time (not real world time), in seconds
	Time float32

	// total trial count -- increments continuously in NewState call *only in Train mode* from whenever it was last reset -- can be used for synchronizing weight updates across nodes
	TrialsTotal int32

	// amount of time to increment per cycle
	TimePerCycle float32 `def:"0.001"`

	// how frequently to perform slow adaptive processes such as synaptic scaling, inhibition adaptation, associated in the brain with sleep, in the SlowAdapt method.  This should be long enough for meaningful changes to accumulate -- 100 is default but could easily be longer in larger models.  Because SlowCtr is incremented by NData, high NData cases (e.g. 16) likely need to increase this value -- e.g., 400 seems to produce overall consistent results in various models.
	SlowInterval int32 `def:"100"`

	// counter for how long it has been since last SlowAdapt step.  Note that this is incremented by NData to maintain consistency across different values of this parameter.
	SlowCtr int32 `inactive:"+"`

	// synaptic calcium counter, which drives the CaUpT synaptic value to optimize updating of this computationally expensive factor. It is incremented by 1 for each cycle, and reset at the SlowInterval, at which point the synaptic calcium values are all reset.
	SynCaCtr float32 `inactive:"+"`

	pad, pad1 float32

	// indexes and sizes of current network
	NetIdxs NetIdxs `view:"inline"`

	// stride offsets for accessing neuron variables
	NeuronVars NeuronVarStrides `view:"-"`

	// stride offsets for accessing neuron average variables
	NeuronAvgVars NeuronAvgVarStrides `view:"-"`

	// stride offsets for accessing neuron indexes
	NeuronIdxs NeuronIdxStrides `view:"-"`

	// stride offsets for accessing synapse variables
	SynapseVars SynapseVarStrides `view:"-"`

	// stride offsets for accessing synapse Ca variables
	SynapseCaVars SynapseCaStrides `view:"-"`

	// stride offsets for accessing synapse indexes
	SynapseIdxs SynapseIdxStrides `view:"-"`

	// random counter -- incremented by maximum number of possible random numbers generated per cycle, regardless of how many are actually used -- this is shared across all layers so must encompass all possible param settings.
	RandCtr slrand.Counter
}

// Defaults sets default values
func (ctx *Context) Defaults() {
	ctx.NetIdxs.NData = 1
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
	ctx.RandCtr.Add(uint32(RandFunIdxN))
}

// SlowInc increments the Slow counter and returns true if time
// to perform SlowAdapt functions (associated with sleep).
func (ctx *Context) SlowInc() bool {
	ctx.SlowCtr += int32(ctx.NetIdxs.NData)
	if ctx.SlowCtr < ctx.SlowInterval {
		return false
	}
	ctx.SlowCtr = 0
	ctx.SynCaCtr = 0
	return true
}

// SetGlobalStrides sets global variable access offsets and strides
func (ctx *Context) SetGlobalStrides() {
	ctx.NetIdxs.GvUSnegOff = ctx.GlobalIdx(0, GvUSneg)
	ctx.NetIdxs.GvUSnegStride = uint32(ctx.NetIdxs.PVLVNNegUSs) * ctx.NetIdxs.MaxData
	ctx.NetIdxs.GvUSposOff = ctx.GlobalUSnegIdx(0, GvUSnegRaw, ctx.NetIdxs.PVLVNNegUSs)
	ctx.NetIdxs.GvUSposStride = uint32(ctx.NetIdxs.PVLVNPosUSs) * ctx.NetIdxs.MaxData
}

// GlobalIdx returns index into main global variables,
// before GvVtaDA
func (ctx *Context) GlobalIdx(di uint32, gvar GlobalVars) uint32 {
	return ctx.NetIdxs.MaxData*uint32(gvar) + di
}

// GlobalUSnegIdx returns index into USneg global variables
func (ctx *Context) GlobalUSnegIdx(di uint32, gvar GlobalVars, negIdx uint32) uint32 {
	return ctx.NetIdxs.GvUSnegOff + uint32(gvar-GvUSneg)*ctx.NetIdxs.GvUSnegStride + negIdx*ctx.NetIdxs.MaxData + di
}

// GlobalUSposIdx returns index into USpos, Drive, VSPatch global variables
func (ctx *Context) GlobalUSposIdx(di uint32, gvar GlobalVars, posIdx uint32) uint32 {
	return ctx.NetIdxs.GvUSposOff + uint32(gvar-GvDrives)*ctx.NetIdxs.GvUSposStride + posIdx*ctx.NetIdxs.MaxData + di
}

// GlobalVNFloats number of floats to allocate for Globals
func (ctx *Context) GlobalVNFloats() uint32 {
	return ctx.GlobalUSposIdx(0, GlobalVarsN, 0)
}

//gosl: end context

// note: following is real code, uncommented by gosl

//gosl: hlsl context

/*

// // NeuronVars

float NrnV(in Context ctx, uint ni, uint di, NeuronVars nvar) {
   return Neurons[ctx.NeuronVars.Idx(ni, di, nvar)];
}

void SetNrnV(in Context ctx, uint ni, uint di, NeuronVars nvar, float val) {
 	Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] = val;
}

void AddNrnV(in Context ctx, uint ni, uint di, NeuronVars nvar, float val) {
 	Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] += val;
}

void MulNrnV(in Context ctx, uint ni, uint di, NeuronVars nvar, float val) {
 	Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] *= val;
}

bool NrnHasFlag(in Context ctx, uint ni, uint di, NeuronFlags flag) {
	return (NeuronFlags(asuint(NrnV(ctx, ni, di, NrnFlags))) & flag) > 0; // weird: != 0 does NOT work on GPU
}

void NrnSetFlag(in Context ctx, uint ni, uint di, NeuronFlags flag) {
	SetNrnV(ctx, ni, di, NrnFlags, asfloat(asuint(NrnV(ctx, ni, di, NrnFlags))|uint(flag)));
}

void NrnClearFlag(in Context ctx, uint ni, uint di, NeuronFlags flag) {
	SetNrnV(ctx, ni, di, NrnFlags, asfloat(asuint(NrnV(ctx, ni, di, NrnFlags))& ~uint(flag)));
}

bool NrnIsOff(in Context ctx, uint ni) {
	return NrnHasFlag(ctx, ni, 0, NeuronOff);
}

// // NeuronAvgVars

float NrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar) {
   return NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)];
}

void SetNrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar, float val) {
 	NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] = val;
}

void AddNrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar, float val) {
 	NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] += val;
}

void MulNrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar, float val) {
 	NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] *= val;
}

// // NeuronIdxs

uint NrnI(in Context ctx, uint ni, NeuronIdxs idx) {
	return NeuronIxs[ctx.NeuronIdxs.Idx(ni, idx)];
}

// // note: no SetNrnI in GPU mode -- all init done in CPU

// // SynapseVars

float SynV(in Context ctx, uint syni, SynapseVars svar) {
	return Synapses[ctx.SynapseVars.Idx(syni, svar)];
}

void SetSynV(in Context ctx, uint syni, SynapseVars svar, float val) {
 	Synapses[ctx.SynapseVars.Idx(syni, svar)] = val;
}

void AddSynV(in Context ctx, uint syni, SynapseVars svar, float val) {
 	Synapses[ctx.SynapseVars.Idx(syni, svar)] += val;
}

void MulSynV(in Context ctx, uint syni, SynapseVars svar, float val) {
 	Synapses[ctx.SynapseVars.Idx(syni, svar)] *= val;
}

// // SynapseCaVars

// // note: with NData repetition, SynCa can easily exceed the nominal 2^31 capacity
// // for buffer access.  Also, if else is significantly faster than switch case here.

float SynCaV(in Context ctx, uint syni, uint di, SynapseCaVars svar) {
	uint64 ix = ctx.SynapseCaVars.Idx(syni, di, svar);
	uint bank = uint(ix / uint64(ctx.NetIdxs.GPUMaxBuffFloats));
	uint res = uint(ix % uint64(ctx.NetIdxs.GPUMaxBuffFloats));
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
	uint64 ix = ctx.SynapseCaVars.Idx(syni, di, svar);
	uint bank = uint(ix / uint64(ctx.NetIdxs.GPUMaxBuffFloats));
	uint res = uint(ix % uint64(ctx.NetIdxs.GPUMaxBuffFloats));
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
	uint64 ix = ctx.SynapseCaVars.Idx(syni, di, svar);
	uint bank = uint(ix / uint64(ctx.NetIdxs.GPUMaxBuffFloats));
	uint res = uint(ix % uint64(ctx.NetIdxs.GPUMaxBuffFloats));
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
	uint64 ix = ctx.SynapseCaVars.Idx(syni, di, svar);
	uint bank = uint(ix / uint64(ctx.NetIdxs.GPUMaxBuffFloats));
	uint res = uint(ix % uint64(ctx.NetIdxs.GPUMaxBuffFloats));
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

// // SynapseIdxs

uint SynI(in Context ctx, uint syni, SynapseIdxs idx) {
	return SynapseIxs[ctx.SynapseIdxs.Idx(syni, idx)];
}

// // note: no SetSynI in GPU mode -- all init done in CPU

// /////////////////////////////////
// //  Global Vars

float GlbV(in Context ctx, uint di, GlobalVars gvar) {
	return Globals[ctx.GlobalIdx(di, gvar)];
}

void SetGlbV(in Context ctx, uint di, GlobalVars gvar, float val) {
	Globals[ctx.GlobalIdx(di, gvar)] = val;
}

void AddGlbV(in Context ctx, uint di, GlobalVars gvar, float val) {
	Globals[ctx.GlobalIdx(di, gvar)] += val;
}

float GlbUSneg(in Context ctx, uint di, GlobalVars gvar, uint negIdx) {
	return Globals[ctx.GlobalUSnegIdx(di, gvar, negIdx)];
}

void SetGlbUSneg(in Context ctx, uint di, GlobalVars gvar, uint negIdx, float val) {
	Globals[ctx.GlobalUSnegIdx(di, gvar, negIdx)] = val;
}

void AddGlbUSneg(in Context ctx, uint di, GlobalVars gvar, uint negIdx, float val) {
	Globals[ctx.GlobalUSnegIdx(di, gvar, negIdx)] += val;
}

float GlbUSposV(in Context ctx, uint di, GlobalVars gvar, uint posIdx) {
	return Globals[ctx.GlobalUSposIdx(di, gvar, posIdx)];
}

void SetGlbUSposV(in Context ctx, uint di, GlobalVars gvar, uint posIdx, float val) {
	Globals[ctx.GlobalUSposIdx(di, gvar, posIdx)] = val;
}

void AddGlbUSposV(in Context ctx, uint di, GlobalVars gvar, uint posIdx, float val) {
	Globals[ctx.GlobalUSposIdx(di, gvar, posIdx)] += val;
}

*/

//gosl: end context

//gosl: start context

// GlobalsReset resets all global values to 0, for all NData
func GlobalsReset(ctx *Context) {
	for di := uint32(0); di < ctx.NetIdxs.MaxData; di++ {
		for vg := GvRew; vg < GvUSneg; vg++ {
			SetGlbV(ctx, di, vg, 0)
		}
		for vn := GvUSneg; vn <= GvUSnegRaw; vn++ {
			for ui := uint32(0); ui < ctx.NetIdxs.PVLVNNegUSs; ui++ {
				SetGlbUSneg(ctx, di, vn, ui, 0)
			}
		}
		for vp := GvDrives; vp < GlobalVarsN; vp++ {
			for ui := uint32(0); ui < ctx.NetIdxs.PVLVNPosUSs; ui++ {
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

// PVLVUSStimVal returns stimulus value for US at given index
// and valence.  If US > 0.01, a full 1 US activation is returned.
func PVLVUSStimVal(ctx *Context, di uint32, usIdx uint32, valence ValenceTypes) float32 {
	us := float32(0)
	if valence == Positive {
		if usIdx < ctx.NetIdxs.PVLVNPosUSs {
			us = GlbUSposV(ctx, di, GvUSpos, usIdx)
		}
	} else {
		if usIdx < ctx.NetIdxs.PVLVNNegUSs {
			us = GlbUSneg(ctx, di, GvUSneg, usIdx)
		}
	}
	return us
}

//gosl: end context

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
