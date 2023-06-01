// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"math"

	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/etime"
	"github.com/goki/gosl/slbool"
	"github.com/goki/gosl/slrand"
	"github.com/goki/ki/bools"
	"github.com/goki/mat32"
)

// Networks is a global list of networks, needed for GPU shader kernel
// compatible variable access in CPU mode.
// This is updated in Network.InitName, which sets NetIdx.
var Networks []*Network

// note: the following nohlsl is included for the Go type inference processing
// but is then excluded from the final .hlsl file.
// this is key for cases where there are alternative versions of functions
// in GPU vs. CPU.

//gosl: nohlsl context

// NeuronVars

// NrnV is the CPU version of the neuron variable accessor
func NrnV(ctx *Context, ni, di uint32, nvar NeuronVars) float32 {
	return Networks[ctx.NetIdxs.NetIdx].Neurons[ctx.NeuronVars.Idx(ni, di, nvar)]
}

// SetNrnV is the CPU version of the neuron variable settor
func SetNrnV(ctx *Context, ni, di uint32, nvar NeuronVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] = val
}

// AddNrnV is the CPU version of the neuron variable addor
func AddNrnV(ctx *Context, ni, di uint32, nvar NeuronVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] += val
}

// MulNrnV is the CPU version of the neuron variable multiplier
func MulNrnV(ctx *Context, ni, di uint32, nvar NeuronVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] *= val
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
	return Networks[ctx.NetIdxs.NetIdx].NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)]
}

// SetNrnAvgV is the CPU version of the neuron variable settor
func SetNrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] = val
}

// AddNrnAvgV is the CPU version of the neuron variable addor
func AddNrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] += val
}

// MulNrnAvgV is the CPU version of the neuron variable multiplier
func MulNrnAvgV(ctx *Context, ni uint32, nvar NeuronAvgVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] *= val
}

// NeuronIdxs

// NrnI is the CPU version of the neuron idx accessor
func NrnI(ctx *Context, ni uint32, idx NeuronIdxs) uint32 {
	return Networks[ctx.NetIdxs.NetIdx].NeuronIxs[ctx.NeuronIdxs.Idx(ni, idx)]
}

// SetNrnI is the CPU version of the neuron idx settor
func SetNrnI(ctx *Context, ni uint32, idx NeuronIdxs, val uint32) {
	Networks[ctx.NetIdxs.NetIdx].NeuronIxs[ctx.NeuronIdxs.Idx(ni, idx)] = val
}

// SynapseVars

// SynV is the CPU version of the synapse variable accessor
func SynV(ctx *Context, syni uint32, svar SynapseVars) float32 {
	return Networks[ctx.NetIdxs.NetIdx].Synapses[ctx.SynapseVars.Idx(syni, svar)]
}

// SetSynV is the CPU version of the synapse variable settor
func SetSynV(ctx *Context, syni uint32, svar SynapseVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Synapses[ctx.SynapseVars.Idx(syni, svar)] = val
}

// AddSynV is the CPU version of the synapse variable addor
func AddSynV(ctx *Context, syni uint32, svar SynapseVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Synapses[ctx.SynapseVars.Idx(syni, svar)] += val
}

// MulSynV is the CPU version of the synapse variable multiplier
func MulSynV(ctx *Context, syni uint32, svar SynapseVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Synapses[ctx.SynapseVars.Idx(syni, svar)] *= val
}

// SynapseCaVars

// SynCaV is the CPU version of the synapse variable accessor
func SynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars) float32 {
	return Networks[ctx.NetIdxs.NetIdx].SynapseCas[ctx.SynapseCaVars.Idx(syni, di, svar)]
}

// SetSynCaV is the CPU version of the synapse variable settor
func SetSynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].SynapseCas[ctx.SynapseCaVars.Idx(syni, di, svar)] = val
}

// AddSynCaV is the CPU version of the synapse variable addor
func AddSynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].SynapseCas[ctx.SynapseCaVars.Idx(syni, di, svar)] += val
}

// MulSynCaV is the CPU version of the synapse variable multiplier
func MulSynCaV(ctx *Context, syni, di uint32, svar SynapseCaVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].SynapseCas[ctx.SynapseCaVars.Idx(syni, di, svar)] *= val
}

// SynapseIdxs

// SynI is the CPU version of the synapse idx accessor
func SynI(ctx *Context, syni uint32, idx SynapseIdxs) uint32 {
	return Networks[ctx.NetIdxs.NetIdx].SynapseIxs[ctx.SynapseIdxs.Idx(syni, idx)]
}

// SetSynI is the CPU version of the synapse idx settor
func SetSynI(ctx *Context, syni uint32, idx SynapseIdxs, val uint32) {
	Networks[ctx.NetIdxs.NetIdx].SynapseIxs[ctx.SynapseIdxs.Idx(syni, idx)] = val
}

/////////////////////////////////
//  Global Vars

// GlbV is the CPU version of the global variable accessor
func GlbV(ctx *Context, di uint32, gvar GlobalVars) float32 {
	return Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalIdx(di, gvar)]
}

// SetGlbV is the CPU version of the global variable settor
func SetGlbV(ctx *Context, di uint32, gvar GlobalVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalIdx(di, gvar)] = val
}

// AddGlbV is the CPU version of the global variable addor
func AddGlbV(ctx *Context, di uint32, gvar GlobalVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalIdx(di, gvar)] += val
}

// GlbVTA is the CPU version of the global VTA variable accessor
func GlbVTA(ctx *Context, di uint32, vtaType GlobalVTAType, gvar GlobalVars) float32 {
	return Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalVTAIdx(di, vtaType, gvar)]
}

// SetGlbVTA is the CPU version of the global VTA variable settor
func SetGlbVTA(ctx *Context, di uint32, vtaType GlobalVTAType, gvar GlobalVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalVTAIdx(di, vtaType, gvar)] = val
}

// GlbUSneg is the CPU version of the global USneg variable accessor
func GlbUSneg(ctx *Context, di uint32, negIdx uint32) float32 {
	return Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalUSnegIdx(di, negIdx)]
}

// SetGlbUSneg is the CPU version of the global USneg variable settor
func SetGlbUSneg(ctx *Context, di uint32, negIdx uint32, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalUSnegIdx(di, negIdx)] = val
}

// GlbDriveV is the CPU version of the global Drive, USpos variable accessor
func GlbDrvV(ctx *Context, di uint32, drIdx uint32, gvar GlobalVars) float32 {
	return Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalDriveIdx(di, drIdx, gvar)]
}

// SetGlbDriveV is the CPU version of the global Drive, USpos variable settor
func SetGlbDrvV(ctx *Context, di uint32, drIdx uint32, gvar GlobalVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalDriveIdx(di, drIdx, gvar)] = val
}

// AddGlbDriveV is the CPU version of the global Drive, USpos variable adder
func AddGlbDrvV(ctx *Context, di uint32, drIdx uint32, gvar GlobalVars, val float32) {
	Networks[ctx.NetIdxs.NetIdx].Globals[ctx.GlobalDriveIdx(di, drIdx, gvar)] += val
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
// #include "pvlv.hlsl"
//gosl: end context

//gosl: start context

// NetIdxs are indexes and sizes for processing network
type NetIdxs struct {
	NData         uint32 `min:"1" desc:"number of data parallel items to process currently"`
	NetIdx        uint32 `inactive:"+" desc:"network index in global Networks list of networks -- needed for GPU shader kernel compatible network variable access functions (e.g., NrnV, SynV etc) in CPU mode"`
	MaxData       uint32 `inactive:"+" desc:"maximum amount of data parallel"`
	NLayers       uint32 `inactive:"+" desc:"number of layers in the network"`
	NNeurons      uint32 `inactive:"+" desc:"total number of neurons"`
	NPools        uint32 `inactive:"+" desc:"total number of pools excluding * MaxData factor"`
	NSyns         uint32 `inactive:"+" desc:"total number of synapses"`
	GvVTAOff      uint32 `inactive:"+" desc:"offset into GlobalVars for VTA values"`
	GvVTAStride   uint32 `inactive:"+" desc:"stride into GlobalVars for VTA values"`
	GvUSnegOff    uint32 `inactive:"+" desc:"offset into GlobalVars for USneg values"`
	GvDriveOff    uint32 `inactive:"+" desc:"offset into GlobalVars for Drive and USpos values"`
	GvDriveStride uint32 `inactive:"+" desc:"stride into GlobalVars for Drive and USpos values"`

	// pad uint32
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
	Mode         etime.Modes `desc:"current evaluation mode, e.g., Train, Test, etc"`
	Testing      slbool.Bool `inactive:"+" desc:"if true, the model is being run in a testing mode, so no weight changes or other associated computations are needed.  this flag should only affect learning-related behavior.  Is automatically updated based on Mode != Train"`
	Phase        int32       `desc:"phase counter: typicaly 0-1 for minus-plus but can be more phases for other algorithms"`
	PlusPhase    slbool.Bool `desc:"true if this is the plus phase, when the outcome / bursting is occurring, driving positive learning -- else minus phase"`
	PhaseCycle   int32       `desc:"cycle within current phase -- minus or plus"`
	Cycle        int32       `desc:"cycle counter: number of iterations of activation updating (settling) on the current state -- this counts time sequentially until reset with NewState"`
	ThetaCycles  int32       `def:"200" desc:"length of the theta cycle in terms of 1 msec Cycles -- some network update steps depend on doing something at the end of the theta cycle (e.g., CTCtxtPrjn)."`
	CyclesTotal  int32       `desc:"total cycle count -- increments continuously from whenever it was last reset -- typically this is number of milliseconds in simulation time -- is int32 and not uint32 b/c used with Synapse CaUpT which needs to have a -1 case for expired update time"`
	Time         float32     `desc:"accumulated amount of time the network has been running, in simulation-time (not real world time), in seconds"`
	TrialsTotal  int32       `desc:"total trial count -- increments continuously in NewState call *only in Train mode* from whenever it was last reset -- can be used for synchronizing weight updates across nodes"`
	TimePerCycle float32     `def:"0.001" desc:"amount of time to increment per cycle"`
	SlowInterval int32       `def:"100" desc:"how frequently to perform slow adaptive processes such as synaptic scaling, inhibition adaptation, associated in the brain with sleep, in the SlowAdapt method.  This should be long enough for meaningful changes to accumulate -- 100 is default but could easily be longer in larger models."`
	SlowCtr      int32       `inactive:"+" desc:"counter for how long it has been since last SlowAdapt step"`
	SynCaCtr     float32     `inactive:"+" desc:"synaptic calcium counter, which drives the CaUpT synaptic value to optimize updating of this computationally expensive factor. It is incremented by 1 for each cycle, and reset at the SlowInterval, at which point the synaptic calcium values are all reset."`

	pad, pad1 float32

	NetIdxs       NetIdxs             `view:"inline" desc:"indexes and sizes of current network"`
	NeuronVars    NeuronVarStrides    `view:"-" desc:"stride offsets for accessing neuron variables"`
	NeuronAvgVars NeuronAvgVarStrides `view:"-" desc:"stride offsets for accessing neuron average variables"`
	NeuronIdxs    NeuronIdxStrides    `view:"-" desc:"stride offsets for accessing neuron indexes"`
	SynapseVars   SynapseVarStrides   `view:"-" desc:"stride offsets for accessing synapse variables"`
	SynapseCaVars SynapseCaStrides    `view:"-" desc:"stride offsets for accessing synapse Ca variables"`
	SynapseIdxs   SynapseIdxStrides   `view:"-" desc:"stride offsets for accessing synapse indexes"`

	RandCtr slrand.Counter `desc:"random counter -- incremented by maximum number of possible random numbers generated per cycle, regardless of how many are actually used -- this is shared across all layers so must encompass all possible param settings."`
	PVLV    PVLV           `desc:"PVLV system for phasic dopamine signaling, including internal drives, US outcomes.  Core LHb (lateral habenula) and VTA (ventral tegmental area) dopamine are computed in equations using inputs from specialized network layers (LDTLayer driven by BLA, CeM layers, VSPatchLayer).  Renders USLayer, PVLayer, DrivesLayer representations based on state updated here."`
}

// Defaults sets default values
func (ctx *Context) Defaults() {
	ctx.NetIdxs.NData = 1
	ctx.TimePerCycle = 0.001
	ctx.ThetaCycles = 200
	ctx.SlowInterval = 100
	ctx.Mode = etime.Train
	ctx.PVLV.Defaults()
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
	ctx.NetIdxs.GvVTAOff = ctx.GlobalIdx(0, GvVtaDA)
	ctx.NetIdxs.GvVTAStride = uint32(GlobalVTATypeN) * ctx.NetIdxs.MaxData
	ctx.NetIdxs.GvUSnegOff = ctx.GlobalVTAIdx(0, GvVtaRaw, GvUSneg)
	ctx.NetIdxs.GvDriveOff = ctx.GlobalUSnegIdx(0, ctx.PVLV.Drive.NNegUSs)
	ctx.NetIdxs.GvDriveStride = uint32(ctx.PVLV.Drive.NActive) * ctx.NetIdxs.MaxData
}

// GlobalIdx returns index into main global variables,
// before GvVtaDA
func (ctx *Context) GlobalIdx(di uint32, gvar GlobalVars) uint32 {
	return ctx.NetIdxs.MaxData*uint32(gvar) + di
}

// GlobalVTAIdx returns index into VTA global variables
func (ctx *Context) GlobalVTAIdx(di uint32, vtaType GlobalVTAType, gvar GlobalVars) uint32 {
	return ctx.NetIdxs.GvVTAOff + uint32(gvar-GvVtaDA)*ctx.NetIdxs.GvVTAStride + uint32(vtaType)*ctx.NetIdxs.MaxData + di
}

// GlobalUSnegIdx returns index into USneg global variables
func (ctx *Context) GlobalUSnegIdx(di uint32, negIdx uint32) uint32 {
	return ctx.NetIdxs.GvUSnegOff + negIdx*ctx.NetIdxs.MaxData + di
}

// GlobalDriveIdx returns index into Drive and USpos, VSPatch global variables
func (ctx *Context) GlobalDriveIdx(di uint32, drIdx uint32, gvar GlobalVars) uint32 {
	return ctx.NetIdxs.GvDriveOff + uint32(gvar-GvDrives)*ctx.NetIdxs.GvDriveStride + drIdx*ctx.NetIdxs.MaxData + di
}

// GlobalVNFloats number of floats to allocate for Globals
func (ctx *Context) GlobalVNFloats() uint32 {
	return ctx.GlobalDriveIdx(0, 0, GlobalVarsN)
}

//gosl: end context

// note: following is real code, uncommented by gosl

// synmem is a way to extend memory capacity under uint limits

//gosl: hlsl synmem
// struct SynMemBlock {
// 	float vals[64];
// };
//gosl: end synmem

//gosl: hlsl context

// // NeuronVars

// // NrnV is GPU version of neuron var accessor into Neurons array
// float NrnV(in Context ctx, uint ni, uint di, NeuronVars nvar) {
//    return Neurons[ctx.NeuronVars.Idx(ni, di, nvar)];
// }
// // SetNrnV is the GPU version of the neuron variable settor
// void SetNrnV(in Context ctx, uint ni, uint di, NeuronVars nvar, float val) {
//  	Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] = val;
// }
// // AddNrnV is the GPU version of the neuron variable addor
// void AddNrnV(in Context ctx, uint ni, uint di, NeuronVars nvar, float val) {
//  	Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] += val;
// }
// // MulNrnV is the GPU version of the neuron variable multor
// void MulNrnV(in Context ctx, uint ni, uint di, NeuronVars nvar, float val) {
//  	Neurons[ctx.NeuronVars.Idx(ni, di, nvar)] *= val;
// }
//
// bool NrnHasFlag(in Context ctx, uint ni, uint di, NeuronFlags flag) {
// 	return (NeuronFlags(asuint(NrnV(ctx, ni, di, NrnFlags))) & flag) > 0; // weird: != 0 does NOT work on GPU
// }
// void NrnSetFlag(in Context ctx, uint ni, uint di, NeuronFlags flag) {
// 	SetNrnV(ctx, ni, di, NrnFlags, asfloat(asuint(NrnV(ctx, ni, di, NrnFlags))|uint(flag)));
// }
// void NrnClearFlag(in Context ctx, uint ni, uint di, NeuronFlags flag) {
// 	SetNrnV(ctx, ni, di, NrnFlags, asfloat(asuint(NrnV(ctx, ni, di, NrnFlags))& ~uint(flag)));
// }
// // NrnIsOff returns true if the neuron has been turned off (lesioned)
// // Only checks the first data item -- all should be consistent.
// bool NrnIsOff(in Context ctx, uint ni) {
// 	return NrnHasFlag(ctx, ni, 0, NeuronOff);
// }

// // NeuronAvgVars

// // NrnAvgV is GPU version of neuron var accessor into Neurons array
// float NrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar) {
//    return NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)];
// }
// // SetNrnAvgV is the GPU version of the neuron variable settor
// void SetNrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar, float val) {
//  	NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] = val;
// }
// // AddNrnAvgV is the GPU version of the neuron variable addor
// void AddNrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar, float val) {
//  	NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] += val;
// }
// // MulNrnAvgV is the GPU version of the neuron variable multor
// void MulNrnAvgV(in Context ctx, uint ni, NeuronAvgVars nvar, float val) {
//  	NeuronAvgs[ctx.NeuronAvgVars.Idx(ni, nvar)] *= val;
// }

// // NeuronIdxs
//
// // NrnI is the GPU version of the neuron idx accessor
// uint NrnI(in Context ctx, uint ni, NeuronIdxs idx) {
// 	return NeuronIxs[ctx.NeuronIdxs.Idx(ni, idx)];
// }
// // note: no SetNrnI in GPU mode -- all init done in CPU
//

// // SynapseVars
// // note: use SynMemBlock to extend memory capacity

// // SynV is GPU version of synapse var accessor into Synapses array
// float SynV(in Context ctx, uint syni, SynapseVars svar) {
// 	uint64 ix = ctx.SynapseVars.Idx(syni, svar);
// 	return Synapses[ uint(ix / 64)].vals[uint(ix % 64)];
// }
// // SetSynV is the GPU version of the synapse variable settor
// void SetSynV(in Context ctx, uint syni, SynapseVars svar, float val) {
// 	uint64 ix = ctx.SynapseVars.Idx(syni, svar);
//  	Synapses[ uint(ix / 64)].vals[uint(ix % 64)] = val;
// }
// // AddSynV is the GPU version of the synapse variable addor
// void AddSynV(in Context ctx, uint syni, SynapseVars svar, float val) {
// 	uint64 ix = ctx.SynapseVars.Idx(syni, svar);
//  	Synapses[ uint(ix / 64)].vals[uint(ix % 64)] += val;
// }
// // MulSynV is the GPU version of the synapse variable multor
// void MulSynV(in Context ctx, uint syni, SynapseVars svar, float val) {
// 	uint64 ix = ctx.SynapseVars.Idx(syni, svar);
//  	Synapses[ uint(ix / 64)].vals[uint(ix % 64)] *= val;
// }

// // SynapseCaVars

// // SynCaV is GPU version of synapse var accessor into Synapses array
// float SynCaV(in Context ctx, uint syni, uint di, SynapseCaVars svar) {
// 	uint64 ix = ctx.SynapseCaVars.Idx(syni, di, svar);
//  	return SynapseCas[ uint(ix / 64)].vals[uint(ix % 64)];
// }
// // SetSynCaV is the GPU version of the synapse variable settor
// void SetSynCaV(in Context ctx, uint syni, uint di, SynapseCaVars svar, float val) {
// 	uint64 ix = ctx.SynapseCaVars.Idx(syni, di, svar);
//  	SynapseCas[ uint(ix / 64)].vals[uint(ix % 64)] = val;
// }
// // AddSynCaV is the GPU version of the synapse variable addor
// void AddSynCaV(in Context ctx, uint syni, uint di, SynapseCaVars svar, float val) {
// 	uint64 ix = ctx.SynapseCaVars.Idx(syni, di, svar);
//  	SynapseCas[ uint(ix / 64)].vals[uint(ix % 64)] += val;
// }
// // MulSynCaV is the GPU version of the synapse variable multor
// void MulSynCaV(in Context ctx, uint syni, uint di, SynapseCaVars svar, float val) {
// 	uint64 ix = ctx.SynapseCaVars.Idx(syni, di, svar);
//  	SynapseCas[ uint(ix / 64)].vals[uint(ix % 64)] *= val;
// }

// // SynapseIdxs
//
// // SynI is the GPU version of the synapse idx accessor
// uint SynI(in Context ctx, uint syni, SynapseIdxs idx) {
// 	return SynapseIxs[ctx.SynapseIdxs.Idx(syni, idx)];
// }
// // note: no SetSynI in GPU mode -- all init done in CPU
//
//
// /////////////////////////////////
// //  Global Vars
//
// // GlbV is the CPU version of the global variable accessor
// float GlbV(in Context ctx, uint di, GlobalVars gvar) {
// 	return Globals[ctx.GlobalIdx(di, gvar)];
// }
//
// // SetGlbV is the CPU version of the global variable settor
// void SetGlbV(in Context ctx, uint di, GlobalVars gvar, float val) {
// 	Globals[ctx.GlobalIdx(di, gvar)] = val;
// }
//
// // AddGlbV is the CPU version of the global variable addor
// void AddGlbV(in Context ctx, uint di, GlobalVars gvar, float val) {
// 	Globals[ctx.GlobalIdx(di, gvar)] += val;
// }
//
// // GlbVTA is the CPU version of the global VTA variable accessor
// float GlbVTA(in Context ctx, uint di, GlobalVTAType vtaType, GlobalVars gvar) {
// 	return Globals[ctx.GlobalVTAIdx(di, vtaType, gvar)];
// }
//
// // SetGlbVTA is the CPU version of the global VTA variable settor
// void SetGlbVTA(in Context ctx, uint di, GlobalVTAType vtaType, GlobalVars gvar, float val) {
// 	Globals[ctx.GlobalVTAIdx(di, vtaType, gvar)] = val;
// }
//
// // GlbUSneg is the CPU version of the global USneg variable accessor
// float GlbUSneg(in Context ctx, uint di, uint negIdx) {
// 	return Globals[ctx.GlobalUSnegIdx(di, negIdx)];
// }
//
// // SetGlbUSneg is the CPU version of the global USneg variable settor
// void SetGlbUSneg(in Context ctx, uint di, uint negIdx, float val) {
// 	Globals[ctx.GlobalUSnegIdx(di, negIdx)] = val;
// }
//
// // GlbDrvV is the CPU version of the global Drive, USpos variable accessor
// float GlbDrvV(in Context ctx, uint di, uint drIdx, GlobalVars gvar) {
// 	return Globals[ctx.GlobalDriveIdx(di, drIdx, gvar)];
// }
//
// // SetGlbDrvV is the CPU version of the global Drive, USpos variable settor
// void SetGlbDrvV(in Context ctx, uint di, uint drIdx, GlobalVars gvar, float val) {
// 	Globals[ctx.GlobalDriveIdx(di, drIdx, gvar)] = val;
// }
//
// // AddGlbDrvV is the CPU version of the global Drive, USpos variable adder
// void AddGlbDrvV(in Context ctx, uint di, uint drIdx, GlobalVars gvar, float val) {
// 	Globals[ctx.GlobalDriveIdx(di, drIdx, gvar)] += val;
// }
//

//gosl: end context

//gosl: start context

/////////////////////////////////////////////////////////
// NeuroMod and PVLV global functions

// note: These global methods are needed for GPU which requires
// a strictly linear order of dependencies, and global var
// access depends on Context, so these can't be methods.

/////////////////////////////////////////////////////////
// 	NeuroMod

// NeuroModInit does neuromod initialization
func NeuroModInit(ctx *Context, di uint32) {
	for ns := GvRew; ns <= GvNotMaint; ns++ {
		if ns != GvPrevPred {
			SetGlbV(ctx, di, ns, 0)
		}
	}
}

// NeuroModSetRew is a convenience function for setting the external reward
func NeuroModSetRew(ctx *Context, di uint32, rew float32, hasRew bool) {
	SetGlbV(ctx, di, GvHasRew, bools.ToFloat32(hasRew))
	if hasRew {
		SetGlbV(ctx, di, GvRew, rew)
	} else {
		SetGlbV(ctx, di, GvRew, 0)
	}
}

/////////////////////////////////////////////////////////
// 	Drives

// DriveVarToZero sets all values of given drive-sized variable to 0
func DriveVarToZero(ctx *Context, di uint32, gvar GlobalVars) {
	nd := ctx.PVLV.Drive.NActive
	for i := uint32(0); i < nd; i++ {
		SetGlbDrvV(ctx, di, i, gvar, 0)
	}
}

// DrivesToZero sets all drives to 0
func DrivesToZero(ctx *Context, di uint32) {
	DriveVarToZero(ctx, di, GvDrives)
}

// DrivesToBaseline sets all drives to their baseline levels
func DrivesToBaseline(ctx *Context, di uint32) {
	nd := ctx.PVLV.Drive.NActive
	for i := uint32(0); i < nd; i++ {
		SetGlbDrvV(ctx, di, i, GvDrives, ctx.PVLV.Drive.Base.Get(i))
	}
}

// USnegToZero sets all values of USneg to zero
func USnegToZero(ctx *Context, di uint32) {
	nn := ctx.PVLV.Drive.NNegUSs
	for i := uint32(0); i < nn; i++ {
		SetGlbUSneg(ctx, di, i, 0)
	}
}

// AddTo increments drive by given amount, subject to 0-1 range clamping.
// Returns new val.
func DrivesAddTo(ctx *Context, di uint32, drv uint32, delta float32) float32 {
	dv := GlbDrvV(ctx, di, drv, GvDrives) + delta
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	SetGlbDrvV(ctx, di, drv, GvDrives, dv)
	return dv
}

// DrivesSoftAdd increments drive by given amount, using soft-bounding to 0-1 extremes.
// if delta is positive, multiply by 1-val, else val.  Returns new val.
func DrivesSoftAdd(ctx *Context, di uint32, drv uint32, delta float32) float32 {
	dv := GlbDrvV(ctx, di, drv, GvDrives)
	if delta > 0 {
		dv += (1 - dv) * delta
	} else {
		dv += dv * delta
	}
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	SetGlbDrvV(ctx, di, drv, GvDrives, dv)
	return dv
}

// DrivesExpStep updates drive with an exponential step with given dt value
// toward given baseline value.
func DrivesExpStep(ctx *Context, di uint32, drv uint32, dt, base float32) float32 {
	dv := GlbDrvV(ctx, di, drv, GvDrives)
	dv += dt * (base - dv)
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	SetGlbDrvV(ctx, di, drv, GvDrives, dv)
	return dv
}

// DrivesExpStepAll updates given drives with an exponential step using dt values
// toward baseline values.
func DrivesExpStepAll(ctx *Context, di uint32) {
	nd := ctx.PVLV.Drive.NActive
	for i := uint32(0); i < nd; i++ {
		DrivesExpStep(ctx, di, i, ctx.PVLV.Drive.Dt.Get(i), ctx.PVLV.Drive.Base.Get(i))
	}
}

// DrivesEffectiveDrive returns the Max of Drives at given index and DriveMin.
// note that index 0 is the novelty / curiosity drive.
func DrivesEffectiveDrive(ctx *Context, di uint32, i uint32) float32 {
	if i == 0 {
		return GlbDrvV(ctx, di, uint32(0), GvDrives)
	}
	return mat32.Max(GlbDrvV(ctx, di, i, GvDrives), ctx.PVLV.Drive.DriveMin)
}

/////////////////////////////////////////////////////////
// 	Effort

// EffortReset resets the raw effort back to zero -- at start of new gating event
func EffortReset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvEffortRaw, 0)
	SetGlbV(ctx, di, GvEffortCurMax, ctx.PVLV.Effort.Max)
	SetGlbV(ctx, di, GvEffortDisc, 1)
}

// EffortDiscFmEffort computes Disc from Raw effort
func EffortDiscFmEffort(ctx *Context, di uint32) float32 {
	disc := ctx.PVLV.Effort.DiscFun(GlbV(ctx, di, GvEffortRaw))
	SetGlbV(ctx, di, GvEffortDisc, disc)
	return disc
}

// EffortAddEffort adds an increment of effort and updates the Disc discount factor
func EffortAddEffort(ctx *Context, di uint32, inc float32) {
	AddGlbV(ctx, di, GvEffortRaw, inc)
	EffortDiscFmEffort(ctx, di)
}

// EffortGiveUp returns true if maximum effort has been exceeded
func EffortGiveUp(ctx *Context, di uint32) bool {
	raw := GlbV(ctx, di, GvEffortRaw)
	curMax := GlbV(ctx, di, GvEffortCurMax)
	if curMax > 0 && raw > curMax {
		return true
	}
	return false
}

/////////////////////////////////////////////////////////
// 	Urgency

// UrgencyReset resets the raw urgency back to zero -- at start of new gating event
func UrgencyReset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvUrgencyRaw, 0)
	SetGlbV(ctx, di, GvUrgency, 0)
}

// UrgeFmUrgency computes Urge from Raw
func UrgeFmUrgency(ctx *Context, di uint32) float32 {
	urge := ctx.PVLV.Urgency.UrgeFun(GlbV(ctx, di, GvUrgencyRaw))
	if urge < ctx.PVLV.Urgency.Thr {
		urge = 0
	}
	SetGlbV(ctx, di, GvUrgency, urge)
	return urge
}

// UrgencyAddEffort adds an effort increment of urgency and updates the Urge factor
func UrgencyAddEffort(ctx *Context, di uint32, inc float32) {
	AddGlbV(ctx, di, GvUrgencyRaw, inc)
	UrgeFmUrgency(ctx, di)
}

/////////////////////////////////////////////////////////
// 	LHb

// LHbReset resets all LHb vars back to 0
func LHbReset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvLHbDip, 0)
	SetGlbV(ctx, di, GvLHbBurst, 0)
	SetGlbV(ctx, di, GvLHbDipSumCur, 0)
	SetGlbV(ctx, di, GvLHbDipSum, 0)
	SetGlbV(ctx, di, GvLHbGiveUp, 0)
}

// LHbFmPVVS computes the overall LHbDip and LHbBurst values from PV (primary value)
// and VSPatch inputs.
func LHbFmPVVS(ctx *Context, di uint32, pvPos, pvNeg, vsPatchPos float32) {
	pos := ctx.PVLV.LHb.PosGain * (vsPatchPos - pvPos)
	neg := ctx.PVLV.LHb.NegGain * pvNeg
	SetGlbV(ctx, di, GvLHbPos, pos)
	SetGlbV(ctx, di, GvLHbNeg, neg)
	netLHb := pos + neg

	if netLHb > 0 {
		SetGlbV(ctx, di, GvLHbDip, netLHb)
		SetGlbV(ctx, di, GvLHbBurst, 0)
	} else {
		SetGlbV(ctx, di, GvLHbBurst, -netLHb)
		SetGlbV(ctx, di, GvLHbDip, 0)
	}
}

// LHbShouldGiveUp increments DipSum and checks if should give up if above threshold
func LHbShouldGiveUp(ctx *Context, di uint32) bool {
	dip := GlbV(ctx, di, GvLHbDip)
	AddGlbV(ctx, di, GvLHbDipSumCur, dip)
	cur := GlbV(ctx, di, GvLHbDipSumCur)
	SetGlbV(ctx, di, GvLHbDipSum, cur)
	SetGlbV(ctx, di, GvLHbGiveUp, 0)
	giveUp := false
	if cur > ctx.PVLV.LHb.GiveUpThr {
		giveUp = true
		SetGlbV(ctx, di, GvLHbGiveUp, 1)
		SetGlbV(ctx, di, GvLHbGiveUp, 1)
		SetGlbV(ctx, di, GvLHbDipSumCur, 0)
	}
	return giveUp
}

/////////////////////////////////////////////////////////
// 	VTA

// VTAZeroVals sets all VTA values to zero for given type
func VTAZeroVals(ctx *Context, di uint32, vtaType GlobalVTAType) {
	for vv := GvVtaDA; vv <= GvVtaVSPatchPos; vv++ {
		SetGlbVTA(ctx, di, vtaType, vv, 0)
	}
}

// VTAReset sets everything to zero
func VTAReset(ctx *Context, di uint32) {
	VTAZeroVals(ctx, di, GvVtaRaw)
	VTAZeroVals(ctx, di, GvVtaVals)
	VTAZeroVals(ctx, di, GvVtaPrev)
}

// VTADAFmRaw computes the intermediate Vals and final DA value from
// Raw values that have been set prior to calling.
// ACh value from LDT is passed as a parameter.
func VTADAFmRaw(ctx *Context, di uint32, ach float32, hasRew bool) {
	SetGlbVTA(ctx, di, GvVtaVals, GvVtaPVpos, ctx.PVLV.VTA.Gain.PVpos*GlbVTA(ctx, di, GvVtaRaw, GvVtaPVpos))
	SetGlbVTA(ctx, di, GvVtaVals, GvVtaCeMpos, ctx.PVLV.VTA.Gain.CeMpos*GlbVTA(ctx, di, GvVtaRaw, GvVtaCeMpos))
	SetGlbVTA(ctx, di, GvVtaVals, GvVtaCeMneg, ctx.PVLV.VTA.Gain.CeMneg*GlbVTA(ctx, di, GvVtaRaw, GvVtaCeMneg))
	SetGlbVTA(ctx, di, GvVtaVals, GvVtaLHbDip, ctx.PVLV.VTA.Gain.LHbDip*GlbVTA(ctx, di, GvVtaRaw, GvVtaLHbDip))
	SetGlbVTA(ctx, di, GvVtaVals, GvVtaLHbBurst, ctx.PVLV.VTA.Gain.LHbBurst*GlbVTA(ctx, di, GvVtaRaw, GvVtaLHbBurst))
	SetGlbVTA(ctx, di, GvVtaVals, GvVtaVSPatchPos, ctx.PVLV.VTA.Gain.VSPatchPos*GlbVTA(ctx, di, GvVtaRaw, GvVtaVSPatchPos))

	if GlbVTA(ctx, di, GvVtaVals, GvVtaVSPatchPos) < 0 {
		SetGlbVTA(ctx, di, GvVtaVals, GvVtaVSPatchPos, 0)
	}
	pvDA := GlbVTA(ctx, di, GvVtaVals, GvVtaPVpos) - GlbVTA(ctx, di, GvVtaVals, GvVtaVSPatchPos) - GlbVTA(ctx, di, GvVtaVals, GvVtaPVneg) - GlbVTA(ctx, di, GvVtaVals, GvVtaLHbDip)
	csNet := GlbVTA(ctx, di, GvVtaVals, GvVtaCeMpos) - GlbVTA(ctx, di, GvVtaVals, GvVtaCeMneg)
	csDA := ach*mat32.Max(csNet, GlbVTA(ctx, di, GvVtaVals, GvVtaLHbBurst)) - GlbVTA(ctx, di, GvVtaVals, GvVtaLHbDip) // restore LHbDip contribution
	// note that ach is only on cs -- should be 1 for PV events anyway..
	netDA := float32(0)
	if hasRew {
		netDA = pvDA
	} else {
		netDA = csDA
	}
	SetGlbVTA(ctx, di, GvVtaVals, GvVtaDA, ctx.PVLV.VTA.Gain.DA*netDA)
}

/////////////////////////////////////////////////////////
// 	PVLV

// PVLVInitUS initializes all the USs to zero
func PVLVInitUS(ctx *Context, di uint32) {
	DriveVarToZero(ctx, di, GvUSpos)
	USnegToZero(ctx, di)
}

// PVLVInitDrives initializes all the Drives to zero
func PVLVInitDrives(ctx *Context, di uint32) {
	DrivesToZero(ctx, di)
}

// PVLVReset resets all PVLV state
func PVLVReset(ctx *Context, di uint32) {
	DrivesToZero(ctx, di)
	EffortReset(ctx, di)
	UrgencyReset(ctx, di)
	LHbReset(ctx, di)
	VTAReset(ctx, di)
	PVLVInitUS(ctx, di)
	DriveVarToZero(ctx, di, GvVSPatch)
	SetGlbV(ctx, di, GvVSMatrixJustGated, 0)
	SetGlbV(ctx, di, GvVSMatrixHasGated, 0)
	SetGlbV(ctx, di, GvHasRewPrev, 0)
	SetGlbV(ctx, di, GvHasPosUS, 0)
	// pp.HasPosUSPrev.SetBool(false) // key to not reset!!
}

// PVLVPosPV returns the reward for current positive US state relative to current drives
func PVLVPosPV(ctx *Context, di uint32) float32 {
	rew := float32(0)
	nd := ctx.PVLV.Drive.NActive
	for i := uint32(0); i < nd; i++ {
		rew += GlbDrvV(ctx, di, i, GvUSpos) * DrivesEffectiveDrive(ctx, di, i)
	}
	return rew
}

// PVLVNegPV returns the reward for current negative US state -- just a sum of USneg
func PVLVNegPV(ctx *Context, di uint32) float32 {
	rew := float32(0)
	nn := ctx.PVLV.Drive.NNegUSs
	for i := uint32(0); i < nn; i++ {
		rew += GlbUSneg(ctx, di, i)
	}
	return rew
}

// PVLVVSPatchMax returns the max VSPatch value across drives
func PVLVVSPatchMax(ctx *Context, di uint32) float32 {
	max := float32(0)
	nd := ctx.PVLV.Drive.NActive
	for i := uint32(0); i < nd; i++ {
		vs := GlbDrvV(ctx, di, i, GvVSPatch)
		if vs > max {
			max = vs
		}
	}
	return max
}

// PVLVHasPosUS returns true if there is at least one non-zero positive US
func PVLVHasPosUS(ctx *Context, di uint32) bool {
	nd := ctx.PVLV.Drive.NActive
	for i := uint32(0); i < nd; i++ {
		if GlbDrvV(ctx, di, i, GvUSpos) > 0 {
			return true
		}
	}
	return false
}

// PVLVHasNegUS returns true if there is at least one non-zero negative US
func PVLVHasNegUS(ctx *Context, di uint32) bool {
	nd := ctx.PVLV.Drive.NActive
	for i := uint32(0); i < nd; i++ {
		if GlbDrvV(ctx, di, i, GvUSpos) > 0 {
			return true
		}
	}
	return false
}

// PVLVNetPV returns VTA.Vals.PVpos - VTA.Vals.PVneg
func PVLVNetPV(ctx *Context, di uint32) float32 {
	return GlbVTA(ctx, di, GvVtaVals, GvVtaPVpos) - GlbVTA(ctx, di, GvVtaVals, GvVtaPVneg)
}

// PVLVPosPVFmDriveEffort returns the net primary value ("reward") based on
// given US value and drive for that value (typically in 0-1 range),
// and total effort, from which the effort discount factor is computed an applied:
// usValue * drive * Effort.DiscFun(effort)
func PVLVPosPVFmDriveEffort(ctx *Context, usValue, drive, effort float32) float32 {
	return usValue * drive * ctx.PVLV.Effort.DiscFun(effort)
}

// PVLVSetPosUS sets given positive US (associated with same-indexed Drive) to given value
func PVLVSetPosUS(ctx *Context, di uint32, usn uint32, val float32) {
	SetGlbDrvV(ctx, di, usn, GvUSpos, val)
}

// PVLVSetNegUS sets given negative US to given value
func PVLVSetNegUS(ctx *Context, di uint32, usn uint32, val float32) {
	SetGlbDrvV(ctx, di, usn, GvUSneg, val)
}

// PVLVSetDrive sets given Drive to given value
func PVLVSetDrive(ctx *Context, di uint32, dr uint32, val float32) {
	SetGlbDrvV(ctx, di, dr, GvDrives, val)
}

// PVLVUSStimVal returns stimulus value for US at given index
// and valence.  If US > 0.01, a full 1 US activation is returned.
func PVLVUSStimVal(ctx *Context, di uint32, usIdx uint32, valence ValenceTypes) float32 {
	us := float32(0)
	if valence == Positive {
		if usIdx < ctx.PVLV.Drive.NActive {
			us = GlbDrvV(ctx, di, usIdx, GvUSpos)
		}
	} else {
		if usIdx < ctx.PVLV.Drive.NNegUSs {
			us = GlbUSneg(ctx, di, usIdx)
		}
	}
	if us > 0.01 { // threshold for presentation to net
		us = 1 // https://github.com/emer/axon/issues/194
	}
	return us
}

// PVLVDAImpl computes the updated dopamine from all the current state,
// including ACh from LDT via Context.
// Call after setting USs, Effort, Drives, VSPatch vals etc.
// Resulting DA is in VTA.Vals.DA, and is returned
// (to be set to Context.NeuroMod.DA)
func PVLVDAImpl(ctx *Context, di uint32, ach float32, hasRew bool) float32 {
	usPos := PVLVPosPV(ctx, di)
	pvNeg := PVLVNegPV(ctx, di)
	giveUp := GlbV(ctx, di, GvLHbGiveUp)
	effDisc := GlbV(ctx, di, GvEffortDisc)
	if giveUp > 0 {
		pvNeg += 1.0 - effDisc // pay effort cost here..
	}
	pvPos := usPos * effDisc
	vsPatchPos := PVLVVSPatchMax(ctx, di)
	LHbFmPVVS(ctx, di, pvPos, pvNeg, vsPatchPos)

	SetGlbVTA(ctx, di, GvVtaRaw, GvVtaUSpos, usPos)
	SetGlbVTA(ctx, di, GvVtaRaw, GvVtaPVpos, pvPos)
	SetGlbVTA(ctx, di, GvVtaRaw, GvVtaPVneg, pvNeg)
	SetGlbVTA(ctx, di, GvVtaRaw, GvVtaLHbDip, GlbV(ctx, di, GvLHbDip))
	SetGlbVTA(ctx, di, GvVtaRaw, GvVtaLHbBurst, GlbV(ctx, di, GvLHbBurst))
	SetGlbVTA(ctx, di, GvVtaRaw, GvVtaVSPatchPos, vsPatchPos)

	VTADAFmRaw(ctx, di, ach, hasRew)
	return GlbVTA(ctx, di, GvVtaVals, GvVtaDA)
}

// PVLVDriveUpdt updates the drives based on the current USs,
// subtracting USDec * US from current Drive,
// and calling ExpStep with the Dt and Base params.
func PVLVDriveUpdt(ctx *Context, di uint32) {
	DrivesExpStepAll(ctx, di)
	nd := ctx.PVLV.Drive.NActive
	for i := uint32(0); i < nd; i++ {
		us := GlbDrvV(ctx, di, i, GvUSpos)
		AddGlbDrvV(ctx, di, i, GvDrives, -us*ctx.PVLV.Drive.USDec.Get(i))
	}
}

// PVLVUrgencyUpdt updates the urgency and urgency based on given effort increment,
// resetting instead if HasRewPrev and HasPosUSPrev is true indicating receipt
// of an actual positive US.
// Call this at the start of the trial, in ApplyPVLV method.
func PVLVUrgencyUpdt(ctx *Context, di uint32, effort float32) {
	if (GlbV(ctx, di, GvHasRewPrev) > 0) && (GlbV(ctx, di, GvHasPosUSPrev) > 0) {
		UrgencyReset(ctx, di)
	} else {
		UrgencyAddEffort(ctx, di, effort)
	}
}

// PVLVDA computes the updated dopamine for PVLV algorithm from all the current state,
// including pptg and vsPatchPos (from RewPred) via Context.
// Call after setting USs, VSPatchVals, Effort, Drives, etc.
// Resulting DA is in VTA.Vals.DA is returned.
func PVLVDA(ctx *Context, di uint32) float32 {
	da := PVLVDAImpl(ctx, di, GlbV(ctx, di, GvACh), (GlbV(ctx, di, GvHasRew) > 0))
	SetGlbV(ctx, di, GvDA, da)
	SetGlbV(ctx, di, GvRewPred, GlbVTA(ctx, di, GvVtaVals, GvVtaVSPatchPos))
	for vv := GvVtaDA; vv <= GvVtaVSPatchPos; vv++ {
		SetGlbVTA(ctx, di, GvVtaPrev, vv, GlbVTA(ctx, di, GvVtaVals, vv)) // avoid race
	}
	if PVLVHasPosUS(ctx, di) {
		NeuroModSetRew(ctx, di, PVLVNetPV(ctx, di), true)
	}
	return da
}

//gosl: end context

// NewState resets counters at start of new state (trial) of processing.
// Pass the evaluation model associated with this new state --
// if !Train then testing will be set to true.
func (ctx *Context) NewState(mode etime.Modes) {
	for di := uint32(0); di < ctx.NetIdxs.MaxData; di++ {
		PVLVNewState(ctx, di, bools.FromFloat32(GlbV(ctx, di, GvHasRew)))
		NeuroModInit(ctx, di)
	}
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

// PVLVInitUS initializes the US state -- call this before calling PVLVSetUS.
func (ctx *Context) PVLVInitUS(di uint32) {
	PVLVInitUS(ctx, di)
	SetGlbV(ctx, di, GvHasRew, 0)
	SetGlbV(ctx, di, GvRew, 0)
}

// PVLVSetUS sets the given unconditioned stimulus (US) state for PVLV algorithm.
// Call PVLVInitUS before calling this, and only call this when a US has been received,
// at the start of a Trial typically.
// This then drives activity of relevant PVLV-rendered inputs, and dopamine.
// The US index is automatically adjusted for the curiosity drive / US for
// positive US outcomes -- i.e., pass in a value with 0 starting index.
// By default, negative USs do not set the overall ctx.NeuroMod.HasRew flag,
// which is the trigger for a full-blown US learning event. Set this yourself
// if the negative US is more of a discrete outcome vs. something that happens
// in the course of goal engaged approach.
func (ctx *Context) PVLVSetUS(di uint32, valence ValenceTypes, usIdx int, magnitude float32) {
	if valence == Positive {
		SetGlbV(ctx, di, GvHasRew, 1)                     // only for positive USs
		PVLVSetPosUS(ctx, di, uint32(usIdx)+1, magnitude) // +1 for curiosity
	} else {
		PVLVSetNegUS(ctx, di, uint32(usIdx), magnitude)
	}
}

// PVLVSetDrives sets current PVLV drives to given magnitude,
// and sets the first curiosity drive to given level.
// Drive indexes are 0 based, so 1 is added automatically to accommodate
// the first curiosity drive.
func (ctx *Context) PVLVSetDrives(di uint32, curiosity, magnitude float32, drives ...int) {
	PVLVInitDrives(ctx, di)
	PVLVSetDrive(ctx, di, 0, curiosity)
	for _, i := range drives {
		PVLVSetDrive(ctx, di, uint32(1+i), magnitude)
	}
}

// PVLVStepStart must be called at start of a new iteration (trial)
// of behavior when using the PVLV framework, after applying USs,
// Drives, and updating Effort (e.g., as last step in ApplyPVLV method).
// Calls PVLVGiveUp (and potentially other things).
func (ctx *Context) PVLVStepStart(di uint32, rnd erand.Rand) {
	ctx.PVLVShouldGiveUp(di, rnd)
}

// PVLVShouldGiveUp tests whether it is time to give up on the current goal,
// based on sum of LHb Dip (missed expected rewards) and maximum effort.
// called in PVLVStepStart.
func (ctx *Context) PVLVShouldGiveUp(di uint32, rnd erand.Rand) {
	giveUp := ctx.PVLV.ShouldGiveUp(ctx, di, rnd, GlbV(ctx, di, GvHasRew) > 0)
	if giveUp {
		NeuroModSetRew(ctx, di, 0, true) // sets HasRew -- drives maint reset, ACh
	}
}

// PVLVNewState is called at start of new state (trial) of processing.
// hadRew indicates if there was a reward state the previous trial.
// It calls LHGiveUpFmSum to trigger a "give up" state on this trial
// if previous expectation of reward exceeds critical sum.
func PVLVNewState(ctx *Context, di uint32, hadRew bool) {
	SetGlbV(ctx, di, GvHasRewPrev, bools.ToFloat32(hadRew))
	SetGlbV(ctx, di, GvHasPosUSPrev, bools.ToFloat32(PVLVHasPosUS(ctx, di)))

	if hadRew {
		SetGlbV(ctx, di, GvVSMatrixHasGated, 0)
	} else if GlbV(ctx, di, GvVSMatrixJustGated) > 0 {
		SetGlbV(ctx, di, GvVSMatrixHasGated, 1)
	}
	SetGlbV(ctx, di, GvVSMatrixJustGated, 0)
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
	for di := uint32(0); di < ctx.NetIdxs.MaxData; di++ {
		PVLVReset(ctx, di)
		NeuroModInit(ctx, di)
	}
}

// NewContext returns a new Time struct with default parameters
func NewContext() *Context {
	ctx := &Context{}
	ctx.Defaults()
	return ctx
}
