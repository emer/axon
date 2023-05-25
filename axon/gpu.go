// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"embed"
	"log"
	"unsafe"

	"github.com/goki/gi/oswin"
	"github.com/goki/vgpu/vgpu"
	vk "github.com/goki/vulkan"
)

//go:embed shaders/*.spv
var content embed.FS

//go:generate gosl -exclude=Update,UpdateParams,Defaults,AllParams github.com/goki/mat32/fastexp.go github.com/emer/etable/minmax ../chans/chans.go ../chans ../kinase ../fsfffb/inhib.go ../fsfffb github.com/emer/emergent/etime github.com/emer/emergent/ringidx rand.go avgmax.go neuromod.go pvlv.go context.go neuron.go synapse.go pool.go layervals.go act.go act_prjn.go inhib.go learn.go layertypes.go layerparams.go deep_layers.go rl_layers.go pvlv_layers.go pcore_layers.go prjntypes.go prjnparams.go deep_prjns.go rl_prjns.go pvlv_prjns.go pcore_prjns.go gpu_hlsl

// Full vars code -- each gpu_*.hlsl uses a subset

/*

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform indexes and prjn params as structured buffers in storage
[[vk::binding(0, 1)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Idxs]
[[vk::binding(1, 1)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(2, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][SendPrjns]
[[vk::binding(3, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(4, 1)]] StructuredBuffer<uint> RecvPrjnIdxs; // [Layer][RecvPrjns]
[[vk::binding(5, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]
[[vk::binding(6, 1)]] StructuredBuffer<uint> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurons][Syns]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] RWStructuredBuffer<Context> Ctx; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools][Data]
[[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer][Data]
[[vk::binding(5, 2)]] RWStructuredBuffer<float> Exts;  // [In / Out Layers][Neurons][Data]

// There might be a limit of 8 buffers per set -- can't remember..

// Set 3: synapse vars
[[vk::binding(0, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 3)]] RWStructuredBuffer<float> SynapseCas;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(2, 3)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1][Data]
[[vk::binding(3, 3)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons][Data]


Set: 0
    Role: Uniform
        Var: 0:	Layers	Struct[4]	(size: 1520)	Vals: 1
Set: 1	Idxs
    Role: Storage
        Var: 0:	NeuronIxs	Uint32[534]	(size: 4)	Vals: 1
        Var: 1:	SynapseIxs	Uint32[38976]	(size: 4)	Vals: 1
        Var: 2:	Prjns	Struct[5]	(size: 352)	Vals: 1
        Var: 3:	SendCon	Struct[242]	(size: 16)	Vals: 1
        Var: 4:	RecvPrjnIdxs	Uint32[5]	(size: 4)	Vals: 1
        Var: 5:	RecvCon	Struct[281]	(size: 16)	Vals: 1
        Var: 6:	RecvSynIdxs	Uint32[12992]	(size: 4)	Vals: 1
Set: 2	Structs
    Role: Storage
        Var: 0:	Ctx	Struct	(size: 864)	Vals: 1
        Var: 1:	Neurons	Float32[14596]	(size: 4)	Vals: 1
        Var: 2:	NeuronAvgs	Float32[890]	(size: 4)	Vals: 1
        Var: 3:	Pools	Struct[4]	(size: 1040)	Vals: 1
        Var: 4:	LayVals	Struct[4]	(size: 128)	Vals: 1
        Var: 5:	Exts	Float32[50]	(size: 4)	Vals: 1
Set: 3	Syns
    Role: Storage
        Var: 0:	Synapses	Float32[64960]	(size: 4)	Vals: 1
        Var: 1:	SynapseCas	Float32[77952]	(size: 4)	Vals: 1
        Var: 2:	GBuf	Int32[843]	(size: 4)	Vals: 1
        Var: 3:	GSyns	Float32[281]	(size: 4)	Vals: 1
*/

// TheGPU is the gpu device, shared across all networks
var TheGPU *vgpu.GPU

// CyclesN is the number of cycles to run as a group
// for ra25, 10 = ~50 msec / trial, 25 = ~48, all 150 / 50 minus / plus = ~44
// 10 is good enough and unlikely to mess with anything else..
const CyclesN = 10

// GPU manages all of the GPU-based computation
type GPU struct {
	On           bool `desc:"if true, actually use the GPU"`
	RecFunTimes  bool `desc:"if true, slower separate shader pipeline runs are used, with a CPU-sync Wait at the end, to enable timing information about each individual shader to be collected using the network FunTimer system.  otherwise, only aggregate information is available about the entire Cycle call.`
	CycleByCycle bool `desc:"if true, process each cycle one at a time.  Otherwise, 10 cycles at a time are processed in one batch."`

	Net        *Network                `view:"-" desc:"the network we operate on -- we live under this net"`
	Ctx        *Context                `view:"-" desc:"the context we use"`
	Sys        *vgpu.System            `view:"-" desc:"the vgpu compute system"`
	Params     *vgpu.VarSet            `view:"-" desc:"VarSet = 0: the uniform LayerParams"`
	Idxs       *vgpu.VarSet            `view:"-" desc:"VarSet = 1: the storage indexes and PrjnParams"`
	Structs    *vgpu.VarSet            `view:"-" desc:"VarSet = 2: the Storage buffer for RW state structs and neuron floats"`
	Syns       *vgpu.VarSet            `view:"-" desc:"Varset = 3: the Storage buffer for synapses"`
	Semaphores map[string]vk.Semaphore `view:"-" desc:"for sequencing commands"`
	NThreads   int                     `view:"-" inactive:"-" def:"64" desc:"number of warp threads -- typically 64 -- must update all hlsl files if changed!"`

	DidBind map[string]bool `view:"-" desc:"tracks var binding"`
}

// ConfigGPUwithGUI turns on GPU mode in context of an active GUI where Vulkan
// has been initialized etc.
// Configures the GPU -- call after Network is Built, initialized, params are set,
// and everything is ready to run.
func (nt *Network) ConfigGPUwithGUI(ctx *Context) {
	oswin.TheApp.RunOnMain(func() {
		nt.GPU.Config(ctx, nt)
	})
}

// ConfigGPUnoGUI turns on GPU mode in case where no GUI is being used.
// This directly accesses the GPU hardware.  It does not work well when GUI also being used.
// Configures the GPU -- call after Network is Built, initialized, params are set,
// and everything is ready to run.
func (nt *Network) ConfigGPUnoGUI(ctx *Context) {
	if err := vgpu.InitNoDisplay(); err != nil {
		panic(err)
	}
	nt.GPU.Config(ctx, nt)
}

// Destroy should be called to release all the resources allocated by the network
func (gp *GPU) Destroy() {
	if gp.Sys != nil {
		gp.Sys.Destroy()
	}
	gp.Sys = nil
}

// Config configures the network -- must call on an already-built network
func (gp *GPU) Config(ctx *Context, net *Network) {
	gp.On = true
	gp.Net = net
	gp.Ctx = ctx
	gp.NThreads = 64

	gp.DidBind = make(map[string]bool)

	if TheGPU == nil {
		TheGPU = vgpu.NewComputeGPU()
		// vgpu.Debug = true
		TheGPU.Config("axon")
	}

	gp.Sys = TheGPU.NewComputeSystem("axon")

	vars := gp.Sys.Vars()
	gp.Params = vars.AddSet()
	gp.Idxs = vars.AddSet()
	gp.Structs = vars.AddSet()
	gp.Syns = vars.AddSet()

	gp.Params.AddStruct("Layers", int(unsafe.Sizeof(LayerParams{})), len(gp.Net.LayParams), vgpu.Uniform, vgpu.ComputeShader)

	// note: prjns must be in Storage here because couldn't have both Layers and Prjns as uniform.
	gp.Idxs.Add("NeuronIxs", vgpu.Uint32, len(gp.Net.NeuronIxs), vgpu.Storage, vgpu.ComputeShader)
	gp.Idxs.Add("SynapseIxs", vgpu.Uint32, len(gp.Net.SynapseIxs), vgpu.Storage, vgpu.ComputeShader)
	gp.Idxs.AddStruct("Prjns", int(unsafe.Sizeof(PrjnParams{})), len(gp.Net.PrjnParams), vgpu.Storage, vgpu.ComputeShader)
	gp.Idxs.AddStruct("SendCon", int(unsafe.Sizeof(StartN{})), len(gp.Net.PrjnSendCon), vgpu.Storage, vgpu.ComputeShader)
	gp.Idxs.Add("RecvPrjnIdxs", vgpu.Uint32, len(gp.Net.RecvPrjnIdxs), vgpu.Storage, vgpu.ComputeShader)
	gp.Idxs.AddStruct("RecvCon", int(unsafe.Sizeof(StartN{})), len(gp.Net.PrjnRecvCon), vgpu.Storage, vgpu.ComputeShader)
	gp.Idxs.Add("RecvSynIdxs", vgpu.Uint32, len(gp.Net.RecvSynIdxs), vgpu.Storage, vgpu.ComputeShader)

	gp.Structs.AddStruct("Ctx", int(unsafe.Sizeof(Context{})), 1, vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("Neurons", vgpu.Float32, len(gp.Net.Neurons), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("NeuronAvgs", vgpu.Float32, len(gp.Net.NeuronAvgs), vgpu.Storage, vgpu.ComputeShader)

	gp.Structs.AddStruct("Pools", int(unsafe.Sizeof(Pool{})), len(gp.Net.Pools), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("LayVals", int(unsafe.Sizeof(LayerVals{})), len(gp.Net.LayVals), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("Exts", vgpu.Float32, len(gp.Net.Exts), vgpu.Storage, vgpu.ComputeShader)

	gp.Syns.Add("Synapses", vgpu.Float32, len(gp.Net.Synapses), vgpu.Storage, vgpu.ComputeShader)
	gp.Syns.Add("SynapseCas", vgpu.Float32, len(gp.Net.SynapseCas), vgpu.Storage, vgpu.ComputeShader)
	gp.Syns.Add("GBuf", vgpu.Int32, len(gp.Net.PrjnGBuf), vgpu.Storage, vgpu.ComputeShader)
	gp.Syns.Add("GSyns", vgpu.Float32, len(gp.Net.PrjnGSyns), vgpu.Storage, vgpu.ComputeShader)

	gp.Params.ConfigVals(1)
	gp.Idxs.ConfigVals(1)
	gp.Structs.ConfigVals(1)
	gp.Syns.ConfigVals(1)

	// pipelines
	gp.Sys.NewComputePipelineEmbed("GatherSpikes", content, "shaders/gpu_gather.spv")
	gp.Sys.NewComputePipelineEmbed("LayGi", content, "shaders/gpu_laygi.spv")
	gp.Sys.NewComputePipelineEmbed("BetweenGi", content, "shaders/gpu_betweengi.spv")
	gp.Sys.NewComputePipelineEmbed("PoolGi", content, "shaders/gpu_poolgi.spv")
	gp.Sys.NewComputePipelineEmbed("Cycle", content, "shaders/gpu_cycle.spv")
	gp.Sys.NewComputePipelineEmbed("CycleInc", content, "shaders/gpu_cycleinc.spv")
	gp.Sys.NewComputePipelineEmbed("SendSpike", content, "shaders/gpu_sendspike.spv")
	gp.Sys.NewComputePipelineEmbed("SynCa", content, "shaders/gpu_synca.spv")
	gp.Sys.NewComputePipelineEmbed("CyclePost", content, "shaders/gpu_cyclepost.spv")

	gp.Sys.NewComputePipelineEmbed("NewState", content, "shaders/gpu_newstate.spv")
	gp.Sys.NewComputePipelineEmbed("MinusPool", content, "shaders/gpu_minuspool.spv")
	gp.Sys.NewComputePipelineEmbed("MinusNeuron", content, "shaders/gpu_minusneuron.spv")
	gp.Sys.NewComputePipelineEmbed("PlusStart", content, "shaders/gpu_plusstart.spv")
	gp.Sys.NewComputePipelineEmbed("PlusPool", content, "shaders/gpu_pluspool.spv")
	gp.Sys.NewComputePipelineEmbed("PlusNeuron", content, "shaders/gpu_plusneuron.spv")
	gp.Sys.NewComputePipelineEmbed("DWt", content, "shaders/gpu_dwt.spv")
	gp.Sys.NewComputePipelineEmbed("DWtFmDi", content, "shaders/gpu_dwtfmdi.spv")
	gp.Sys.NewComputePipelineEmbed("WtFmDWt", content, "shaders/gpu_wtfmdwt.spv")
	gp.Sys.NewComputePipelineEmbed("DWtSubMean", content, "shaders/gpu_dwtsubmean.spv")
	gp.Sys.NewComputePipelineEmbed("ApplyExts", content, "shaders/gpu_applyext.spv")

	gp.Sys.NewEvent("MemCopyTo")
	gp.Sys.NewEvent("MemCopyFm")
	gp.Sys.NewEvent("CycleEnd")
	gp.Sys.NewEvent("CycleInc")
	gp.Sys.NewEvent("GatherSpikes")
	gp.Sys.NewEvent("LayGi")
	gp.Sys.NewEvent("BetweenGi")
	gp.Sys.NewEvent("PoolGi")
	gp.Sys.NewEvent("Cycle")
	gp.Sys.NewEvent("CyclePost")
	gp.Sys.NewEvent("SendSpike")

	gp.Sys.Config()

	gp.CopyParamsToStaging()
	gp.CopyIdxsToStaging()
	gp.CopyExtsToStaging()
	gp.CopyContextToStaging()
	gp.CopyStateToStaging()
	gp.CopySynapsesToStaging()

	gp.Sys.Mem.SyncToGPU()

	// todo: add a convenience method to vgpu to do this for everything
	vars.BindDynValIdx(0, "Layers", 0)

	vars.BindDynValIdx(1, "NeuronIxs", 0)
	vars.BindDynValIdx(1, "SynapseIxs", 0)
	vars.BindDynValIdx(1, "Prjns", 0)
	vars.BindDynValIdx(1, "SendCon", 0)
	vars.BindDynValIdx(1, "RecvPrjnIdxs", 0)
	vars.BindDynValIdx(1, "RecvCon", 0)
	vars.BindDynValIdx(1, "RecvSynIdxs", 0)

	vars.BindDynValIdx(2, "Ctx", 0)
	vars.BindDynValIdx(2, "Neurons", 0)
	vars.BindDynValIdx(2, "NeuronAvgs", 0)
	vars.BindDynValIdx(2, "Pools", 0)
	vars.BindDynValIdx(2, "LayVals", 0)
	vars.BindDynValIdx(2, "Exts", 0)

	vars.BindDynValIdx(3, "Synapses", 0)
	vars.BindDynValIdx(3, "SynapseCas", 0)
	vars.BindDynValIdx(3, "GBuf", 0)
	vars.BindDynValIdx(3, "GSyns", 0)

}

///////////////////////////////////////////////////////////////////////
// 	Sync To
//			the CopyFromBytes call automatically flags updated regions
// 		and SyncMemToGPU does all updated

// SyncMemToGPU synchronizes any staging memory buffers that have been updated with
// a Copy function, actually sending the updates from the staging -> GPU.
// The CopyTo commands just copy Network-local data to a staging buffer,
// and this command then actually moves that onto the GPU.
// In unified GPU memory architectures, this staging buffer is actually the same
// one used directly by the GPU -- otherwise it is a separate staging buffer.
func (gp *GPU) SyncMemToGPU() {
	gp.Sys.Mem.SyncToGPU()
}

// CopyParamsToStaging copies the LayerParams and PrjnParams to staging from CPU.
// Must call SyncMemToGPU after this (see SyncParamsToGPU).
func (gp *GPU) CopyParamsToStaging() {
	if !gp.On {
		return
	}
	_, layv, _ := gp.Params.ValByIdxTry("Layers", 0)
	layv.CopyFromBytes(unsafe.Pointer(&gp.Net.LayParams[0]))

	_, pjnv, _ := gp.Idxs.ValByIdxTry("Prjns", 0)
	pjnv.CopyFromBytes(unsafe.Pointer(&gp.Net.PrjnParams[0]))
}

// SyncParamsToGPU copies the LayerParams and PrjnParams to the GPU from CPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncParamsToGPU() {
	if !gp.On {
		return
	}
	gp.CopyParamsToStaging()
	gp.SyncMemToGPU()
}

// CopyIdxsToStaging is only called when the network is built
// to copy the indexes specifying connectivity etc to staging from CPU.
func (gp *GPU) CopyIdxsToStaging() {
	if !gp.On {
		return
	}

	_, neuriv, _ := gp.Idxs.ValByIdxTry("NeuronIxs", 0)
	neuriv.CopyFromBytes(unsafe.Pointer(&gp.Net.NeuronIxs[0]))

	_, syniv, _ := gp.Idxs.ValByIdxTry("SynapseIxs", 0)
	syniv.CopyFromBytes(unsafe.Pointer(&gp.Net.SynapseIxs[0]))

	_, sconv, _ := gp.Idxs.ValByIdxTry("SendCon", 0)
	sconv.CopyFromBytes(unsafe.Pointer(&gp.Net.PrjnSendCon[0]))

	_, spiv, _ := gp.Idxs.ValByIdxTry("RecvPrjnIdxs", 0)
	spiv.CopyFromBytes(unsafe.Pointer(&gp.Net.RecvPrjnIdxs[0]))

	_, rconv, _ := gp.Idxs.ValByIdxTry("RecvCon", 0)
	rconv.CopyFromBytes(unsafe.Pointer(&gp.Net.PrjnRecvCon[0]))

	_, ssiv, _ := gp.Idxs.ValByIdxTry("RecvSynIdxs", 0)
	ssiv.CopyFromBytes(unsafe.Pointer(&gp.Net.RecvSynIdxs[0]))
}

// CopyExtsToStaging copies external inputs to staging from CPU.
// Typically used in RunApplyExts which also does the Sync.
func (gp *GPU) CopyExtsToStaging() {
	if !gp.On {
		return
	}
	_, extv, _ := gp.Structs.ValByIdxTry("Exts", 0)
	extv.CopyFromBytes(unsafe.Pointer(&gp.Net.Exts[0]))
}

// CopyContextToStaging copies current context to staging from CPU.
// Must call SyncMemToGPU after this (see SyncContextToGPU).
// See SetContext if there is a new one.
func (gp *GPU) CopyContextToStaging() {
	if !gp.On {
		return
	}
	_, ctxv, _ := gp.Structs.ValByIdxTry("Ctx", 0)
	ctxv.CopyFromBytes(unsafe.Pointer(gp.Ctx))
}

// SyncContextToGPU copies current context to GPU from CPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
// See SetContext if there is a new one.
func (gp *GPU) SyncContextToGPU() {
	if !gp.On {
		return
	}
	gp.CopyContextToStaging()
	gp.SyncMemToGPU()
}

// SetContext sets our context to given context and syncs it to the GPU.
// Typically a single context is used as it must be synced into the GPU.
// The GPU never writes to the CPU
func (gp *GPU) SetContext(ctx *Context) {
	if !gp.On {
		return
	}
	gp.Ctx = ctx
	gp.SyncContextToGPU()
}

// CopyLayerValsToStaging copies LayerVals to staging from CPU.
// Must call SyncMemToGPU after this (see SyncLayerValsToGPU).
func (gp *GPU) CopyLayerValsToStaging() {
	if !gp.On {
		return
	}
	_, layv, _ := gp.Structs.ValByIdxTry("LayVals", 0)
	layv.CopyFromBytes(unsafe.Pointer(&gp.Net.LayVals[0]))
}

// SyncLayerValsToGPU copies LayerVals to GPU from CPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncLayerValsToGPU() {
	if !gp.On {
		return
	}
	gp.CopyLayerValsToStaging()
	gp.SyncMemToGPU()
}

// CopyPoolsToStaging copies Pools to staging from CPU.
// Must call SyncMemToGPU after this (see SyncPoolsToGPU).
func (gp *GPU) CopyPoolsToStaging() {
	if !gp.On {
		return
	}
	_, poolv, _ := gp.Structs.ValByIdxTry("Pools", 0)
	poolv.CopyFromBytes(unsafe.Pointer(&gp.Net.Pools[0]))
}

// SyncPoolsToGPU copies Pools to GPU from CPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncPoolsToGPU() {
	if !gp.On {
		return
	}
	gp.CopyPoolsToStaging()
	gp.SyncMemToGPU()
}

// CopyNeuronsToStaging copies neuron state up to staging from CPU.
// Must call SyncMemToGPU after this (see SyncNeuronsToGPU).
func (gp *GPU) CopyNeuronsToStaging() {
	if !gp.On {
		return
	}
	_, neurv, _ := gp.Structs.ValByIdxTry("Neurons", 0)
	neurv.CopyFromBytes(unsafe.Pointer(&gp.Net.Neurons[0]))
	_, neurav, _ := gp.Structs.ValByIdxTry("NeuronAvgs", 0)
	neurav.CopyFromBytes(unsafe.Pointer(&gp.Net.NeuronAvgs[0]))
}

// SyncNeuronsToGPU copies neuron state up to GPU from CPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncNeuronsToGPU() {
	if !gp.On {
		return
	}
	gp.CopyNeuronsToStaging()
	gp.SyncMemToGPU()
}

// CopyStateToStaging copies LayerVals, Pools, Neurons state to staging from CPU.
// this is typically sufficient for most syncing --
// only missing the Synapses which must be copied separately.
// Must call SyncMemToGPU after this (see SyncStateToGPU).
func (gp *GPU) CopyStateToStaging() {
	if !gp.On {
		return
	}
	gp.CopyLayerValsToStaging()
	gp.CopyPoolsToStaging()
	gp.CopyNeuronsToStaging()
}

// SyncStateToGPU copies LayVals, Pools, Neurons state to GPU
// this is typically sufficient for most syncing --
// only missing the Synapses which must be copied separately.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncStateToGPU() {
	if !gp.On {
		return
	}
	gp.CopyStateToStaging()
	gp.SyncMemToGPU()
}

// SyncAllToGPU copies LayerVals, Pools, Neurons, Synapses to GPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncAllToGPU() {
	if !gp.On {
		return
	}
	gp.CopyStateToStaging()
	gp.CopySynapsesToStaging()
	gp.SyncMemToGPU()
}

// CopySynapsesToStaging copies the synapse memory to staging (large).
// This is not typically needed except when weights are initialized or
// for the Slow weight update processes that are not on GPU.
// Must call SyncMemToGPU after this (see SyncSynapsesToGPU).
func (gp *GPU) CopySynapsesToStaging() {
	if !gp.On {
		return
	}
	_, synv, _ := gp.Syns.ValByIdxTry("Synapses", 0)
	synv.CopyFromBytes(unsafe.Pointer(&gp.Net.Synapses[0]))
	_, syncv, _ := gp.Syns.ValByIdxTry("SynapseCas", 0)
	syncv.CopyFromBytes(unsafe.Pointer(&gp.Net.SynapseCas[0]))
}

// SyncSynapsesToGPU copies the synapse memory to GPU (large).
// This is not typically needed except when weights are initialized or
// for the Slow weight update processes that are not on GPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncSynapsesToGPU() {
	if !gp.On {
		return
	}
	gp.CopySynapsesToStaging()
	gp.SyncMemToGPU()
}

// SyncGBufToGPU copies the GBuf and GSyns memory to the GPU.
// This is a temporary measure to be replaced with a simple kernel to init gbuf,
// needed for InitActs.
func (gp *GPU) SyncGBufToGPU() {
	if !gp.On {
		return
	}
	_, gbv, _ := gp.Syns.ValByIdxTry("GBuf", 0)
	gbv.CopyFromBytes(unsafe.Pointer(&gp.Net.PrjnGBuf[0]))
	_, gsv, _ := gp.Syns.ValByIdxTry("GSyns", 0)
	gsv.CopyFromBytes(unsafe.Pointer(&gp.Net.PrjnGSyns[0]))
	gp.SyncMemToGPU()
}

///////////////////////////////////////////////////////////////////////
// 	Sync From
//			unlike Sync To, need to specify the regions to sync from first
// 		and then copy from staging to get into CPU memory

// SyncRegionStruct returns the SyncRegion with error panic
func (gp *GPU) SyncRegionStruct(vnm string) vgpu.MemReg {
	r, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, vnm, 0)
	if err != nil {
		panic(err)
	}
	return r
}

// SyncRegionSyns returns the SyncRegion with error panic
func (gp *GPU) SyncRegionSyns(vnm string) vgpu.MemReg {
	r, err := gp.Sys.Mem.SyncRegionValIdx(gp.Syns.Set, vnm, 0)
	if err != nil {
		panic(err)
	}
	return r
}

// CopyContextFmStaging copies Context from staging to CPU, after Sync back down.
func (gp *GPU) CopyContextFmStaging() {
	if !gp.On {
		return
	}
	_, cxv, _ := gp.Structs.ValByIdxTry("Ctx", 0)
	cxv.CopyToBytes(unsafe.Pointer(gp.Ctx))
}

// SyncContextFmGPU copies Context from GPU to CPU.
// This is done at the end of each cycle to get state back from GPU for CPU-side computations.
// Use only when only thing being copied -- more efficient to get all at once.
// e.g. see SyncStateFmGPU
func (gp *GPU) SyncContextFmGPU() {
	if !gp.On {
		return
	}
	cxr := gp.SyncRegionStruct("Ctx")
	gp.Sys.Mem.SyncStorageRegionsFmGPU(cxr)
	gp.CopyContextFmStaging()
}

// CopyLayerValsFmStaging copies LayerVals from staging to CPU, after Sync back down.
func (gp *GPU) CopyLayerValsFmStaging() {
	if !gp.On {
		return
	}
	_, layv, _ := gp.Structs.ValByIdxTry("LayVals", 0)
	layv.CopyToBytes(unsafe.Pointer(&gp.Net.LayVals[0]))
}

// SyncLayerValsFmGPU copies LayerVals from GPU to CPU.
// This is done at the end of each cycle to get state back from staging for CPU-side computations.
// Use only when only thing being copied -- more efficient to get all at once.
// e.g. see SyncStateFmGPU
func (gp *GPU) SyncLayerValsFmGPU() {
	if !gp.On {
		return
	}
	lvr := gp.SyncRegionStruct("LayVals")
	gp.Sys.Mem.SyncStorageRegionsFmGPU(lvr)
	gp.CopyLayerValsFmStaging()
}

// CopyPoolsFmStaging copies Pools from staging to CPU, after Sync back down.
func (gp *GPU) CopyPoolsFmStaging() {
	if !gp.On {
		return
	}
	_, plv, _ := gp.Structs.ValByIdxTry("Pools", 0)
	plv.CopyToBytes(unsafe.Pointer(&gp.Net.Pools[0]))
}

// SyncPoolsFmGPU copies Pools from GPU to CPU.
// Use only when only thing being copied -- more efficient to get all at once.
// e.g. see SyncStateFmGPU
func (gp *GPU) SyncPoolsFmGPU() {
	if !gp.On {
		return
	}
	plr := gp.SyncRegionStruct("Pools")
	gp.Sys.Mem.SyncStorageRegionsFmGPU(plr)
	gp.CopyPoolsFmStaging()
}

// CopyNeuronsFmStaging copies Neurons from staging to CPU, after Sync back down.
func (gp *GPU) CopyNeuronsFmStaging() {
	if !gp.On {
		return
	}
	_, neurv, _ := gp.Structs.ValByIdxTry("Neurons", 0)
	neurv.CopyToBytes(unsafe.Pointer(&gp.Net.Neurons[0]))
	_, neurav, _ := gp.Structs.ValByIdxTry("NeuronAvgs", 0)
	neurav.CopyToBytes(unsafe.Pointer(&gp.Net.NeuronAvgs[0]))
	// note: don't need to get indexes back down
}

// SyncNeuronsFmGPU copies Neurons from GPU to CPU.
// Use only when only thing being copied -- more efficient to get all at once.
// e.g. see SyncStateFmGPU
func (gp *GPU) SyncNeuronsFmGPU() {
	if !gp.On {
		return
	}
	nrr := gp.SyncRegionStruct("Neurons")
	nrar := gp.SyncRegionStruct("NeuronAvgs")
	// note: don't need to get indexes back down
	gp.Sys.Mem.SyncStorageRegionsFmGPU(nrr, nrar)
	gp.CopyNeuronsFmStaging()
}

// CopySynapsesFmStaging copies Synapses from staging to CPU, after Sync back down.
func (gp *GPU) CopySynapsesFmStaging() {
	if !gp.On {
		return
	}
	_, synv, _ := gp.Syns.ValByIdxTry("Synapses", 0)
	synv.CopyToBytes(unsafe.Pointer(&gp.Net.Synapses[0]))
	_, syncv, _ := gp.Syns.ValByIdxTry("SynapseCas", 0)
	syncv.CopyToBytes(unsafe.Pointer(&gp.Net.SynapseCas[0]))
}

// SyncSynapsesFmGPU copies Synapses from GPU to CPU.
// Use only when only thing being copied -- more efficient to get all at once.
func (gp *GPU) SyncSynapsesFmGPU() {
	if !gp.On {
		return
	}
	syr := gp.SyncRegionSyns("Synapses")
	sycr := gp.SyncRegionSyns("SynapseCas")
	gp.Sys.Mem.SyncStorageRegionsFmGPU(syr, sycr)
	gp.CopySynapsesFmStaging()
}

// CopyLayerStateFmStaging copies Context, LayerVals and Pools from staging to CPU, after Sync.
func (gp *GPU) CopyLayerStateFmStaging() {
	gp.CopyContextFmStaging()
	gp.CopyLayerValsFmStaging()
	gp.CopyPoolsFmStaging()
}

// SyncLayerStateFmCPU copies Context, LayerVals, and Pools from GPU to CPU.
// This is the main GPU->CPU sync step automatically called after each Cycle.
func (gp *GPU) SyncLayerStateFmGPU() {
	if !gp.On {
		return
	}
	cxr := gp.SyncRegionStruct("Ctx")
	lvr := gp.SyncRegionStruct("LayVals")
	plr := gp.SyncRegionStruct("Pools")
	gp.Sys.Mem.SyncStorageRegionsFmGPU(cxr, lvr, plr)
	gp.CopyLayerStateFmStaging()
}

// CopyStateFmStaging copies Context, LayerVals, Pools, and Neurons from staging to CPU, after Sync.
func (gp *GPU) CopyStateFmStaging() {
	gp.CopyLayerStateFmStaging()
	gp.CopyNeuronsFmStaging()
}

// SyncStateFmCPU copies Neurons, LayerVals, and Pools from GPU to CPU.
// This is the main GPU->CPU sync step automatically called in PlusPhase.
func (gp *GPU) SyncStateFmGPU() {
	if !gp.On {
		return
	}
	cxr := gp.SyncRegionStruct("Ctx")
	lvr := gp.SyncRegionStruct("LayVals")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")
	nrar := gp.SyncRegionStruct("NeuronAvgs")
	gp.Sys.Mem.SyncStorageRegionsFmGPU(cxr, lvr, plr, nrr, nrar)
	gp.CopyStateFmStaging()
}

// SyncAllFmCPU copies State except Context plus Synapses from GPU to CPU.
// This is called before SlowAdapt, which is run CPU-side
func (gp *GPU) SyncAllFmGPU() {
	if !gp.On {
		return
	}
	lvr := gp.SyncRegionStruct("LayVals")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")
	nrar := gp.SyncRegionStruct("NeuronAvgs")
	syr := gp.SyncRegionSyns("Synapses")
	sycr := gp.SyncRegionSyns("SynapseCas")
	gp.Sys.Mem.SyncStorageRegionsFmGPU(lvr, plr, nrr, nrar, syr, sycr)
	gp.CopyLayerValsFmStaging()
	gp.CopyPoolsFmStaging()
	gp.CopyNeuronsFmStaging()
	gp.CopySynapsesFmStaging()
}

///////////////////////////////////////////////////////////////////////
// 	Run

// RunPipelineWait runs given pipeline in "single shot" mode,
// which is maximally inefficient if multiple commands need to be run.
// This is the only mode in which timer information is available.
func (gp *GPU) RunPipelineWait(name string, n int) {
	pl, err := gp.Sys.PipelineByNameTry(name)
	if err != nil {
		panic(err)
	}
	gnm := "GPU:" + name
	gp.Net.FunTimerStart(gnm)
	gp.Sys.ComputeResetBindVars(0)
	pl.ComputeCommand1D(n, gp.NThreads)
	gp.Sys.ComputeSubmitWait()
	gp.Net.FunTimerStop(gnm)
}

// StartRun resets the command buffer in preparation for recording commands
// for a multi-step run.
// It is much more efficient to record all commands to one buffer, and use
// Events to synchronize the steps between them, rather than using semaphores.
// The submit call is by far the most expensive so that should only happen once!
func (gp *GPU) StartRun() {
	gp.Sys.ComputeResetBindVars(0)
}

// StartRunCmd resets the given command buffer in preparation
// for recording commands for a multi-step run.
// It is much more efficient to record all commands to one buffer, and use
// Events to synchronize the steps between them, rather than using semaphores.
// The submit call is by far the most expensive so that should only happen once!
func (gp *GPU) StartRunCmd(cmd vk.CommandBuffer) {
	gp.Sys.ComputeResetBindVarsCmd(cmd, 0)
}

// RunPipeline records command to run given pipeline with
// optional wait & signal event names
func (gp *GPU) RunPipeline(name string, n int, wait, signal string) {
	pl, err := gp.Sys.PipelineByNameTry(name)
	if err != nil {
		panic(err)
	}
	if wait != "" {
		gp.Sys.ComputeWaitEvents(wait)
	}
	pl.ComputeCommand1D(n, gp.NThreads)
	if signal != "" {
		gp.Sys.ComputeSetEvent(signal)
	}
}

// RunPipelineCmd records command to run given pipeline with
// optional wait & signal event names
func (gp *GPU) RunPipelineCmd(cmd vk.CommandBuffer, name string, n int, wait, signal string) {
	pl, err := gp.Sys.PipelineByNameTry(name)
	if err != nil {
		panic(err)
	}
	if wait != "" {
		gp.Sys.ComputeWaitEventsCmd(cmd, wait)
	}
	pl.ComputeCommand1DCmd(cmd, n, gp.NThreads)
	if signal != "" {
		gp.Sys.ComputeSetEventCmd(cmd, signal)
	}
}

// RunApplyExts copies Exts external input memory to the GPU and then
// runs the ApplyExts shader that applies those external inputs to the
// GPU-side neuron state.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunApplyExts() {
	cmd := gp.RunApplyExtsCmd()
	gnm := "GPU:ApplyExts"
	gp.Net.FunTimerStart(gnm)
	gp.CopyExtsToStaging()
	gp.CopyContextToStaging()
	gp.Sys.ComputeSubmitWaitCmd(cmd)
	gp.Net.FunTimerStop(gnm)
}

// RunApplyExtsCmd returns the commands to
// copy Exts external input memory to the GPU and then
// runs the ApplyExts shader that applies those external inputs to the
// GPU-side neuron state.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunApplyExtsCmd() vk.CommandBuffer {
	cnm := "RunApplyExts"
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)

	exr, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Exts", 0)
	if err != nil {
		log.Println(err)
	}
	cxr := gp.SyncRegionStruct("Ctx")
	gp.StartRunCmd(cmd)
	gp.Sys.ComputeCmdCopyToGPUCmd(cmd, exr)
	gp.Sys.ComputeCmdCopyToGPUCmd(cmd, cxr) // staging -> GPU
	gp.Sys.ComputeSetEventCmd(cmd, "MemCopyTo")
	gp.RunPipelineCmd(cmd, "ApplyExts", neurDataN, "MemCopyTo", "")
	gp.Sys.ComputeCmdEndCmd(cmd)
	return cmd
}

// RunCycle is the main cycle-level update loop for updating one msec of neuron state.
// It copies current Context up to GPU for updated Cycle counter state and random number state,
// Different versions of the code are run depending on various flags.
// By default, it will run the entire minus and plus phase in one big chunk.
// The caller must check the On flag before running this, to use CPU vs. GPU.
func (gp *GPU) RunCycle() {
	if gp.RecFunTimes { // must use Wait calls here.
		gp.RunCycleSeparateFuns()
		return
	}
	if gp.CycleByCycle {
		gp.RunCycleOne()
		return
	}
	if gp.Ctx.Cycle%CyclesN == 0 {
		gp.RunCycles()
	}
}

// RunCycleOne does one cycle of updating in an optimized manner using Events to
// sequence each of the pipeline calls.  It is for CycleByCycle mode and syncs back
// full state every cycle.
func (gp *GPU) RunCycleOne() {
	cmd := gp.RunCycleOneCmd()
	gnm := "GPU:CycleOne"
	gp.Net.FunTimerStart(gnm)
	gp.CopyContextToStaging()
	gp.Sys.ComputeSubmitWaitCmd(cmd)
	gp.CopyStateFmStaging()
	gp.Net.FunTimerStop(gnm)
}

// RunCycleOneCmd returns commands to
// do one cycle of updating in an optimized manner using Events to
// sequence each of the pipeline calls.
// It is for CycleByCycle mode and syncs back full state every cycle.
func (gp *GPU) RunCycleOneCmd() vk.CommandBuffer {
	cnm := "RunCycleOne"
	if gp.Ctx.Testing.IsTrue() {
		cnm += "Testing"
	}
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	cxr := gp.SyncRegionStruct("Ctx")
	lvr := gp.SyncRegionStruct("LayVals")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")

	layDataN := gp.Net.NLayers() * int(gp.Net.MaxData)
	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)
	poolDataN := len(gp.Net.Pools)

	gp.StartRunCmd(cmd)
	gp.Sys.ComputeCmdCopyToGPUCmd(cmd, cxr) // staging -> GPU
	gp.Sys.ComputeSetEventCmd(cmd, "MemCopyTo")
	gp.RunPipelineCmd(cmd, "GatherSpikes", neurDataN, "MemCopyTo", "GatherSpikes")

	gp.RunPipelineCmd(cmd, "LayGi", layDataN, "GatherSpikes", "LayGi")
	gp.RunPipelineCmd(cmd, "BetweenGi", layDataN, "LayGi", "BetweenGi")
	gp.RunPipelineCmd(cmd, "PoolGi", poolDataN, "BetweenGi", "PoolGi")

	gp.RunPipelineCmd(cmd, "Cycle", neurDataN, "PoolGi", "Cycle")

	// todo: do over SendCon instead
	gp.RunPipelineCmd(cmd, "SendSpike", neurDataN, "Cycle", "SendSpike")
	if gp.Ctx.Testing.IsTrue() {
		gp.RunPipelineCmd(cmd, "CyclePost", 1, "SendSpike", "CycleEnd")
	} else {
		gp.RunPipelineCmd(cmd, "CyclePost", 1, "SendSpike", "CyclePost")
		gp.RunPipelineCmd(cmd, "SynCa", neurDataN, "CyclePost", "CycleEnd")
	}

	gp.Sys.ComputeWaitEventsCmd(cmd, "CycleEnd")
	gp.Sys.ComputeCmdCopyFmGPUCmd(cmd, cxr, lvr, plr, nrr)
	gp.Sys.ComputeCmdEndCmd(cmd)
	return cmd
}

// RunCycles does multiple cycles of updating in one chunk
func (gp *GPU) RunCycles() {
	cmd := gp.RunCyclesCmd()
	gnm := "GPU:Cycles"
	gp.Net.FunTimerStart(gnm)
	stCtx := *gp.Ctx // save starting state to restore below
	gp.CopyContextToStaging()
	gp.Sys.ComputeSubmitWaitCmd(cmd)
	gp.CopyLayerStateFmStaging()
	stCtx.NeuroMod = gp.Ctx.NeuroMod // only state that is updated separately -- counters are full ncyc ahead
	stCtx.PVLV = gp.Ctx.PVLV
	*gp.Ctx = stCtx
	gp.Net.FunTimerStop(gnm)
}

// RunCyclesCmd returns the RunCycles commands to
// do multiple cycles of updating in one chunk
func (gp *GPU) RunCyclesCmd() vk.CommandBuffer {
	cnm := "RunCycles"
	if gp.Ctx.Testing.IsTrue() {
		cnm += "Testing"
	}
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	cxr := gp.SyncRegionStruct("Ctx")
	lvr := gp.SyncRegionStruct("LayVals")
	plr := gp.SyncRegionStruct("Pools")

	layDataN := gp.Net.NLayers() * int(gp.Net.MaxData)
	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)
	poolDataN := len(gp.Net.Pools)

	gp.StartRunCmd(cmd)
	gp.Sys.ComputeCmdCopyToGPUCmd(cmd, cxr) // staging -> GPU
	gp.Sys.ComputeSetEventCmd(cmd, "MemCopyTo")
	gp.RunPipelineCmd(cmd, "GatherSpikes", neurDataN, "MemCopyTo", "GatherSpikes")

	for ci := 0; ci < CyclesN; ci++ {
		if ci > 0 {
			gp.RunPipelineCmd(cmd, "GatherSpikes", neurDataN, "CycleInc", "GatherSpikes")
		}

		gp.RunPipelineCmd(cmd, "LayGi", layDataN, "GatherSpikes", "LayGi")
		gp.RunPipelineCmd(cmd, "BetweenGi", layDataN, "LayGi", "BetweenGi")
		gp.RunPipelineCmd(cmd, "PoolGi", poolDataN, "BetweenGi", "PoolGi")

		gp.RunPipelineCmd(cmd, "Cycle", neurDataN, "PoolGi", "Cycle")

		gp.RunPipelineCmd(cmd, "SendSpike", neurDataN, "Cycle", "SendSpike")
		if gp.Ctx.Testing.IsTrue() {
			gp.RunPipelineCmd(cmd, "CyclePost", 1, "SendSpike", "CycleEnd")
		} else {
			gp.RunPipelineCmd(cmd, "CyclePost", 1, "SendSpike", "CyclePost")
			gp.RunPipelineCmd(cmd, "SynCa", neurDataN, "CyclePost", "CycleEnd")
		}
		if ci < CyclesN-1 {
			gp.RunPipelineCmd(cmd, "CycleInc", 1, "CycleEnd", "CycleInc") // we do
		}
	}
	gp.Sys.ComputeWaitEventsCmd(cmd, "CycleEnd")
	gp.Sys.ComputeCmdCopyFmGPUCmd(cmd, cxr, lvr, plr)
	gp.Sys.ComputeCmdEndCmd(cmd)
	return cmd
}

// RunCycleSeparateFuns does one cycle of updating in a very slow manner
// that allows timing to be recorded for each function call, for profiling.
func (gp *GPU) RunCycleSeparateFuns() {
	gp.SyncContextToGPU()

	layDataN := gp.Net.NLayers() * int(gp.Net.MaxData)
	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)
	poolDataN := len(gp.Net.Pools)

	gp.RunPipelineWait("GatherSpikes", neurDataN)

	gp.RunPipelineWait("LayGi", layDataN)
	gp.RunPipelineWait("BetweenGi", layDataN)
	gp.RunPipelineWait("PoolGi", poolDataN)

	gp.RunPipelineWait("Cycle", neurDataN)

	if gp.Ctx.Testing.IsTrue() {
		gp.RunPipelineWait("SendSpike", neurDataN)
		gp.RunPipelineWait("CyclePost", 1)
	} else {
		gp.RunPipelineWait("SendSpike", neurDataN)
		gp.RunPipelineWait("CyclePost", 1)
		gp.RunPipelineWait("SynCa", neurDataN)
	}
	gp.SyncLayerStateFmGPU()
}

// RunNewState runs the NewState shader to initialize state at start of new
// ThetaCycle trial.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunNewState() {
	poolDataN := len(gp.Net.Pools)
	gp.RunPipelineWait("NewState", poolDataN)
}

// RunMinusPhase runs the MinusPhase shader to update snapshot variables
// at the end of the minus phase.
// All non-synapse state is copied back down after this, so it is available
// for action calls
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunMinusPhase() {
	cmd := gp.RunMinusPhaseCmd()
	gnm := "GPU:MinusPhase"
	gp.Net.FunTimerStart(gnm)
	gp.CopyContextToStaging()
	gp.Sys.ComputeSubmitWaitCmd(cmd)
	gp.CopyStateFmStaging()
	gp.Net.FunTimerStop(gnm)
}

// RunMinusPhaseCmd returns the commands to
// run the MinusPhase shader to update snapshot variables
// at the end of the minus phase.
// All non-synapse state is copied back down after this, so it is available
// for action calls
func (gp *GPU) RunMinusPhaseCmd() vk.CommandBuffer {
	cnm := "RunMinusPhase"
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	cxr := gp.SyncRegionStruct("Ctx")
	lvr := gp.SyncRegionStruct("LayVals")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")

	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)
	poolDataN := len(gp.Net.Pools)

	gp.StartRunCmd(cmd)
	gp.Sys.ComputeCmdCopyToGPUCmd(cmd, cxr) // staging -> GPU
	gp.Sys.ComputeSetEventCmd(cmd, "MemCopyTo")
	gp.RunPipelineCmd(cmd, "MinusPool", poolDataN, "MemCopyTo", "PoolGi")
	gp.RunPipelineCmd(cmd, "MinusNeuron", neurDataN, "PoolGi", "MemCopyFm")
	gp.Sys.ComputeWaitEventsCmd(cmd, "MemCopyFm")
	gp.Sys.ComputeCmdCopyFmGPUCmd(cmd, cxr, lvr, plr, nrr)
	gp.Sys.ComputeCmdEndCmd(cmd)
	return cmd
}

// RunPlusPhaseStart runs the PlusPhaseStart shader
// does updating at the start of the plus phase:
// applies Target inputs as External inputs.
func (gp *GPU) RunPlusPhaseStart() {
	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)
	gp.RunPipelineWait("PlusStart", neurDataN)
}

// RunPlusPhase runs the PlusPhase shader to update snapshot variables
// and do additional stats-level processing at end of the plus phase.
// All non-synapse state is copied back down after this.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunPlusPhase() {
	cmd := gp.RunPlusPhaseCmd()
	gnm := "GPU:PlusPhase"
	gp.Net.FunTimerStart(gnm)
	gp.CopyContextToStaging()
	gp.Sys.ComputeSubmitWaitCmd(cmd)
	gp.CopyStateFmStaging()
	gp.Net.FunTimerStop(gnm)
}

// RunPlusPhaseCmd returns the commands to
// run the PlusPhase shader to update snapshot variables
// and do additional stats-level processing at end of the plus phase.
// All non-synapse state is copied back down after this.
func (gp *GPU) RunPlusPhaseCmd() vk.CommandBuffer {
	cnm := "RunPlusPhase"
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	cxr := gp.SyncRegionStruct("Ctx")
	lvr := gp.SyncRegionStruct("LayVals")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")

	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)
	poolDataN := len(gp.Net.Pools)

	gp.StartRunCmd(cmd)
	gp.Sys.ComputeCmdCopyToGPUCmd(cmd, cxr) // staging -> GPU
	gp.Sys.ComputeSetEventCmd(cmd, "MemCopyTo")
	gp.RunPipelineCmd(cmd, "PlusPool", poolDataN, "MemCopyTo", "PoolGi")
	gp.RunPipelineCmd(cmd, "PlusNeuron", neurDataN, "PoolGi", "MemCopyFm")

	// note: could use atomic add to accumulate CorSim stat values in LayVals tmp vars for Cosv, ssm and ssp
	// from which the overall val is computed
	// use float atomic add for this case b/c not so time critical
	// also, matrix gated could be computed all on GPU without too much difficulty.
	// this would put all standard computation on the GPU for entire ThetaCycle

	gp.Sys.ComputeWaitEventsCmd(cmd, "MemCopyFm")
	gp.Sys.ComputeCmdCopyFmGPUCmd(cmd, cxr, lvr, plr, nrr)
	gp.Sys.ComputeCmdEndCmd(cmd)
	return cmd
}

// RunDWt runs the DWt shader to compute weight changes.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunDWt() {
	cmd := gp.RunDWtCmd()
	gnm := "GPU:DWt"
	gp.Net.FunTimerStart(gnm)
	gp.Sys.ComputeSubmitWaitCmd(cmd)
	gp.Net.FunTimerStop(gnm)
}

// RunDWtCmd returns the commands to run the DWt shader
// to compute weight changes.
func (gp *GPU) RunDWtCmd() vk.CommandBuffer {
	cnm := "RunDWt"
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	// note: not * MaxData
	synN := int(gp.Net.NSyns)
	synDataN := int(gp.Net.NSyns) * int(gp.Net.MaxData)

	gp.StartRunCmd(cmd)
	gp.RunPipelineCmd(cmd, "DWt", synDataN, "", "LayGi")
	gp.RunPipelineCmd(cmd, "DWtFmDi", synN, "LayGi", "")
	gp.Sys.ComputeCmdEndCmd(cmd)
	return cmd
}

// RunWtFmDWt runs the WtFmDWt shader to update weights from weigh changes.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunWtFmDWt() {
	cmd := gp.RunWtFmDWtCmd()
	gnm := "GPU:WtFmDWt"
	gp.Net.FunTimerStart(gnm)
	gp.CopyNeuronsToStaging()
	gp.Sys.ComputeSubmitWaitCmd(cmd)
	gp.Net.FunTimerStop(gnm)
}

// RunWtFmDWtCmd returns the commands to
// run the WtFmDWt shader to update weights from weight changes.
// This also syncs neuron state from CPU -> GPU because TrgAvgFmD
// has updated that state.
func (gp *GPU) RunWtFmDWtCmd() vk.CommandBuffer {
	cnm := "RunWtFmDWt"
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	nrr := gp.SyncRegionStruct("Neurons")

	// note: not * MaxData
	synN := int(gp.Net.NSyns)
	neurN := int(gp.Net.NNeurons)

	gp.StartRunCmd(cmd)
	gp.Sys.ComputeCmdCopyToGPUCmd(cmd, nrr)                   // staging -> GPU
	gp.RunPipelineCmd(cmd, "DWtSubMean", neurN, "", "PoolGi") // using poolgi for kicks
	gp.RunPipelineCmd(cmd, "WtFmDWt", synN, "PoolGi", "")
	gp.Sys.ComputeCmdEndCmd(cmd)
	return cmd
}
