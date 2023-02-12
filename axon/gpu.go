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

//go:generate gosl -exclude=Update,UpdateParams,Defaults,AllParams github.com/goki/mat32/fastexp.go github.com/emer/etable/minmax ../chans/chans.go ../chans ../kinase ../fsfffb/inhib.go ../fsfffb github.com/emer/emergent/etime github.com/emer/emergent/ringidx rand.go avgmax.go neuromod.go context.go neuron.go synapse.go pool.go layervals.go act.go act_prjn.go inhib.go learn.go layertypes.go layerparams.go deep_layers.go rl_layers.go pvlv_layers.go pcore_layers.go prjntypes.go prjnparams.go deep_prjns.go rl_prjns.go pvlv_prjns.go pcore_prjns.go gpu_hlsl

// Full vars code -- each gpu_*.hlsl uses a subset

/*

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
[[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][RecvPrjns]
[[vk::binding(1, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]
[[vk::binding(2, 1)]] StructuredBuffer<uint> SendPrjnIdxs; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(3, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(4, 1)]] StructuredBuffer<uint> SendSynIdxs; // [Layer][SendPrjns][SendNeurons][Syns]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
[[vk::binding(5, 2)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
[[vk::binding(6, 2)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs
[[vk::binding(0, 3)]] RWStructuredBuffer<float> Exts;  // [In / Out Layers][Neurons]


Set: 0
    Role: Uniform
        Var: 0:	Layers	Struct[4]	(size: 1280)	Vals: 1
Set: 1
    Role: Storage
        Var: 0:	Prjns	Struct[5]	(size: 336)	Vals: 1
        Var: 1:	RecvCon	Struct[281]	(size: 16)	Vals: 1
        Var: 2:	SendPrjnIdxs	Uint32[5]	(size: 4)	Vals: 1
        Var: 3:	SendCon	Struct[242]	(size: 16)	Vals: 1
        Var: 4:	SendSynIdxs	Uint32[12992]	(size: 4)	Vals: 1
Set: 2
    Role: Storage
        Var: 0:	Ctxt	Struct	(size: 112)	Vals: 1
        Var: 1:	Neurons	Struct[178]	(size: 368)	Vals: 1
        Var: 2:	Pools	Struct[4]	(size: 720)	Vals: 1
        Var: 3:	LayVals	Struct[4]	(size: 112)	Vals: 1
        Var: 4:	Synapses	Struct[12992]	(size: 64)	Vals: 1
        Var: 5:	GBuf	Int32[843]	(size: 4)	Vals: 1
        Var: 6:	GSyns	Float32[281]	(size: 4)	Vals: 1
Set: 3
    Role: Storage
        Var: 0:	Exts	Float32[50]	(size: 4)	Vals: 1
*/

// TheGPU is the gpu device, shared across all networks
var TheGPU *vgpu.GPU

// GPU manages all of the GPU-based computation
type GPU struct {
	On           bool `desc:"if true, actually use the GPU"`
	RecFunTimes  bool `desc:"if true, slower separate shader pipeline runs are used, with a CPU-sync Wait at the end, to enable timing information about each individual shader to be collected using the network FunTimer system.  otherwise, only aggregate information is available about the entire Cycle call.`
	CycleByCycle bool `desc:"if true, process each cycle one at a time.  Otherwise, 10 cycles at a time are processed in one batch."`

	Net        *Network                `view:"-" desc:"the network we operate on -- we live under this net"`
	Ctx        *Context                `view:"-" desc:"the context we use"`
	Sys        *vgpu.System            `view:"-" desc:"the vgpu compute system"`
	Params     *vgpu.VarSet            `view:"-" desc:"VarSet = 0: the uniform LayerParams"`
	Prjns      *vgpu.VarSet            `view:"-" desc:"VarSet = 1: the storage PrjnParams, RecvCon, Send*"`
	Structs    *vgpu.VarSet            `view:"-" desc:"VarSet = 2: the Storage buffer for RW state structs "`
	Exts       *vgpu.VarSet            `view:"-" desc:"Varset = 3: the Storage buffer for external inputs -- sync frequently"`
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
	if vgpu.InitNoDisplay() != nil {
		return
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

	ctx.NLayers = int32(gp.Net.NLayers())
	gp.DidBind = make(map[string]bool)

	if TheGPU == nil {
		TheGPU = vgpu.NewComputeGPU()
		// vgpu.Debug = true
		TheGPU.Config("axon")
	}

	gp.Sys = TheGPU.NewComputeSystem("axon")

	vars := gp.Sys.Vars()
	gp.Params = vars.AddSet()
	gp.Prjns = vars.AddSet()
	gp.Structs = vars.AddSet()
	gp.Exts = vars.AddSet()

	gp.Params.AddStruct("Layers", int(unsafe.Sizeof(LayerParams{})), len(gp.Net.LayParams), vgpu.Uniform, vgpu.ComputeShader)

	// note: prjns must be in Storage here because couldn't have both Layers and Prjns as uniform.
	gp.Prjns.AddStruct("Prjns", int(unsafe.Sizeof(PrjnParams{})), len(gp.Net.PrjnParams), vgpu.Storage, vgpu.ComputeShader)
	gp.Prjns.AddStruct("RecvCon", int(unsafe.Sizeof(StartN{})), len(gp.Net.PrjnRecvCon), vgpu.Storage, vgpu.ComputeShader)
	gp.Prjns.Add("SendPrjnIdxs", vgpu.Uint32, len(gp.Net.SendPrjnIdxs), vgpu.Storage, vgpu.ComputeShader)
	gp.Prjns.AddStruct("SendCon", int(unsafe.Sizeof(StartN{})), len(gp.Net.PrjnSendCon), vgpu.Storage, vgpu.ComputeShader)
	gp.Prjns.Add("SendSynIdxs", vgpu.Uint32, len(gp.Net.SendSynIdxs), vgpu.Storage, vgpu.ComputeShader)

	gp.Structs.AddStruct("Ctx", int(unsafe.Sizeof(Context{})), 1, vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("Neurons", int(unsafe.Sizeof(Neuron{})), len(gp.Net.Neurons), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("Pools", int(unsafe.Sizeof(Pool{})), len(gp.Net.Pools), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("LayVals", int(unsafe.Sizeof(LayerVals{})), len(gp.Net.LayVals), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("Synapses", int(unsafe.Sizeof(Synapse{})), len(gp.Net.Synapses), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("GBuf", vgpu.Int32, len(gp.Net.PrjnGBuf), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("GSyns", vgpu.Float32, len(gp.Net.PrjnGSyns), vgpu.Storage, vgpu.ComputeShader)

	gp.Exts.Add("Exts", vgpu.Float32, len(gp.Net.Exts), vgpu.Storage, vgpu.ComputeShader)

	gp.Params.ConfigVals(1)
	gp.Prjns.ConfigVals(1)
	gp.Structs.ConfigVals(1)
	gp.Exts.ConfigVals(1)

	// pipelines
	gp.Sys.NewComputePipelineEmbed("GatherSpikes", content, "shaders/gpu_gather.spv")
	gp.Sys.NewComputePipelineEmbed("LayGi", content, "shaders/gpu_laygi.spv")
	gp.Sys.NewComputePipelineEmbed("BetweenGi", content, "shaders/gpu_betweengi.spv")
	gp.Sys.NewComputePipelineEmbed("PoolGi", content, "shaders/gpu_poolgi.spv")
	gp.Sys.NewComputePipelineEmbed("Cycle", content, "shaders/gpu_cycle.spv")
	gp.Sys.NewComputePipelineEmbed("CycleInc", content, "shaders/gpu_cycleinc.spv")
	gp.Sys.NewComputePipelineEmbed("SendSpike", content, "shaders/gpu_sendspike.spv")
	gp.Sys.NewComputePipelineEmbed("SynCa", content, "shaders/gpu_synca.spv")
	gp.Sys.NewComputePipelineEmbed("SynCaRecv", content, "shaders/gpu_syncarecv.spv")
	gp.Sys.NewComputePipelineEmbed("SynCaSend", content, "shaders/gpu_syncasend.spv")
	gp.Sys.NewComputePipelineEmbed("CyclePost", content, "shaders/gpu_cyclepost.spv")

	gp.Sys.NewComputePipelineEmbed("NewState", content, "shaders/gpu_newstate.spv")
	gp.Sys.NewComputePipelineEmbed("MinusPool", content, "shaders/gpu_minuspool.spv")
	gp.Sys.NewComputePipelineEmbed("MinusNeuron", content, "shaders/gpu_minusneuron.spv")
	gp.Sys.NewComputePipelineEmbed("PlusPool", content, "shaders/gpu_pluspool.spv")
	gp.Sys.NewComputePipelineEmbed("PlusNeuron", content, "shaders/gpu_plusneuron.spv")
	gp.Sys.NewComputePipelineEmbed("DWt", content, "shaders/gpu_dwt.spv")
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
	gp.Sys.NewEvent("SynCaSend")

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

	vars.BindDynValIdx(1, "Prjns", 0)
	vars.BindDynValIdx(1, "RecvCon", 0)
	vars.BindDynValIdx(1, "SendPrjnIdxs", 0)
	vars.BindDynValIdx(1, "SendCon", 0)
	vars.BindDynValIdx(1, "SendSynIdxs", 0)

	vars.BindDynValIdx(2, "Ctx", 0)
	vars.BindDynValIdx(2, "Neurons", 0)
	vars.BindDynValIdx(2, "Pools", 0)
	vars.BindDynValIdx(2, "LayVals", 0)
	vars.BindDynValIdx(2, "Synapses", 0)
	vars.BindDynValIdx(2, "GBuf", 0)
	vars.BindDynValIdx(2, "GSyns", 0)

	vars.BindDynValIdx(3, "Exts", 0)
}

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

	_, pjnv, _ := gp.Prjns.ValByIdxTry("Prjns", 0)
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
	_, rconv, _ := gp.Prjns.ValByIdxTry("RecvCon", 0)
	rconv.CopyFromBytes(unsafe.Pointer(&gp.Net.PrjnRecvCon[0]))

	_, spiv, _ := gp.Prjns.ValByIdxTry("SendPrjnIdxs", 0)
	spiv.CopyFromBytes(unsafe.Pointer(&gp.Net.SendPrjnIdxs[0]))

	_, sconv, _ := gp.Prjns.ValByIdxTry("SendCon", 0)
	sconv.CopyFromBytes(unsafe.Pointer(&gp.Net.PrjnSendCon[0]))

	_, ssiv, _ := gp.Prjns.ValByIdxTry("SendSynIdxs", 0)
	ssiv.CopyFromBytes(unsafe.Pointer(&gp.Net.SendSynIdxs[0]))
}

// CopyExtsToStaging copies external inputs to staging from CPU.
// Typically used in RunApplyExts which also does the Sync.
func (gp *GPU) CopyExtsToStaging() {
	if !gp.On {
		return
	}
	_, extv, _ := gp.Exts.ValByIdxTry("Exts", 0)
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

// CopyStateToStaging copies Neurons, Pools, and LayVals state to staging from CPU.
// this is typically sufficient for most syncing --
// only missing the Synapses which must be copied separately.
// Must call SyncMemToGPU after this (see SyncStateToGPU).
func (gp *GPU) CopyStateToStaging() {
	if !gp.On {
		return
	}
	gp.CopyNeuronsToStaging()
	gp.CopyLayerValsToStaging()
	gp.CopyPoolsToStaging()
}

// SyncStateToGPU copies Neurons, Pools, and LayVals state to GPU
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

// SyncAllToGPU copies Neurons, Pools, LayVals, Synapses to GPU.
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
	_, synv, _ := gp.Structs.ValByIdxTry("Synapses", 0)
	synv.CopyFromBytes(unsafe.Pointer(&gp.Net.Synapses[0]))
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

///////////////////////////////////////////////////////////////////////
// 	Sync From

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
	cxr, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Ctx", 0)
	if err != nil {
		log.Println(err)
		return
	}
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
	lvl, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "LayVals", 0)
	if err != nil {
		log.Println(err)
		return
	}
	gp.Sys.Mem.SyncStorageRegionsFmGPU(lvl)
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
	pl, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Pools", 0)
	if err != nil {
		log.Println(err)
		return
	}
	gp.Sys.Mem.SyncStorageRegionsFmGPU(pl)
	gp.CopyPoolsFmStaging()
}

// CopyNeuronsFmStaging copies Neurons from staging to CPU, after Sync back down.
func (gp *GPU) CopyNeuronsFmStaging() {
	if !gp.On {
		return
	}
	_, neurv, _ := gp.Structs.ValByIdxTry("Neurons", 0)
	neurv.CopyToBytes(unsafe.Pointer(&gp.Net.Neurons[0]))
}

// SyncNeuronsFmGPU copies Neurons from GPU to CPU.
// Use only when only thing being copied -- more efficient to get all at once.
// e.g. see SyncStateFmGPU
func (gp *GPU) SyncNeuronsFmGPU() {
	if !gp.On {
		return
	}
	nrn, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Neurons", 0)
	if err != nil {
		log.Println(err)
		return
	}
	gp.Sys.Mem.SyncStorageRegionsFmGPU(nrn)
	gp.CopyNeuronsFmStaging()
}

// CopySynapsesFmStaging copies Synapses from staging to CPU, after Sync back down.
func (gp *GPU) CopySynapsesFmStaging() {
	if !gp.On {
		return
	}
	_, synv, _ := gp.Structs.ValByIdxTry("Synapses", 0)
	synv.CopyToBytes(unsafe.Pointer(&gp.Net.Synapses[0]))
}

// SyncSynapsesFmGPU copies Synapses from GPU to CPU.
// Use only when only thing being copied -- more efficient to get all at once.
func (gp *GPU) SyncSynapsesFmGPU() {
	if !gp.On {
		return
	}
	syn, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Synapses", 0)
	if err != nil {
		log.Println(err)
		return
	}
	gp.Sys.Mem.SyncStorageRegionsFmGPU(syn)
	gp.CopySynapsesFmStaging()
}

// CopyStateFmStaging copies Neurons, LayerVals, and Pools from staging to CPU, after Sync.
func (gp *GPU) CopyStateFmStaging() {
	gp.CopyNeuronsFmStaging()
	gp.CopyLayerValsFmStaging()
	gp.CopyPoolsFmStaging()
}

// SyncStateFmCPU copies Neurons, LayerVals, and Pools from GPU to CPU.
// This is the main GPU->CPU sync step automatically called in PlusPhase.
func (gp *GPU) SyncStateFmGPU() {
	if !gp.On {
		return
	}
	nrn, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Neurons", 0)
	if err != nil {
		log.Println(err)
		return
	}
	lvl, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "LayVals", 0)
	if err != nil {
		log.Println(err)
		return
	}
	pl, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Pools", 0)
	if err != nil {
		log.Println(err)
		return
	}
	gp.Sys.Mem.SyncStorageRegionsFmGPU(nrn, lvl, pl)
	gp.CopyStateFmStaging()
}

// SyncAllFmCPU copies Neurons, LayerVals, Pools, and Synapses from GPU to CPU.
// This is the main GPU->CPU sync step automatically called in PlusPhase.
func (gp *GPU) SyncAllFmGPU() {
	if !gp.On {
		return
	}
	nrn, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Neurons", 0)
	if err != nil {
		log.Println(err)
		return
	}
	lvl, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "LayVals", 0)
	if err != nil {
		log.Println(err)
		return
	}
	pl, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Pools", 0)
	if err != nil {
		log.Println(err)
		return
	}
	syn, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Synapses", 0)
	if err != nil {
		log.Println(err)
		return
	}
	gp.Sys.Mem.SyncStorageRegionsFmGPU(nrn, lvl, pl, syn)
	gp.CopyStateFmStaging()
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

// RunPipelineWaitSignal records command to run given pipeline with
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

// RunApplyExts copies Exts external input memory to the GPU and then
// runs the ApplyExts shader that applies those external inputs to the
// GPU-side neuron state.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunApplyExts() {
	gnm := "GPU:ApplyExts"
	gp.Net.FunTimerStart(gnm)
	gp.CopyExtsToStaging()
	exr, err := gp.Sys.Mem.SyncRegionValIdx(gp.Exts.Set, "Exts", 0)
	if err != nil {
		log.Println(err)
		return
	}
	gp.StartRun()
	gp.Sys.ComputeCmdCopyToGPU(exr)
	gp.Sys.ComputeSetEvent("MemCopyTo")
	gp.RunPipeline("ApplyExts", len(gp.Net.Neurons), "MemCopyTo", "")
	gp.Sys.ComputeSubmitWait()
	gp.Net.FunTimerStop(gnm)
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
	if gp.Ctx.Cycle%10 == 0 { // for ra25, 10 = ~50 msec / trial, 25 = ~48, all 150 / 50 minus / plus = ~44
		// 10 is good enough and unlikely to mess with anything else..
		gp.RunCycles(10)
	}
}

// RunCycleOne does one cycle of updating in an optimized manner using Events to
// sequence each of the pipeline calls.
func (gp *GPU) RunCycleOne() {
	gnm := "GPU:CycleOne"
	gp.Net.FunTimerStart(gnm)
	cxr, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Ctx", 0)
	if err != nil {
		log.Println(err)
		return
	}
	lvr, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "LayVals", 0)
	if err != nil {
		log.Println(err)
		return
	}

	gp.CopyContextToStaging()
	gp.StartRun()
	gp.Sys.ComputeCmdCopyToGPU(cxr) // staging -> GPU
	gp.Sys.ComputeSetEvent("MemCopyTo")
	gp.RunPipeline("GatherSpikes", len(gp.Net.Neurons), "MemCopyTo", "GatherSpikes")

	gp.RunPipeline("LayGi", len(gp.Net.Layers), "GatherSpikes", "LayGi")
	gp.RunPipeline("BetweenGi", len(gp.Net.Layers), "LayGi", "BetweenGi")
	gp.RunPipeline("PoolGi", len(gp.Net.Pools), "BetweenGi", "PoolGi")

	gp.RunPipeline("Cycle", len(gp.Net.Neurons), "PoolGi", "Cycle")

	if gp.Ctx.Testing.IsTrue() {
		gp.RunPipeline("SendSpike", len(gp.Net.Neurons), "Cycle", "SendSpike")
		gp.RunPipeline("CyclePost", 1, "SendSpike", "CycleEnd")
	} else {
		gp.RunPipeline("SendSpike", len(gp.Net.Neurons), "Cycle", "SendSpike")
		gp.RunPipeline("CyclePost", 1, "SendSpike", "CyclePost")
		gp.RunPipeline("SynCaSend", len(gp.Net.Neurons), "CyclePost", "SynCaSend")
		// use send first b/c just did SendSpike -- tiny bit faster
		gp.RunPipeline("SynCaRecv", len(gp.Net.Neurons), "SynCaSend", "CycleEnd")
	}

	gp.Sys.ComputeWaitEvents("CycleEnd")
	gp.Sys.ComputeCmdCopyFmGPU(cxr, lvr) // ctx updated in cycle post

	gp.Sys.ComputeSubmitWait()
	gp.CopyLayerValsFmStaging()
	gp.CopyContextFmStaging()
	gp.Net.FunTimerStop(gnm)
}

// RunCycles does multiple cycles of updating in one chunk
func (gp *GPU) RunCycles(ncyc int) {
	gnm := "GPU:Cycles"
	gp.Net.FunTimerStart(gnm)
	cxr, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Ctx", 0)
	if err != nil {
		log.Println(err)
		return
	}
	lvr, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "LayVals", 0)
	if err != nil {
		log.Println(err)
		return
	}

	stCtx := *gp.Ctx // save starting state to restore below

	gp.CopyContextToStaging()
	gp.StartRun()
	gp.Sys.ComputeCmdCopyToGPU(cxr) // staging -> GPU
	gp.Sys.ComputeSetEvent("MemCopyTo")
	gp.RunPipeline("GatherSpikes", len(gp.Net.Neurons), "MemCopyTo", "GatherSpikes")

	for ci := 0; ci < ncyc; ci++ {
		if ci > 0 {
			gp.RunPipeline("GatherSpikes", len(gp.Net.Neurons), "CycleInc", "GatherSpikes")
		}

		gp.RunPipeline("LayGi", len(gp.Net.Layers), "GatherSpikes", "LayGi")
		gp.RunPipeline("BetweenGi", len(gp.Net.Layers), "LayGi", "BetweenGi")
		gp.RunPipeline("PoolGi", len(gp.Net.Pools), "BetweenGi", "PoolGi")

		gp.RunPipeline("Cycle", len(gp.Net.Neurons), "PoolGi", "Cycle")

		if gp.Ctx.Testing.IsTrue() {
			gp.RunPipeline("SendSpike", len(gp.Net.Neurons), "Cycle", "SendSpike")
			gp.RunPipeline("CyclePost", 1, "SendSpike", "CycleEnd")
		} else {
			gp.RunPipeline("SendSpike", len(gp.Net.Neurons), "Cycle", "SendSpike")
			gp.RunPipeline("CyclePost", 1, "SendSpike", "CyclePost")
			gp.RunPipeline("SynCaSend", len(gp.Net.Neurons), "CyclePost", "SynCaSend")
			// use send first b/c just did SendSpike -- tiny bit faster
			gp.RunPipeline("SynCaRecv", len(gp.Net.Neurons), "SynCaSend", "CycleEnd")
		}
		if ci < ncyc-1 {
			gp.RunPipeline("CycleInc", len(gp.Net.Neurons), "CycleEnd", "CycleInc") // we do
		}
	}
	gp.Sys.ComputeWaitEvents("CycleEnd")
	gp.Sys.ComputeCmdCopyFmGPU(cxr, lvr)
	gp.Sys.ComputeSubmitWait()

	gp.CopyLayerValsFmStaging()
	gp.CopyContextFmStaging()        // gp.Ctx is now in the future
	stCtx.NeuroMod = gp.Ctx.NeuroMod // only state that is updated separately
	*gp.Ctx = stCtx

	gp.Net.FunTimerStop(gnm)
}

// RunCycleSeparateFuns does one cycle of updating in a very slow manner
// that allows timing to be recorded for each function call, for profiling.
func (gp *GPU) RunCycleSeparateFuns() {
	gp.SyncContextToGPU()

	gp.RunPipelineWait("GatherSpikes", len(gp.Net.Neurons))

	gp.RunPipelineWait("LayGi", len(gp.Net.Layers))
	gp.RunPipelineWait("BetweenGi", len(gp.Net.Layers))
	gp.RunPipelineWait("PoolGi", len(gp.Net.Pools))

	gp.RunPipelineWait("Cycle", len(gp.Net.Neurons))

	if gp.Ctx.Testing.IsTrue() {
		gp.RunPipelineWait("SendSpike", len(gp.Net.Neurons))
		gp.RunPipelineWait("CyclePost", 1)
	} else {
		gp.RunPipelineWait("SendSpike", len(gp.Net.Neurons))
		gp.RunPipelineWait("CyclePost", 1)
		gp.RunPipelineWait("SynCaSend", len(gp.Net.Neurons))
		gp.RunPipelineWait("SynCaRecv", len(gp.Net.Neurons))
	}
	gp.SyncLayerValsFmGPU() // only thing we get back
}

// RunNewState runs the NewState shader to initialize state at start of new
// ThetaCycle trial.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunNewState() {
	gp.RunPipelineWait("NewState", len(gp.Net.Pools))
}

// RunMinusPhase runs the MinusPhase shader to update snapshot variables
// at the end of the minus phase.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunMinusPhase() {
	gnm := "GPU:MinusPhase"
	gp.Net.FunTimerStart(gnm)
	gp.StartRun()
	gp.RunPipeline("MinusPool", len(gp.Net.Pools), "", "PoolGi")
	gp.RunPipeline("MinusNeuron", len(gp.Net.Neurons), "PoolGi", "")
	gp.Sys.ComputeSubmitWait()
	gp.Net.FunTimerStop(gnm)
}

// RunPlusPhase runs the PlusPhase shader to update snapshot variables
// and do additional stats-level processing at end of the plus phase.
// All non-synapse state is copied back down after this.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunPlusPhase() {
	gnm := "GPU:PlusPhase"
	gp.Net.FunTimerStart(gnm)
	nrn, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Neurons", 0)
	if err != nil {
		log.Println(err)
		return
	}
	lvl, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "LayVals", 0)
	if err != nil {
		log.Println(err)
		return
	}
	pl, err := gp.Sys.Mem.SyncRegionValIdx(gp.Structs.Set, "Pools", 0)
	if err != nil {
		log.Println(err)
		return
	}
	gp.StartRun()
	gp.RunPipeline("PlusPool", len(gp.Net.Pools), "", "PoolGi")
	gp.RunPipeline("PlusNeuron", len(gp.Net.Neurons), "PoolGi", "MemCopyFm")
	gp.Sys.ComputeWaitEvents("MemCopyFm")
	gp.Sys.ComputeCmdCopyFmGPU(nrn, lvl, pl)
	gp.Sys.ComputeSubmitWait()
	gp.CopyStateFmStaging()
	gp.Net.FunTimerStop(gnm)
}

// RunDWt runs the DWt shader to compute weight changes.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunDWt() {
	gp.RunPipelineWait("DWt", len(gp.Net.Synapses))
}

// RunWtFmDWt runs the WtFmDWt shader to update weights from weigh changes.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunWtFmDWt() {
	gnm := "GPU:WtFmDWt"
	gp.Net.FunTimerStart(gnm)
	gp.StartRun()
	gp.RunPipeline("DWtSubMean", len(gp.Net.Neurons), "", "PoolGi")
	gp.RunPipeline("WtFmDWt", len(gp.Net.Synapses), "PoolGi", "")
	gp.Sys.ComputeSubmitWait()
	gp.Net.FunTimerStop(gnm)
}
