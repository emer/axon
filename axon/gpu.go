// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"embed"
	"unsafe"

	"github.com/goki/gi/oswin"
	"github.com/goki/vgpu/vgpu"
)

//go:embed shaders/*.spv
var content embed.FS

//go:generate gosl -exclude=Update,UpdateParams,Defaults,AllParams github.com/goki/mat32/fastexp.go github.com/emer/etable/minmax ../chans/chans.go ../chans ../kinase ../fsfffb/inhib.go ../fsfffb github.com/emer/emergent/etime github.com/emer/emergent/ringidx neuromod.go context.go neuron.go synapse.go pool.go layervals.go act.go act_prjn.go inhib.go learn.go layertypes.go layerparams.go deep_layers.go rl_layers.go pvlv_layers.go pcore_layers.go prjntypes.go prjnparams.go deep_prjns.go rl_prjns.go pvlv_prjns.go pcore_prjns.go gpu_hlsl

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
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
[[vk::binding(5, 2)]] RWStructuredBuffer<float> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
[[vk::binding(6, 2)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs
[[vk::binding(0, 3)]] RWStructuredBuffer<float> Exts;  // [In / Out Layers][Neurons]


Set: 0
    Role: Uniform
        Var: 0:	Layers	Struct[3]	(size: 1264)	Vals: 1
Set: 1
    Role: Storage
        Var: 0:	Prjns	Struct[2]	(size: 320)	Vals: 1
        Var: 1:	RecvCon	Struct[8]	(size: 16)	Vals: 1
Set: 2
    Role: Storage
        Var: 0:	Ctxt	Struct	(size: 112)	Vals: 1
        Var: 1:	Neurons	Struct[12]	(size: 368)	Vals: 1
        Var: 2:	Pools	Struct[3]	(size: 704)	Vals: 1
        Var: 3:	LayVals	Struct[3]	(size: 112)	Vals: 1
        Var: 4:	Synapses	Struct[8]	(size: 64)	Vals: 1
        Var: 5:	GBuf	Float32[24]	(size: 4)	Vals: 1
        Var: 6:	GSyns	Float32[8]	(size: 4)	Vals: 1
Set: 3
    Role: Storage
        Var: 0:	Exts	Float32[8]	(size: 4)	Vals: 1
*/

/*
cycle update:

GatherSpikes & RecvSpikes:
gpu_gather.hlsl 	[Neurons]

GiFmSpikes:
gpu_poolgemax.hlsl [Pools] -- does GeToPool and AvgMax Update, calls LayPoolGiFmSpikes
gpu_poolgi.hlsl 	  [Pools] -- only operates on sub-Pools, does SubPoolGiFmSpikes

CycleNeuron:
gpu_cycle.hlsl 	[Neurons]

gpu_sendspike.hlsl	[Neurons]

gpu_synca.hlsl		[Synapses]
// gpu_syncarecv.hlsl	[Neurons]
// gpu_syncasend.hlsl	[Neurons]

DWt:
gpu_dwt.hlsl		[Synapses]
todo: submean
gpu_wtfmdwt.hlsl	[Synapses]

todo: plus phase!

todo: need apply inputs tensors!

*/

// TheGPU is the gpu device, shared across all networks
var TheGPU *vgpu.GPU

// GPU manages all of the GPU-based computation
type GPU struct {
	On           bool           `desc:"if true, actually use the GPU"`
	Sys          *vgpu.System   `desc:"the vgpu compute system"`
	Params       *vgpu.VarSet   `desc:"VarSet = 0: the uniform LayerParams"`
	Prjns        *vgpu.VarSet   `desc:"VarSet = 1: the storage PrjnParams, RecvCon, Send*"`
	Structs      *vgpu.VarSet   `desc:"VarSet = 2: the Storage buffer for RW state structs "`
	Exts         *vgpu.VarSet   `desc:"Varset = 3: the Storage buffer for external inputs -- sync frequently"`
	GatherSpikes *vgpu.Pipeline `desc:"GatherSpikes pipeline"`
	PoolGeMax    *vgpu.Pipeline `desc:"PoolGeMax pipeline"`
	BetweenGi    *vgpu.Pipeline `desc:"BetweenGi pipeline"`
	PoolGi       *vgpu.Pipeline `desc:"PoolG pipeline"`
	Cycle        *vgpu.Pipeline `desc:"Cycle pipeline"`
	SendSpike    *vgpu.Pipeline `desc:"SendSpike pipeline"`
	SynCa        *vgpu.Pipeline `desc:"SynCa pipeline"`
	SynCaRecv    *vgpu.Pipeline `desc:"SynCa pipeline"`
	SynCaSend    *vgpu.Pipeline `desc:"SynCa pipeline"`
	NewState     *vgpu.Pipeline `desc:"new state pipeline"`
	MinusPhase   *vgpu.Pipeline `desc:"minus phase pipeline"`
	PlusPhase    *vgpu.Pipeline `desc:"plus phase pipeline"`
	DWt          *vgpu.Pipeline `desc:"DWt pipeline"`
	WtFmDWt      *vgpu.Pipeline `desc:"WtFmDWt pipeline"`
	ApplyExts    *vgpu.Pipeline `desc:"ApplyExts pipeline"`
	NThreads     int            `def:"64" desc:"number of warp threads -- typically 64 -- must update all hlsl files if changed!"`
}

// GPUOnGUI turns on GPU mode in context of GUI active, configures the GPU -- call after all built,
// initialized, params are set, and ready to run
func (nt *Network) GPUOnGUI(ctx *Context) {
	oswin.TheApp.RunOnMain(func() {
		nt.GPU.Config(ctx, nt)
	})
	nt.GPU.On = true
}

// GPUOnNoGUI turns on GPU mode in context of NO GUI active,
// configures the GPU -- call after all built,
// initialized, params are set, and ready to run
func (nt *Network) GPUOnNoGUI(ctx *Context) {
	if vgpu.InitNoDisplay() != nil {
		return
	}
	nt.GPU.Config(ctx, nt)
	nt.GPU.On = true
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
	gp.NThreads = 64

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

	gp.Params.AddStruct("Layers", int(unsafe.Sizeof(LayerParams{})), len(net.LayParams), vgpu.Uniform, vgpu.ComputeShader)

	// note: prjns must be in Storage here because couldn't have both Layers and Prjns as uniform.
	gp.Prjns.AddStruct("Prjns", int(unsafe.Sizeof(PrjnParams{})), len(net.PrjnParams), vgpu.Storage, vgpu.ComputeShader)
	gp.Prjns.AddStruct("RecvCon", int(unsafe.Sizeof(StartN{})), len(net.PrjnRecvCon), vgpu.Storage, vgpu.ComputeShader)
	gp.Prjns.Add("SendPrjnIdxs", vgpu.Uint32, len(net.SendPrjnIdxs), vgpu.Storage, vgpu.ComputeShader)
	gp.Prjns.AddStruct("SendCon", int(unsafe.Sizeof(StartN{})), len(net.PrjnSendCon), vgpu.Storage, vgpu.ComputeShader)
	gp.Prjns.Add("SendSynIdxs", vgpu.Uint32, len(net.SendSynIdxs), vgpu.Storage, vgpu.ComputeShader)

	gp.Structs.AddStruct("Ctxt", int(unsafe.Sizeof(Context{})), 1, vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("Neurons", int(unsafe.Sizeof(Neuron{})), len(net.Neurons), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("Pools", int(unsafe.Sizeof(Pool{})), len(net.Pools), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("LayVals", int(unsafe.Sizeof(LayerVals{})), len(net.LayVals), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("Synapses", int(unsafe.Sizeof(Synapse{})), len(net.Synapses), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("GBuf", vgpu.Float32, len(net.PrjnGBuf), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("GSyns", vgpu.Float32, len(net.PrjnGSyns), vgpu.Storage, vgpu.ComputeShader)

	gp.Exts.Add("Exts", vgpu.Float32, len(net.Exts), vgpu.Storage, vgpu.ComputeShader)

	gp.Params.ConfigVals(1)
	gp.Prjns.ConfigVals(1)
	gp.Structs.ConfigVals(1)
	gp.Exts.ConfigVals(1)

	// pipelines
	gp.GatherSpikes = gp.Sys.NewComputePipelineEmbed("GatherSpikes", content, "shaders/gpu_gather.spv")
	gp.PoolGeMax = gp.Sys.NewComputePipelineEmbed("PoolGeMax", content, "shaders/gpu_poolgemax.spv")
	gp.BetweenGi = gp.Sys.NewComputePipelineEmbed("BetweenGi", content, "shaders/gpu_betweengi.spv")
	gp.PoolGi = gp.Sys.NewComputePipelineEmbed("PoolGi", content, "shaders/gpu_poolgi.spv")
	gp.Cycle = gp.Sys.NewComputePipelineEmbed("Cycle", content, "shaders/gpu_cycle.spv")
	gp.SendSpike = gp.Sys.NewComputePipelineEmbed("SendSpikes", content, "shaders/gpu_sendspike.spv")
	gp.SynCa = gp.Sys.NewComputePipelineEmbed("SynCa", content, "shaders/gpu_synca.spv")
	gp.SynCaRecv = gp.Sys.NewComputePipelineEmbed("SynCaRecv", content, "shaders/gpu_syncarecv.spv")
	gp.SynCaSend = gp.Sys.NewComputePipelineEmbed("SynCaSend", content, "shaders/gpu_syncasend.spv")
	gp.NewState = gp.Sys.NewComputePipelineEmbed("NewState", content, "shaders/gpu_newstate.spv")
	gp.MinusPhase = gp.Sys.NewComputePipelineEmbed("MinusPhase", content, "shaders/gpu_minusphase.spv")
	gp.PlusPhase = gp.Sys.NewComputePipelineEmbed("PlusPhase", content, "shaders/gpu_plusphase.spv")
	gp.DWt = gp.Sys.NewComputePipelineEmbed("DWt", content, "shaders/gpu_dwt.spv")
	gp.WtFmDWt = gp.Sys.NewComputePipelineEmbed("WtFmDWt", content, "shaders/gpu_wtfmdwt.spv")
	gp.ApplyExts = gp.Sys.NewComputePipelineEmbed("ApplyExts", content, "shaders/gpu_applyext.spv")

	gp.Sys.Config()

	gp.CopyParamsToGPU(ctx, net)
	gp.CopyExtsToGPU(ctx, net)
	gp.CopyContextToGPU(ctx, net)
	gp.CopyNeuronsToGPU(ctx, net)
	gp.CopyStateToGPU(ctx, net)

	gp.Sys.Mem.SyncToGPU()

	// todo: add a convenience method to vgpu to do this for everything
	vars.BindDynValIdx(0, "Layers", 0)

	vars.BindDynValIdx(1, "Prjns", 0)
	vars.BindDynValIdx(1, "RecvCon", 0)
	vars.BindDynValIdx(1, "SendPrjnIdxs", 0)
	vars.BindDynValIdx(1, "SendCon", 0)
	vars.BindDynValIdx(1, "SendSynIdxs", 0)

	vars.BindDynValIdx(2, "Ctxt", 0)
	vars.BindDynValIdx(2, "Neurons", 0)
	vars.BindDynValIdx(2, "Pools", 0)
	vars.BindDynValIdx(2, "LayVals", 0)
	vars.BindDynValIdx(2, "Synapses", 0)
	vars.BindDynValIdx(2, "GBuf", 0)
	vars.BindDynValIdx(2, "GSyns", 0)

	vars.BindDynValIdx(3, "Exts", 0)
}

func (gp *GPU) CopyParamsToGPU(ctx *Context, net *Network) {
	_, layv, _ := gp.Params.ValByIdxTry("Layers", 0)
	layv.CopyFromBytes(unsafe.Pointer(&net.LayParams[0]))

	_, pjnv, _ := gp.Prjns.ValByIdxTry("Prjns", 0)
	pjnv.CopyFromBytes(unsafe.Pointer(&net.PrjnParams[0]))

	_, rconv, _ := gp.Prjns.ValByIdxTry("RecvCon", 0)
	rconv.CopyFromBytes(unsafe.Pointer(&net.PrjnRecvCon[0]))

	_, spiv, _ := gp.Prjns.ValByIdxTry("SendPrjnIdxs", 0)
	spiv.CopyFromBytes(unsafe.Pointer(&net.SendPrjnIdxs[0]))

	_, sconv, _ := gp.Prjns.ValByIdxTry("SendCon", 0)
	sconv.CopyFromBytes(unsafe.Pointer(&net.PrjnSendCon[0]))

	_, ssiv, _ := gp.Prjns.ValByIdxTry("SendSynIdxs", 0)
	ssiv.CopyFromBytes(unsafe.Pointer(&net.SendSynIdxs[0]))
}

func (gp *GPU) CopyExtsToGPU(ctx *Context, net *Network) {
	_, extv, _ := gp.Exts.ValByIdxTry("Exts", 0)
	extv.CopyFromBytes(unsafe.Pointer(&net.Exts[0]))
}

func (gp *GPU) CopyContextToGPU(ctx *Context, net *Network) {
	_, ctxv, _ := gp.Structs.ValByIdxTry("Ctxt", 0)
	ctxv.CopyFromBytes(unsafe.Pointer(ctx))
}

func (gp *GPU) CopyNeuronsToGPU(ctx *Context, net *Network) {
	_, neurv, _ := gp.Structs.ValByIdxTry("Neurons", 0)
	neurv.CopyFromBytes(unsafe.Pointer(&net.Neurons[0]))
}

func (gp *GPU) CopyNeuronsFromGPU(ctx *Context, net *Network) {
	if !gp.On {
		return
	}
	gp.Sys.Mem.SyncValIdxFmGPU(gp.Structs.Set, "Neurons", 0)
	_, neurv, _ := gp.Structs.ValByIdxTry("Neurons", 0)
	neurv.CopyToBytes(unsafe.Pointer(&net.Neurons[0]))
}

func (gp *GPU) CopySynapsesFromGPU(ctx *Context, net *Network) {
	if !gp.On {
		return
	}
	gp.Sys.Mem.SyncValIdxFmGPU(gp.Structs.Set, "Synapses", 0)
	_, synv, _ := gp.Structs.ValByIdxTry("Synapses", 0)
	synv.CopyToBytes(unsafe.Pointer(&net.Synapses[0]))
}

func (gp *GPU) CopyStateToGPU(ctx *Context, net *Network) {
	_, poolv, _ := gp.Structs.ValByIdxTry("Pools", 0)
	poolv.CopyFromBytes(unsafe.Pointer(&net.Pools[0]))

	_, layv, _ := gp.Structs.ValByIdxTry("LayVals", 0)
	layv.CopyFromBytes(unsafe.Pointer(&net.LayVals[0]))

	_, synv, _ := gp.Structs.ValByIdxTry("Synapses", 0)
	synv.CopyFromBytes(unsafe.Pointer(&net.Synapses[0]))
}

func (gp *GPU) SyncMemToGPU() {
	gp.Sys.Mem.SyncToGPU()
}

func (gp *GPU) RunPipeline(net *Network, name string, pl *vgpu.Pipeline, n int) {
	// todo: need to bind vars again?  try just doing it once the first time
	// and compare times..
	net.FunTimerStart(name)
	gp.Sys.CmdResetBindVars(gp.Sys.CmdPool.Buff, 0)
	// gp.Sys.ComputeResetBegin()
	pl.ComputeCommand1D(n, gp.NThreads)
	gp.Sys.ComputeSubmitWait()
	net.FunTimerStop(name)
}

func (gp *GPU) RunApplyExts(ctx *Context, net *Network) {
	gp.CopyExtsToGPU(ctx, net)
	gp.SyncMemToGPU()
	gp.RunPipeline(net, "GPU:ApplyExt", gp.ApplyExts, len(net.Neurons))
}

func (gp *GPU) RunCycle(ctx *Context, net *Network) {
	gp.CopyContextToGPU(ctx, net)
	gp.SyncMemToGPU()
	gp.RunPipeline(net, "GPU:GatherSpikes", gp.GatherSpikes, len(net.Neurons))

	// todo: use semaphors for all of these instead of waits
	// todo: need to bind vars again?
	gp.RunPipeline(net, "GPU:PoolGeMax", gp.PoolGeMax, len(net.Pools))
	gp.RunPipeline(net, "GPU:BetweenGi", gp.BetweenGi, len(net.Pools))
	gp.RunPipeline(net, "GPU:PoolGi", gp.PoolGi, len(net.Pools))

	gp.RunPipeline(net, "GPU:Cycle", gp.Cycle, len(net.Neurons))

	gp.RunPipeline(net, "GPU:SendSpike", gp.SendSpike, len(net.Neurons))

	if ctx.Testing.IsFalse() {
		// todo: test in larger networks!
		gp.RunPipeline(net, "GPU:SynCa", gp.SynCa, len(net.Synapses)) // faster?
		// gp.RunPipeline(net, "GPU:SynCaRecv", gp.SynCaRecv, len(net.Neurons)) // recv first as faster
		// gp.RunPipeline(net, "GPU:SynCaSend", gp.SynCaSend, len(net.Neurons))
	}
}

func (gp *GPU) RunNewState(ctx *Context, net *Network) {
	gp.RunPipeline(net, "GPU:NewState", gp.NewState, len(net.Pools))
}

func (gp *GPU) RunMinusPhase(ctx *Context, net *Network) {
	gp.RunPipeline(net, "GPU:MinusPhase", gp.MinusPhase, len(net.Pools))
}

func (gp *GPU) RunPlusPhase(ctx *Context, net *Network) {
	gp.RunPipeline(net, "GPU:PlusPhase", gp.PlusPhase, len(net.Pools))
}

func (gp *GPU) RunDWt(ctx *Context, net *Network) {
	gp.RunPipeline(net, "GPU:DWt", gp.DWt, len(net.Synapses))
}

func (gp *GPU) RunWtFmDWt(ctx *Context, net *Network) {
	gp.RunPipeline(net, "GPU:WtFmDWt", gp.WtFmDWt, len(net.Synapses))
}
