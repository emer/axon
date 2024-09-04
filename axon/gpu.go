// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !offscreen && ((darwin && !ios) || windows || (linux && !android) || dragonfly || openbsd)

package axon

import (
	"embed"
	"fmt"
	"math"
	"unsafe"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/system"
	"cogentcore.org/core/vgpu"
	vk "github.com/goki/vulkan"
)

//go:embed shaders/*.wgsl
var content embed.FS

//go:generate gosl -exclude=Update,UpdateParams,Defaults,AllParams,ShouldDisplay cogentcore.org/core/math32/fastexp.go cogentcore.org/core/math32/minmax ../chans/chans.go ../chans ../kinase ../fsfffb/inhib.go ../fsfffb github.com/emer/emergent/v2/etime github.com/emer/emergent/v2/ringidx rand.go avgmax.go neuromod.go globals.go context.go neuron.go synapse.go pool.go layervals.go act.go act_path.go inhib.go learn.go layertypes.go layerparams.go deep_layers.go rl_layers.go rubicon_layers.go pcore_layers.go pathtypes.go pathparams.go deep_paths.go rl_paths.go rubicon_paths.go pcore_paths.go hip_paths.go gpu_wgsl/gpu_applyext.wgsl

// Full vars code -- each gpu_*.wgsl uses a subset

/*

// note: binding is var, set

// Set 0: uniform layer params -- could not have paths also be uniform..
@group(0) @binding(0)
var<storage, read_write> Layers: array<LayerParams>;
@group(0) @binding(1)
var<storage, read_write> Paths: array<PathParams>;

// Set 1: effectively uniform indexes and path params as structured buffers in storage
@group(1) @binding(0)
var<storage, read_write> NeuronIxs: array<u32>; // [Neurons][Indexes]
@group(1) @binding(1)
var<storage, read_write> SynapseIxs: array<u32>; // [Layer][SendPaths][SendNeurons][Syns]
@group(1) @binding(2)
var<storage, read_write> SendCon: array<StartN>; // [Layer][SendPaths][SendNeurons]
@group(1) @binding(3)
var<storage, read_write> RecvPathIndexes: array<u32>; // [Layer][RecvPaths]
@group(1) @binding(4)
var<storage, read_write> RecvCon: array<StartN>; // [Layer][RecvPaths][RecvNeurons]
@group(1) @binding(5)
var<storage, read_write> RecvSynIndexes: array<u32>; // [Layer][RecvPaths][RecvNeurons][Syns]

// Set 2: main network structs and vals -- all are writable
@group(2) @binding(0)
var<storage, read_write> Ctx: array<Context>; // [0]
@group(2) @binding(1)
var<storage, read_write> Neurons: array<f32>; // [Neurons][Vars][Data]
@group(2) @binding(2)
var<storage, read_write> NeuronAvgs: array<f32>; // [Neurons][Vars]
@group(2) @binding(3)
var<storage, read_write> Pools: array<f32>; // [Layer][Pools][Data]
@group(2) @binding(4)
var<storage, read_write> LayValues: array<LayerValues>; // [Layer][Data]
@group(2) @binding(5)
var<storage, read_write> Globals: array<f32>; // [NGlobals]
@group(2) @binding(6)
var<storage, read_write> Exts: array<f32>; // [In / Out Layers][Neurons][Data]

// There might be a limit of 8 buffers per set -- can't remember..

// Set 3: synapse vars
@group(3) @binding(0)
var<storage, read_write> GBuf: array<i32>; // [Layer][RecvPaths][RecvNeurons][MaxDel+1][Data]
@group(3) @binding(1)
var<storage, read_write> GSyns: array<f32>; // [Layer][RecvPaths][RecvNeurons][Data]
@group(3) @binding(2)
var<storage, read_write> Synapses: array<f32>; // [Layer][SendPaths][SendNeurons][Syns]

// todo: future expansion to add more tranches of Synapses

// Set 4: SynCa -- can only access in 2^31 chunks
@group(4) @binding(0)
var<storage, read_write> SynapseCas: array<f32>; // [Layer][SendPaths][SendNeurons][Syns][Data]
@group(4) @binding(1)
var<storage, read_write> SynapseCas1: array<f32>;
@group(4) @binding(2)
var<storage, read_write> SynapseCas2: array<f32>;
@group(4) @binding(3)
var<storage, read_write> SynapseCas3: array<f32>;
@group(4) @binding(4)
var<storage, read_write> SynapseCas4: array<f32>;
@group(4) @binding(5)
var<storage, read_write> SynapseCas5: array<f32>;
@group(4) @binding(6)
var<storage, read_write> SynapseCas6: array<f32>;
@group(4) @binding(7)
var<storage, read_write> SynapseCas7: array<f32>;

Set: 0
    Role: Storage
        Var: 0:	Layers	Struct[4]	(size: 1520)	Values: 1
        Var: 1:	Paths		Struct[5]	(size: 352)	Values: 1
Set: 1
    Role: Storage
        Var: 0:	NeuronIxs		Uint32[534]	(size: 4)	Values: 1
        Var: 1:	SynapseIxs	Uint32[38976]	(size: 4)	Values: 1
        Var: 2:	SendCon		Struct[242]	(size: 16)	Values: 1
        Var: 3:	RecvPathIndexes	Uint32[5]	(size: 4)	Values: 1
        Var: 4:	RecvCon		Struct[281]	(size: 16)	Values: 1
        Var: 5:	RecvSynIndexes	Uint32[12992]	(size: 4)	Values: 1
Set: 2
    Role: Storage
        Var: 0:	Ctx		Struct	(size: 512)	Values: 1
        Var: 1:	Neurons	Float32[227840]	(size: 4)	Values: 1
        Var: 2:	NeuronAvgs	Float32[1246]	(size: 4)	Values: 1
        Var: 3:	Pools		Struct[64]	(size: 1040)	Values: 1
        Var: 4:	LayValues	Struct[64]	(size: 80)	Values: 1
        Var: 5:	Globals	Float32[976]	(size: 4)	Values: 1
        Var: 6:	Exts		Float32[800]	(size: 4)	Values: 1
Set: 3
    Role: Storage
        Var: 0:	GBuf		Int32[13488]	(size: 4)	Values: 1
        Var: 1:	GSyns		Float32[4496]	(size: 4)	Values: 1
        Var: 2:	Synapses	Float32[64960]	(size: 4)	Values: 1
Set: 4
    Role: Storage
        Var: 0:	SynapseCas0	Float32[1455104]	(size: 4)	Values: 1
        Var: 1:	SynapseCas1	Float32	(size: 4)	Values: 1
        Var: 2:	SynapseCas2	Float32	(size: 4)	Values: 1
        Var: 3:	SynapseCas3	Float32	(size: 4)	Values: 1
        Var: 4:	SynapseCas4	Float32	(size: 4)	Values: 1
        Var: 5:	SynapseCas5	Float32	(size: 4)	Values: 1
        Var: 6:	SynapseCas6	Float32	(size: 4)	Values: 1
*/

// TheGPU is the gpu device, shared across all networks
var TheGPU *vgpu.GPU

// CyclesN is the number of cycles to run as a group
// for ra25, 10 = ~50 msec / trial, 25 = ~48, all 150 / 50 minus / plus = ~44
// 10 is good enough and unlikely to mess with anything else..
const CyclesN = 10

// PushOff has push constants for setting offset into compute shader
type PushOff struct {

	// offset
	Off uint32

	pad, pad1, pad2 uint32
}

// GPU manages all of the GPU-based computation for a given Network.
// Lives within the network.
type GPU struct {

	// if true, actually use the GPU
	On bool

	// if true, slower separate shader pipeline runs are used, with a CPU-sync Wait at the end, to enable timing information about each individual shader to be collected using the network FunTimer system.  otherwise, only aggregate information is available about the entire Cycle call.
	RecFunTimes bool

	// if true, process each cycle one at a time.  Otherwise, 10 cycles at a time are processed in one batch.
	CycleByCycle bool

	// the network we operate on -- we live under this net
	Net *Network `display:"-"`

	// the context we use
	Ctx *Context `display:"-"`

	// the vgpu compute system
	Sys *vgpu.System `display:"-"`

	// VarSet = 0: the uniform LayerParams
	Params *vgpu.VarSet `display:"-"`

	// VarSet = 1: the storage indexes and PathParams
	Indexes *vgpu.VarSet `display:"-"`

	// VarSet = 2: the Storage buffer for RW state structs and neuron floats
	Structs *vgpu.VarSet `display:"-"`

	// Varset = 3: the Storage buffer for synapses
	Syns *vgpu.VarSet `display:"-"`

	// Varset = 4: the Storage buffer for SynCa banks
	SynCas *vgpu.VarSet `display:"-"`

	// for sequencing commands
	Semaphores map[string]vk.Semaphore `display:"-"`

	// number of warp threads -- typically 64 -- must update all wgsl files if changed!
	NThreads int `display:"-" inactive:"-" default:"64"`

	// maximum number of bytes per individual storage buffer element, from GPUProps.Limits.MaxStorageBufferRange
	MaxBufferBytes uint32 `display:"-"`

	// bank of floats for GPU access
	SynapseCas0 []float32 `display:"-"`

	// bank of floats for GPU access
	SynapseCas1 []float32 `display:"-"`

	// bank of floats for GPU access
	SynapseCas2 []float32 `display:"-"`

	// bank of floats for GPU access
	SynapseCas3 []float32 `display:"-"`

	// bank of floats for GPU access
	SynapseCas4 []float32 `display:"-"`

	// bank of floats for GPU access
	SynapseCas5 []float32 `display:"-"`

	// bank of floats for GPU access
	SynapseCas6 []float32 `display:"-"`

	// bank of floats for GPU access
	SynapseCas7 []float32 `display:"-"`

	// tracks var binding
	DidBind map[string]bool `display:"-"`
}

// ConfigGPUwithGUI turns on GPU mode in context of an active GUI where Vulkan
// has been initialized etc.
// Configures the GPU -- call after Network is Built, initialized, params are set,
// and everything is ready to run.
func (nt *Network) ConfigGPUwithGUI(ctx *Context) {
	system.TheApp.RunOnMain(func() {
		nt.GPU.Config(ctx, nt)
	})
	fmt.Printf("Running on GPU: %s\n", TheGPU.DeviceName)
}

// ConfigGPUnoGUI turns on GPU mode in case where no GUI is being used.
// This directly accesses the GPU hardware.  It does not work well when GUI also being used.
// Configures the GPU -- call after Network is Built, initialized, params are set,
// and everything is ready to run.
func (nt *Network) ConfigGPUnoGUI(ctx *Context) {
	if TheGPU == nil {
		if err := vgpu.InitNoDisplay(); err != nil {
			panic(err)
		}
	}
	nt.GPU.Config(ctx, nt)
	mpi.AllPrintf("Running on GPU: %s\n", TheGPU.DeviceName)
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
		opts := vgpu.NewRequiredOpts(vgpu.OptShaderInt64) // , vgpu.OptRobustBufferAccess
		TheGPU.Config("axon", &opts)
	}

	gp.MaxBufferBytes = TheGPU.GPUProperties.Limits.MaxStorageBufferRange - 16
	gp.Sys = TheGPU.NewComputeSystem("axon")
	gp.Sys.StaticVars = true // no diff in perf..

	gp.ConfigSynCaBuffs()

	vars := gp.Sys.Vars()

	pcset := vars.AddPushSet()
	gp.Params = vars.AddSet()
	gp.Indexes = vars.AddSet()
	gp.Structs = vars.AddSet()
	gp.Syns = vars.AddSet()
	gp.SynCas = vars.AddSet()

	pcset.AddStruct("PushOff", int(unsafe.Sizeof(PushOff{})), 1, vgpu.Push, vgpu.ComputeShader)

	gp.Params.AddStruct("Layers", int(unsafe.Sizeof(LayerParams{})), len(gp.Net.LayParams), vgpu.Storage, vgpu.ComputeShader)
	gp.Params.AddStruct("Paths", int(unsafe.Sizeof(PathParams{})), len(gp.Net.PathParams), vgpu.Storage, vgpu.ComputeShader)

	// note: paths must be in Storage here because couldn't have both Layers and Paths as uniform.
	gp.Indexes.Add("NeuronIxs", vgpu.Uint32, len(gp.Net.NeuronIxs), vgpu.Storage, vgpu.ComputeShader)
	gp.Indexes.Add("SynapseIxs", vgpu.Uint32, len(gp.Net.SynapseIxs), vgpu.Storage, vgpu.ComputeShader)
	gp.Indexes.AddStruct("SendCon", int(unsafe.Sizeof(StartN{})), len(gp.Net.PathSendCon), vgpu.Storage, vgpu.ComputeShader)
	gp.Indexes.Add("RecvPathIndexes", vgpu.Uint32, len(gp.Net.RecvPathIndexes), vgpu.Storage, vgpu.ComputeShader)
	gp.Indexes.AddStruct("RecvCon", int(unsafe.Sizeof(StartN{})), len(gp.Net.PathRecvCon), vgpu.Storage, vgpu.ComputeShader)
	gp.Indexes.Add("RecvSynIndexes", vgpu.Uint32, len(gp.Net.RecvSynIndexes), vgpu.Storage, vgpu.ComputeShader)

	gp.Structs.AddStruct("Ctx", int(unsafe.Sizeof(Context{})), 1, vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("Neurons", vgpu.Float32, len(gp.Net.Neurons), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("NeuronAvgs", vgpu.Float32, len(gp.Net.NeuronAvgs), vgpu.Storage, vgpu.ComputeShader)

	gp.Structs.AddStruct("Pools", int(unsafe.Sizeof(Pool{})), len(gp.Net.Pools), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.AddStruct("LayValues", int(unsafe.Sizeof(LayerValues{})), len(gp.Net.LayValues), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("Globals", vgpu.Float32, len(gp.Net.Globals), vgpu.Storage, vgpu.ComputeShader)
	gp.Structs.Add("Exts", vgpu.Float32, len(gp.Net.Exts), vgpu.Storage, vgpu.ComputeShader)

	gp.Syns.Add("GBuf", vgpu.Int32, len(gp.Net.PathGBuf), vgpu.Storage, vgpu.ComputeShader)
	gp.Syns.Add("GSyns", vgpu.Float32, len(gp.Net.PathGSyns), vgpu.Storage, vgpu.ComputeShader)
	gp.Syns.Add("Synapses", vgpu.Float32, len(gp.Net.Synapses), vgpu.Storage, vgpu.ComputeShader)

	gp.SynCas.Add("SynapseCas0", vgpu.Float32, len(gp.SynapseCas0), vgpu.Storage, vgpu.ComputeShader)
	gp.SynCas.Add("SynapseCas1", vgpu.Float32, len(gp.SynapseCas1), vgpu.Storage, vgpu.ComputeShader)
	gp.SynCas.Add("SynapseCas2", vgpu.Float32, len(gp.SynapseCas2), vgpu.Storage, vgpu.ComputeShader)
	gp.SynCas.Add("SynapseCas3", vgpu.Float32, len(gp.SynapseCas3), vgpu.Storage, vgpu.ComputeShader)
	gp.SynCas.Add("SynapseCas4", vgpu.Float32, len(gp.SynapseCas4), vgpu.Storage, vgpu.ComputeShader)
	gp.SynCas.Add("SynapseCas5", vgpu.Float32, len(gp.SynapseCas5), vgpu.Storage, vgpu.ComputeShader)
	gp.SynCas.Add("SynapseCas6", vgpu.Float32, len(gp.SynapseCas6), vgpu.Storage, vgpu.ComputeShader)
	gp.SynCas.Add("SynapseCas7", vgpu.Float32, len(gp.SynapseCas7), vgpu.Storage, vgpu.ComputeShader)

	gp.Params.ConfigValues(1)
	gp.Indexes.ConfigValues(1)
	gp.Structs.ConfigValues(1)
	gp.Syns.ConfigValues(1)
	gp.SynCas.ConfigValues(1)

	// pipelines
	gp.Sys.NewComputePipelineEmbed("GatherSpikes", content, "shaders/gpu_gather.spv")
	gp.Sys.NewComputePipelineEmbed("LayGi", content, "shaders/gpu_laygi.spv")
	gp.Sys.NewComputePipelineEmbed("BetweenGi", content, "shaders/gpu_betweengi.spv")
	gp.Sys.NewComputePipelineEmbed("PoolGi", content, "shaders/gpu_poolgi.spv")
	gp.Sys.NewComputePipelineEmbed("Cycle", content, "shaders/gpu_cycle.spv")
	gp.Sys.NewComputePipelineEmbed("CycleInc", content, "shaders/gpu_cycleinc.spv")
	gp.Sys.NewComputePipelineEmbed("SendSpike", content, "shaders/gpu_sendspike.spv")
	gp.Sys.NewComputePipelineEmbed("CyclePost", content, "shaders/gpu_cyclepost.spv")

	gp.Sys.NewComputePipelineEmbed("NewStatePool", content, "shaders/gpu_newstate_pool.spv")
	gp.Sys.NewComputePipelineEmbed("NewStateNeuron", content, "shaders/gpu_newstate_neuron.spv")
	gp.Sys.NewComputePipelineEmbed("MinusPool", content, "shaders/gpu_minuspool.spv")
	gp.Sys.NewComputePipelineEmbed("MinusNeuron", content, "shaders/gpu_minusneuron.spv")
	gp.Sys.NewComputePipelineEmbed("PlusStart", content, "shaders/gpu_plusstart.spv")
	gp.Sys.NewComputePipelineEmbed("PlusPool", content, "shaders/gpu_pluspool.spv")
	gp.Sys.NewComputePipelineEmbed("PlusNeuron", content, "shaders/gpu_plusneuron.spv")
	gp.Sys.NewComputePipelineEmbed("DWt", content, "shaders/gpu_dwt.spv")
	gp.Sys.NewComputePipelineEmbed("DWtFromDi", content, "shaders/gpu_dwtfmdi.spv")
	gp.Sys.NewComputePipelineEmbed("WtFromDWt", content, "shaders/gpu_wtfmdwt.spv")
	gp.Sys.NewComputePipelineEmbed("DWtSubMean", content, "shaders/gpu_dwtsubmean.spv")
	gp.Sys.NewComputePipelineEmbed("ApplyExts", content, "shaders/gpu_applyext.spv")

	gp.Sys.NewComputePipelineEmbed("TestSynCa", content, "shaders/gpu_test_synca.spv")

	gp.Sys.Config()

	gp.CopyParamsToStaging()
	gp.CopyIndexesToStaging()
	gp.CopyExtsToStaging()
	gp.CopyContextToStaging()
	gp.CopyStateToStaging()
	gp.CopySynapsesToStaging()
	gp.CopySynCaToStaging()

	gp.Sys.Mem.SyncToGPU()
}

// ConfigSynCaBuffs configures special SynapseCas buffers needed for larger memory access
func (gp *GPU) ConfigSynCaBuffs() {
	bufMax := gp.MaxBufferBytes
	floatMax := int(bufMax) / 4 // 32 bit floats for now

	ctx := gp.Ctx
	net := gp.Net
	ctx.NetIndexes.GPUMaxBuffFloats = uint32(floatMax)
	net.Ctx.NetIndexes.GPUMaxBuffFloats = uint32(floatMax)

	nSynCaFloat := len(net.SynapseCas)
	nCaBanks := nSynCaFloat / int(floatMax)
	caLast := nSynCaFloat % int(floatMax)
	if caLast > 0 {
		nCaBanks++
	}
	ctx.NetIndexes.GPUSynCaBanks = uint32(nCaBanks)
	net.Ctx.NetIndexes.GPUSynCaBanks = uint32(nCaBanks)

	// fmt.Printf("banks %d: MaxBuffFloats: %X\n", ctx.NetIndexes.GPUSynCaBanks, ctx.NetIndexes.GPUMaxBuffFloats)

	if nCaBanks > 8 {
		panic(fmt.Sprintf("SynapseCas only supports 8 banks of %X floats -- needs: %d banks\n", floatMax, nCaBanks))
	}
	base := 0
	if nCaBanks > 1 {
		gp.SynapseCas0 = net.SynapseCas[base : base+floatMax]
	} else if nCaBanks == 1 {
		gp.SynapseCas0 = net.SynapseCas[base : base+caLast]
	}
	base += floatMax
	if nCaBanks > 2 {
		gp.SynapseCas1 = net.SynapseCas[base : base+floatMax]
	} else if nCaBanks == 2 {
		gp.SynapseCas1 = net.SynapseCas[base : base+caLast]
	} else {
		gp.SynapseCas1 = make([]float32, 4) // dummy
	}
	base += floatMax
	if nCaBanks > 3 {
		gp.SynapseCas2 = net.SynapseCas[base : base+floatMax]
	} else if nCaBanks == 3 {
		gp.SynapseCas2 = net.SynapseCas[base : base+caLast]
	} else {
		gp.SynapseCas2 = make([]float32, 4) // dummy
	}
	base += floatMax
	if nCaBanks > 4 {
		gp.SynapseCas3 = net.SynapseCas[base : base+floatMax]
	} else if nCaBanks == 4 {
		gp.SynapseCas3 = net.SynapseCas[base : base+caLast]
	} else {
		gp.SynapseCas3 = make([]float32, 4) // dummy
	}
	base += floatMax
	if nCaBanks > 5 {
		gp.SynapseCas4 = net.SynapseCas[base : base+floatMax]
	} else if nCaBanks == 5 {
		gp.SynapseCas4 = net.SynapseCas[base : base+caLast]
	} else {
		gp.SynapseCas4 = make([]float32, 4) // dummy
	}
	base += floatMax
	if nCaBanks > 6 {
		gp.SynapseCas5 = net.SynapseCas[base : base+floatMax]
	} else if nCaBanks == 6 {
		gp.SynapseCas5 = net.SynapseCas[base : base+caLast]
	} else {
		gp.SynapseCas5 = make([]float32, 4) // dummy
	}
	base += floatMax
	if nCaBanks > 7 {
		gp.SynapseCas6 = net.SynapseCas[base : base+floatMax]
	} else if nCaBanks == 7 {
		gp.SynapseCas6 = net.SynapseCas[base : base+caLast]
	} else {
		gp.SynapseCas6 = make([]float32, 4) // dummy
	}
	base += floatMax
	if nCaBanks > 8 {
		gp.SynapseCas7 = net.SynapseCas[base : base+floatMax]
	} else if nCaBanks == 8 {
		gp.SynapseCas7 = net.SynapseCas[base : base+caLast]
	} else {
		gp.SynapseCas7 = make([]float32, 4) // dummy
	}
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

// CopyParamsToStaging copies the LayerParams and PathParams to staging from CPU.
// Must call SyncMemToGPU after this (see SyncParamsToGPU).
func (gp *GPU) CopyParamsToStaging() {
	if !gp.On {
		return
	}
	_, layv, _ := gp.Params.ValueByIndexTry("Layers", 0)
	layv.CopyFromBytes(unsafe.Pointer(&gp.Net.LayParams[0]))

	_, pjnv, _ := gp.Params.ValueByIndexTry("Paths", 0)
	pjnv.CopyFromBytes(unsafe.Pointer(&gp.Net.PathParams[0]))
}

// SyncParamsToGPU copies the LayerParams and PathParams to the GPU from CPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncParamsToGPU() {
	if !gp.On {
		return
	}
	gp.CopyParamsToStaging()
	gp.SyncMemToGPU()
}

// CopyIndexesToStaging is only called when the network is built
// to copy the indexes specifying connectivity etc to staging from CPU.
func (gp *GPU) CopyIndexesToStaging() {
	if !gp.On {
		return
	}

	_, neuriv, _ := gp.Indexes.ValueByIndexTry("NeuronIxs", 0)
	neuriv.CopyFromBytes(unsafe.Pointer(&gp.Net.NeuronIxs[0]))

	_, syniv, _ := gp.Indexes.ValueByIndexTry("SynapseIxs", 0)
	syniv.CopyFromBytes(unsafe.Pointer(&gp.Net.SynapseIxs[0]))

	_, sconv, _ := gp.Indexes.ValueByIndexTry("SendCon", 0)
	sconv.CopyFromBytes(unsafe.Pointer(&gp.Net.PathSendCon[0]))

	_, spiv, _ := gp.Indexes.ValueByIndexTry("RecvPathIndexes", 0)
	spiv.CopyFromBytes(unsafe.Pointer(&gp.Net.RecvPathIndexes[0]))

	_, rconv, _ := gp.Indexes.ValueByIndexTry("RecvCon", 0)
	rconv.CopyFromBytes(unsafe.Pointer(&gp.Net.PathRecvCon[0]))

	_, ssiv, _ := gp.Indexes.ValueByIndexTry("RecvSynIndexes", 0)
	ssiv.CopyFromBytes(unsafe.Pointer(&gp.Net.RecvSynIndexes[0]))
}

// CopyExtsToStaging copies external inputs to staging from CPU.
// Typically used in RunApplyExts which also does the Sync.
func (gp *GPU) CopyExtsToStaging() {
	if !gp.On {
		return
	}
	_, extv, _ := gp.Structs.ValueByIndexTry("Exts", 0)
	extv.CopyFromBytes(unsafe.Pointer(&gp.Net.Exts[0]))
}

// CopyContextToStaging copies current context to staging from CPU.
// Must call SyncMemToGPU after this (see SyncContextToGPU).
// See SetContext if there is a new one.
func (gp *GPU) CopyContextToStaging() {
	if !gp.On {
		return
	}
	_, ctxv, _ := gp.Structs.ValueByIndexTry("Ctx", 0)
	_, glbv, _ := gp.Structs.ValueByIndexTry("Globals", 0)
	ctxv.CopyFromBytes(unsafe.Pointer(gp.Ctx))
	glbv.CopyFromBytes(unsafe.Pointer(&gp.Net.Globals[0]))
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

// CopyLayerValuesToStaging copies LayerValues to staging from CPU.
// Must call SyncMemToGPU after this (see SyncLayerValuesToGPU).
func (gp *GPU) CopyLayerValuesToStaging() {
	if !gp.On {
		return
	}
	_, layv, _ := gp.Structs.ValueByIndexTry("LayValues", 0)
	layv.CopyFromBytes(unsafe.Pointer(&gp.Net.LayValues[0]))
}

// SyncLayerValuesToGPU copies LayerValues to GPU from CPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncLayerValuesToGPU() {
	if !gp.On {
		return
	}
	gp.CopyLayerValuesToStaging()
	gp.SyncMemToGPU()
}

// CopyPoolsToStaging copies Pools to staging from CPU.
// Must call SyncMemToGPU after this (see SyncPoolsToGPU).
func (gp *GPU) CopyPoolsToStaging() {
	if !gp.On {
		return
	}
	_, poolv, _ := gp.Structs.ValueByIndexTry("Pools", 0)
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
	_, neurv, _ := gp.Structs.ValueByIndexTry("Neurons", 0)
	neurv.CopyFromBytes(unsafe.Pointer(&gp.Net.Neurons[0]))
	_, neurav, _ := gp.Structs.ValueByIndexTry("NeuronAvgs", 0)
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

// CopyStateToStaging copies LayerValues, Pools, Neurons state to staging from CPU.
// this is typically sufficient for most syncing --
// only missing the Synapses which must be copied separately.
// Must call SyncMemToGPU after this (see SyncStateToGPU).
func (gp *GPU) CopyStateToStaging() {
	if !gp.On {
		return
	}
	gp.CopyLayerValuesToStaging()
	gp.CopyPoolsToStaging()
	gp.CopyNeuronsToStaging()
}

// SyncStateToGPU copies LayValues, Pools, Neurons state to GPU
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

// SyncStateGBufToGPU copies LayValues, Pools, Neurons, GBuf state to GPU
// this is typically sufficient for most syncing --
// only missing the Synapses which must be copied separately.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncStateGBufToGPU() {
	if !gp.On {
		return
	}
	gp.CopyStateToStaging()
	gp.CopyGBufToStaging()
	gp.SyncMemToGPU()
}

// SyncAllToGPU copies LayerValues, Pools, Neurons, Synapses to GPU.
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
// Does not copy SynCa synapse state -- see SynCa methods.
// This is not typically needed except when weights are initialized or
// for the Slow weight update processes that are not on GPU.
// Must call SyncMemToGPU after this (see SyncSynapsesToGPU).
func (gp *GPU) CopySynapsesToStaging() {
	if !gp.On {
		return
	}
	_, synv, _ := gp.Syns.ValueByIndexTry("Synapses", 0)
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

// CopySynCaToStaging copies the SynCa variables to GPU, which are per-Di (even larger).
// This is only used for initialization -- SynCa vars otherwise managed entirely on GPU.
// Must call SyncMemToGPU after this (see SyncSynCaToGPU).
func (gp *GPU) CopySynCaToStaging() {
	if !gp.On {
		return
	}
	// note: do not need these except in GUI or tests
	_, syncv, _ := gp.SynCas.ValueByIndexTry("SynapseCas0", 0)
	syncv.CopyFromBytes(unsafe.Pointer(&gp.SynapseCas0[0]))
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 1 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas1", 0)
		syncv.CopyFromBytes(unsafe.Pointer(&gp.SynapseCas1[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 2 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas2", 0)
		syncv.CopyFromBytes(unsafe.Pointer(&gp.SynapseCas2[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 3 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas3", 0)
		syncv.CopyFromBytes(unsafe.Pointer(&gp.SynapseCas3[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 4 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas4", 0)
		syncv.CopyFromBytes(unsafe.Pointer(&gp.SynapseCas4[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 5 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas5", 0)
		syncv.CopyFromBytes(unsafe.Pointer(&gp.SynapseCas5[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 6 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas6", 0)
		syncv.CopyFromBytes(unsafe.Pointer(&gp.SynapseCas6[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 7 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas7", 0)
		syncv.CopyFromBytes(unsafe.Pointer(&gp.SynapseCas7[0]))
	}
}

// SyncSynCaToGPU copies the SynCa variables to GPU, which are per-Di (even larger).
// This is only used for initialization -- SynCa vars otherwise managed entirely on GPU.
// Calls SyncMemToGPU -- use when this is the only copy taking place.
func (gp *GPU) SyncSynCaToGPU() {
	if !gp.On {
		return
	}
	gp.CopySynCaToStaging()
	gp.SyncMemToGPU()
}

// CopyGBufToStaging copies the GBuf and GSyns memory to staging.
func (gp *GPU) CopyGBufToStaging() {
	if !gp.On {
		return
	}
	_, gbv, _ := gp.Syns.ValueByIndexTry("GBuf", 0)
	gbv.CopyFromBytes(unsafe.Pointer(&gp.Net.PathGBuf[0]))
	_, gsv, _ := gp.Syns.ValueByIndexTry("GSyns", 0)
	gsv.CopyFromBytes(unsafe.Pointer(&gp.Net.PathGSyns[0]))
}

// SyncGBufToGPU copies the GBuf and GSyns memory to the GPU.
func (gp *GPU) SyncGBufToGPU() {
	if !gp.On {
		return
	}
	gp.CopyGBufToStaging()
	gp.SyncMemToGPU()
}

///////////////////////////////////////////////////////////////////////
// 	Sync From
//			unlike Sync To, need to specify the regions to sync from first
// 		and then copy from staging to get into CPU memory

// SyncRegionStruct returns the SyncRegion with error panic
func (gp *GPU) SyncRegionStruct(vnm string) vgpu.MemReg {
	r, err := gp.Sys.Mem.SyncRegionValueIndex(gp.Structs.Set, vnm, 0)
	if err != nil {
		panic(err)
	}
	return r
}

// SyncRegionSyns returns the SyncRegion with error panic
func (gp *GPU) SyncRegionSyns(vnm string) vgpu.MemReg {
	r, err := gp.Sys.Mem.SyncRegionValueIndex(gp.Syns.Set, vnm, 0)
	if err != nil {
		panic(err)
	}
	return r
}

// SyncRegionSynCas returns the SyncRegion with error panic
func (gp *GPU) SyncRegionSynCas(vnm string) vgpu.MemReg {
	r, err := gp.Sys.Mem.SyncRegionValueIndex(gp.SynCas.Set, vnm, 0)
	if err != nil {
		panic(err)
	}
	return r
}

// CopyContextFromStaging copies Context from staging to CPU, after Sync back down.
func (gp *GPU) CopyContextFromStaging() {
	if !gp.On {
		return
	}
	_, ctxv, _ := gp.Structs.ValueByIndexTry("Ctx", 0)
	_, glbv, _ := gp.Structs.ValueByIndexTry("Globals", 0)
	ctxv.CopyToBytes(unsafe.Pointer(gp.Ctx))
	glbv.CopyToBytes(unsafe.Pointer(&gp.Net.Globals[0]))
}

// SyncContextFromGPU copies Context from GPU to CPU.
// This is done at the end of each cycle to get state back from GPU for CPU-side computations.
// Use only when only thing being copied -- more efficient to get all at once.
// e.g. see SyncStateFromGPU
func (gp *GPU) SyncContextFromGPU() {
	if !gp.On {
		return
	}
	cxr := gp.SyncRegionStruct("Ctx")
	glr := gp.SyncRegionStruct("Globals")
	gp.Sys.Mem.SyncStorageRegionsFromGPU(cxr, glr)
	gp.CopyContextFromStaging()
}

// CopyLayerValuesFromStaging copies LayerValues from staging to CPU, after Sync back down.
func (gp *GPU) CopyLayerValuesFromStaging() {
	if !gp.On {
		return
	}
	_, layv, _ := gp.Structs.ValueByIndexTry("LayValues", 0)
	layv.CopyToBytes(unsafe.Pointer(&gp.Net.LayValues[0]))
}

// SyncLayerValuesFromGPU copies LayerValues from GPU to CPU.
// This is done at the end of each cycle to get state back from staging for CPU-side computations.
// Use only when only thing being copied -- more efficient to get all at once.
// e.g. see SyncStateFromGPU
func (gp *GPU) SyncLayerValuesFromGPU() {
	if !gp.On {
		return
	}
	lvr := gp.SyncRegionStruct("LayValues")
	gp.Sys.Mem.SyncStorageRegionsFromGPU(lvr)
	gp.CopyLayerValuesFromStaging()
}

// CopyPoolsFromStaging copies Pools from staging to CPU, after Sync back down.
func (gp *GPU) CopyPoolsFromStaging() {
	if !gp.On {
		return
	}
	_, plv, _ := gp.Structs.ValueByIndexTry("Pools", 0)
	plv.CopyToBytes(unsafe.Pointer(&gp.Net.Pools[0]))
}

// SyncPoolsFromGPU copies Pools from GPU to CPU.
// Use only when only thing being copied -- more efficient to get all at once.
// e.g. see SyncStateFromGPU
func (gp *GPU) SyncPoolsFromGPU() {
	if !gp.On {
		return
	}
	plr := gp.SyncRegionStruct("Pools")
	gp.Sys.Mem.SyncStorageRegionsFromGPU(plr)
	gp.CopyPoolsFromStaging()
}

// CopyNeuronsFromStaging copies Neurons from staging to CPU, after Sync back down.
func (gp *GPU) CopyNeuronsFromStaging() {
	if !gp.On {
		return
	}
	_, neurv, _ := gp.Structs.ValueByIndexTry("Neurons", 0)
	neurv.CopyToBytes(unsafe.Pointer(&gp.Net.Neurons[0]))
	_, neurav, _ := gp.Structs.ValueByIndexTry("NeuronAvgs", 0)
	neurav.CopyToBytes(unsafe.Pointer(&gp.Net.NeuronAvgs[0]))
	// note: don't need to get indexes back down
}

// SyncNeuronsFromGPU copies Neurons from GPU to CPU.
// Use only when only thing being copied -- more efficient to get all at once.
// e.g. see SyncStateFromGPU
func (gp *GPU) SyncNeuronsFromGPU() {
	if !gp.On {
		return
	}
	nrr := gp.SyncRegionStruct("Neurons")
	nrar := gp.SyncRegionStruct("NeuronAvgs")
	// note: don't need to get indexes back down
	gp.Sys.Mem.SyncStorageRegionsFromGPU(nrr, nrar)
	gp.CopyNeuronsFromStaging()
}

// CopySynapsesFromStaging copies Synapses from staging to CPU, after Sync back down.
// Does not copy SynCa synapse state -- see SynCa methods.
func (gp *GPU) CopySynapsesFromStaging() {
	if !gp.On {
		return
	}
	_, synv, _ := gp.Syns.ValueByIndexTry("Synapses", 0)
	synv.CopyToBytes(unsafe.Pointer(&gp.Net.Synapses[0]))
}

// SyncSynapsesFromGPU copies Synapses from GPU to CPU.
// Does not copy SynCa synapse state -- see SynCa methods.
// Use only when only thing being copied -- more efficient to get all at once.
func (gp *GPU) SyncSynapsesFromGPU() {
	if !gp.On {
		return
	}
	syr := gp.SyncRegionSyns("Synapses")
	gp.Sys.Mem.SyncStorageRegionsFromGPU(syr)
	gp.CopySynapsesFromStaging()
}

// CopySynCaFromStaging copies the SynCa variables to GPU, which are per-Di (even larger).
// This is only used for GUI viewing -- SynCa vars otherwise managed entirely on GPU.
func (gp *GPU) CopySynCaFromStaging() {
	if !gp.On {
		return
	}
	_, syncv, _ := gp.SynCas.ValueByIndexTry("SynapseCas0", 0)
	syncv.CopyToBytes(unsafe.Pointer(&gp.SynapseCas0[0]))
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 1 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas1", 0)
		syncv.CopyToBytes(unsafe.Pointer(&gp.SynapseCas1[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 2 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas2", 0)
		syncv.CopyToBytes(unsafe.Pointer(&gp.SynapseCas2[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 3 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas3", 0)
		syncv.CopyToBytes(unsafe.Pointer(&gp.SynapseCas3[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 4 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas4", 0)
		syncv.CopyToBytes(unsafe.Pointer(&gp.SynapseCas4[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 5 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas5", 0)
		syncv.CopyToBytes(unsafe.Pointer(&gp.SynapseCas5[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 6 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas6", 0)
		syncv.CopyToBytes(unsafe.Pointer(&gp.SynapseCas6[0]))
	}
	if gp.Ctx.NetIndexes.GPUSynCaBanks > 7 {
		_, syncv, _ = gp.SynCas.ValueByIndexTry("SynapseCas7", 0)
		syncv.CopyToBytes(unsafe.Pointer(&gp.SynapseCas7[0]))
	}
}

func (gp *GPU) SynCaBuff(idx uint32) []float32 {
	switch idx {
	case 0:
		return gp.SynapseCas0
	case 1:
		return gp.SynapseCas1
	case 2:
		return gp.SynapseCas2
	case 3:
		return gp.SynapseCas3
	case 4:
		return gp.SynapseCas4
	case 5:
		return gp.SynapseCas5
	case 6:
		return gp.SynapseCas6
	case 7:
		return gp.SynapseCas7
	}
	return nil
}

// SyncSynCaFromGPU copies the SynCa variables to GPU, which are per-Di (even larger).
// This is only used for GUI viewing -- SynCa vars otherwise managed entirely on GPU.
// Use only when only thing being copied -- more efficient to get all at once.
func (gp *GPU) SyncSynCaFromGPU() {
	if !gp.On {
		return
	}
	ctx := gp.Ctx
	nBanks := int(ctx.NetIndexes.GPUSynCaBanks)
	regs := make([]vgpu.MemReg, nBanks)
	for i := range regs {
		reg := fmt.Sprintf("SynapseCas%d", i)
		regs[i] = gp.SyncRegionSynCas(reg)
	}
	gp.Sys.Mem.SyncStorageRegionsFromGPU(regs...)
	gp.CopySynCaFromStaging()
}

// CopyLayerStateFromStaging copies Context, LayerValues and Pools from staging to CPU, after Sync.
func (gp *GPU) CopyLayerStateFromStaging() {
	gp.CopyContextFromStaging()
	gp.CopyLayerValuesFromStaging()
	gp.CopyPoolsFromStaging()
}

// SyncLayerStateFromCPU copies Context, LayerValues, and Pools from GPU to CPU.
// This is the main GPU->CPU sync step automatically called after each Cycle.
func (gp *GPU) SyncLayerStateFromGPU() {
	if !gp.On {
		return
	}
	cxr := gp.SyncRegionStruct("Ctx")
	glr := gp.SyncRegionStruct("Globals")
	lvr := gp.SyncRegionStruct("LayValues")
	plr := gp.SyncRegionStruct("Pools")
	gp.Sys.Mem.SyncStorageRegionsFromGPU(cxr, glr, lvr, plr)
	gp.CopyLayerStateFromStaging()
}

// CopyStateFromStaging copies Context, LayerValues, Pools, and Neurons from staging to CPU, after Sync.
func (gp *GPU) CopyStateFromStaging() {
	gp.CopyLayerStateFromStaging()
	gp.CopyNeuronsFromStaging()
}

// SyncStateFromCPU copies Neurons, LayerValues, and Pools from GPU to CPU.
// This is the main GPU->CPU sync step automatically called in PlusPhase.
func (gp *GPU) SyncStateFromGPU() {
	if !gp.On {
		return
	}
	cxr := gp.SyncRegionStruct("Ctx")
	glr := gp.SyncRegionStruct("Globals")
	lvr := gp.SyncRegionStruct("LayValues")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")
	nrar := gp.SyncRegionStruct("NeuronAvgs")
	gp.Sys.Mem.SyncStorageRegionsFromGPU(cxr, glr, lvr, plr, nrr, nrar)
	gp.CopyStateFromStaging()
}

// SyncAllFromCPU copies State except Context plus Synapses from GPU to CPU.
// This is called before SlowAdapt, which is run CPU-side
func (gp *GPU) SyncAllFromGPU() {
	if !gp.On {
		return
	}
	lvr := gp.SyncRegionStruct("LayValues")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")
	nrar := gp.SyncRegionStruct("NeuronAvgs")
	syr := gp.SyncRegionSyns("Synapses")
	gp.Sys.Mem.SyncStorageRegionsFromGPU(lvr, plr, nrr, nrar, syr)
	gp.CopyLayerValuesFromStaging()
	gp.CopyPoolsFromStaging()
	gp.CopyNeuronsFromStaging()
	gp.CopySynapsesFromStaging()
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
	cmd := gp.Sys.ComputeCmdBuff()
	gp.Sys.ComputeResetBindVars(cmd, 0)
	pl.ComputeDispatch1D(cmd, n, gp.NThreads)
	gp.Sys.ComputeCmdEnd(cmd)
	gp.Sys.ComputeSubmitWait(cmd)
	gp.Net.FunTimerStop(gnm)
}

// StartRun resets the given command buffer in preparation
// for recording commands for a multi-step run.
// It is much more efficient to record all commands to one buffer, and use
// Events to synchronize the steps between them, rather than using semaphores.
// The submit call is by far the most expensive so that should only happen once!
func (gp *GPU) StartRun(cmd vk.CommandBuffer) {
	gp.Sys.ComputeResetBindVars(cmd, 0)
}

// RunPipelineMemWait records command to run given pipeline
// with a WaitMemWriteRead after it, so subsequent pipeline run will
// have access to values updated by this command.
func (gp *GPU) RunPipelineMemWait(cmd vk.CommandBuffer, name string, n int) {
	pl, err := gp.Sys.PipelineByNameTry(name)
	if err != nil {
		panic(err)
	}
	pl.ComputeDispatch1D(cmd, n, gp.NThreads)
	gp.Sys.ComputeWaitMemWriteRead(cmd)
}

// RunPipelineNoWait records command to run given pipeline
// without any waiting after it for writes to complete.
// This should be the last command in the sequence.
func (gp *GPU) RunPipelineNoWait(cmd vk.CommandBuffer, name string, n int) {
	pl, err := gp.Sys.PipelineByNameTry(name)
	if err != nil {
		panic(err)
	}
	pl.ComputeDispatch1D(cmd, n, gp.NThreads)
}

// RunPipelineOffset records command to run given pipeline
// with a push constant offset for the starting index to compute.
// This is needed when the total number of dispatch indexes exceeds
// GPU.MaxComputeWorkGroupCount1D.  Does NOT wait for writes,
// assuming a parallel launch of all.
func (gp *GPU) RunPipelineOffset(cmd vk.CommandBuffer, name string, n, off int) {
	pl, err := gp.Sys.PipelineByNameTry(name)
	if err != nil {
		panic(err)
	}
	vars := gp.Sys.Vars()
	pvar, _ := vars.VarByNameTry(int(vgpu.PushSet), "PushOff")
	pl.Push(cmd, pvar, unsafe.Pointer(&PushOff{Off: uint32(off)}))
	pl.ComputeDispatch1D(cmd, n, gp.NThreads)
}

///////////////////////////////////////////////////////////////
//  Actual Network computation functions

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
	gp.Sys.ComputeSubmitWait(cmd)
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

	exr := gp.SyncRegionStruct("Exts")
	cxr := gp.SyncRegionStruct("Ctx")
	glr := gp.SyncRegionStruct("Globals")
	gp.StartRun(cmd)
	gp.Sys.ComputeCopyToGPU(cmd, exr, cxr, glr)
	gp.Sys.ComputeWaitMemHostToShader(cmd)
	gp.RunPipelineNoWait(cmd, "ApplyExts", neurDataN)
	gp.Sys.ComputeCmdEnd(cmd)
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
	gp.Sys.ComputeSubmitWait(cmd)
	gp.CopyStateFromStaging()
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
	glr := gp.SyncRegionStruct("Globals")
	lvr := gp.SyncRegionStruct("LayValues")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")
	nrar := gp.SyncRegionStruct("NeuronAvgs") // not strictly needed but consistency..

	maxData := int(gp.Net.MaxData)
	layDataN := gp.Net.EmerNetwork.NumLayers() * maxData
	neurDataN := int(gp.Net.NNeurons) * maxData
	poolDataN := len(gp.Net.Pools)

	gp.StartRun(cmd)
	gp.Sys.ComputeCopyToGPU(cmd, cxr, glr) // staging -> GPU
	gp.Sys.ComputeWaitMemHostToShader(cmd)
	gp.RunPipelineMemWait(cmd, "GatherSpikes", neurDataN)

	gp.RunPipelineMemWait(cmd, "LayGi", layDataN)
	gp.RunPipelineMemWait(cmd, "BetweenGi", layDataN)
	gp.RunPipelineMemWait(cmd, "PoolGi", poolDataN)

	gp.RunPipelineMemWait(cmd, "Cycle", neurDataN)

	gp.RunPipelineMemWait(cmd, "SendSpike", neurDataN)
	if gp.Ctx.Testing.IsTrue() {
		gp.RunPipelineMemWait(cmd, "CyclePost", maxData)
	} else {
		gp.RunPipelineMemWait(cmd, "CyclePost", maxData)
	}

	gp.Sys.ComputeCopyFromGPU(cmd, cxr, glr, lvr, plr, nrr, nrar)
	gp.Sys.ComputeCmdEnd(cmd)
	return cmd
}

// RunCycles does multiple cycles of updating in one chunk
func (gp *GPU) RunCycles() {
	cmd := gp.RunCyclesCmd()
	gnm := "GPU:Cycles"
	gp.Net.FunTimerStart(gnm)
	stCtx := *gp.Ctx
	gp.CopyContextToStaging()
	gp.Sys.ComputeSubmitWait(cmd)
	gp.CopyLayerStateFromStaging()
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
	glr := gp.SyncRegionStruct("Globals")
	lvr := gp.SyncRegionStruct("LayValues")
	plr := gp.SyncRegionStruct("Pools")

	maxData := int(gp.Net.MaxData)
	layDataN := gp.Net.EmerNetwork.NumLayers() * maxData
	neurDataN := int(gp.Net.NNeurons) * maxData
	poolDataN := len(gp.Net.Pools)

	gp.StartRun(cmd)
	gp.Sys.ComputeCopyToGPU(cmd, cxr, glr) // staging -> GPU
	gp.Sys.ComputeWaitMemHostToShader(cmd)
	gp.RunPipelineMemWait(cmd, "GatherSpikes", neurDataN)

	for ci := 0; ci < CyclesN; ci++ {
		if ci > 0 {
			gp.RunPipelineMemWait(cmd, "GatherSpikes", neurDataN)
		}

		gp.RunPipelineMemWait(cmd, "LayGi", layDataN)
		gp.RunPipelineMemWait(cmd, "BetweenGi", layDataN)
		gp.RunPipelineMemWait(cmd, "PoolGi", poolDataN)

		gp.RunPipelineMemWait(cmd, "Cycle", neurDataN)

		gp.RunPipelineMemWait(cmd, "SendSpike", neurDataN)
		if gp.Ctx.Testing.IsTrue() {
			gp.RunPipelineMemWait(cmd, "CyclePost", maxData)
		} else {
			gp.RunPipelineMemWait(cmd, "CyclePost", maxData)
		}
		if ci < CyclesN-1 {
			gp.RunPipelineMemWait(cmd, "CycleInc", 1) // we do
		}
	}
	gp.Sys.ComputeCopyFromGPU(cmd, cxr, glr, lvr, plr)
	gp.Sys.ComputeCmdEnd(cmd)
	return cmd
}

// RunCycleSeparateFuns does one cycle of updating in a very slow manner
// that allows timing to be recorded for each function call, for profiling.
func (gp *GPU) RunCycleSeparateFuns() {
	gp.SyncContextToGPU()

	maxData := int(gp.Net.MaxData)
	layDataN := gp.Net.EmerNetwork.NumLayers() * maxData
	neurDataN := int(gp.Net.NNeurons) * maxData
	poolDataN := len(gp.Net.Pools)

	gp.RunPipelineWait("GatherSpikes", neurDataN)

	gp.RunPipelineWait("LayGi", layDataN)
	gp.RunPipelineWait("BetweenGi", layDataN)
	gp.RunPipelineWait("PoolGi", poolDataN)

	gp.RunPipelineWait("Cycle", neurDataN)

	gp.RunPipelineWait("SendSpike", neurDataN)
	gp.RunPipelineWait("CyclePost", maxData)
	if !gp.Ctx.Testing.IsTrue() {
		gp.RunPipelineWait("CyclePost", maxData)
	}
	gp.SyncLayerStateFromGPU()
}

// RunNewState runs the NewState shader to initialize state at start of new
// ThetaCycle trial.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunNewState() {
	// todo: we're not actually calling this now, due to bug in NewStateNeuron
	cmd := gp.RunNewStateCmd()
	gnm := "GPU:NewState"
	gp.Net.FunTimerStart(gnm)
	gp.Sys.ComputeSubmitWait(cmd)
	gp.Net.FunTimerStop(gnm)
}

// RunNewStateCmd returns the commands to
// run the NewState shader to update variables
// at the start of a new trial.
func (gp *GPU) RunNewStateCmd() vk.CommandBuffer {
	cnm := "RunNewState"
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)
	poolDataN := len(gp.Net.Pools)

	gp.StartRun(cmd)
	gp.RunPipelineMemWait(cmd, "NewStatePool", poolDataN)
	gp.RunPipelineNoWait(cmd, "NewStateNeuron", neurDataN) // todo: this has NrnV read = 0 bug
	gp.Sys.ComputeCmdEnd(cmd)
	return cmd
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
	gp.Sys.ComputeSubmitWait(cmd)
	gp.CopyStateFromStaging()
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
	glr := gp.SyncRegionStruct("Globals")
	lvr := gp.SyncRegionStruct("LayValues")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")

	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)
	poolDataN := len(gp.Net.Pools)

	gp.StartRun(cmd)
	gp.Sys.ComputeCopyToGPU(cmd, cxr, glr) // staging -> GPU
	gp.Sys.ComputeWaitMemHostToShader(cmd)
	gp.RunPipelineMemWait(cmd, "MinusPool", poolDataN)
	gp.RunPipelineNoWait(cmd, "MinusNeuron", neurDataN)
	gp.Sys.ComputeCopyFromGPU(cmd, cxr, glr, lvr, plr, nrr)
	gp.Sys.ComputeCmdEnd(cmd)
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
	gp.Sys.ComputeSubmitWait(cmd)
	gp.CopyStateFromStaging()
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
	glr := gp.SyncRegionStruct("Globals")
	lvr := gp.SyncRegionStruct("LayValues")
	plr := gp.SyncRegionStruct("Pools")
	nrr := gp.SyncRegionStruct("Neurons")

	neurDataN := int(gp.Net.NNeurons) * int(gp.Net.MaxData)
	poolDataN := len(gp.Net.Pools)

	gp.StartRun(cmd)
	gp.Sys.ComputeCopyToGPU(cmd, cxr, glr) // staging -> GPU
	gp.Sys.ComputeWaitMemHostToShader(cmd)
	gp.RunPipelineMemWait(cmd, "PlusPool", poolDataN)
	gp.RunPipelineNoWait(cmd, "PlusNeuron", neurDataN)

	// note: could use atomic add to accumulate CorSim stat values in LayValues tmp vars for Cosv, ssm and ssp
	// from which the overall val is computed
	// use float atomic add for this case b/c not so time critical
	// also, matrix gated could be computed all on GPU without too much difficulty.
	// this would put all standard computation on the GPU for entire ThetaCycle

	gp.Sys.ComputeCopyFromGPU(cmd, cxr, glr, lvr, plr, nrr)
	gp.Sys.ComputeCmdEnd(cmd)
	return cmd
}

///////////////////////////////////////////////////////////
//    Synaptic level computation

// SynDataNs returns the numbers for processing SynapseCas vars =
// Synapses * MaxData.  Can exceed thread count limit and require
// multiple command launches with different offsets.
// The offset is in terms of synapse index, so everything is computed
// in terms of synapse indexes, with MaxData then multiplied to get final values.
// nCmd = number of command launches, nPer = number of synapses per cmd,
// nLast = number of synapses for last command launch.
func (gp *GPU) SynDataNs() (nCmd, nPer, nLast int) {
	synN := int(gp.Net.NSyns)
	maxData := int(gp.Net.MaxData)
	maxTh := int(TheGPU.MaxComputeWorkGroupCount1D)
	maxThSyn := maxTh / maxData

	// fmt.Printf("synN: %X  maxTh: %X  maxThSyn: %X\n", synN, maxTh, maxThSyn)
	if synN < maxThSyn {
		nCmd = 1
		nPer = synN
		nLast = synN
		return
	}
	nCmd = synN / maxThSyn
	if synN%maxThSyn > 0 {
		nCmd++
	}
	nPer = synN / nCmd
	nLast = synN - (nCmd * nPer)
	// sanity checks:
	if nPer*maxData > maxTh {
		panic("axon.GPU.SynDataNs allocated too many nPer threads!")
	}
	if nLast*maxData > maxTh {
		panic(fmt.Sprintf("axon.GPU.SynDataNs allocated too many nLast threads. maxData: %d  nCmd: %d  synN: %X  nPer: %X  nLast: %X MaxComputeWorkGroupCount1D: %d", maxData, nCmd, synN, nPer, nLast, maxTh))
	}
	return
}

// RunDWt runs the DWt shader to compute weight changes.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunDWt() {
	cmd := gp.RunDWtCmd()
	gnm := "GPU:DWt"
	gp.Net.FunTimerStart(gnm)
	gp.Sys.ComputeSubmitWait(cmd)
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

	maxData := int(gp.Net.MaxData)
	nCmd, nPer, nLast := gp.SynDataNs()
	gp.StartRun(cmd)
	off := 0
	for i := 0; i < nCmd; i++ {
		n := nPer
		if i == nCmd-1 {
			n = nLast
		}
		if nCmd == 1 {
			gp.RunPipelineOffset(cmd, "DWt", n*maxData, off) // note: no wait
		} else {
			gp.RunPipelineOffset(cmd, "DWt", n*maxData, off) // note: no wait
		}
		off += n
	}
	gp.Sys.ComputeWaitMemWriteRead(cmd)
	gp.RunPipelineNoWait(cmd, "DWtFromDi", synN)
	gp.Sys.ComputeCmdEnd(cmd)
	return cmd
}

// RunWtFromDWt runs the WtFromDWt shader to update weights from weigh changes.
// The caller must check the On flag before running this, to use CPU vs. GPU
func (gp *GPU) RunWtFromDWt() {
	cmd := gp.RunWtFromDWtCmd()
	gnm := "GPU:WtFromDWt"
	gp.Net.FunTimerStart(gnm)
	gp.CopyNeuronsToStaging()
	gp.Sys.ComputeSubmitWait(cmd)
	gp.Net.FunTimerStop(gnm)
}

// RunWtFromDWtCmd returns the commands to
// run the WtFromDWt shader to update weights from weight changes.
// This also syncs neuron state from CPU -> GPU because TrgAvgFromD
// has updated that state.
func (gp *GPU) RunWtFromDWtCmd() vk.CommandBuffer {
	cnm := "RunWtFromDWt"
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	nrr := gp.SyncRegionStruct("Neurons")
	nrar := gp.SyncRegionStruct("NeuronAvgs")

	// note: not * MaxData
	synN := int(gp.Net.NSyns)
	neurN := int(gp.Net.NNeurons)

	gp.StartRun(cmd)
	gp.Sys.ComputeCopyToGPU(cmd, nrr, nrar) // staging -> GPU
	gp.Sys.ComputeWaitMemHostToShader(cmd)
	gp.RunPipelineMemWait(cmd, "DWtSubMean", neurN) // using poolgi for kicks
	gp.RunPipelineNoWait(cmd, "WtFromDWt", synN)
	gp.Sys.ComputeCmdEnd(cmd)
	return cmd
}

/////////////////////////////////////////////
//   Tests

func (gp *GPU) TestSynCaCmd() vk.CommandBuffer {
	cnm := "TestSynCa"
	cmd, err := gp.Sys.CmdBuffByNameTry(cnm)
	if err == nil {
		return cmd
	}
	cmd = gp.Sys.NewCmdBuff(cnm)

	maxData := int(gp.Net.MaxData)
	nCmd, nPer, nLast := gp.SynDataNs()
	fmt.Printf("nCmd: %d  nPer: %X  nLast: %X\n", nCmd, nPer, nLast)
	gp.StartRun(cmd)
	off := 0
	for i := 0; i < nCmd; i++ {
		n := nPer
		if i == nCmd-1 {
			n = nLast
		}
		fmt.Printf("proc i: %d  n: %X  off: %X\n", i, n, off)
		gp.RunPipelineOffset(cmd, cnm, n*maxData, off)
		off += n
	}
	gp.Sys.ComputeCmdEnd(cmd)
	return cmd
}

// TestSynCa tests writing to SynCa -- returns true if passed
func (gp *GPU) TestSynCa() bool {
	ctx := gp.Ctx
	cmd := gp.TestSynCaCmd()
	gnm := "GPU:TestSynCa"
	gp.Net.FunTimerStart(gnm)
	gp.Sys.ComputeSubmitWait(cmd)
	gp.Net.FunTimerStop(gnm)

	gp.SyncSynCaFromGPU()

	// for synapse-ordered memory
	// nCmd, nPer, nLast := gp.SynDataNs()
	// off := 0
	// for i := 0; i < nCmd; i++ {
	// 	n := nPer
	// 	if i == nCmd-1 {
	// 		n = nLast
	// 	}
	// 	ix := ctx.SynapseCaVars.Index(uint32(off), 0, CaM)
	// 	bank := uint32(ix / uint64(ctx.NetIndexes.GPUMaxBuffFloats))
	// 	res := uint32(ix % uint64(ctx.NetIndexes.GPUMaxBuffFloats))
	// 	fmt.Printf("proc: %d  ix: %X  bank: %d  res: %X\n", i, ix, bank, res)
	// 	off += n
	// }

	// for var-ordered memory:
	// for vr := CaM; vr < SynapseCaVarsN; vr++ {
	// 	ix := ctx.SynapseCaVars.Index(0, 0, vr)
	// 	bank := uint32(ix / uint64(ctx.NetIndexes.GPUMaxBuffFloats))
	// 	res := uint32(ix % uint64(ctx.NetIndexes.GPUMaxBuffFloats))
	// 	fmt.Printf("var: %d  %s   \tix: %X  bank: %d  res: %X\n", vr, vr.String(), ix, bank, res)
	// }

	limit := 2
	failed := false

	for vr := Tr; vr < SynapseCaVarsN; vr++ {
		nfail := 0
		for syni := uint32(0); syni < uint32(4); syni++ {
			for di := uint32(0); di < gp.Net.MaxData; di++ {
				ix := ctx.SynapseCaVars.Index(syni, di, vr)
				bank := uint32(ix / uint32(ctx.NetIndexes.GPUMaxBuffFloats)) // TODO:gosl was 64
				res := uint32(ix % uint32(ctx.NetIndexes.GPUMaxBuffFloats))
				iix := math.Float32bits(SynCaV(ctx, syni, di, vr))
				ix32 := uint32(ix % 0xFFFFFFFF)
				if ix32 != iix {
					fmt.Printf("FAIL: var: %d  %s   \t syni: %X  di: %d  bank: %d  res: %x  ix: %X  ixb: %X  iix: %X\n", vr, vr.String(), syni, di, bank, res, ix, 4*ix, iix)
					nfail++
					failed = true
					if nfail > limit {
						break
					}
				}
			}
			if nfail > limit {
				break
			}
		}
	}

	return !failed
}
