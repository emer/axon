# GPU: graphical processing unit implementation

This document provides detailed info about the GPU implementation of axon, which allows the same Go codebase to run on CPU and GPU.  [gosl](https://github.com/goki/gosl) converts the existing Go code into HLSL shader code, along with hand-written HLSL glue code in `gpu_wgsl`, all of which ends up in the `shaders` directory.  The `go generate` command in the `axon` subdirectory, or equivalent `make all` target in the `shaders` directory, must be called whenever the main codebase changes.  The `.wgsl` files are compiled via `glslc` into SPIR-V `.spv` files that are embedded into the axon library and loaded by the [vgpu](https://github.com/goki/vgpu) Vulkan GPU framework.

To add GPU support to an existing simulation, add these lines to the end of the `ConfigGUI` method to run in GUI mode:

```Go
	ss.GUI.FinalizeGUI(false) // existing -- insert below vvv
	if GPU { // GPU is global bool var flag at top -- or true if always using
		ss.Net.ConfigGPUwithGUI(&TheSim.Context)
		core.TheApp.AddQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
	return ss.GUI.Win // existing -- insert above ^^^
```

And in the `CmdArgs()` function for `-nogui` command-line execution, there is a default `-gpu` arg added by the `ecmd.Args` package, which can be passed to use gpu with this additional code:

```Go
	ss.NewRun() // existing -- insert below vvv
	if ss.Args.Bool("gpu") {
		ss.Net.ConfigGPUnoGUI(&TheSim.Context)
	}
```

and at the very end of that function:
```Go
	ss.Net.GPU.Destroy()
```

You must also add this at the end of the `ApplyInputs` method:
```Go
	ss.Net.ApplyExts(&ss.Context)
```
which actually applies the external inputs to the network in GPU mode (and does nothing in CPU mode).

If using `mpi` in combination with GPU, you need to add:
```Go
    ss.Net.SyncAllToGPU()
```
prior to calling `WtFmDWt` after sync'ing the dwt across nodes.

The `Net.GPU` object manages all the GPU functionality and the `Net.GPU.On` is checked to determine whether to use GPU vs. CPU for all the relevant function calls -- that flag can be toggled to switch.  Direct calls to GPU methods automatically check the flag and return immediately if not `On`, so it is safe to just include these directly, e.g., for additional places where memory is `Sync`'d.

The network state (everything except Synapses) is automatically synchronized at the end of the Plus Phase, so it will be visible in the netview.

Also, the `axon.LooperUpdateNetView` method now requires a `ss.Net` argument, to enable grabbing Neuron state if updating at the Cycle level.

The new `LayerTypes` and `PathTypes` enums require renaming type selectors in params and other places.  There is now just one layer type, `Layer`, so cases that were different types are now specified by the Class selector (`.` prefix) based on the LayerTypes enum, which is automatically added for each layer.

```Go
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer", "TargetLayer")
```

And it is recommended to use `LayersByType` with type enums instead of `LayersByClass` with strings, where appropriate.

* Here are the types that changed:
    * .Hidden -> .SuperLayer
    * .Inhib -> .InhibPath
    * SuperLayer -> .SuperLayer
    * CTLayer -> .CTLayer
    * PulvLayer -> .PulvinarLayer

Finally, you must move the calls to `net.Defaults`, `Params.SetObject` *after* the `net.Build()` call -- a network must be built before the params are allocated.

# GPU Variable layout:

```
Set: 0
    Role: Uniform
        Var: 0:	Layers	Struct[4]	(size: 1520)	Values: 1
Set: 1	Indexes
    Role: Storage
        Var: 0:	NeuronIxs	Uint32[534]	(size: 4)	Values: 1
        Var: 1:	SynapseIxs	Uint32[38976]	(size: 4)	Values: 1
        Var: 2:	Paths	Struct[5]	(size: 352)	Values: 1
        Var: 3:	SendCon	Struct[242]	(size: 16)	Values: 1
        Var: 4:	RecvPathIxs	Uint32[5]	(size: 4)	Values: 1
        Var: 5:	RecvCon	Struct[281]	(size: 16)	Values: 1
        Var: 6:	RecvSynIxs	Uint32[12992]	(size: 4)	Values: 1
Set: 2	Structs
    Role: Storage
        Var: 0:	Ctx	Struct	(size: 864)	Values: 1
        Var: 1:	Neurons	Float32[14596]	(size: 4)	Values: 1
        Var: 2:	NeuronAvgs	Float32[890]	(size: 4)	Values: 1
        Var: 3:	Pools	Struct[4]	(size: 1040)	Values: 1
        Var: 4:	LayValues	Struct[4]	(size: 128)	Values: 1
        Var: 5:	Exts	Float32[50]	(size: 4)	Values: 1
Set: 3	Syns
    Role: Storage
        Var: 0:	Synapses	Float32[64960]	(size: 4)	Values: 1
        Var: 1:	SynapseCas	Float32[77952]	(size: 4)	Values: 1
        Var: 2:	GBuf	Int32[843]	(size: 4)	Values: 1
        Var: 3:	GSyns	Float32[281]	(size: 4)	Values: 1
```

# General issues and strategies

* See [Managing Limits](#managing_limits) for issues with overcoming various limits imposed by the GPU architecture.

* Everything must be stored in top-level arrays of structs as shown above -- these are the only variable length data structures.

* The pathways and synapses are now organized in a receiver-based ordering, with indexes used to access in a sender-based order. 

* The core computation is invoked via the `RunCycle` method, which can actually run 10 cycles at a time in one GPU command submission, which greatly reduces overhead.  The cycle-level computation is fully implemented all on the GPU, but basic layer-level state is copied back down by default because it is very fast to do so and might be needed.

* `Context` (was Time) is copied *from CPU -> GPU* at the start of `RunCycle` and back down *from GPU -> CPU* at the end.  The GPU can update the `NeuroMod` values during the Cycle call, while Context can be updated with on the GPU side to encode global reward values and other relevant context.

* `LayerValues` and `Pool`s are copied *from GPU -> CPU* at the end of `RunCycle`, and can be used for logging, stats or other functions during the 200 cycle update.  There is a `Special` field for `LaySpecialValues` that holds generic special values that can be used for different cases.

* At end of the 200 cycle ThetaCycle, the above state plus Neurons are grabbed back from GPU -> CPU, so further stats etc can be computed on this Neuron level data.

* The GPU must have a separate impl for any code that involves accessing state outside of what it is directly processing (e.g., an individual Neuron) -- this access must go through the global state variables on the GPU, while it can be accessed via local pointer-based fields in the CPU-side.

* In a few cases, it was useful to leverage atomic add operations to aggregate values, which are fully supported on all platforms for ints, but not floats.  Thus, we convert floats to ints by multiplying by a large number (e.g., `1 << 24`) and then dividing that back out after the aggregation, effectively a fixed-precision floating point implementation, which works well given the normalized values typically present.  A signed `int` is used and the sign checked to detect overflow.  

## Type / Code organization

### CPU-side

`Layer`:

* `LayerBase` in `layerbase.go` has basic "generic" (non-algorithm-specific) infrastructure stuff for managing all of the layer-specific state (as pointers / sub-slices of global network-level arrays) and most of the `emer.Layer` API that enables the `NetView` etc.  It can be ignored for anyone focused on algorithm-specific code.  Basically, if you were writing a whole new algorithm, you should be able to just copy layerbase.go and use without much modification.

* `Layer` in `layer.go` manages algorithm-specific code (plus some infrastructure that is algorithm-dependent) and is strictly CPU-side.  It manages all the configuration, parameterization (e.g. `Defaults`), initialization (e.g., `InitWts`, `InitActs`).

* `Layer` in `layer_compute.go` pulls out the core algorithm-specific computational code that is also run GPU-side.  It mostly iterates over Neurons and calls `LayerParams` methods -- GPU-specific code does this on global vars. Also has `CyclePost` which has special computation that only happens on the CPU.

`Path`: -- mirrors the same structure as layer

* `PathBase` in `pathbase.go` manages basic infrastructure.

* `Path` in `paths.go` does algorithm-specific CPU-side stuff.

* `Path` in `path_compute.go` pulls out core algorithm-specific code that is also run on the GPU, making calls into the `PathParams` methods.

`Network`: likewise has `NetworkBase` etc.

### GPU-side 

The following Layer and Path level types contain most of the core algorithm specific code, and are used as a `uniform` constant data structure in the GPU shader code:

* `LayerParams` in `layerparams.go` has all the core algorithm parameters and methods that run on both the GPU and the CPU.  This file is converted to `shaders/layerparams.wgsl` by [gosl](https://github.com/goki/gosl).  All the methods must have args providing all of the state that is needed for the computation, which is supplied either by the GPU or CPU.  The overall layer-level parameters are further defined in:
    + `ActParams` in `act.go` -- for computing spiking neural activation.
    + `InhibParams` in `inhib.go` -- for simulated inhibitory interneuron inhibition.
    + `LearnNeuronParams` in `learn.go` -- learning-related functions at the neuron level.

* `PathParams` in `pathparams.go` has all the core algorithm parameters and methods that run on both the GPU and CPU, likewise converted to `shaders/pathparams.wgsl`.  The specific params are in:
    + `SynComParams` at the bottom of `act.go` -- synaptic communication params used in computing spiking activation.
    + `PathScaleParams` also at end of `act.go` -- pathway scaling params, for `GScale` overall value.
    + `SWtParams` in `learn.go` -- for initializing the slow and regular weight values -- most of the initial weight variation goes into SWt.
    + `LearnSynParams` in `learn.go` -- core learning algorithm at the synapse level.
    + `GScaleValues` -- these are computed from `PathScaleParams` and not user-set directly, but remain constant so are put here.

Each class of special algorithms has its own set of mostly GPU-side code:

* `deep` predictive learning in pulvinar and cortical deep layers: `deep_layers.go, deep_paths.go, deep_net.go`

* `rl` reinforcement learning (TD and Rescorla Wagner): `rl_layers.go, rl_paths.go, rl_net.go`

* `pcore` pallidal core BG model

* `pvlv` primary value, learned value conditioning model, also includes special BOA net configs

* `hip` hippocampus

# Managing Limits

Standard backprop-based neural network operate on a "tensor" as the basic unit, which corresponds to a "layer" of units.  Each layer must be processed in full before moving on to the next layer, so that forms the natural computational chunk size, and it isn't typically that big.  In axon, the entire network's worth of neurons and synapses is processed in parallel, so as we scale up the models, we run up against the limits imposed by GPU hardware and software, which are typically in the 32 bit range or lower, in contrast to the 64 bit memory addressing capability of most current CPUs.  Thus, extra steps are required to work around these limits.

**The current implementation (v1.8.x, from June 2023) has an overall hard uint32 limit (4 Gi) on number of synapses**.  Significant additional work will be required to support more than this number.  Also, a single  `Synapses` memory buffer is currently being used, with a 4 GiB memory limit, and each synapse takes 5 * 4 bytes of memory, so the effective limit is 214,748,364.  It is easy to increase this up to the 4 Gi limit, following the example of `SynapseCas`.  For reference, the LVis model has 32,448,512 synapses.

Most of the relevant limits in Vulkan are listed in the [PhysicalDeviceLimits](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html) struct which is available in the [vgpu](https://github.com/goki/vgpu) `GPU` type as `GPUProps.Limits`.  In general the limits in Vulkan are also present in [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications), so these are things that have to be worked around regardless of the implementation.

## Compute threads

In HLSL, a compute shader is parameterized by `Dispatch` indexes (equivalent to the `thread block` concept in CUDA), which determine the total number and shape of parallel compute threads that run the given compute shader (kernel in CUDA).  The threads are grouped together into a *Warp*, which shares memory access and is the minimum chunk of computation.  Each HLSL shader has a `[numthreads(x, y, z)]` directive right before the `main` function specifying how many threads per dimension are executed in each warp: [HLSL numthreads doc](https://learn.microsoft.com/en-us/windows/win32/direct3dwgsl/sm5-attributes-numthreads).  According to this [reddit/r/GraphicsProgramming post](https://www.reddit.com/r/GraphicsProgramming/comments/aeyfkh/for_compute_shaders_is_there_an_ideal_numthreads/), 
the hardware typically has 32 (NVIDIA, M1, M2) or 64 (AMD) hardware threads per warp, so 64 is typically used as a default product of threads per warp across all of the dimensions.  Here's more [HLSL docs on dispatch](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-dispatch).

Because of this lower hardware limit, the upper bounds on threads per warp (numthreads x*y*z) is not that relevant, but it is given by `maxComputeWorkGroupInvocations`, and is typically 1024 for relevant hardware: [vulkan gpuinfo browser](https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxComputeWorkGroupInvocations&platform=all).

The limit on *total number of threads* in any invocation is the main relevant limit, and is given by `maxComputeWorkGroupCount[x,y,z] * numthreads`.  The 1D `x` dimension is generally larger than the other two (`y, z`), which are almost always 2^16-1 (64k), and it varies widely across platforms: [vulkan gpuinfo browser](https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxComputeWorkGroupCount[0]&platform=all)with ~2^16 (64k) or ~2^31 (2 Gi) being the modal values.  It appears to be a largely software-defined value, as all macs with a variety of discrete GPU types, including the M1, have 2^30 (1 Gi), whereas that same chip on other platforms can have a lower value (e.g., 64k).  The modern desktop NVIDIA chips generally have 2 Gi.

Given that this limit is specified per dimension, it remains unclear exactly how all the dimensions add up into an overall total limit.  Empirically, for both the Mac M1 and NVIDIA A100, the actual hard limit was 2 Gi for a 1D case -- invoking with more than that many threads resulted in a failure on the `gpu_test_synca.wgsl` shader run in the `TestGPUSynCa` test in `bench_lvis` varying `ndata` to push the memory and compute limits, without any diagnostic warning from the `vgpu.Debug = true` mode that activates Vulkan validation.  [vgpu](https://github.com/goki/vgpu) now has a MaxComputeWorkGroupCount1D for the max threads when using just 1D (typical case) -- it is set to 2 Gi for Mac and NVIDIA.

To work around the limit, we are just launching multiple kernels with a push constant starting offset set for each, to cover the full space.

This is a useful [stackoverflow answer](https://stackoverflow.com/questions/68653519/maximum-number-of-threads-of-vulkan-compute-shader) explaining these vulkan compute shader compute thread limits.

## Memory buffers

There is a hard max storage buffer limit of 4 GiB (uint32), and `MaxStorageBufferRange` in `PhysicalDeviceLimits` has the specific number for a given device.  We use this `GPUProps.Limits.MaxStorageBufferRange` in constraining our memory allocations, but the typical GPU targets have the 4 GiB limit.

* [vulkan gpu info](https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxStorageBufferRange)
* [vulkan gpu info, mac](https://vulkan.gpuinfo.org/listreports.php?limit=maxStorageBufferRange&value=4294967295&platform=macos)
* related discussion: https://github.com/gpuweb/gpuweb/issues/1371, https://github.com/KhronosGroup/Vulkan-Docs/issues/1016, https://github.com/openxla/iree/issues/13196

# GPU Quirks

* cannot have a struct field with the same name as a NeuronVar enum in the same method context -- results in: `error: 'NrnV' : no matching overloaded function found`
    + Can rename the field where appropriate (e.g., Spike -> Spikes, GABAB -> GabaB to avoid conflicts).  
    + Can also call method at a higher or lower level to avoid the name conflict (e.g., get the variable value at the outer calling level, then pass it in as an arg) -- this is preferable if name changes are not easy (e.g., Pool.AvgMaxUpdate)

* List of param changes:
    + `Layer.Act.` -> `Layer.Acts.`
    + `Layer.Acts.GABAB.` -> `Layer.Acts.GabaB.`
    + `Layer.Acts.Spike.` -> `Layer.Acts.Spikes.`
    + `Layer.Learn.LearnCa.` -> `Layer.Learn.CaLearn.`
    
    

