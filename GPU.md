# GPU: graphical processing unit implementation

This document provides detailed info about the GPU implementation of axon, which allows the same Go codebase to run on CPU and GPU.  [gosl](https://github.com/goki/gosl) converts the existing Go code into HLSL shader code, along with hand-written HLSL glue code in `gpu_hlsl`, all of which ends up in the `shaders` directory.  The `go generate` command in the `axon` subdirectory, or equivalent `make all` target in the `shaders` directory, must be called whenever the main codebase changes.  The `.hlsl` files are compiled via `glslc` into SPIR-V `.spv` files that are embedded into the axon library and loaded by the [vgpu](https://github.com/goki/vgpu) Vulkan GPU framework.

To add GPU support to an existing simulation, add these lines to the end of the `ConfigGUI` method to run in GUI mode:

```Go
	ss.GUI.FinalizeGUI(false) // existing -- insert below vvv
	if GPU { // GPU is global bool var flag at top -- or true if always using
		ss.Net.ConfigGPUwithGUI(&TheSim.Context)
		gi.SetQuitCleanFunc(func() {
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

Also, the `axon.LooperUpdtNetView` method now requires a `ss.Net` argument, to enable grabbing Neuron state if updating at the Cycle level.

The new `LayerTypes` and `PrjnTypes` enums require renaming type selectors in params and other places.  There is now just one layer type, `Layer`, so cases that were different types are now specified by the Class selector (`.` prefix) based on the LayerTypes enum, which is automatically added for each layer.

```Go
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer", "TargetLayer")
```

And it is recommended to use `LayersByType` with type enums instead of `LayersByClass` with strings, where appropriate.

* Here are the types that changed:
    * .Hidden -> .SuperLayer
    * .Inhib -> .InhibPrjn
    * SuperLayer -> .SuperLayer
    * CTLayer -> .CTLayer
    * PulvLayer -> .PulvinarLayer

Finally, you must move the calls to `net.Defaults`, `Params.SetObject` *after* the `net.Build()` call -- a network must be built before the params are allocated.

# GPU Variable layout:

Set: 0 "Params" -- read only
    Role: Uniform
        Var: 0:	Layers	Struct[4]	(size: 1280)	Vals: 1
Set: 1 "Prjns" -- read only
    Role: Storage
        Var: 0:	Prjns	Struct[5]	(size: 336)	Vals: 1
        Var: 1:	RecvCon	Struct[281]	(size: 16)	Vals: 1
        Var: 2:	RecvPrjnIdxs	Uint32[5]	(size: 4)	Vals: 1
        Var: 3:	SendCon	Struct[242]	(size: 16)	Vals: 1
        Var: 4:	RecvSynIdxs	Uint32[12992]	(size: 4)	Vals: 1
Set: 2 "Structs" -- read-write
    Role: Storage
        Var: 0:	Ctx	Struct	(size: 112)	Vals: 1
        Var: 1:	Neurons	Struct[178]	(size: 368)	Vals: 1
        Var: 2:	Pools	Struct[4]	(size: 720)	Vals: 1
        Var: 3:	LayVals	Struct[4]	(size: 112)	Vals: 1
        Var: 4:	Synapses	Struct[12992]	(size: 64)	Vals: 1
        Var: 5:	GBuf	Int32[843]	(size: 4)	Vals: 1
        Var: 6:	GSyns	Float32[281]	(size: 4)	Vals: 1
Set: 3 "Exts" -- read-only
    Role: Storage
        Var: 0:	Exts	Float32[50]	(size: 4)	Vals: 1

# General issues and strategies

* Everything must be stored in top-level arrays of structs as shown above -- these are the only variable length data structures.

* The projections and synapses are now organized in a receiver-based ordering, with indexes used to access in a sender-based order. 

* The core computation is invoked via the `RunCycle` method, which can actually run 10 cycles at a time in one GPU command submission, which greatly reduces overhead.  The cycle-level computation is fully implemented all on the GPU, but basic layer-level state is copied back down by default because it is very fast to do so and might be needed.

* `Context` (was Time) is copied *from CPU -> GPU* at the start of `RunCycle` and back down *from GPU -> CPU* at the end.  The GPU can update the `NeuroMod` values during the Cycle call, while Context can be updated with on the GPU side to encode global reward values and other relevant context.

* `LayerVals` and `Pool`s are copied *from GPU -> CPU* at the end of `RunCycle`, and can be used for logging, stats or other functions during the 200 cycle update.  There is a `Special` field for `LaySpecialVals` that holds generic special values that can be used for different cases.

* At end of the 200 cycle ThetaCycle, the above state plus Neurons are grabbed back from GPU -> CPU, so further stats etc can be computed on this Neuron level data.

* The GPU must have a separate impl for any code that involves accessing state outside of what it is directly processing (e.g., an individual Neuron) -- this access must go through the global state variables on the GPU, while it can be accessed via local pointer-based fields in the CPU-side.

* In a few cases, it was useful to leverage atomic add operations to aggregate values, which are fully supported on all platforms for ints, but not floats.  Thus, we convert floats to ints by multiplying by a large number (e.g., `1 << 24`) and then dividing that back out after the aggregation, effectively a fixed-precision floating point implementation, which works well given the normalized values typically present.  A signed `int` is used and the sign checked to detect overflow.  

## Type / Code organization

### CPU-side

`Layer`:

* `LayerBase` in `layerbase.go` has basic "generic" (non-algorithm-specific) infrastructure stuff for managing all of the layer-specific state (as pointers / sub-slices of global network-level arrays) and most of the `emer.Layer` API that enables the `NetView` etc.  It can be ignored for anyone focused on algorithm-specific code.  Basically, if you were writing a whole new algorithm, you should be able to just copy layerbase.go and use without much modification.

* `Layer` in `layer.go` manages algorithm-specific code (plus some infrastructure that is algorithm-dependent) and is strictly CPU-side.  It manages all the configuration, parameterization (e.g. `Defaults`), initialization (e.g., `InitWts`, `InitActs`).

* `Layer` in `layer_compute.go` pulls out the core algorithm-specific computational code that is also run GPU-side.  It mostly iterates over Neurons and calls `LayerParams` methods -- GPU-specific code does this on global vars. Also has `CyclePost` which has special computation that only happens on the CPU.

`Prjn`: -- mirrors the same structure as layer

* `PrjnBase` in `prjnbase.go` manages basic infrastructure.

* `Prjn` in `prjn.go` does algorithm-specific CPU-side stuff.

* `Prjn` in `prjn_compute.go` pulls out core algorithm-specific code that is also run on the GPU, making calls into the `PrjnParams` methods.

`Network`: likewise has `NetworkBase` etc.

### GPU-side 

The following Layer and Prjn level types contain most of the core algorithm specific code, and are used as a `uniform` constant data structure in the GPU shader code:

* `LayerParams` in `layerparams.go` has all the core algorithm parameters and methods that run on both the GPU and the CPU.  This file is converted to `shaders/layerparams.hlsl` by [gosl](https://github.com/goki/gosl).  All the methods must have args providing all of the state that is needed for the computation, which is supplied either by the GPU or CPU.  The overall layer-level parameters are further defined in:
    + `ActParams` in `act.go` -- for computing spiking neural activation.
    + `InhibParams` in `inhib.go` -- for simulated inhibitory interneuron inhibition.
    + `LearnNeurParams` in `learn.go` -- learning-related functions at the neuron level.

* `PrjnParams` in `prjnparams.go` has all the core algorithm parameters and methods that run on both the GPU and CPU, likewise converted to `shaders/prjnparams.hlsl`.  The specific params are in:
    + `SynComParams` at the bottom of `act.go` -- synaptic communication params used in computing spiking activation.
    + `PrjnScaleParams` also at end of `act.go` -- projection scaling params, for `GScale` overall value.
    + `SWtParams` in `learn.go` -- for initializing the slow and regular weight values -- most of the initial weight variation goes into SWt.
    + `LearnSynParams` in `learn.go` -- core learning algorithm at the synapse level.
    + `GScaleVals` -- these are computed from `PrjnScaleParams` and not user-set directly, but remain constant so are put here.

Each class of special algorithms has its own set of mostly GPU-side code:

* `deep` predictive learning in pulvinar and cortical deep layers: `deep_layers.go, deep_prjns.go, deep_net.go`

* `rl` reinforcement learning (TD and Rescorla Wagner): `rl_layers.go, rl_prjns.go, rl_net.go`

# GPU Quirks

* cannot have a struct field with the same name as a NeuronVar enum in the same method context -- results in: `error: 'NrnV' : no matching overloaded function found`
    + Can rename the field where appropriate (e.g., Spike -> Spikes, GABAB -> GabaB to avoid conflicts).  
    + Can also call method at a higher or lower level to avoid the name conflict (e.g., get the variable value at the outer calling level, then pass it in as an arg) -- this is preferable if name changes are not easy (e.g., Pool.AvgMaxUpdate)

* List of param changes:
    + `Layer.Act.` -> `Layer.Acts.`
    + `Layer.Acts.GABAB.` -> `Layer.Acts.GabaB.`
    + `Layer.Acts.Spike.` -> `Layer.Acts.Spikes.`
    + `Layer.Acts.Attn.` -> `Layer.Acts.AttnMod.`
    + `Layer.Learn.CaLrn.` -> `Layer.Learn.CaLearn.`
    
    
# TODO:

* HebbPrjn type
    
* DWt is using Context.NeuroMod for all DA, ACh values -- in principle should use LayerVals.NeuroMod in case a layer does something different.  can fix later as needed.


