# GPU: graphical processing unit implementation

This document provides detailed info about the GPU implementation of axon, using [gosl](https://github.com/goki/gosl) to convert the existing Go code into HLSL shader code, in the `shaders` directory, which is then compiled and loaded by the [vgpu](https://github.com/goki/vgpu) Vulkan GPU framework.  This allows the same Go codebase to run on CPU and GPU.

# GPU Variable layout:

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

# General issues and strategies

* Everything must be stored in top-level arrays of structs as shown above -- these are the only variable length data structures.

* Efficient access requires Start, N indexes in data structs and there must be a contiguous layout for each different way of iterating over the data (otherwise require a singleton index array to indirect through, which is not efficient) -- this means both recv and send versions of the PrjnParams, which are constant and not a big deal to duplicate.

* `Context` (was Time) is the *only* state that is copied **from CPU -> GPU** every cycle.  At end of ThetaCycle, Neurons are grabbed back from GPU -> CPU.

* `LayerVals` are copied **from GPU -> CPU** every cycle, so anything that a layer computes that needs to be accessed in the CPU must be in LayerVals.  There is a `Special` field for `LaySpecialVals` that holds generic special values that can be used for different cases.

* Anything involving direct copying of values between different layers, as happens especially in RL algorithms, should be done in the CPU and copied into Context, or via LayerVals.  There will be a 1 cycle delay but that is fine.

* `Pools` are copied **from GPU -> CPU** at the start of the plus phase, so that detailed pool-level AvgMax stats are available for plus-phase computations prior to doing DWt.

* Note that Aggregating any values, e.g., layer-level AvgMax, must happen at the *aggregator* level - cannot do this in parallel at the lower level as memory coherency / mutex is not there.  Thus, we are doing all this aggregation in `gpu_laygi` and `gpu_poolgi` at the cycle level, and then copying snapshots from there.

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

# TODO:

* DWtSubMean!

* general renaming for params selectors:
    * .Hidden -> .SuperLayer
    * .Inhib -> .InhibPrjn
    * SuperLayer -> .SuperLayer
    * CTLayer -> .CTLayer
    * PulvLayer -> .PulvinarLayer

    * LayersByClass needs fixed if in sim
    * MUST move Defaults, Params.SetObject *after* Build()!
    
* HebbPrjn type
    
* DWt is using Context.NeuroMod for all DA, ACh values -- in principle should use LayerVals.NeuroMod in case a layer does something different.  can fix later as needed.

# BOA notes:

* BOA: need a time-integration on ACh from RSal -- can be very up-and-down and the end when used for learning can randomly be off..

* BOA: ACh is not reliable as a US -> decay factor for BG learning.  Need to have something that uniquely identifies ground-truth outcome, and updates learning only then.  For now, can just use hack..

* BOA: also the double-gating at US is a bit weird -- major over gating in general..


