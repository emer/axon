# GPU: gpu strategy notes

# GPU variable layout:

Set 0:  Uniforms
    0. LayerParams -- array by layer index
    1. PrjnParams -- array by prjn index -- need flat prjn indexes!

Set 1:  non-RW Storage of indexes -- can't use uniform or ConstantBuffer
    0. SendNeurSynIdxs
    1. RecvNeurSynIdxs
    2. RecvSynIdxs
    
Set 2:  Storage
    0. Time
    1. Neurons -- array by global neuron index
    2. Synapses -- array by global synapse index?
    3. Pools -- multi-dim array?
    4. LayerVals -- array by layer index
    5. PrjnVals -- array by prjn index


# General issues and strategies

* Everything must be stored in top-level arrays of structs as shown above -- these are the only variable length data structures.

* Efficient access requires Start, N indexes in data structs and there must be a contiguous layout for each different way of iterating over the data (otherwise require a singleton index array to indirect through, which is not efficient) -- this means both recv and send versions of the PrjnParams, which are constant and not a big deal to duplicate.

* Context (was Time) is the *only* state that is copied from CPU to GPU every cycle.  At end of ThetaCycle, Neurons are grabbed back from GPU -> CPU.  Also, LayerVals are copied down from GPU -> CPU, so anything that a layer computes that needs to be accessed in the CPU must be in LayerVals.

* Anything involving direct copying of values between different layers, as happens especially in RL algorithms, should be done in the CPU and copied into Context.  There will be a 1 cycle delay but that is fine.

    
# TODO:

* general renaming for params selectors:
    * .Hidden -> .SuperLayer
    * .Inhib -> .InhibPrjn
    * SuperLayer -> .SuperLayer
    * CTLayer -> .CTLayer
    * PulvLayer -> .PulvinarLayer

    * LayersByClass needs fixed if in sim
    * MUST move Defaults, Params.SetObject *after* Build()!
    
* HebbPrjn type
    
* build WarpSize = 64 default into vgpu compute command

* pass n layers, n prjns as fast buffer thing to shader

* Neuron.SubPoolG
* Pool.IsLayPool
* Layer.LayerIdxs
* Prjn.Idxs

* TEST more: NewState and PlusPhase need to update pool.Inhib.Clamped state -- input is now set up front.

* Figure out global vars per above

* Build all cortical variants into base type
    + Deep worked well -- keep going!
    + SendCtxtGe, RecvCtxtGe in GPU

