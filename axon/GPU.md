# GPU: gpu strategy notes

GPU variable layout:

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


    
TODO:

* build WarpSize = 64 default into vgpu compute command

* pass n layers, n prjns as fast buffer thing to shader

* Neuron.SubPoolG
* Pool.IsLayPool
* Layer.LayerIdxs
* Prjn.Idxs

* TEST more: NewState and PlusPhase need to update pool.Inhib.Clamped state -- input is now set up front.

* Figure out global vars per above
* Breakout prjn integration functions -- currently not working in Go mode because of this.

* Build all cortical variants into base type



