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

* Figure out global vars per above
* Breakout prjn integration functions
* IsInputOrTarget needs to be compiled into layer params
* Params need to apply directly to Params field
* Params support for slbool
* GUI support for slbool
* slbool has string conversion methods
* Build all cortical variants into base type


