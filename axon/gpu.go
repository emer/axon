// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//go:generate gosl -exclude=Update,UpdateParams,Defaults,AllParams github.com/goki/mat32/fastexp.go github.com/emer/etable/minmax ../chans/chans.go ../chans ../kinase ../fsfffb/inhib.go ../fsfffb github.com/emer/emergent/etime github.com/emer/emergent/ringidx neuromod.go context.go neuron.go synapse.go pool.go layervals.go act.go act_prjn.go inhib.go learn.go layertypes.go layerparams.go deep_layers.go rl_layers.go pvlv_layers.go pcore_layers.go prjntypes.go prjnparams.go deep_prjns.go rl_prjns.go pvlv_prjns.go pcore_prjns.go gpu_gather.hlsl gpu_poolgi.hlsl gpu_cycle.hlsl gpu_synca.hlsl gpu_dwt.hlsl gpu_wtfmdwt.hlsl

// Full vars code -- each gpu_*.hlsl uses a subset

/*

// note: binding is var, set

// Set 0: uniforms -- these are constant
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
[[vk::binding(1, 0)]] uniform PrjnParams Prjns[]; // [Layer][RecvPrjns]

// Set 1: main network structs and vals
[[vk::binding(0, 1)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 1)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 1)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 1)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 1)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
[[vk::binding(5, 1)]] RWStructuredBuffer<float> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
[[vk::binding(6, 1)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 2: prjn, synapse level indexes and buffer values
[[vk::binding(0, 2)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]

*/

// cycle update:

// PrjnGatherSpikes:
// gpu_gfmspikes.hlsl [Prjns]

// GiFmSpikes:
// gpu_exttopool.hlsl [Neurons] // todo: can this be done in send spike??
// gpu_laygi.hlsl [Layers]
// gpu_poolgi.hlsl [Pools]

// CycleNeuron:
// gpu_cycle.hlsl [Neurons]

// gpu_sendsynca.hlsl [SendSynapses]
// gpu_recvsynca.hlsl [RecvSynapses]

// DWt
// gpu_dwt.hlsl
// later: submean
// gpu_wtfmdwt.hlsl
