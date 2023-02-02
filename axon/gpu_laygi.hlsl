// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
// [[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][RecvPrjns]
// [[vk::binding(1, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
// [[vk::binding(5, 2)]] RWStructuredBuffer<float> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
// [[vk::binding(6, 2)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs
// [[vk::binding(0, 3)]] RWStructuredBuffer<float> Exts;  // [In / Out Layers][Neurons]

void CaSpkPAvgMax(in Context ctx, uint li, in LayerParams ly, inout Pool lpl, inout LayerVals vals) {
	lpl.AvgMax.Init();
	for(uint ni = 0; ni < ly.Idxs.NeurN; ni++) {
		lpl.AvgMax.UpdateVals(Neurons[ly.Idxs.NeurSt+ni], int(ni));
	}
	lpl.AvgMax.CalcAvg();
}

void LayGi(in Context ctx, uint li, in LayerParams ly) {
	ly.LayPoolGiFmSpikes(ctx, Pools[ly.Idxs.PoolSt], LayVals[li]);
	CaSpkPAvgMax(ctx, li, ly, Pools[ly.Idxs.PoolSt], LayVals[li]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {
	// todo: need NLayers, NPrjns as fast params
	// if(idx.x < ns) {
	LayGi(Ctxt[0], idx.x, Layers[idx.x]);
	// }
}

