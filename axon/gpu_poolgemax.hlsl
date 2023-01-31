// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// calls GeToPool, AvgMax Update on each Pool

// note: binding is var, set

// Set 0: uniforms -- these are constant
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
// [[vk::binding(1, 0)]] uniform PrjnParams Prjns[]; // [Layer][RecvPrjns]

// Set 1: main network structs and vals
[[vk::binding(0, 1)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 1)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 1)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 1)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(4, 1)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
// [[vk::binding(5, 1)]] RWStructuredBuffer<float> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
// [[vk::binding(6, 1)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 2: prjn, synapse level indexes and buffer values
// [[vk::binding(0, 2)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]

void PoolGeMaxNeuron(in Context ctx, in LayerParams ly, uint ni, in Neuron nrn, inout Pool pl) {
	ly.GeToPool(ctx, ni, nrn, pl);
	pl.AvgMax.UpdateVals(nrn, int(ni));
}

void PoolGeMax2(in Context ctx, uint pi, inout Pool pl, in LayerParams ly, in LayerVals vals) {
	pl.AvgMax.Init();
	for (uint ni = pl.StIdx; ni < pl.EdIdx; ni++) {
		PoolGeMaxNeuron(ctx, ly, ni, Neurons[ly.Idxs.NeurSt+ni], pl);
	}
	pl.AvgMax.CalcAvg();
	if (pl.IsLayPool == 1) { // must do layer-level first in this round, then used in next round
		ly.LayPoolGiFmSpikes(ctx, pl, vals);
	}
}

void PoolGeMax(in Context ctx, uint pi, inout Pool pl) {
	PoolGeMax2(ctx, pi, pl, Layers[pl.LayIdx], LayVals[pl.LayIdx]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Pools
	uint ns;
	uint st;
	Pools.GetDimensions(ns, st);
	if(idx.x < ns) {
		PoolGeMax(Ctxt[0], idx.x, Pools[idx.x]);
	}
}

