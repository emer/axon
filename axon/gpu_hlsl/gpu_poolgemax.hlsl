// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// calls GeToPool, AvgMax Update on each Pool
// NOTE: this is receiver-based on Pools and is very slow compared to 
// the atomic add based code in gpu_gather and is NOT being used now

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
// [[vk::binding(5, 2)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
// [[vk::binding(6, 2)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs
// [[vk::binding(0, 3)]] RWStructuredBuffer<float> Exts;  // [In / Out Layers][Neurons]

void PoolGeMaxNeuron(in Context ctx, in LayerParams ly, uint ni, inout Neuron nrn, inout Pool pl) {
	pl.Inhib.RawIncr(nrn.Spike, nrn.GeRaw, nrn.GeExt);
	pl.AvgMax.UpdateVals(nrn);
}

void PoolGeMax2(in Context ctx, uint pi, inout Pool pl, in LayerParams ly, in LayerVals vals) {
	pl.AvgMax.Init();
	for (uint ni = pl.StIdx; ni < pl.EdIdx; ni++) {
		PoolGeMaxNeuron(ctx, ly, ni, Neurons[ly.Idxs.NeurSt+ni], pl);
	}
	pl.AvgMax.Calc();
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

