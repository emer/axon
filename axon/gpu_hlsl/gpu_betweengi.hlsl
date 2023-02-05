// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// computes BetweenLayer GI on layer pools, after poolgemax has been called.

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

float BetweenLayerGiMax(float maxGi, int layIdx) {
	if (layIdx < 0) {
		return maxGi;
	}
	float ogi = Pools[Layers[layIdx].Idxs.PoolSt].Inhib.Gi;
	if (ogi > maxGi) {
		maxGi = ogi;
	}
	return maxGi;
}

void BetweenGi2(in Context ctx, uint pi, inout Pool pl, in LayerParams ly, float giMult) {
	if (pl.IsLayPool == 0) {
		return;
	}
	float maxGi = pl.Inhib.Gi;
	maxGi = BetweenLayerGiMax(maxGi, ly.LayInhib1Idx);
	maxGi = BetweenLayerGiMax(maxGi, ly.LayInhib2Idx);
	maxGi = BetweenLayerGiMax(maxGi, ly.LayInhib3Idx);
	maxGi = BetweenLayerGiMax(maxGi, ly.LayInhib4Idx);
	pl.Inhib.Gi = maxGi; // our inhib is max of us and everyone in the layer pool
}

void BetweenGi(in Context ctx, uint pi, inout Pool pl) {
	BetweenGi2(ctx, pi, pl, Layers[pl.LayIdx], LayVals[pl.LayIdx].ActAvg.GiMult);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Pools
	uint ns;
	uint st;
	Pools.GetDimensions(ns, st);
	if(idx.x < ns) {
		BetweenGi(Ctxt[0], idx.x, Pools[idx.x]);
	}
}

