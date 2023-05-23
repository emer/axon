// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// computes BetweenLayer GI on layer pools, after poolgemax has been called.

// note: all must be visible always because accessor methods refer to them
[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(3, 2)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Idxs]
[[vk::binding(0, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 3)]] RWStructuredBuffer<float> SynapseCas;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(2, 3)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPrjns][SendNeurons][Syns]

#include "context.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(4, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools][Data]


float BetweenLayerGiMax(int layIdx, uint di, float maxGi) {
	if (layIdx < 0) {
		return maxGi;
	}
	float ogi = Pools[Layers[layIdx].Idxs.PoolIdx(0, di)].Inhib.Gi;
	if (ogi > maxGi) {
		maxGi = ogi;
	}
	return maxGi;
}

void BetweenGi2(in Context ctx, in LayerParams ly, uint di, inout Pool lpl) {
	float maxGi = lpl.Inhib.Gi;
	maxGi = BetweenLayerGiMax(ly.LayInhib.Idx1, di, maxGi);
	maxGi = BetweenLayerGiMax(ly.LayInhib.Idx2, di, maxGi);
	maxGi = BetweenLayerGiMax(ly.LayInhib.Idx3, di, maxGi);
	maxGi = BetweenLayerGiMax(ly.LayInhib.Idx4, di, maxGi);
	lpl.Inhib.Gi = maxGi; // our inhib is max of us and everyone in the layer pool
}

void BetweenGi(in Context ctx, in LayerParams ly, uint li, uint di) {
	BetweenGi2(ctx, ly, di, Pools[ly.Idxs.PoolIdx(0, di)]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Layers * Data
	uint li = Ctx[0].NetIdxs.ItemIdx(idx.x);
	if (!Ctx[0].NetIdxs.LayerIdxIsValid(li)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	BetweenGi(Ctx[0], Layers[li], li, di);
}

