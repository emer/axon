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

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]

// Set 3: external inputs

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

void BetweenGi2(in Context ctx, in LayerParams ly, inout Pool lpl) {
	float maxGi = lpl.Inhib.Gi;
	maxGi = BetweenLayerGiMax(maxGi, ly.LayInhib.Idx1);
	maxGi = BetweenLayerGiMax(maxGi, ly.LayInhib.Idx2);
	maxGi = BetweenLayerGiMax(maxGi, ly.LayInhib.Idx3);
	maxGi = BetweenLayerGiMax(maxGi, ly.LayInhib.Idx4);
	lpl.Inhib.Gi = maxGi; // our inhib is max of us and everyone in the layer pool
}

void BetweenGi(in Context ctx, uint li, in LayerParams ly) {
	BetweenGi2(ctx, ly, Pools[ly.Idxs.PoolSt]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Layers
	if (idx.x < Ctx[0].NLayers) {
		BetweenGi(Ctx[0], idx.x, Layers[idx.x]);
	}
}

