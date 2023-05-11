// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

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

void LayGi2(in Context ctx, uint li, in LayerParams ly, inout Pool pl, inout LayerVals vals) {
	pl.AvgMax.Calc(int(li));
	pl.Inhib.IntToRaw();
	ly.LayPoolGiFmSpikes(ctx, pl, vals); // also updates LayerVals with NeuroMod
}

void LayGi(in Context ctx, uint li, in LayerParams ly) {
	LayGi2(ctx, li, ly, Pools[ly.Idxs.PoolSt], LayVals[li]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Layers
	if (idx.x < Ctx[0].NLayers) {
		LayGi(Ctx[0], idx.x, Layers[idx.x]);
	}
}

