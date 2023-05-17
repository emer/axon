// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// does MinusPhase Update on each Pool

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
// [[vk::binding(1, 2)]] StructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]

// Set 3: external inputs

void MinusPool2(in Context ctx, in LayerParams ly, inout Pool pl, inout LayerVals vals) {
	ly.MinusPhasePool(ctx, pl);
	if (pl.IsLayPool != 0) {
		ly.AvgGeM(ctx, pl, vals);
	}
}

void MinusPool(in Context ctx, uint pi, inout Pool pl) {
	MinusPool2(ctx, Layers[pl.LayIdx], pl, LayVals[pl.LayIdx]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Pools
	uint ns;
	uint st;
	Pools.GetDimensions(ns, st);
	if(idx.x < ns) {
		MinusPool(Ctx[0], idx.x, Pools[idx.x]);
	}
}

