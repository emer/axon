// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// calls SubPoolGiFmSpikes on sub-pools, after poolgemax has been called.

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

void PoolGi2(in Context ctx, uint pi, inout Pool pl, in LayerParams ly, float giMult) {
	if(pl.IsLayPool == 0) {
		pl.AvgMax.Calc(pl.LayIdx);
		pl.Inhib.IntToRaw();
		ly.SubPoolGiFmSpikes(ctx, pl, Pools[ly.Idxs.PoolSt], ly.Inhib.Layer.On == 1, giMult);
	}
}

void PoolGi(in Context ctx, uint pi, inout Pool pl) {
	PoolGi2(ctx, pi, pl, Layers[pl.LayIdx], LayVals[pl.LayIdx].ActAvg.GiMult);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Pools
	uint ns;
	uint st;
	Pools.GetDimensions(ns, st);
	if(idx.x < ns) {
		PoolGi(Ctx[0], idx.x, Pools[idx.x]);
	}
}

