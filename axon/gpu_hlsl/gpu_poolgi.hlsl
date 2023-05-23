// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// calls SubPoolGiFmSpikes on sub-pools, after poolgemax has been called.

// note: all must be visible always because accessor methods refer to them
[[vk::binding(0, 1)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Idxs]
[[vk::binding(1, 1)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(0, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 3)]] RWStructuredBuffer<float> SynapseCas;  // [Layer][SendPrjns][SendNeurons][Syns][Data]

#include "context.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform indexes and prjn params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools][Data]
[[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer][Data]


void PoolGi2(in Context ctx, in LayerParams ly, uint di, inout Pool pl, float giMult) {
	if(pl.IsLayPool == 0) {
		pl.AvgMax.Calc(pl.LayIdx);
		pl.Inhib.IntToRaw();
		ly.SubPoolGiFmSpikes(ctx, di, pl, Pools[ly.Idxs.PoolIdx(0, di)], ly.Inhib.Layer.On == 1, giMult);
	}
}

void PoolGi(in Context ctx, uint di, inout Pool pl) {
	PoolGi2(ctx, Layers[pl.LayIdx], di, pl, LayVals[ctx.NetIdxs.ValsIdx(pl.LayIdx, di)].ActAvg.GiMult);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Pools * Data (all pools)
	uint npi = idx.x; // network pi
	if (!Ctx[0].NetIdxs.PoolDataIdxIsValid(npi)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	PoolGi(Ctx[0], di, Pools[npi]);
}

