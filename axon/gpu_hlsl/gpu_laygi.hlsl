// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// updates layer inhibition

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
[[vk::binding(4, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(5, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]


void LayGi2(in Context ctx, in LayerParams ly, uint li, uint di, inout Pool lpl, inout LayerVals vals) {
	lpl.AvgMax.Calc(int(li));
	lpl.Inhib.IntToRaw();
	ly.LayPoolGiFmSpikes(ctx, lpl, vals); // also updates LayerVals with NeuroMod
}

void LayGi(in Context ctx, in LayerParams ly, uint li, uint di) {
	LayGi2(ctx, ly, li, di, Pools[ly.Idxs.PoolIdx(0, di)], LayVals[ly.Idxs.ValsIdx(di)]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Layers * Data
	uint li = Ctx[0].NetIdxs.ItemIdx(idx.x);
	if (!Ctx[0].NetIdxs.LayerIdxIsValid(li)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	LayGi(Ctx[0], Layers[li], li, di);
}

