// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// calls ApplyExt on neurons

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
[[vk::binding(6, 2)]] StructuredBuffer<float> Exts;  // [In / Out Layers][Neurons]

void ApplyExt2(in Context ctx, in LayerParams ly, uint ni, uint di) {
	uint lni = ni - ly.Idxs.NeurSt; // layer-based 
	ly.InitExt(ctx, ni, di);
	if (IsExtLayerType(ly.LayType)) {
		ly.ApplyExtVal(ctx, ni, di, Exts[ly.Idxs.ExtsSt + lni]);
	}
}

void ApplyExt(in Context ctx, uint ni, uint di) {
	ApplyExt2(ctx, Layers[NrnI(ctx, ni, NrnLayIdx)], ni, di);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons x Data
	uint ni = Ctx[0].NetIdxs.ItemIdx(idx.x);
	if (!Ctx[0].NetIdxs.NeurIdxIsValid(ni)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	ApplyExt(Ctx[0], ni, di);
}

