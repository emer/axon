// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the DWtFromDiDWt function on all sending projections

// note: all must be visible always because accessor methods refer to them
[[vk::binding(0, 1)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Indexes]
[[vk::binding(1, 1)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(5, 2)]] RWStructuredBuffer<float> Globals;  // [NGlobals]

[[vk::binding(2, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(0, 4)]] RWStructuredBuffer<float> SynapseCas0;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(1, 4)]] RWStructuredBuffer<float> SynapseCas1;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(2, 4)]] RWStructuredBuffer<float> SynapseCas2;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(3, 4)]] RWStructuredBuffer<float> SynapseCas3;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(4, 4)]] RWStructuredBuffer<float> SynapseCas4;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(5, 4)]] RWStructuredBuffer<float> SynapseCas5;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(6, 4)]] RWStructuredBuffer<float> SynapseCas6;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(7, 4)]] RWStructuredBuffer<float> SynapseCas7;  // [Layer][SendPrjns][SendNeurons][Syns][Data]

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] StructuredBuffer<LayerParams> Layers; // [Layer]
[[vk::binding(1, 0)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][SendPrjns]

// Set 1: effectively uniform indexes and prjn params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]


void DWtFromDiDWtSyn2(in Context ctx, in PrjnParams pj, uint syni) {
	if(pj.Learn.Learn == 0) {
		return;
	}
	pj.DWtFromDiDWtSyn(ctx, syni);
}

void DWtFromDiDWtSyn(in Context ctx, uint syni) {
	uint pi = SynI(ctx, syni, SynPrjnIndex);
	DWtFromDiDWtSyn2(ctx, Prjns[pi], syni);
}


[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Synapses NOT * Data
	uint syni = idx.x;
	if (!Ctx[0].NetIndexes.SynIndexIsValid(syni)) {
		return;
	}
	DWtFromDiDWtSyn(Ctx[0], syni);
}


