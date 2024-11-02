// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the DWtSubMean synaptic Ca integration function on all sending projections

// note: all must be visible always because accessor methods refer to them
[[vk::binding(0, 1)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Indexes]
[[vk::binding(1, 1)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPaths][SendNeurons][Syns]
[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(5, 2)]] RWStructuredBuffer<float> Globals;  // [NGlobals]

[[vk::binding(2, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPaths][SendNeurons][Syns]
[[vk::binding(0, 4)]] RWStructuredBuffer<float> SynapseCas0;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(1, 4)]] RWStructuredBuffer<float> SynapseCas1;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(2, 4)]] RWStructuredBuffer<float> SynapseCas2;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(3, 4)]] RWStructuredBuffer<float> SynapseCas3;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(4, 4)]] RWStructuredBuffer<float> SynapseCas4;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(5, 4)]] RWStructuredBuffer<float> SynapseCas5;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(6, 4)]] RWStructuredBuffer<float> SynapseCas6;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(7, 4)]] RWStructuredBuffer<float> SynapseCas7;  // [Layer][SendPaths][SendNeurons][Syns][Data]

#include "context.hlsl"
#include "layerparams.hlsl"
#include "pathparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have paths also be uniform..
[[vk::binding(0, 0)]] StructuredBuffer<LayerParams> Layers; // [Layer]
[[vk::binding(1, 0)]] StructuredBuffer<PathParams> Paths; // [Layer][SendPaths]

// Set 1: effectively uniform indexes and path params as structured buffers in storage
// [[vk::binding(2, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPaths][SendNeurons]
[[vk::binding(3, 1)]] StructuredBuffer<uint> RecvPathIndexes; // [Layer][RecvPaths][RecvNeurons]
[[vk::binding(4, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPaths][RecvNeurons]
[[vk::binding(5, 1)]] StructuredBuffer<uint> RecvSynIndexes; // [Layer][RecvPaths][RecvNeurons][Syns]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]


void DWtSubMeanPath(in Context ctx, in PathParams pj, in LayerParams ly, uint ri, uint lni) {
	float sm = pj.Learn.Trace.SubMean;
	if (sm == 0) {
		return;
	}
	uint cni = pj.Indexes.RecvConSt + lni;
	uint synst = pj.Indexes.RecvSynSt + RecvCon[cni].Start;
	uint synn = RecvCon[cni].N;

	float sumDWt = 0;
	int nnz = 0;
	for (uint ci = 0; ci < synn; ci++) {
		uint syni = RecvSynIndexes[synst + ci];
		float dw = SynV(ctx, syni, DWt);
		if (dw != 0) {
			sumDWt += dw;
			nnz++;
		}
	}
	if (nnz <= 1) {
		return;
	}
	sumDWt /= float(nnz);
	for (uint sci = 0; sci < synn; sci++) {
		uint syni = RecvSynIndexes[synst + sci];
		float dw = SynV(ctx, syni, DWt);
		if (dw != 0) {
			AddSynV(ctx, syni, DWt, -sm * sumDWt);
		}
	}	
}

void DWtSubMean2(in Context ctx, in LayerParams ly, uint ri) {
	uint lni = ri - ly.Indexes.NeurSt; // layer-based as in Go
	for (uint pi = 0; pi < ly.Indexes.RecvN; pi++) {
		DWtSubMeanPath(ctx, Paths[RecvPathIndexes[ly.Indexes.RecvSt + pi]], ly, ri, lni);
	}
}

void DWtSubMean(in Context ctx, uint ri) {
	uint li = NrnI(ctx, ri, NrnLayIndex);
	DWtSubMean2(ctx, Layers[li], ri);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons -- NOT Data
	uint ri = idx.x;
	if (!Ctx[0].NetIndexes.NeurIndexIsValid(ri)) {
		return;
	}
	DWtSubMean(Ctx[0], ri);
}



