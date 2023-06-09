// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// does NewState Update on each Pool

#include "synmem.hlsl"

// note: all must be visible always because accessor methods refer to them
[[vk::binding(0, 1)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Idxs]
[[vk::binding(1, 1)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(5, 2)]] RWStructuredBuffer<float> Globals;  // [NGlobals]
[[vk::binding(0, 3)]] RWStructuredBuffer<SynMemBlock> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 3)]] RWStructuredBuffer<SynMemBlock> SynapseCas;  // [Layer][SendPrjns][SendNeurons][Syns][Data]

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
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools][Data]
[[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer][Data]

[[vk::binding(2, 3)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1][Data]
[[vk::binding(3, 3)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons][Data]


void InitPrjnGBuffs(in Context ctx, in PrjnParams pj) {
	uint dlen = pj.Com.DelLen;
	uint maxData = ctx.NetIdxs.MaxData;
	for (uint i = 0; i < pj.Idxs.RecvNeurN; i++) {
		for (uint di = 0; di < maxData; di++) {
			GSyns[pj.Idxs.GSynSt + i*maxData + di] = 0;
			for (uint dl = 0; dl < dlen; dl++) {
				GBuf[pj.Idxs.GBufSt + i*dlen*maxData + dl*maxData + di] = 0;
			}
		}
	}
}

void NewState2(in Context ctx, in LayerParams ly, uint di, inout Pool pl, inout LayerVals vals) {
	ly.NewStatePool(ctx, pl);
	if (pl.IsLayPool == 0) {
		return;
	}
	// todo: important -- see layer_compute.go Layer::NewState method for updates 
	// calling NewStateLayerActAvg with aggregated actMinusAvg and actPlusAvg
	// need to impl that here!
	ly.NewStateLayer(ctx, pl, vals);
	for (uint pi = 0; pi < ly.Idxs.RecvN; pi++) {
		InitPrjnGBuffs(ctx, Prjns[ly.Idxs.RecvSt + pi]);
	}
}

void NewState(in Context ctx, uint di, inout Pool pl) {
	NewState2(ctx, Layers[pl.LayIdx], di, pl, LayVals[ctx.NetIdxs.ValsIdx(pl.LayIdx, di)]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Pools * Data (all pools)
	uint npi = idx.x; // network pool
	if (!Ctx[0].NetIdxs.PoolDataIdxIsValid(npi)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	if (!Ctx[0].NetIdxs.DataIdxIsValid(di)) {
		return;
	}
	NewState(Ctx[0], di, Pools[npi]);
}

