// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the GatherSpikes function on all recv neurons

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
// [[vk::binding(2, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(3, 1)]] StructuredBuffer<uint> RecvPrjnIdxs; // [Layer][RecvPrjns][RecvNeurons]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools][Data]
// [[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer][Data]

[[vk::binding(2, 3)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1][Data]
[[vk::binding(3, 3)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons][Data]

void GatherSpikesPrjn(in Context ctx, in PrjnParams pj, in LayerParams ly, uint ni, uint di, uint lni) {
	uint bi = pj.Idxs.GBufSt + pj.Com.ReadIdx(lni, di, ctx.CyclesTotal, pj.Idxs.RecvNeurN, ly.Idxs.MaxData);
	float gRaw = pj.Com.FloatFromGBuf(GBuf[bi]);
	GBuf[bi] = 0;
	uint gsi = lni*ly.Idxs.MaxData + di;
	float gSyn = GSyns[pj.Idxs.GSynSt + gsi];
	pj.GatherSpikes(ctx, ly, ni, di, gRaw, gSyn); // integrates into G*Raw; gSyn modified in fun
	GSyns[pj.Idxs.GSynSt + gsi] = gSyn;	
}

// Note: Interlocked* methods can ONLY operate directly on the
// RWStructuredBuffer items, not on arg variables.  Furthermore,
// if you pass the pools as inout arg values, it dutifully
// writes over any changes with the unchanged args on output!

void NeuronAvgMax2(in Context ctx, in LayerParams ly, uint pi, uint lpi, uint ni, uint di) {
	float nrnSpike = NrnV(ctx, ni, di, Spike);
	float nrnGeRaw = NrnV(ctx, ni, di, GeRaw);
	float nrnGeExt = NrnV(ctx, ni, di, GeExt);
	AtomicInhibRawIncr(Pools[pi].Inhib, nrnSpike, nrnGeRaw, nrnGeExt);
	AtomicUpdatePoolAvgMax(Pools[pi].AvgMax, ctx, ni, di);
	if (Pools[pi].IsLayPool == 0) { // also update layer pool if I am a subpool
		AtomicInhibRawIncr(Pools[lpi].Inhib, nrnSpike, nrnGeRaw, nrnGeExt);
		AtomicUpdatePoolAvgMax(Pools[lpi].AvgMax, ctx, ni, di);
	}
}

void NeuronAvgMax(in Context ctx, in LayerParams ly, uint ni, uint di) {
	uint pi = NrnI(ctx, ni, NrnSubPool);
	NeuronAvgMax2(ctx, ly, ly.Idxs.PoolIdx(pi, di), ly.Idxs.PoolIdx(0, di), ni, di);
}

void GatherSpikes2(in Context ctx, LayerParams ly, uint ni, uint di) {
	uint lni = ni - ly.Idxs.NeurSt; // layer-based

	ly.GatherSpikesInit(ctx, ni, di);
	
	for (uint pi = 0; pi < ly.Idxs.RecvN; pi++) {
		GatherSpikesPrjn(ctx, Prjns[RecvPrjnIdxs[ly.Idxs.RecvSt + pi]], ly, ni, di, lni);
	}
	
	NeuronAvgMax(ctx, ly, ni, di);
}

void GatherSpikes(in Context ctx, uint ni, uint di) {
	uint li = NrnI(ctx, ni, NrnLayIdx);
	GatherSpikes2(ctx, Layers[li], ni, di);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons * Data
	uint ni = Ctx[0].NetIdxs.ItemIdx(idx.x);
	if (!Ctx[0].NetIdxs.NeurIdxIsValid(ni)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	if (!Ctx[0].NetIdxs.DataIdxIsValid(di)) {
		return;
	}
	GatherSpikes(Ctx[0], ni, di);
}

