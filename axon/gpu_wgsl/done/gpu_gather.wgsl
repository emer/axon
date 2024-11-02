// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the GatherSpikes function on all recv neurons

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

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools][Data]
// [[vk::binding(4, 2)]] RWStructuredBuffer<LayerValues> LayValues; // [Layer][Data]

[[vk::binding(0, 3)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPaths][RecvNeurons][MaxDel+1][Data]
[[vk::binding(1, 3)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPaths][RecvNeurons][Data]

void GatherSpikesPath(in Context ctx, in PathParams pj, in LayerParams ly, uint ni, uint di, uint lni) {
	uint bi = pj.Indexes.GBufSt + pj.Com.ReadIndex(lni, di, ctx.CyclesTotal, pj.Indexes.RecvNeurN, ly.Indexes.MaxData);
	float gRaw = pj.Com.FloatFromGBuf(GBuf[bi]);
	GBuf[bi] = 0;
	uint gsi = lni*ly.Indexes.MaxData + di;
	float gSyn = GSyns[pj.Indexes.GSynSt + gsi];
	pj.GatherSpikes(ctx, ly, ni, di, gRaw, gSyn); // integrates into G*Raw; gSyn modified in fun
	GSyns[pj.Indexes.GSynSt + gsi] = gSyn;	
}

// Note: Interlocked* methods can ONLY operate directly on the
// RWStructuredBuffer items, not on arg variables.  Furthermore,
// if you pass the pools as inout arg values, it dutifully
// writes over any changes with the unchanged args on output!

void NeuronAvgMax2(in Context ctx, in LayerParams ly, uint pi, uint lpi, uint ni, uint di) {
	float nrnSpike = NrnV(ctx, ni, di, Spike);
	float nrnGeRaw = NrnV(ctx, ni, di, GeRaw);
	float nrnGeExt = NrnV(ctx, ni, di, GeExt);
	AtomicInhibRawIncr(Pools[pi].Inhib, nrnSpike, nrnGeRaw, nrnGeExt, Pools[pi].NNeurons());
	AtomicUpdatePoolAvgMax(Pools[pi].AvgMax, ctx, ni, di);
	if (Pools[pi].IsLayPool == 0) { // also update layer pool if I am a subpool
		AtomicInhibRawIncr(Pools[lpi].Inhib, nrnSpike, nrnGeRaw, nrnGeExt, Pools[lpi].NNeurons());
		AtomicUpdatePoolAvgMax(Pools[lpi].AvgMax, ctx, ni, di);
	}
}

void NeuronAvgMax(in Context ctx, in LayerParams ly, uint ni, uint di) {
	uint pi = NrnI(ctx, ni, NrnSubPool);
	NeuronAvgMax2(ctx, ly, ly.Indexes.PoolIndex(pi, di), ly.Indexes.PoolIndex(0, di), ni, di);
}

void GatherSpikes2(in Context ctx, LayerParams ly, uint ni, uint di) {
	uint lni = ni - ly.Indexes.NeurSt; // layer-based

	ly.GatherSpikesInit(ctx, ni, di);
	
	for (uint pi = 0; pi < ly.Indexes.RecvN; pi++) {
		GatherSpikesPath(ctx, Paths[RecvPathIndexes[ly.Indexes.RecvSt + pi]], ly, ni, di, lni);
	}
	
	NeuronAvgMax(ctx, ly, ni, di);
}

void GatherSpikes(in Context ctx, uint ni, uint di) {
	uint li = NrnI(ctx, ni, NrnLayIndex);
	GatherSpikes2(ctx, Layers[li], ni, di);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons * Data
	uint ni = Ctx[0].NetIndexes.ItemIndex(idx.x);
	if (!Ctx[0].NetIndexes.NeurIndexIsValid(ni)) {
		return;
	}
	uint di = Ctx[0].NetIndexes.DataIndex(idx.x);
	if (!Ctx[0].NetIndexes.DataIndexIsValid(di)) {
		return;
	}
	GatherSpikes(Ctx[0], ni, di);
}

