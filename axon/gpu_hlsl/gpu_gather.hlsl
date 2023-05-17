// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the GatherSpikes function on all recv neurons

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
[[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][SendPrjns]
// [[vk::binding(1, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(2, 1)]] StructuredBuffer<uint> RecvPrjnIdxs; // [Layer][RecvPrjns][RecvNeurons]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(5, 2)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
[[vk::binding(6, 2)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs

void GatherSpikesPrjn(in Context ctx, in PrjnParams pj, in LayerParams ly, uint ni, inout Neuron nrn) {
	uint bi = pj.Idxs.GBufSt + pj.Com.ReadIdx(ni, ctx.CyclesTotal, pj.Idxs.RecvNeurN);
	float gRaw = pj.Com.FloatFromGBuf(GBuf[bi]);
	GBuf[bi] = 0;
	float gSyn = GSyns[pj.Idxs.GSynSt + ni];
	pj.GatherSpikes(ctx, ly, ni, nrn, gRaw, gSyn); // integrates into G*Raw; gSyn modified in fun
	GSyns[pj.Idxs.GSynSt + ni] = gSyn;	
}

void NeuronAvgMax(in Context ctx, in LayerParams ly, uint ni, in Neuron nrn) {
	AtomicInhibRawIncr(Pools[nrn.SubPoolN].Inhib, nrn.Spike, nrn.GeRaw, nrn.GeExt);
	AtomicUpdatePoolAvgMax(Pools[nrn.SubPoolN].AvgMax, nrn);
	if (Pools[nrn.SubPoolN].IsLayPool == 0) { // also update layer pool if I am a subpool
		AtomicInhibRawIncr(Pools[ly.Idxs.PoolSt].Inhib, nrn.Spike, nrn.GeRaw, nrn.GeExt);
		AtomicUpdatePoolAvgMax(Pools[ly.Idxs.PoolSt].AvgMax, nrn);
	}
}

void GatherSpikes2(in Context ctx, LayerParams ly, uint nin, inout Neuron nrn) {
	uint ni = nin - ly.Idxs.NeurSt; // layer-based as in Go

	ly.GatherSpikesInit(nrn);
	
	for (uint pi = 0; pi < ly.Idxs.RecvN; pi++) {
		GatherSpikesPrjn(ctx, Prjns[RecvPrjnIdxs[ly.Idxs.RecvSt + pi]], ly, ni, nrn);
	}
	
	// Note: Interlocked* methods can ONLY operate directly on the
	// RWStructuredBuffer items, not on arg variables.  Furthermore,
	// if you pass the pools as inout arg values, it dutifully
	// writes over any changes with the unchanged args on output!
	NeuronAvgMax(ctx, ly, ni, nrn);
}

void GatherSpikes(in Context ctx, uint nin, inout Neuron nrn) {
	GatherSpikes2(ctx, Layers[nrn.LayIdx], nin, nrn);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if (idx.x < ns) {
		GatherSpikes(Ctx[0], idx.x, Neurons[idx.x]);
	}
}

