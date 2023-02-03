// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// does MinusPhase Update on each Pool

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
// [[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][RecvPrjns]
// [[vk::binding(1, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
// [[vk::binding(5, 2)]] RWStructuredBuffer<float> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
// [[vk::binding(6, 2)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs
// [[vk::binding(0, 3)]] RWStructuredBuffer<float> Exts;  // [In / Out Layers][Neurons]

void MinusPhaseNeuron(in Context ctx, in LayerParams ly, uint ni, inout Neuron nrn, in Pool lpl, in LayerVals vals) {
	ly.MinusPhaseNeuron(ctx, ni, nrn, Pools[nrn.SubPoolN], lpl, vals);
}

void MinusPhase2(in Context ctx, uint pi, inout Pool pl, in LayerParams ly, inout LayerVals vals) {
	ly.MinusPhasePool(ctx, pl);
	if (pl.IsLayPool == 0) {
		return;
	}
	for (uint ni = pl.StIdx; ni < pl.EdIdx; ni++) {
		MinusPhaseNeuron(ctx, ly, ni, Neurons[ly.Idxs.NeurSt+ni], pl, vals);
	}
	ly.AvgGeM(ctx, pl, vals);
}

void MinusPhase(in Context ctx, uint pi, inout Pool pl) {
	MinusPhase2(ctx, pi, pl, Layers[pl.LayIdx], LayVals[pl.LayIdx]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Pools
	uint ns;
	uint st;
	Pools.GetDimensions(ns, st);
	if(idx.x < ns) {
		MinusPhase(Ctxt[0], idx.x, Pools[idx.x]);
	}
}

