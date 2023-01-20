// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set

// Set 0: uniforms -- these are constant
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
// [[vk::binding(1, 0)]] uniform PrjnParams SendPrjns[]; // [Layer][SendPrjns]
// [[vk::binding(2, 0)]] uniform PrjnParams RecvPrjns[]; // [Layer][RecvPrjns]

// Set 1: main network structs and vals
[[vk::binding(0, 1)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 1)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 1)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(3, 1)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(4, 1)]] RWStructuredBuffer<PrjnVals> PrjVals; // [Layer][SendPrjns]
// [[vk::binding(5, 1)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurs][Syns]

// Set 2: prjn, synapse level indexes and buffer values
// [[vk::binding(0, 2)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][SendNeurs]
// [[vk::binding(1, 2)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][RecvNeurs]
// [[vk::binding(2, 2)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurs][Syns]
// [[vk::binding(3, 2)]] RWStructuredBuffer<PrjnGVals> RecvPrjnGVals; // [Layer][RecvPrjns][RecvNeurs]


void GeExtToPool2(in Context ctx, in LayerParams ly, inout Pool pl, uint ni, in Neuron nrn) {
	uint lni = ni - ly.Idxs.NeurSt; // layer-based as in Go
	bool subPool = (pl.IsLayPool==0);
	if(subPool) {
		ly.GeExtToPool(ctx, lni, nrn, pl, Pools[pl.LayPoolIdx], subPool);
	} else {
		ly.GeExtToPool(ctx, lni, nrn, pl, pl, subPool);
	}
}

void GeExtToPool(in Context ctx, uint ni, inout Neuron nrn) {
	GeExtToPool2(ctx, Layers[nrn.LayIdx], Pools[nrn.SubPoolG], ni, nrn);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if(idx.x < ns) {
		GeExtToPool(Ctxt[0], idx.x, Neurons[idx.x]);
	}
}

