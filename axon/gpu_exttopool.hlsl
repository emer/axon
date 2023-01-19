// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
// [[vk::binding(1, 0)]] uniform PrjnParams Prjns[]; // [Layer][SendPrjns]

// [[vk::binding(0, 1)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][Send Neurs]
// [[vk::binding(1, 1)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][Recv Neurs]
// [[vk::binding(2, 1)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][Recv Neurs][Syns]

[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
// [[vk::binding(2, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][Send Neurs][Syns]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(5, 2)]] RWStructuredBuffer<PrjnVals> PrjnVals; // [Layer][SendPrjns]

void GeExtToPool2(in LayerParams ly, inout Pool pl, uint ni, in Neuron nrn, in Context ctxt) {
	uint lni = ni - ly.Idxs.NeurSt; // layer-based as in Go
	bool subPool = (pl.IsLayPool==0);
	if(subPool) {
		ly.GeExtToPool(lni, nrn, pl, Pools[pl.LayPoolIdx], subPool, ctxt);
	} else {
		ly.GeExtToPool(lni, nrn, pl, pl, subPool, ctxt);
	}
}

void GeExtToPool(uint ni, inout Neuron nrn, in Context ctxt) {
	GeExtToPool2(Layers[nrn.LayIdx], Pools[nrn.SubPoolG], ni, nrn, ctxt);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if(idx.x < ns) {
		GeExtToPool(idx.x, Neurons[idx.x], Ctxt[0]);
	}
}

