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
[[vk::binding(3, 1)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(4, 1)]] RWStructuredBuffer<PrjnVals> PrjVals; // [Layer][SendPrjns]
// [[vk::binding(5, 1)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurs][Syns]

// Set 2: prjn, synapse level indexes and buffer values
// [[vk::binding(0, 2)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][SendNeurs]
// [[vk::binding(1, 2)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][RecvNeurs]
// [[vk::binding(2, 2)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurs][Syns]
// [[vk::binding(3, 2)]] RWStructuredBuffer<PrjnGVals> RecvPrjnGVals; // [Layer][RecvPrjns][RecvNeurs]

void CaSpkPAvgMax(in Context ctx, uint li, in LayerParams ly, inout Pool pl) {
	pl.AvgMax.Init();
	for(uint ni = pl.StIdx; ni < pl.EdIdx; ni++) {
		pl.AvgMax.UpdateVals(Neurons[ly.Idxs.NeurSt+ni], int(ni));
	}
	pl.AvgMax.CalcAvg();
}

void PoolGi2(in Context ctx, uint pi, inout Pool pl, in LayerParams ly, float giMult) {
	if(pl.IsLayPool == 0) {
		ly.SubPoolGiFmSpikes(ctx, pl, Pools[pl.LayPoolIdx], ly.Inhib.Layer.On == 1, giMult);
	}
}

void PoolGi(in Context ctx, uint pi, inout Pool pl) {
	PoolGi2(ctx, pi, pl, Layers[pl.LayIdx], LayVals[pl.LayIdx].ActAvg.GiMult);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {
	uint ns;
	uint st;
	Pools.GetDimensions(ns, st);
	if(idx.x < ns) {
		PoolGi(Ctxt[0], idx.x, Pools[idx.x]);
	}
}

