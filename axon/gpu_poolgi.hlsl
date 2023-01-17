// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "time.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
// [[vk::binding(1, 0)]] uniform PrjnParams Prjns[]; // [Layer][SendPrjns]

// [[vk::binding(0, 1)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][Send Neurs]
// [[vk::binding(1, 1)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][Recv Neurs]
// [[vk::binding(2, 1)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][Recv Neurs][Syns]

[[vk::binding(0, 2)]] StructuredBuffer<Time> CTime; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
// [[vk::binding(2, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][Send Neurs][Syns]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(5, 2)]] RWStructuredBuffer<PrjnVals> PrjnVals; // [Layer][SendPrjns]

void PoolGi2(uint pi, inout Pool pl, in LayerParams ly, float giMult, in Time ctime) {
	if(pl.IsLayPool == 0) {
		ly.SubPoolGiFmSpikes(pl, Pools[pl.LayPoolIdx], ly.Inhib.Layer.On == 1, giMult, ctime);
	}
}

void PoolGi(uint pi, inout Pool pl, in Time ctime) {
	PoolGi2(pi, pl, Layers[pl.LayIdx], LayVals[pl.LayIdx].ActAvg.GiMult, ctime);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {
	uint ns;
	uint st;
	Pools.GetDimensions(ns, st);
	if(idx.x < ns) {
		PoolGi(idx.x, Pools[idx.x], CTime[0]);
	}
}

