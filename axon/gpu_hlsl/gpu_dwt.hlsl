// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the DWt function on all sending projections

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
[[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][SendPrjns]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]

// Set 3: external inputs

void DWtSyn2(in Context ctx, in LayerParams rlay, in PrjnParams pj, uint ci, inout Synapse sy, in Neuron sn, in Neuron rn) {
	if(pj.Learn.Learn == 0) {
		return;
	}
	bool isTarget = (rlay.Act.Clamp.IsTarget == 1);

	pj.DWtSyn(ctx, sy, sn, rn, Pools[rlay.Idxs.PoolSt], Pools[rn.SubPoolN], isTarget);
}

void DWtSyn(in Context ctx, uint ci, inout Synapse sy) {
	DWtSyn2(ctx, Layers[Prjns[sy.PrjnIdx].Idxs.RecvLay], Prjns[sy.PrjnIdx], ci, sy, Neurons[sy.SendIdx], Neurons[sy.RecvIdx]);
}


[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Synapses
	uint ns;
	uint st;
	Synapses.GetDimensions(ns, st);
	if(idx.x < ns) {
		DWtSyn(Ctx[0], idx.x, Synapses[idx.x]);
	}
}



