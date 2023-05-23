// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the DWt function on all sending projections

// note: all must be visible always because accessor methods refer to them

[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(3, 2)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Idxs]
[[vk::binding(0, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 3)]] RWStructuredBuffer<float> SynapseCas;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(2, 3)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPrjns][SendNeurons][Syns]

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
[[vk::binding(4, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(5, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]


void DWtSyn2(in Context ctx, in LayerParams rlay, in PrjnParams pj, uint syni, uint di, uint si, uint ri) {
	if(pj.Learn.Learn == 0) {
		return;
	}
	bool isTarget = (rlay.Acts.Clamp.IsTarget == 1);
	uint pi = NrnI(ctx, ri, NrnSubPool);

	pj.DWtSyn(ctx, syni, si, ri, di, Pools[rlay.Idxs.PoolIdx(0, di)], Pools[rlay.Idxs.PoolIdx(pi, di)], isTarget);
}

void DWtSyn(in Context ctx, uint syni, uint di) {
	uint pi = SynI(ctx, syni, SynPrjnIdx);
	uint si = SynI(ctx, syni, SynSendIdx);
	uint ri = SynI(ctx, syni, SynRecvIdx);
	DWtSyn2(ctx, Layers[Prjns[pi].Idxs.RecvLay], Prjns[pi], syni, di, si, ri);
}


[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Synapses * Data
	uint syni = Ctx[0].NetIdxs.ItemIdx(idx.x);
	if (!Ctx[0].NetIdxs.SynIdxIsValid(syni)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	DWtSyn(Ctx[0], syni, di);
}



