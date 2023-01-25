// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the WtFmDWt function on all sending projections

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set
// Set 0: uniforms -- these are constant
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
[[vk::binding(1, 0)]] uniform PrjnParams SendPrjns[]; // [Layer][SendPrjns]
// [[vk::binding(2, 0)]] uniform PrjnParams RecvPrjns[]; // [Layer][RecvPrjns]

// Set 1: main network structs and vals
[[vk::binding(0, 1)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 1)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 1)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(3, 1)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(4, 1)]] RWStructuredBuffer<PrjnVals> PrjVals; // [Layer][SendPrjns]
[[vk::binding(5, 1)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurs][Syns]

// Set 2: prjn, synapse level indexes and buffer values
[[vk::binding(0, 2)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][SendNeurs]
// [[vk::binding(1, 2)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][RecvNeurs]
// [[vk::binding(2, 2)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurs][Syns]
// [[vk::binding(3, 2)]] RWStructuredBuffer<PrjnGVals> RecvPrjnGVals; // [Layer][RecvPrjns][RecvNeurs]

void WtFmDWtSyn(in Context ctx, in PrjnParams pj, inout Synapse sy) {
	pj.WtFmDWtSyn(ctx, sy);
}

void WtFmDWtSendNeurSyn2(in Context ctx, in NeurSynIdx nsi, in PrjnParams pj) {
	if(pj.Learn.Learn == 0) {
		return;
	}
	uint nc = nsi.SynN;
	uint st = nsi.SynSt;
	for(uint si = 0; si < nc; si++) {
		uint sia = si + st;
		WtFmDWtSyn(ctx, pj, Synapses[sia]);
	}
}

void WtFmDWtSendNeurSyn(in Context ctx, uint snsi, in NeurSynIdx nsi) {
	WtFmDWtSendNeurSyn2(ctx, nsi, SendPrjns[nsi.PrjnIdx]);
}


[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over SendNeurSynIdxs
	uint ns;
	uint st;
	SendNeurSynIdxs.GetDimensions(ns, st);
	if(idx.x < ns) {
		WtFmDWtSendNeurSyn(Ctxt[0], idx.x, SendNeurSynIdxs[idx.x]);
	}
}



