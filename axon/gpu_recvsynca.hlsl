// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the RecvSynCa function on all sending projections

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set
// Set 0: uniforms -- these are constant
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
// [[vk::binding(1, 0)]] uniform PrjnParams SendPrjns[]; // [Layer][SendPrjns]
[[vk::binding(2, 0)]] uniform PrjnParams RecvPrjns[]; // [Layer][RecvPrjns]

// Set 1: main network structs and vals
[[vk::binding(0, 1)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 1)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 1)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(3, 1)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(4, 1)]] RWStructuredBuffer<PrjnVals> PrjVals; // [Layer][SendPrjns]
[[vk::binding(5, 1)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurs][Syns]

// Set 2: prjn, synapse level indexes and buffer values
// [[vk::binding(0, 2)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][SendNeurs]
// [[vk::binding(1, 2)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][RecvNeurs]
// [[vk::binding(2, 2)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurs][Syns]
// [[vk::binding(3, 2)]] RWStructuredBuffer<PrjnGVals> RecvPrjnGVals; // [Layer][RecvPrjns][RecvNeurs]

void RecvSynCaSyn(in Context ctx, in PrjnParams pj, in SynIdx sidx, float rnCaSyn, float updtThr) {
	pj.RecvSynCaSyn(ctx, Synapses[sidx.SynIdx], Neurons[sidx.SendNeurIdx], rnCaSyn, updtThr);
}

void RecvSynCaSendNeurSyn2(in Context ctx, in NeurSynIdx nsi, in Neuron rn, in LayerParams slay, in PrjnParams pj) {
	if (pj.Learn.Learn == 0) {
		return;
	}
	if (rn.Spike == 0) {
		return;
	}
	float updtThr = pj.Learn.KinaseCa.UpdtThr;
	if (rn.CaSpkP < updtThr && rn.CaSpkD < updtThr) {
		return;
	}
	float rnCaSyn = rn.CaSyn * pj.Learn.KinaseCa.SpikeG * slay.Learn.CaSpk.SynSpkG;
	uint nc = nsi.SynN;
	uint st = nsi.SynSt;
	for(uint ri = 0; ri < nc; ri++) {
		uint sia = ri + st;
		RecvSynCaSyn(ctx, pj, RecvSynIdxs[sia], rnCaSyn, updtThr);
	}
}

void RecvSynCaSendNeurSyn(in Context ctx, in NeurSynIdx nsi) {
	RecvSynCaSendNeurSyn2(ctx, nsi, Neurons[nsi.NeurIdx], Layers[RecvPrjns[nsi.PrjnIdx].Idxs.SendLay], RecvPrjns[nsi.PrjnIdx]);
}


[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over RecvNeurSynIdxs
	uint ns;
	uint st;
	RecvNeurSynIdxs.GetDimensions(ns, st);
	if(idx.x < ns) {
		RecvSynCaSendNeurSyn(Ctxt[0], RecvNeurSynIdxs[idx.x]);
	}
}



