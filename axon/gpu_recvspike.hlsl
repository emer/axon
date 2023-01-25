// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the SendSpike function on all sending projections

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
[[vk::binding(4, 1)]] RWStructuredBuffer<PrjnVals> PrjVals; // [Layer][SendPrjns]
[[vk::binding(5, 1)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurs][Syns]

// Set 2: prjn, synapse level indexes and buffer values
[[vk::binding(0, 2)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][SendNeurs]
// [[vk::binding(1, 2)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][RecvNeurs]
// [[vk::binding(2, 2)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurs][Syns]

[[vk::binding(3, 2)]] RWStructuredBuffer<PrjnGVals> SendPrjnGVals; // [Layer][SendPrjns][RecvNeurs]
[[vk::binding(4, 2)]] RWStructuredBuffer<PrjnGBuf> SendPrjnGBuf; // [Layer][SendPrjns][RecvNeurs*]

// [[vk::binding(3, 2)]] RWStructuredBuffer<PrjnGVals> RecvPrjnGVals; // [Layer][RecvPrjns][RecvNeurs]


void SendSpikeSyn2(in Context ctx, in PrjnParams pj, in Synapse sy, uint ridx, uint rnPool, float scale) {
	pj.SendSpikeSyn(ctx, sy, rn, snCaSyn, updtThr);
}


void SendSpikeSyn(in Context ctx, in PrjnParams pj, in Synapse sy, uint ri, uint rnPool, float scale) {
	float sv = scale * sy.Wt;
	
}

void SendSpikeSendNeurSyn2(in Context ctx, in PrjnParams pj, in PrjnVals pvals, in LayerParams rlay, in Neuron sn) {
	if (sn.Spike == 0) {
		return;
	}
	uint rlst = rlay.NeurStIdx;
	float scale = pj.GScale.Scale;
	uint maxDelay pj.Com.Delay;
	uint delayBufSize maxDelay + 1;
	uint currDelayIdx pvals.Gidx.Idx(maxDelay); // index in ringbuffer to put new values -- end of line.
	bool excite = pj.IsExcitatory()
	uint nc = nsi.SynN;
	uint st = nsi.SynSt;
	for (uint si = 0; si < nc; si++) {
		uint sia = si + st;
		uint riG = Synapses[sia].RecvNeurIdx; // global recv index
		uint ri = riG - rlst; // layer-relative recv idx, used for indexing into GBuf
		SendSpikeSyn(ctx, pj, Synapses[sia], ri, Neurons[riG].SubPool, scale);
	}
}

void SendSpikePrjn(in Context ctx, in PrjnParams pj, in PrjnVals pvals, in LayerParams slay) {
	uint nn = slay.Idxs.NeurN;
	uint st = slay.Idxs.NeurSt;
	for (uint si = 0; si < nn; si++) {
		SendSpikeSendNeur(ctx, pj, pvals, Layers[pj.Idxs.RecvLay], Neurons[st + si]);
	}
}

// note: this can only be parallel over SendPrjns because we
// are aggregating into per-prjn buffers, over different 
// sending neurons, multiple of which could send to the same
// recv neuron, meaning that the GBuf which is per-recv neuron 
// within the projection will get hit across diff threads if
// we tried to do it over SendNeurSynIdxs..

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over SendPrjns
	uint ns;
	uint st;
	SendPrjns.GetDimensions(ns, st);
	if(idx.x < ns) {
		SendSpikePrjn(Ctxt[0], SendPrjns[idx.x], PrjVals[idx.s], Layers[SendPrjns[idx.x].Idxs.SendLay]);
	}
}



