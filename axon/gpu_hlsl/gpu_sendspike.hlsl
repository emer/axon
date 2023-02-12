// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// #extension GL_EXT_shader_atomic_float : enable

// performs the SendSpike function on all sending neurons

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
[[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][RecvPrjns]
// [[vk::binding(1, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]
[[vk::binding(2, 1)]] StructuredBuffer<uint> SendPrjnIdxs; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(3, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(4, 1)]] StructuredBuffer<uint> SendSynIdxs; // [Layer][SendPrjns][SendNeurons][Syns]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] StructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 2)]] StructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
[[vk::binding(5, 2)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][MaxDel+1][RecvNeurons]
// [[vk::binding(6, 2)]] StructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs
// [[vk::binding(0, 3)]] StructuredBuffer<float> Exts;  [In / Out Layers][Neurons]

void SendSpikeSyn(in Context ctx, in PrjnParams pj, in Synapse sy, in float sendVal, in uint recvNeurSt) {
	uint bi = pj.Idxs.GBufSt + pj.Com.WriteIdx(sy.RecvIdx - recvNeurSt, ctx.CycleTot, pj.Idxs.RecvNeurN);
	InterlockedAdd(GBuf[bi], int(sendVal * sy.Wt));
}

void SendSpikePrjn(in Context ctx, in PrjnParams pj, uint sendIdx, in Neuron sn) {
	float sendVal = pj.GScale.Scale * pj.Com.FloatToIntFactor(); // baked in
	if (pj.PrjnType == CTCtxtPrjn) {
		if (ctx.Cycle != ctx.ThetaCycles-1-int(pj.Com.DelLen)) {
			return;
		}
		sendVal *= sn.Burst;
	} else {
		if (sn.Spike == 0) {
			return;
		}
	}
	uint recvNeurSt = pj.Idxs.RecvNeurSt;
	uint cni = pj.Idxs.SendConSt + sendIdx;
	uint synst = pj.Idxs.SendSynSt + SendCon[cni].Start;
	uint synn = SendCon[cni].N;
	for (uint ci = 0; ci < synn; ci++) {
		SendSpikeSyn(ctx, pj, Synapses[SendSynIdxs[synst + ci]], sendVal, recvNeurSt);
	}
}

void PostSpike(in Context ctx, in LayerParams ly, uint ni, inout Neuron nrn, in Pool pl, in Pool lpl, inout LayerVals vals) {
	ly.PostSpikeSpecial(ctx, ni, nrn, pl, lpl, vals); // this writes to vals
	ly.PostSpike(ctx, ni, nrn, pl, vals);
}


void SendSpike2(in Context ctx, LayerParams ly, uint nin, inout Neuron sn) {
	uint ni = nin - ly.Idxs.NeurSt;

	PostSpike(ctx, ly, ni, sn, Pools[sn.SubPoolN], Pools[Layers[sn.LayIdx].Idxs.PoolSt], LayVals[sn.LayIdx]);
	
	for (uint pi = 0; pi < ly.Idxs.SendN; pi++) {
		SendSpikePrjn(ctx, Prjns[SendPrjnIdxs[ly.Idxs.SendSt + pi]], ni, sn);
	}
}

void SendSpike(in Context ctx, uint nin, inout Neuron sn) {
	SendSpike2(ctx, Layers[sn.LayIdx], nin, sn);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Send Neurons
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if (idx.x < ns) {
		SendSpike(Ctx[0], idx.x, Neurons[idx.x]);
	}
}

