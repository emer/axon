// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the SendSpike function on all sending neurons

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
[[vk::binding(1, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPrjns][SendNeurons]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(4, 2)]] StructuredBuffer<Pool> Pools; // [Layer][Paools]
[[vk::binding(5, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]

[[vk::binding(3, 3)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][MaxDel+1][RecvNeurons]
// [[vk::binding(4, 3)]] StructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs

void SendSpikeSyn(in Context ctx, in PrjnParams pj, uint syni, uint di, in float sendVal, in uint recvNeurSt) {
	uint ri = SynI(ctx, syni, SynRecvIdx);
	uint bi = pj.Idxs.GBufSt + pj.Com.WriteIdx(ri - recvNeurSt, di, ctx.CyclesTotal, pj.Idxs.RecvNeurN, ctx.NetIdxs.MaxData);
	InterlockedAdd(GBuf[bi], int(sendVal * SynV(ctx, syni, Wt)));
}

void SendSpikePrjn(in Context ctx, in PrjnParams pj, uint ni, uint lni, uint di) {
	float sendVal = pj.GScale.Scale * pj.Com.FloatToIntFactor(); // baked in
	if (pj.PrjnType == CTCtxtPrjn) {
		if (ctx.Cycle != ctx.ThetaCycles-1-int(pj.Com.DelLen)) {
			return;
		}
		sendVal *= NrnV(ctx, ni, di, Burst);
	} else {
		if (NrnV(ctx, ni, di, Spike) == 0) {
			return;
		}
	}
	uint recvNeurSt = pj.Idxs.RecvNeurSt;
	uint cni = pj.Idxs.SendConSt + lni;
	uint synst = pj.Idxs.SynapseSt + SendCon[cni].Start;
	uint synn = SendCon[cni].N;
	for (uint ci = 0; ci < synn; ci++) {
		SendSpikeSyn(ctx, pj, synst + ci, di, sendVal, recvNeurSt);
	}
}

void PostSpike(in Context ctx, in LayerParams ly, uint ni, uint di, in Pool pl, in Pool lpl, inout LayerVals vals) {
	ly.PostSpikeSpecial(ctx, ni, di, pl, lpl, vals); // this writes to vals
	ly.PostSpike(ctx, ni, di, pl, vals);
}

void SendSpike2(in Context ctx, LayerParams ly, uint ni, uint di) {
	uint pi = NrnI(ctx, ni, NrnSubPool);
	uint lni = ni - ly.Idxs.NeurSt;
	
	PostSpike(ctx, ly, ni, di, Pools[ly.Idxs.PoolIdx(pi, di)], Pools[ly.Idxs.PoolIdx(0, di)], LayVals[ly.Idxs.ValsIdx(di)]);
	
	for (uint pi = 0; pi < ly.Idxs.SendN; pi++) {
		SendSpikePrjn(ctx, Prjns[ly.Idxs.SendSt + pi], ni, lni, di);
	}
}

void SendSpike(in Context ctx, uint ni, uint di) {
	uint li = NrnI(ctx, ni, NrnLayIdx);
	SendSpike2(ctx, Layers[li], ni, di);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Send Neurons * Data
	uint ni = Ctx[0].NetIdxs.ItemIdx(idx.x);
	if (!Ctx[0].NetIdxs.NeurIdxIsValid(ni)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	SendSpike(Ctx[0], ni, di);
}

