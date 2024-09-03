// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the SendSpike function on all sending neurons

// note: all must be visible always because accessor methods refer to them
[[vk::binding(0, 1)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Indexes]
[[vk::binding(1, 1)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPaths][SendNeurons][Syns]
[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(5, 2)]] RWStructuredBuffer<float> Globals;  // [NGlobals]

[[vk::binding(2, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPaths][SendNeurons][Syns]
[[vk::binding(0, 4)]] RWStructuredBuffer<float> SynapseCas0;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(1, 4)]] RWStructuredBuffer<float> SynapseCas1;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(2, 4)]] RWStructuredBuffer<float> SynapseCas2;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(3, 4)]] RWStructuredBuffer<float> SynapseCas3;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(4, 4)]] RWStructuredBuffer<float> SynapseCas4;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(5, 4)]] RWStructuredBuffer<float> SynapseCas5;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(6, 4)]] RWStructuredBuffer<float> SynapseCas6;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(7, 4)]] RWStructuredBuffer<float> SynapseCas7;  // [Layer][SendPaths][SendNeurons][Syns][Data]

#include "context.hlsl"
#include "layerparams.hlsl"
#include "pathparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have paths also be uniform..
[[vk::binding(0, 0)]] StructuredBuffer<LayerParams> Layers; // [Layer]
[[vk::binding(1, 0)]] StructuredBuffer<PathParams> Paths; // [Layer][SendPaths]

// Set 1: effectively uniform indexes and path params as structured buffers in storage
[[vk::binding(2, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPaths][SendNeurons]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(3, 2)]] StructuredBuffer<Pool> Pools; // [Layer][Pools][Data]
[[vk::binding(4, 2)]] RWStructuredBuffer<LayerValues> LayValues; // [Layer][Data]

[[vk::binding(0, 3)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPaths][RecvNeurons][MaxDel+1][Data]
// [[vk::binding(1, 3)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPaths][RecvNeurons][Data]


void SendSpikeSyn(in Context ctx, in PathParams pj, uint syni, uint di, in float sendVal, in uint recvNeurSt) {
	uint ri = SynI(ctx, syni, SynRecvIndex);
	uint bi = pj.Indexes.GBufSt + pj.Com.WriteIndex(ri - recvNeurSt, di, ctx.CyclesTotal, pj.Indexes.RecvNeurN, ctx.NetIndexes.MaxData);
	InterlockedAdd(GBuf[bi], int(sendVal * SynV(ctx, syni, Wt)));
}

void SendSpikePath(in Context ctx, in PathParams pj, uint ni, uint lni, uint di) {
	float sendVal = pj.GScale.Scale * pj.Com.FloatToIntFactor(); // baked in
	if (pj.PathType == CTCtxtPath) {
		if (ctx.Cycle != ctx.ThetaCycles-1-int(pj.Com.DelLen)) {
			return;
		}
		sendVal *= NrnV(ctx, ni, di, Burst);
	} else {
		if (NrnV(ctx, ni, di, Spike) == 0) {
			return;
		}
	}
	uint recvNeurSt = pj.Indexes.RecvNeurSt;
	uint cni = pj.Indexes.SendConSt + lni;
	uint synst = pj.Indexes.SynapseSt + SendCon[cni].Start;
	uint synn = SendCon[cni].N;
	for (uint ci = 0; ci < synn; ci++) {
		SendSpikeSyn(ctx, pj, synst + ci, di, sendVal, recvNeurSt);
	}
}

void PostSpike(in Context ctx, in LayerParams ly, uint ni, uint di, in Pool pl, in Pool lpl, inout LayerValues vals) {
	ly.PostSpikeSpecial(ctx, ni, di, pl, lpl, vals); // this writes to vals
	ly.PostSpike(ctx, ni, di, pl, vals);
}

void SendSpike2(in Context ctx, LayerParams ly, uint ni, uint di) {
	uint spi = NrnI(ctx, ni, NrnSubPool);
	uint lni = ni - ly.Indexes.NeurSt;
	
	PostSpike(ctx, ly, ni, di, Pools[ly.Indexes.PoolIndex(spi, di)], Pools[ly.Indexes.PoolIndex(0, di)], LayValues[ly.Indexes.ValuesIndex(di)]);
	
	for (uint pi = 0; pi < ly.Indexes.SendN; pi++) {
		SendSpikePath(ctx, Paths[ly.Indexes.SendSt + pi], ni, lni, di);
	}
}

void SendSpike(in Context ctx, uint ni, uint di) {
	uint li = NrnI(ctx, ni, NrnLayIndex);
	SendSpike2(ctx, Layers[li], ni, di);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons * Data
	uint ni = Ctx[0].NetIndexes.ItemIndex(idx.x);
	if (!Ctx[0].NetIndexes.NeurIndexIsValid(ni)) {
		return;
	}
	uint di = Ctx[0].NetIndexes.DataIndex(idx.x);
	if (!Ctx[0].NetIndexes.DataIndexIsValid(di)) {
		return;
	}
	SendSpike(Ctx[0], ni, di);
}

