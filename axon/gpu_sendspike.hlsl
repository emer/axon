// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#extension GL_EXT_shader_atomic_float : enable

// performs the SendSpike function on all sending neurons

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
[[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][RecvPrjns]
[[vk::binding(1, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
// [[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
[[vk::binding(5, 2)]] RWStructuredBuffer<uint> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
[[vk::binding(6, 2)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs
// [[vk::binding(0, 3)]] RWStructuredBuffer<float> Exts;  [In / Out Layers][Neurons]

/*
void SendSpikeSyn(in Context ctx, in Synapse sy, in float sendval, in uint recvNeurSt) {
	uint bi = pj.Idxs.GBufSt + pj.Com.WriteIdx(sy.RecvIdx - recvNeurSt, ctx.CycleTot;
	GBuf[bi] +=  * scale * sy.Wt;
}

void SendSpikePrjn(in Context ctx, in PrjnParams pj, uint sendIdx, in Neuron nrn) {
	float sendVal = pj.GScale.Scale;
	if (pj.PrjnType == CTCtxtPrjn) {
		if (ctx.Cycle != ctx.ThetaCycles-1-int(pj.Com.DelLen)) {
			return;
		}
		sendVal *= Neurons[sendIdx].Burst;
	} else {
		if (nrn.Spike == 0) {
			return;
		}
	}
	uint recvNeurSt = pj.Idxs.RecvNeurSt;
	uint cni = pj.Idxs.SendConSt + sendIdx;
	// todo: need SendCon for everything!
	uint synst = pj.Idxs.SynapseSt + RecvCon[cni].Start;
	uint synn = RecvCon[cni].N;
	for (uint ci = 0; ci < synn; ci++) {
		// todo: need synapse indexes to indirect through
		SendSpikeSyn(ctx, Synapses[synst + ci], scale);
	}
}

void SendSpike2(in Context ctx, LayerParams ly, uint nin, inout Neuron nrn) {
	for (uint pi = 0; pi < ly.Idxs.SendN; pi++) {
		// todo: need an indirection list for sending prjns
		SendSpikePrjn(ctx, Prjns[ly.Idxs.SendSt + pi], ly, nin, nrn);
	}
}

*/

void SendSpike(in Context ctx, uint nin, inout Neuron nrn) {
	// SendSpike2(ctx, Layers[nrn.LayIdx], nin, nrn);
	InterlockedAdd(GBuf[nin], 1);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Send Neurons
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if (idx.x < ns) {
		SendSpike(Ctxt[0], idx.x, Neurons[idx.x]);
	}
}

