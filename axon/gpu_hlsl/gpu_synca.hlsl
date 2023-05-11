// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the SynCa synaptic Ca integration function on all sending projections

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
[[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][SendPrjns]
[[vk::binding(1, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(2, 1)]] StructuredBuffer<uint> RecvPrjnIdxs; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(3, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]
[[vk::binding(4, 1)]] StructuredBuffer<uint> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurons][Syns]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
// [[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]

// Set 3: external inputs

void SynCaSendSyn(in Context ctx, in PrjnParams pj, inout Synapse sy, float snCaSyn, float updtThr) {
	pj.SynCaSyn(ctx, sy, Neurons[sy.RecvIdx], snCaSyn, updtThr);
}

void SynCaRecvSyn(in Context ctx, in PrjnParams pj, inout Synapse sy, float rnCaSyn, float updtThr) {
	if (Neurons[sy.SendIdx].Spike > 0) { // already processed in Send
		return;
	}
	pj.SynCaSyn(ctx, sy, Neurons[sy.SendIdx], rnCaSyn, updtThr);
}

void SynCaSendPrjn(in Context ctx, in PrjnParams pj, in LayerParams ly, uint ni, in Neuron sn, float updtThr) {
	if (pj.Learn.Learn == 0) {
		return;
	}
	
	float snCaSyn = pj.Learn.KinaseCa.SpikeG * sn.CaSyn;
	uint cni = pj.Idxs.SendConSt + ni;
	uint synst = pj.Idxs.SynapseSt + SendCon[cni].Start;
	uint synn = SendCon[cni].N;
	
	for (uint ci = 0; ci < synn; ci++) {
		SynCaSendSyn(ctx, pj, Synapses[synst + ci], snCaSyn, updtThr);
	}
}

void SynCaRecvPrjn(in Context ctx, in PrjnParams pj, in LayerParams ly, uint ni, in Neuron rn, float updtThr) {
	if (pj.Learn.Learn == 0) {
		return;
	}
	
	float rnCaSyn = pj.Learn.KinaseCa.SpikeG * rn.CaSyn;
	uint cni = pj.Idxs.RecvConSt + ni;
	uint synst = pj.Idxs.RecvSynSt + RecvCon[cni].Start;
	uint synn = RecvCon[cni].N;
	
	for (uint ci = 0; ci < synn; ci++) {
		SynCaRecvSyn(ctx, pj, Synapses[RecvSynIdxs[synst + ci]], rnCaSyn, updtThr);
	}
}

void SynCa2(in Context ctx, in LayerParams ly, uint nin, in Neuron nrn) {
	float updtThr = ly.Learn.CaLrn.UpdtThr;

	if ((nrn.CaSpkP < updtThr) && (nrn.CaSpkD < updtThr)) {
		return;
	}
	uint ni = nin - ly.Idxs.NeurSt; // layer-based as in Go
	
	for (uint pi = 0; pi < ly.Idxs.SendN; pi++) {
		SynCaSendPrjn(ctx, Prjns[ly.Idxs.SendSt + pi], ly, ni, nrn, updtThr);
	}
	for (uint pi = 0; pi < ly.Idxs.RecvN; pi++) {
		SynCaRecvPrjn(ctx, Prjns[RecvPrjnIdxs[ly.Idxs.RecvSt + pi]], ly, ni, nrn, updtThr);
	}
}

void SynCa(in Context ctx, uint nin, in Neuron nrn) {
	if (nrn.Spike == 0) {
		return;
	}
	SynCa2(ctx, Layers[nrn.LayIdx], nin, nrn);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if(idx.x < ns) {
		SynCa(Ctx[0], idx.x, Neurons[idx.x]);
	}
}


