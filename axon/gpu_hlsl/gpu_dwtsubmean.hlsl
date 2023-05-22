// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the DWtSubMean synaptic Ca integration function on all sending projections

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
// [[vk::binding(1, 1)]] StructuredBuffer<StartN> SendCon; // [Layer][SendPrjns][SendNeurons]
[[vk::binding(2, 1)]] StructuredBuffer<uint> RecvPrjnIdxs; // [Layer][RecvPrjns][RecvNeurons]
[[vk::binding(3, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]
[[vk::binding(4, 1)]] StructuredBuffer<uint> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurons][Syns]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]


void DWtSubMeanPrjn(in Context ctx, in PrjnParams pj, in LayerParams ly, uint ni, in Neuron rn) {
	float sm = pj.Learn.Trace.SubMean;
	if (sm == 0) {
		return;
	}
	float rnCaSyn = pj.Learn.KinaseCa.SpikeG * rn.CaSyn;
	uint cni = pj.Idxs.RecvConSt + ni;
	uint synst = pj.Idxs.RecvSynSt + RecvCon[cni].Start;
	uint synn = RecvCon[cni].N;

	float sumDWt = 0;
	int nnz = 0;
	for (uint ci = 0; ci < synn; ci++) {
		float dw = Synapses[RecvSynIdxs[synst + ci]].DWt;
		if (dw != 0) {
			sumDWt += dw;
			nnz++;
		}
	}
	if (nnz <= 1) {
		return;
	}
	sumDWt /= float(nnz);
	for (uint ci = 0; ci < synn; ci++) {
		float dw = Synapses[RecvSynIdxs[synst + ci]].DWt;
		if (dw != 0) {
			Synapses[RecvSynIdxs[synst + ci]].DWt -= sm * sumDWt;
		}
	}	
}

void DWtSubMean2(in Context ctx, in LayerParams ly, uint nin, in Neuron rn) {
	uint ni = nin - ly.Idxs.NeurSt; // layer-based as in Go
	
	for (uint pi = 0; pi < ly.Idxs.RecvN; pi++) {
		DWtSubMeanPrjn(ctx, Prjns[RecvPrjnIdxs[ly.Idxs.RecvSt + pi]], ly, ni, rn);
	}
}

void DWtSubMean(in Context ctx, uint nin, in Neuron rn) {
	if (rn.Spike == 0) {
		return;
	}
	DWtSubMean2(ctx, Layers[rn.LayIdx], nin, rn);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if(idx.x < ns) {
		DWtSubMean(Ctx[0], idx.x, Neurons[idx.x]);
	}
}



