// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the PostSpike function on all sending neurons

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

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(2, 2)]] StructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]


void PostSpikeLay(in Context ctx, in LayerParams ly, uint ni, inout Neuron nrn, in Pool pl, in Pool lpl, inout LayerVals vals) {
	ly.PostSpikeSpecial(ctx, ni, nrn, pl, lpl, vals); // warning: only 1 layer type can write to vals!
	ly.PostSpike(ctx, ni, nrn, pl, vals);
}

void PostSpike2(in Context ctx, LayerParams ly, uint nin, inout Neuron sn) {
	uint ni = nin - ly.Idxs.NeurSt;
	PostSpikeLay(ctx, ly, ni, sn, Pools[sn.SubPoolN], Pools[Layers[sn.LayIdx].Idxs.PoolSt], LayVals[sn.LayIdx]);
}

void PostSpike(in Context ctx, uint nin, inout Neuron sn) {
	PostSpike2(ctx, Layers[sn.LayIdx], nin, sn);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons
	uint nn;
	uint st;
	Neurons.GetDimensions(nn, st);
	if (idx.x < nn) {
		PostSpike(Ctx[0], idx.x, Neurons[idx.x]);
	}
}

