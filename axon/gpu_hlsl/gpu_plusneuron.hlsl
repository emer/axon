// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// does PlusPhase Update on each Pool

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]

// Set 3: external inputs

void PlusPhaseNeuron2(in Context ctx, in LayerParams ly, uint nin, inout Neuron nrn, inout Pool pl) {
	uint ni = nin - ly.Idxs.NeurSt; // layer-based as in Go
	ly.PlusPhaseNeuron(ctx, ni, nrn, pl, Pools[ly.Idxs.PoolSt], LayVals[pl.LayIdx]);
}

void PlusPhaseNeuron(in Context ctx, uint nin, inout Neuron nrn) {
	PlusPhaseNeuron2(ctx, Layers[nrn.LayIdx], nin, nrn, Pools[nrn.SubPoolN]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Neurons
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if(idx.x < ns) {
		PlusPhaseNeuron(Ctx[0], idx.x, Neurons[idx.x]);
	}
}

