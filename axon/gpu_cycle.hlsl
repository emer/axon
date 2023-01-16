// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the CycleNeuron function on all neurons

#include "time.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set
[[vk::binding(0, 0)]] uniform LayerParams Layers[];
// [[vk::binding(1, 0)]] uniform PrjnParams Prjns[];

// [[vk::binding(0, 1)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs;
// [[vk::binding(1, 1)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs;
// [[vk::binding(2, 1)]] StructuredBuffer<SynIdx> RecvSynIdxs;

[[vk::binding(0, 2)]] StructuredBuffer<Time> CTime;
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons;
// [[vk::binding(2, 2)]] RWStructuredBuffer<Synapse> Synapses;
// todo: pools, etc

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {
	// Lay.CycleNeuron(idx.x, Neurons[idx.x], time[0], time[0].RandCtr.Uint2());
	// if(idx.x == 0) {
	// 	Lay.CycleTimeInc(time[0]);
	// }
}

