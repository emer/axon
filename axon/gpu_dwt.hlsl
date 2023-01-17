// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the DWt function on all sending projections

#include "time.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set
// [[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
[[vk::binding(1, 0)]] uniform PrjnParams Prjns[]; // [Layer][SendPrjns]

[[vk::binding(0, 1)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][Send Neurs]
// [[vk::binding(1, 1)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][Recv Neurs]
// [[vk::binding(2, 1)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][Recv Neurs][Syns]

[[vk::binding(0, 2)]] StructuredBuffer<Time> CTime; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][Send Neurs][Syns]
// [[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(5, 2)]] RWStructuredBuffer<PrjnVals> PrjnVals; // [Layer][SendPrjns]

void DWtSyn(uint si, in PrjnParams pj, inout Synapse sy, in Neuron sn, in Neuron rn, in Time ctime) {
	pj.DWtTraceSynSpkThetaSyn(sy, sn, rn, ctime);
}

void DWtSendNeurSyn(uint snsi, in NeurSynIdx nsi, in Time ctime) {
	uint nc = nsi.SynN;
	uint st = nsi.SynSt;
	for(uint si = 0; si < nc; si++) {
		uint sia = si + st;
		DWtSyn(sia, Prjns[nsi.PrjnIdx], Synapses[sia], Neurons[nsi.NeurIdx], Neurons[Synapses[sia].RecvNeurIdx], ctime);
	}
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over SendNeurSynIdxs
	uint ns;
	uint st;
	SendNeurSynIdxs.GetDimensions(ns, st);
	if(idx.x < ns) {
		DWtSendNeurSyn(idx.x, SendNeurSynIdxs[idx.x], CTime[0]);
	}
}



