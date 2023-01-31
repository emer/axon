// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the SynCa synaptic Ca integration function on all sending projections

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set
// Set 0: uniforms -- these are constant
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
[[vk::binding(1, 0)]] uniform PrjnParams Prjns[]; // [Layer][RecvPrjns]

// Set 1: main network structs and vals
[[vk::binding(0, 1)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 1)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
// [[vk::binding(2, 1)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
// [[vk::binding(3, 1)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 1)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurs][Syns]

// Set 2: prjn, synapse level indexes and buffer values
// [[vk::binding(0, 2)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][SendNeurs]
// [[vk::binding(1, 2)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][RecvNeurs]
// [[vk::binding(2, 2)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurs][Syns]
// [[vk::binding(3, 2)]] RWStructuredBuffer<PrjnGVals> RecvPrjnGVals; // [Layer][RecvPrjns][RecvNeurs]

void SynCa2(in Context ctx, in PrjnParams pj, uint ci, inout Synapse sy, in Neuron sn, in Neuron rn) {
	if(pj.Learn.Learn == 0) {
		return;
	}
	pj.CycleSynCaSyn(ctx, sy, sn, rn);
	/*
	float ca = 0;
	if (rn.Spike != 0 || sn.Spike == 0) {
		ca = sn.CaSyn * rn.CaSyn * pj.Learn.KinaseCa.SpikeG;
		sy.Ca = ca;
	}
	*/
	// sy.CaM = .4; // even just doing this takes same amount of time!
	// sy.CaUpT = ctx.CycleTot;
	// pj.Learn.KinaseCa.FmCa(ca, sy.CaM, sy.CaP, sy.CaD);
}

void SynCa(in Context ctx, uint ci, inout Synapse sy) {
	SynCa2(ctx, Prjns[sy.PrjnIdx], ci, sy, Neurons[sy.SendIdx], Neurons[sy.RecvIdx]);
	// sy.CaM = 0.4; // even here..  same time
}


[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Synapses
	uint ns;
	uint st;
	Synapses.GetDimensions(ns, st);
	if(idx.x < ns) {
		SynCa(Ctxt[0], idx.x, Synapses[idx.x]);
	}
}



