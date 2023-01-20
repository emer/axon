// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the CycleNeuron function on all neurons

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniforms -- these are constant
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
[[vk::binding(1, 0)]] uniform PrjnParams SendPrjns[]; // [Layer][SendPrjns]
[[vk::binding(2, 0)]] uniform PrjnParams RecvPrjns[]; // [Layer][RecvPrjns]

// Set 1: main network structs and vals
[[vk::binding(0, 1)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 1)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 1)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 1)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
[[vk::binding(4, 1)]] RWStructuredBuffer<PrjnVals> PrjVals; // [Layer][SendPrjns]
// [[vk::binding(5, 1)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurs][Syns]

// Set 2: prjn, synapse level indexes and buffer values
// [[vk::binding(0, 2)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][SendNeurs]
// [[vk::binding(1, 2)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][RecvNeurs]
// [[vk::binding(2, 2)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurs][Syns]
[[vk::binding(3, 2)]] RWStructuredBuffer<PrjnGVals> RecvPrjnGVals; // [Layer][RecvPrjns][RecvNeurs]

void NeuronGatherSpikesPrjn(in Context ctx, in LayerParams ly, in PrjnParams pj, uint ni, inout Neuron nrn) {
	pj.NeuronGatherSpikesPrjn(ctx, RecvPrjnGVals[pj.Idxs.RecvPrjnGVSt + ni], ni, nrn);
}

void NeuronGatherSpikes(in Context ctx, in LayerParams ly, uint ni, inout Neuron nrn) {
	ly.NeuronGatherSpikesInit(ctx, ni, nrn);
	for (uint pi = 0; pi < ly.Idxs.RecvN; pi++) {
		NeuronGatherSpikesPrjn(ctx, ly, RecvPrjns[ly.Idxs.RecvSt + pi], ni, nrn);
	}
}

void PulvinarDriver(in LayerParams ly, in LayerParams dly, in LayerVals vals, uint ni, out float drvGe, out float nonDrvPct) {
	float drvMax = vals.ActAvg.CaSpkP.Max;
	nonDrvPct = ly.Pulv.NonDrivePct(drvMax); // how much non-driver to keep
	uint gni = ni + ly.Idxs.NeurSt;
	if (dly.LayType == SuperLayer) {
		drvGe = ly.Pulv.DriveGe(Neurons[gni].Burst);
	} else {
		drvGe = ly.Pulv.DriveGe(Neurons[gni].CaSpkP);
	}
}

void CycleNeuron2(in Context ctx, in LayerParams ly, uint ni, inout Neuron nrn, in Pool pl, float giMult) {
	// Note: following is same as Layer.GInteg
	uint lni = ni - ly.Idxs.NeurSt; // layer-based as in Go
	uint2 randctr = ctx.RandCtr.Uint2();
	
	NeuronGatherSpikes(ctx, ly, lni, nrn);
	
	float drvGe = 0;
	float nonDrvPct = 0;
	if (ly.LayType == PulvinarLayer) {
		PulvinarDriver(ly, Layers[ly.Pulv.DriveLayIdx], LayVals[ly.Pulv.DriveLayIdx], ni, drvGe, nonDrvPct);
	}

	float saveVal = ly.SpecialPreGs(ctx, ni, nrn, drvGe, nonDrvPct, randctr);
	
	ly.GFmRawSyn(ctx, lni, nrn, randctr);
	ly.GiInteg(ctx, lni, nrn, pl, giMult);
	
	ly.SpecialPostGs(ctx, ni, nrn, randctr, 0);
	// end GInteg
	
	ly.SpikeFmG(ctx, lni, nrn);
}

void CycleNeuron(in Context ctx, uint ni, inout Neuron nrn) {
	CycleNeuron2(ctx, Layers[nrn.LayIdx], ni, nrn, Pools[nrn.SubPoolG], LayVals[nrn.LayIdx].ActAvg.GiMult);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if (idx.x < ns) {
		CycleNeuron(Ctxt[0], idx.x, Neurons[idx.x]);
	}
}

