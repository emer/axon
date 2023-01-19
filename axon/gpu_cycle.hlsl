// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the CycleNeuron function on all neurons

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]
// [[vk::binding(1, 0)]] uniform PrjnParams SendPrjns[]; // [Layer][SendPrjns]
[[vk::binding(2, 0)]] uniform PrjnParams RecvPrjns[]; // [Layer][RecvPrjns]

// [[vk::binding(0, 1)]] StructuredBuffer<NeurSynIdx> SendNeurSynIdxs; // [Layer][SendPrjns][SendNeurs]
// [[vk::binding(1, 1)]] StructuredBuffer<NeurSynIdx> RecvNeurSynIdxs; // [Layer][RecvPrjns][RecvNeurs]
// [[vk::binding(2, 1)]] StructuredBuffer<SynIdx> RecvSynIdxs; // [Layer][RecvPrjns][RecvNeurs][Syns]

[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctxt; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
// [[vk::binding(2, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][SendPrjns][SendNeurs][Syns]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(5, 2)]] RWStructuredBuffer<PrjnVals> PrjVals; // [Layer][SendPrjns]
[[vk::binding(6, 2)]] RWStructuredBuffer<PrjnGVals> RecvPrjnGVals; // [Layer][RecvPrjns][RecvNeurs]

void NeuronGatherSpikesPrjn(in LayerParams ly, in PrjnParams pj, uint ni, inout Neuron nrn, in Context ctxt) {
	pj.NeuronGatherSpikesPrjn(RecvPrjnGVals[pj.Idxs.RecvPrjnGVSt + ni], ni, nrn, ctxt);
}

void NeuronGatherSpikes(in LayerParams ly, uint ni, inout Neuron nrn, in Context ctxt) {
	ly.NeuronGatherSpikesInit(ni, nrn, ctxt);
	for (uint pi = 0; pi < ly.Idxs.RecvN; pi++) {
		NeuronGatherSpikesPrjn(ly, RecvPrjns[ly.Idxs.RecvSt + pi], ni, nrn, ctxt);
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

void CycleNeuron2(in LayerParams ly, uint ni, inout Neuron nrn, in Pool pl, float giMult, in Context ctxt) {
	// Note: following is same as Layer.GInteg
	uint lni = ni - ly.Idxs.NeurSt; // layer-based as in Go
	uint2 randctr = ctxt.RandCtr.Uint2();
	
	NeuronGatherSpikes(ly, lni, nrn, ctxt);
	
	float drvGe = 0;
	float nonDrvPct = 0;
	if (ly.LayType == PulvinarLayer) {
		PulvinarDriver(ly, Layers[ly.Pulv.DriveLayIdx], LayVals[ly.Pulv.DriveLayIdx], ni, drvGe, nonDrvPct);
	}

	float saveVal = ly.SpecialPreGs(ni, nrn, drvGe, nonDrvPct, ctxt, randctr);
	
	ly.GFmRawSyn(lni, nrn, ctxt, randctr);
	ly.GiInteg(lni, nrn, pl, giMult, ctxt);
	
	ly.SpecialPostGs(ni, nrn, ctxt, randctr, 0);
	// end GInteg
	
	ly.SpikeFmG(lni, nrn, ctxt);
}

void CycleNeuron(uint ni, inout Neuron nrn, in Context ctxt) {
	CycleNeuron2(Layers[nrn.LayIdx], ni, nrn, Pools[nrn.SubPoolG], LayVals[nrn.LayIdx].ActAvg.GiMult, ctxt);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if (idx.x < ns) {
		CycleNeuron(idx.x, Neurons[idx.x], Ctxt[0]);
	}
}

