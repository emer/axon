// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the CycleNeuron function on all neurons

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
[[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][RecvPrjns]
// [[vk::binding(1, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] RWStructuredBuffer<Context> Ctx; // [0]
[[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
// [[vk::binding(5, 2)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
// [[vk::binding(6, 2)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs, etc
// [[vk::binding(0, 3)]] RWStructuredBuffer<float> Exts;  // [In / Out Layers][Neurons]
[[vk::binding(1, 3)]] RWStructuredBuffer<uint> Spikers;  // [[Neurons]] -- indexes of those that spiked


void PulvinarDriver(in LayerParams ly, in LayerParams dly, in Pool dlpl, uint ni, uint nin, out float drvGe, out float nonDrvPct) {
	float drvMax = dlpl.AvgMax.CaSpkP.Cycle.Max;
	nonDrvPct = ly.Pulv.NonDrivePct(drvMax); // how much non-driver to keep
	uint pnin = ni + dly.Idxs.NeurSt;
	drvGe = ly.Pulv.DriveGe(Neurons[pnin].Burst);
}

// GInteg integrates conductances G over time (Ge, NMDA, etc).
// calls NeuronGatherSpikes, GFmRawSyn, GiInteg
void GInteg(in Context ctx, in LayerParams ly, uint ni, uint nin, inout Neuron nrn, in Pool pl, in LayerVals vals) {
	float drvGe = 0;
	float nonDrvPct = 0;
	if (ly.LayType == PulvinarLayer) {
		PulvinarDriver(ly, Layers[ly.Pulv.DriveLayIdx], Pools[Layers[ly.Pulv.DriveLayIdx].Idxs.PoolSt], ni, nin, drvGe, nonDrvPct);
	}

	float saveVal = ly.SpecialPreGs(ctx, ni, nrn, drvGe, nonDrvPct);
	
	ly.GFmRawSyn(ctx, ni, nrn);
	ly.GiInteg(ctx, ni, nrn, pl, vals);
	ly.GNeuroMod(ctx, ni, nrn, vals);
	
	ly.SpecialPostGs(ctx, ni, nrn, saveVal);
}

void CycleNeuron2(in LayerParams ly, uint nin, inout Neuron nrn, in Pool pl, in Pool lpl, in LayerVals vals) {
	uint ni = nin - ly.Idxs.NeurSt; // layer-based as in Go
	
	GInteg(Ctx[0], ly, ni, nin, nrn, pl, vals);
	ly.SpikeFmG(Ctx[0], ni, nrn);
	
	// important: because we have to refer to Ctx[0] directly here for atomic, and can't use 
	// a ctx arg, we *can't* also have that ctx arg -- it will overwrite anything we do!
	float updtThr = ly.Learn.CaLrn.UpdtThr;
	if ((nrn.Spike > 0) && !((nrn.CaSpkP < updtThr) && (nrn.CaSpkD < updtThr))) {
		int spIdx = InterlockedAdd(Ctx[0].NSpiked, 1);
		Spikers[spIdx] = nin; // used later for SendSpike, SynCa
	}
}

void CycleNeuron(uint ni, inout Neuron nrn) {
	CycleNeuron2(Layers[nrn.LayIdx], ni, nrn, Pools[nrn.SubPoolN], Pools[Layers[nrn.LayIdx].Idxs.PoolSt], LayVals[nrn.LayIdx]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {  // over Recv Neurons
	uint ns;
	uint st;
	Neurons.GetDimensions(ns, st);
	if (idx.x < ns) {
		CycleNeuron(idx.x, Neurons[idx.x]);
	}
}

