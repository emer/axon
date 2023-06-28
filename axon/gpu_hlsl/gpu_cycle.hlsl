// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the CycleNeuron function on all neurons

// note: all must be visible always because accessor methods refer to them
[[vk::binding(0, 1)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Idxs]
[[vk::binding(1, 1)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(5, 2)]] RWStructuredBuffer<float> Globals;  // [NGlobals]

[[vk::binding(2, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(0, 4)]] RWStructuredBuffer<float> SynapseCas0;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(1, 4)]] RWStructuredBuffer<float> SynapseCas1;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(2, 4)]] RWStructuredBuffer<float> SynapseCas2;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(3, 4)]] RWStructuredBuffer<float> SynapseCas3;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(4, 4)]] RWStructuredBuffer<float> SynapseCas4;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(5, 4)]] RWStructuredBuffer<float> SynapseCas5;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(6, 4)]] RWStructuredBuffer<float> SynapseCas6;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(7, 4)]] RWStructuredBuffer<float> SynapseCas7;  // [Layer][SendPrjns][SendNeurons][Syns][Data]

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] StructuredBuffer<LayerParams> Layers; // [Layer]
[[vk::binding(1, 0)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][SendPrjns]

// Set 1: effectively uniform indexes and prjn params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools][Data]
[[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer][Data]


void PulvinarDriver2(in Context ctx, in LayerParams ly, in LayerParams dly, in Pool dlpl, uint ni, uint di, out float drvGe, out float nonDrvPct) {
	float drvMax = dlpl.AvgMax.CaSpkP.Cycle.Max;
	nonDrvPct = ly.Pulv.NonDrivePct(drvMax); // how much non-driver to keep
	uint pni = (ni - ly.Idxs.NeurSt) + dly.Idxs.NeurSt;
	drvGe = ly.Pulv.DriveGe(NrnV(ctx, pni, di, Burst));
}

void PulvinarDriver(in Context ctx, in LayerParams ly, in LayerParams dly, uint ni, uint di, out float drvGe, out float nonDrvPct) {
	PulvinarDriver2(ctx, ly, dly, Pools[dly.Idxs.PoolIdx(0, di)], ni, di, drvGe, nonDrvPct);
}

// GInteg integrates conductances G over time (Ge, NMDA, etc).
// calls NeuronGatherSpikes, GFmRawSyn, GiInteg
void GInteg(in Context ctx, in LayerParams ly, uint ni, uint di, in Pool pl, in LayerVals vals) {
	float drvGe = 0;
	float nonDrvPct = 0;
	if (ly.LayType == PulvinarLayer) {
		PulvinarDriver(ctx, ly, Layers[ly.Pulv.DriveLayIdx], ni, di, drvGe, nonDrvPct);
	}

	float saveVal = ly.SpecialPreGs(ctx, ni, di, pl, vals, drvGe, nonDrvPct);
	
	ly.GFmRawSyn(ctx, ni, di);
	ly.GiInteg(ctx, ni, di, pl, vals);
	ly.GNeuroMod(ctx, ni, di, vals);
	
	ly.SpecialPostGs(ctx, ni, di, saveVal);
}

void CycleNeuron3(in Context ctx, in LayerParams ly, uint ni, uint di, in Pool pl, in Pool lpl, in LayerVals vals) {
	uint lni = ni - ly.Idxs.NeurSt; // layer-based as in Go
	
	GInteg(ctx, ly, ni, di, pl, vals);
	ly.SpikeFmG(ctx, ni, di);
}

void CycleNeuron2(in Context ctx, in LayerParams ly, uint ni, uint di) {
	uint pi = NrnI(ctx, ni, NrnSubPool);
	CycleNeuron3(ctx, ly, ni, di, Pools[ly.Idxs.PoolIdx(pi, di)], Pools[ly.Idxs.PoolIdx(0, di)], LayVals[ly.Idxs.ValsIdx(di)]);
}

void CycleNeuron(in Context ctx, uint ni, uint di) {
	uint li = NrnI(ctx, ni, NrnLayIdx);
	CycleNeuron2(ctx, Layers[li], ni, di);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {  // over Neurons * Data
	uint ni = Ctx[0].NetIdxs.ItemIdx(idx.x);
	if (!Ctx[0].NetIdxs.NeurIdxIsValid(ni)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	if (!Ctx[0].NetIdxs.DataIdxIsValid(di)) {
		return;
	}
	CycleNeuron(Ctx[0], ni, di);
}

