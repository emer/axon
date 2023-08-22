// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// does MinusPhase Update on each Pool

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

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] StructuredBuffer<LayerParams> Layers; // [Layer]

// Set 1: effectively uniform indexes and prjn params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools][Data]
[[vk::binding(4, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer][Data]

void PulvinarDriver2(in Context ctx, in LayerParams ly, in LayerParams dly, in Pool dlpl, uint di, out float nonDrvPct) {
	float drvMax = dlpl.AvgMax.CaSpkP.Cycle.Max;
	nonDrvPct = ly.Pulv.NonDrivePct(drvMax); // how much non-driver to keep
}

void PulvinarDriver(in Context ctx, in LayerParams ly, in LayerParams dly, uint di, out float nonDrvPct) {
	PulvinarDriver2(ctx, ly, dly, Pools[dly.Idxs.PoolIdx(0, di)], di, nonDrvPct);
}

void MinusPool2(in Context ctx, in LayerParams ly, uint di, inout Pool pl, inout LayerVals vals) {
	ly.MinusPhasePool(ctx, pl);
	if (pl.IsLayPool != 0) {
		float geIntMinusMax = 0;
		float giIntMinusMax = 0;
		for (uint di = 0; di < ctx.NetIdxs.NData; di++) {
			geIntMinusMax = max(geIntMinusMax, Pools[ly.Idxs.PoolIdx(0, di)].AvgMax.GeInt.Cycle.Max);
			giIntMinusMax = max(giIntMinusMax, Pools[ly.Idxs.PoolIdx(0, di)].AvgMax.GiInt.Cycle.Max);
		}
		ly.AvgGeM(ctx, vals, geIntMinusMax, giIntMinusMax);
	}
	if (ly.LayType == PulvinarLayer) {
		float nonDrvPct = 0;
		PulvinarDriver(ctx, ly, Layers[ly.Pulv.DriveLayIdx], di, nonDrvPct);
		if (nonDrvPct < 0.5) {
			pl.Inhib.Clamped = 1;
		} else { // if more non-drive, then must not use clamped
			pl.Inhib.Clamped = 0;
		}
	}
}

void MinusPool(in Context ctx, uint di, inout Pool pl) {
	MinusPool2(ctx, Layers[pl.LayIdx], di, pl, LayVals[ctx.NetIdxs.ValsIdx(pl.LayIdx, di)]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Pools * Data (all pools)
	uint npi = idx.x; // network pi
	if (!Ctx[0].NetIdxs.PoolDataIdxIsValid(npi)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	if (!Ctx[0].NetIdxs.DataIdxIsValid(di)) {
		return;
	}
	MinusPool(Ctx[0], di, Pools[npi]);
}

