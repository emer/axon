// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// tests writing values to SynCas

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

struct PushConst {
	uint Off;
 	uint pad;
	uint pad1;
	uint pad2;
};

[[vk::push_constant]] PushConst PushOff;

#include "context.hlsl"
#include "layerparams.hlsl"
#include "prjnparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..

// Set 1: effectively uniform indexes and prjn params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] StructuredBuffer<Context> Ctx; // [0]

void WriteSynCa(in Context ctx, uint syni, uint di) {
	// uint64_t ix = ctx.SynapseCaVars.Idx(syni, di, DTr);
	// uint bank = uint(ix / uint64_t(ctx.NetIdxs.GPUMaxBuffFloats));
	// uint res = uint(ix % uint64_t(ctx.NetIdxs.GPUMaxBuffFloats));
	// uint pi = SynI(ctx, syni, SynPrjnIdx);
	// uint si = SynI(ctx, syni, SynSendIdx);
	// uint ri = SynI(ctx, syni, SynRecvIdx);
	// SetSynCaV(ctx, syni, di, CaM, asfloat(pi));
	// SetSynCaV(ctx, syni, di, CaP, asfloat(si));
	// SetSynCaV(ctx, syni, di, CaD, asfloat(ri));
	// SetSynCaV(ctx, syni, di, CaUpT, asfloat(syni));
	// SetSynCaV(ctx, syni, di, Tr, asfloat(di));
	// SetSynCaV(ctx, syni, di, DTr, asfloat(bank));
	// SetSynCaV(ctx, syni, di, DiDWt, asfloat(res));

	SetSynCaV(ctx, syni, di, CaM, asfloat(uint(ctx.SynapseCaVars.Idx(syni, di, CaM) % 0xFFFFFFFF)));
	SetSynCaV(ctx, syni, di, CaP, asfloat(uint(ctx.SynapseCaVars.Idx(syni, di, CaP) % 0xFFFFFFFF)));
	SetSynCaV(ctx, syni, di, CaD, asfloat(uint(ctx.SynapseCaVars.Idx(syni, di, CaD) % 0xFFFFFFFF)));
	SetSynCaV(ctx, syni, di, CaUpT, asfloat(uint(ctx.SynapseCaVars.Idx(syni, di, CaUpT) % 0xFFFFFFFF)));
	SetSynCaV(ctx, syni, di, Tr, asfloat(uint(ctx.SynapseCaVars.Idx(syni, di, Tr) % 0xFFFFFFFF)));
	SetSynCaV(ctx, syni, di, DTr, asfloat(uint(ctx.SynapseCaVars.Idx(syni, di, DTr) % 0xFFFFFFFF)));
	SetSynCaV(ctx, syni, di, DiDWt, asfloat(uint(ctx.SynapseCaVars.Idx(syni, di, DiDWt) % 0xFFFFFFFF)));

	// SetSynCaV(ctx, syni, di, DiDWt, 42.22);
	// uint64_t ix = ctx.SynapseCaVars.Idx(syni, di, DiDWt);
	// uint res = uint(ix - uint64_t(ctx.NetIdxs.GPUMaxBuffFloats));
	// uint res = syni * ctx.NetIdxs.MaxData + di;
	// SynapseCas1[res] = asfloat(0x6666666);
	// SynapseCas0[res] = asfloat(0x4444444);
}


[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Synapses * Data
	uint syni = Ctx[0].NetIdxs.ItemIdx(idx.x) + PushOff.Off;
	if (!Ctx[0].NetIdxs.SynIdxIsValid(syni)) {
		return;
	}
	uint di = Ctx[0].NetIdxs.DataIdx(idx.x);
	if (!Ctx[0].NetIdxs.DataIdxIsValid(di)) {
		return;
	}
	WriteSynCa(Ctx[0], syni, di);
}



