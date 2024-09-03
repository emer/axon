// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// does CyclePost: iterates over data parallel -- handles all special context updates

// note: all must be visible always because accessor methods refer to them
[[vk::binding(0, 1)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Indexes]
[[vk::binding(1, 1)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPaths][SendNeurons][Syns]
[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(5, 2)]] RWStructuredBuffer<float> Globals;  // [NGlobals]

[[vk::binding(2, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPaths][SendNeurons][Syns]
[[vk::binding(0, 4)]] RWStructuredBuffer<float> SynapseCas0;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(1, 4)]] RWStructuredBuffer<float> SynapseCas1;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(2, 4)]] RWStructuredBuffer<float> SynapseCas2;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(3, 4)]] RWStructuredBuffer<float> SynapseCas3;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(4, 4)]] RWStructuredBuffer<float> SynapseCas4;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(5, 4)]] RWStructuredBuffer<float> SynapseCas5;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(6, 4)]] RWStructuredBuffer<float> SynapseCas6;  // [Layer][SendPaths][SendNeurons][Syns][Data]
[[vk::binding(7, 4)]] RWStructuredBuffer<float> SynapseCas7;  // [Layer][SendPaths][SendNeurons][Syns][Data]

#include "context.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have paths also be uniform..
[[vk::binding(0, 0)]] StructuredBuffer<LayerParams> Layers; // [Layer]

// Set 1: effectively uniform indexes and path params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] RWStructuredBuffer<Context> Ctx; // [0]
[[vk::binding(3, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools][Data]
[[vk::binding(4, 2)]] RWStructuredBuffer<LayerValues> LayValues; // [Layer][Data]


float LDTSrcLayAct(int layIndex, uint di) {
	if (layIndex < 0) {
		return 0.0;
	}
	return Pools[Layers[layIndex].Indexes.PoolIndex(0, di)].AvgMax.CaSpkP.Cycle.Avg;
}

void CyclePostLDT(inout Context ctx, uint di, in LayerParams ly, inout LayerValues vals) {
	float srcLay1Act = LDTSrcLayAct(ly.LDT.SrcLay1Index, di);
	float srcLay2Act = LDTSrcLayAct(ly.LDT.SrcLay2Index, di);
	float srcLay3Act = LDTSrcLayAct(ly.LDT.SrcLay3Index, di);
	float srcLay4Act = LDTSrcLayAct(ly.LDT.SrcLay4Index, di);
	ly.CyclePostLDTLayer(ctx, di, vals, srcLay1Act, srcLay2Act, srcLay3Act, srcLay4Act);
}

void CyclePost2(inout Context ctx, in LayerParams ly, uint li, uint di, inout LayerValues vals, in Pool lpl) {
	ly.CyclePostLayer(ctx, di, lpl, vals);
	switch (ly.LayType) {
	case CeMLayer: {
		ly.CyclePostCeMLayer(ctx, di, lpl);
		break;
	}
	case VSPatchLayer: {
		int npl = ly.Indexes.ShpPlY * ly.Indexes.ShpPlX;
		for (int pi = 0; pi < npl; pi++) {
			ly.CyclePostVSPatchLayer(ctx, di, pi+1, Pools[ly.Indexes.PoolIndex(1+pi, di)], vals);
		}
		break;
	}
	case LDTLayer: {
		CyclePostLDT(ctx, di, ly, vals);
		break;
	}
	case VTALayer: {
		ly.CyclePostVTALayer(ctx, di);
		break;
	}
	case RWDaLayer: {
		ly.CyclePostRWDaLayer(ctx, di, vals, LayValues[ctx.NetIndexes.ValuesIndex(ly.RWDa.RWPredLayIndex, di)]);
		break;
	}
	case TDPredLayer: {
		ly.CyclePostTDPredLayer(ctx, di, vals);
		break;
	}
	case TDIntegLayer: {
		ly.CyclePostTDIntegLayer(ctx, di, vals, LayValues[ctx.NetIndexes.ValuesIndex(ly.TDInteg.TDPredLayIndex, di)]);
		break;
	}
	case TDDaLayer: {
		ly.CyclePostTDDaLayer(ctx, di, vals, LayValues[ctx.NetIndexes.ValuesIndex(ly.TDDa.TDIntegLayIndex, di)]);
		break;
	}
	}
}

void CyclePost(inout Context ctx, in LayerParams ly, int li, uint di) {
	CyclePost2(ctx, ly, uint(li), di, LayValues[ly.Indexes.ValuesIndex(di)], Pools[ly.Indexes.PoolIndex(0, di)]);
}

void CyclePostAll2(inout Context ctx, in LayerParams ly, uint li, uint di, inout LayerValues vals, in Pool lpl) {
	ly.CyclePostLayer(ctx, di, lpl, vals); // does reaction time
}

void CyclePostAll(inout Context ctx, in LayerParams ly, int li, uint di) {
	CyclePostAll2(ctx, ly, uint(li), di, LayValues[ly.Indexes.ValuesIndex(di)], Pools[ly.Indexes.PoolIndex(0, di)]);
}

[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) {
	if (idx.x >= Ctx[0].NetIndexes.NData) {
		return;
	}

	uint di = idx.x;
	
	// note: this bizarre logic is only way to get multiple writes to Context
	// to actually stick -- needs to be done sequentially within one thread
	// and not even in a for loop for some reason.
	int pnmi = -1;
	int cmpi = -1;
	int cmni = -1;
	int ldti = -1;
	int vspi1 = -1;
	int vspi2 = -1;
	int vtai = -1;
	int rwdi = -1;
	int tdpi = -1;
	int tdii = -1;
	int tddi = -1;
	for (int li = 0; li < Ctx[0].NetIndexes.NLayers; li++) {
		CyclePostAll(Ctx[0], Layers[li], li, di);
		switch (Layers[li].LayType) {
		case CeMLayer:
			if (Layers[li].Learn.NeuroMod.Valence == Positive) {
				cmpi = li;
			} else {
				cmni = li;
			}
			break;
		case VSPatchLayer:
			if (Layers[li].Learn.NeuroMod.DAMod == D1Mod) {
				vspi1 = li;
			} else {
				vspi2 = li;
			}
			break;
		case LDTLayer:
			ldti = li;
			break;
		case VTALayer:
			vtai = li;
			break;
		case RWDaLayer:
			rwdi = li;
			break;
		case TDPredLayer:
			tdpi = li;
			break;
		case TDIntegLayer:
			tdii = li;
			break;
		case TDDaLayer:
			tddi = li;
			break;
		}
	}
	if (pnmi >= 0) {
		CyclePost(Ctx[0], Layers[pnmi], pnmi, di);
	}
	if (cmpi >= 0) {
		CyclePost(Ctx[0], Layers[cmpi], cmpi, di);
	}                                      
	if (cmni >= 0) {                       
		CyclePost(Ctx[0], Layers[cmni], cmni, di);
	}                                      
	if (ldti >= 0) { // depends on pn note:mi
		CyclePost(Ctx[0], Layers[ldti], ldti, di);
	}                                      
	if (vspi1 >= 0) {                       
		CyclePost(Ctx[0], Layers[vspi1], vspi1, di);
	}                                      
	if (vspi2 >= 0) {                       
		CyclePost(Ctx[0], Layers[vspi2], vspi2, di);
	}                                      
	if (rwdi >= 0) {                       
		CyclePost(Ctx[0], Layers[rwdi], rwdi, di);
	}                                      
	if (tdpi >= 0) {                       
		CyclePost(Ctx[0], Layers[tdpi], tdpi, di);
	}                                      
	if (tdii >= 0) {                       
		CyclePost(Ctx[0], Layers[tdii], tdii, di);
	}                                      
	if (tddi >= 0) {                       
		CyclePost(Ctx[0], Layers[tddi], tddi, di);
	}                                      
	// note: this depends vspi, cm*i, ldds on ti
	if (vtai >= 0) {                       
		CyclePost(Ctx[0], Layers[vtai], vtai, di);
	}
}

