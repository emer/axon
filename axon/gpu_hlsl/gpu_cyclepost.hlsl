// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// note: all must be visible always because accessor methods refer to them

[[vk::binding(1, 2)]] RWStructuredBuffer<float> Neurons; // [Neurons][Vars][Data]
[[vk::binding(2, 2)]] RWStructuredBuffer<float> NeuronAvgs; // [Neurons][Vars]
[[vk::binding(3, 2)]] StructuredBuffer<uint> NeuronIxs; // [Neurons][Idxs]
[[vk::binding(0, 3)]] RWStructuredBuffer<float> Synapses;  // [Layer][SendPrjns][SendNeurons][Syns]
[[vk::binding(1, 3)]] RWStructuredBuffer<float> SynapseCas;  // [Layer][SendPrjns][SendNeurons][Syns][Data]
[[vk::binding(2, 3)]] StructuredBuffer<uint> SynapseIxs;  // [Layer][SendPrjns][SendNeurons][Syns]

#include "context.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] RWStructuredBuffer<Context> Ctx; // [0]
// [[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]

// Set 3: external inputs

void CyclePostVSPatch(inout Context ctx, uint li, in LayerParams ly, int pi, in Pool pl) {
	ly.CyclePostVSPatchLayer(ctx, pi, pl);
}

float LDTSrcLayAct(int layIdx) {
	if (layIdx < 0) {
		return 0.0;
	}
	return Pools[Layers[layIdx].Idxs.PoolSt].AvgMax.CaSpkP.Cycle.Avg;
}

void CyclePostLDT(inout Context ctx, in LayerParams ly, inout LayerVals vals) {
	float srcLay1Act = LDTSrcLayAct(ly.LDT.SrcLay1Idx);
	float srcLay2Act = LDTSrcLayAct(ly.LDT.SrcLay2Idx);
	float srcLay3Act = LDTSrcLayAct(ly.LDT.SrcLay3Idx);
	float srcLay4Act = LDTSrcLayAct(ly.LDT.SrcLay4Idx);
	ly.CyclePostLDTLayer(ctx, vals, srcLay1Act, srcLay2Act, srcLay3Act, srcLay4Act);
}

void CyclePost(inout Context ctx, uint li, in LayerParams ly, inout LayerVals vals, in Pool lpl) {
	switch (ly.LayType) {
	case PTNotMaintLayer:
		ly.CyclePostPTNotMaintLayer(ctx, lpl);
		break;
	case CeMLayer:
		ly.CyclePostCeMLayer(ctx, lpl);
		break;
	case VSPatchLayer:
		int npl = ly.Idxs.ShpPlY * ly.Idxs.ShpPlX;
		for (int pi = 0; pi < npl; pi++) {
			CyclePostVSPatch(ctx, li, ly, pi+1, Pools[ly.Idxs.PoolSt+1+pi]);
		}
		break;
	case LDTLayer:
		CyclePostLDT(ctx, ly, vals);
		break;
	case VTALayer:
		ly.CyclePostVTALayer(ctx);
		break;
	case RWDaLayer:
		ly.CyclePostRWDaLayer(ctx, vals, LayVals[ly.RWDa.RWPredLayIdx]);
		break;
	case TDPredLayer:
		ly.CyclePostTDPredLayer(ctx, vals);
		break;
	case TDIntegLayer:
		ly.CyclePostTDIntegLayer(ctx, vals, LayVals[ly.TDInteg.TDPredLayIdx]);
		break;
	case TDDaLayer:
		ly.CyclePostTDDaLayer(ctx, vals, LayVals[ly.TDDa.TDIntegLayIdx]);
		break;
	}
}


[numthreads(1, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // Just one!
	if (idx.x == 0) {
		// note: this bizarre logic is only way to get multiple writes to Context
		// to actually stick -- needs to be done sequentially within one thread
		// and not even in a for loop for some reason.
		int pnmi = -1;
		int cmpi = -1;
		int cmni = -1;
		int ldti = -1;
		int vspi = -1;
		int vtai = -1;
		int rwdi = -1;
		int tdpi = -1;
		int tdii = -1;
		int tddi = -1;
		for (int li = 0; li < Ctx[0].NLayers; li++) {
			switch (Layers[li].LayType) {
			case PTNotMaintLayer:
				pnmi = li;
				break;
			case CeMLayer:
				if (Layers[li].Learn.NeuroMod.Valence == Positive) {
					cmpi = li;
				} else {
					cmni = li;
				}
				break;
			case VSPatchLayer:
				vspi = li;
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
			CyclePost(Ctx[0], pnmi, Layers[pnmi], LayVals[pnmi], Pools[Layers[pnmi].Idxs.PoolSt]);
		}
		if (cmpi >= 0) {
			CyclePost(Ctx[0], cmpi, Layers[cmpi], LayVals[cmpi], Pools[Layers[cmpi].Idxs.PoolSt]);
		}
		if (cmni >= 0) {
			CyclePost(Ctx[0], cmni, Layers[cmni], LayVals[cmni], Pools[Layers[cmni].Idxs.PoolSt]);
		}
		if (ldti >= 0) { // note: depends on pnmi
			CyclePost(Ctx[0], ldti, Layers[ldti], LayVals[ldti], Pools[Layers[ldti].Idxs.PoolSt]);
		}
		if (vspi >= 0) {
			CyclePost(Ctx[0], vspi, Layers[vspi], LayVals[vspi], Pools[Layers[vspi].Idxs.PoolSt]);
		}
		if (rwdi >= 0) {
			CyclePost(Ctx[0], rwdi, Layers[rwdi], LayVals[rwdi], Pools[Layers[rwdi].Idxs.PoolSt]);
		}
		if (tdpi >= 0) {
			CyclePost(Ctx[0], tdpi, Layers[tdpi], LayVals[tdpi], Pools[Layers[tdpi].Idxs.PoolSt]);
		}
		if (tdii >= 0) {
			CyclePost(Ctx[0], tdii, Layers[tdii], LayVals[tdii], Pools[Layers[tdii].Idxs.PoolSt]);
		}
		if (tddi >= 0) {
			CyclePost(Ctx[0], tddi, Layers[tddi], LayVals[tddi], Pools[Layers[tddi].Idxs.PoolSt]);
		}
		// note: this depends on vspi, cm*i, ldti
		if (vtai >= 0) {
			CyclePost(Ctx[0], vtai, Layers[vtai], LayVals[vtai], Pools[Layers[vtai].Idxs.PoolSt]);
		}
	}
}

