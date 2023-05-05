// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "context.hlsl"
#include "layerparams.hlsl"

// note: binding is var, set

// Set 0: uniform layer params -- could not have prjns also be uniform..
[[vk::binding(0, 0)]] uniform LayerParams Layers[]; // [Layer]

// Set 1: effectively uniform prjn params as structured buffers in storage
// [[vk::binding(0, 1)]] StructuredBuffer<PrjnParams> Prjns; // [Layer][RecvPrjns]
// [[vk::binding(1, 1)]] StructuredBuffer<StartN> RecvCon; // [Layer][RecvPrjns][RecvNeurons]

// Set 2: main network structs and vals -- all are writable
[[vk::binding(0, 2)]] RWStructuredBuffer<Context> Ctx; // [0]
// [[vk::binding(1, 2)]] RWStructuredBuffer<Neuron> Neurons; // [Layer][Neuron]
[[vk::binding(2, 2)]] RWStructuredBuffer<Pool> Pools; // [Layer][Pools]
[[vk::binding(3, 2)]] RWStructuredBuffer<LayerVals> LayVals; // [Layer]
// [[vk::binding(4, 2)]] RWStructuredBuffer<Synapse> Synapses;  // [Layer][RecvPrjns][RecvNeurons][Syns]
// [[vk::binding(5, 2)]] RWStructuredBuffer<int> GBuf;  // [Layer][RecvPrjns][RecvNeurons][MaxDel+1]
// [[vk::binding(6, 2)]] RWStructuredBuffer<float> GSyns;  // [Layer][RecvPrjns][RecvNeurons]

// Set 3: external inputs
// [[vk::binding(0, 3)]] RWStructuredBuffer<float> Exts;  // [In / Out Layers][Neurons]

void CyclePostVSPatch(inout Context ctx, uint li, in LayerParams ly, int pi, in Pool pl) {
	ly.CyclePostVSPatchLayer(ctx, pi, pl);
}

void CyclePost(inout Context ctx, uint li, in LayerParams ly, inout LayerVals vals, in Pool lpl) {
	switch (ly.LayType) {
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
	case CeMLayer:
		ly.CyclePostCeMLayer(ctx, lpl);
		break;
	case VSPatchLayer:
		int npl = ly.Idxs.ShpPlY * ly.Idxs.ShpPlX;
		for (int pi = 0; pi < npl; pi++) {
			CyclePostVSPatch(ctx, li, ly, pi+1, Pools[ly.Idxs.PoolSt+1+pi]);
		}
		break;
	case PTNotMaintLayer:
		ly.CyclePostPTNotMaintLayer(ctx, lpl);
		break;
	}
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

void CyclePostVTA(inout Context ctx, in LayerParams ly) {
	ly.CyclePostVTALayer(ctx);
}


[numthreads(1, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // Just one!
	if (idx.x == 0) {
		int vtaLay = -1;
		int ldtLay = -1;
		for (int li = 0; li < Ctx[0].NLayers; li++) {
			if (Layers[li].LayType == VTALayer) {
				vtaLay = li;
			} else if (Layers[li].LayType == LDTLayer) {
				ldtLay = li;
			} else {
				CyclePost(Ctx[0], li, Layers[li], LayVals[li], Pools[Layers[li].Idxs.PoolSt]);
			}
		}
		// depends on others above
		if (ldtLay >= 0) {
			CyclePostLDT(Ctx[0], Layers[ldtLay], LayVals[ldtLay]);
		}
		if (vtaLay >= 0) {
			CyclePostVTA(Ctx[0], Layers[vtaLay]);
		}
	}
}

