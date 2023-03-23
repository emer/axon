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

float RSalAChLayMaxAct(int layIdx) {
	if (layIdx < 0) {
		return 0.0;
	}
	return Pools[Layers[layIdx].Idxs.PoolSt].AvgMax.Act.Cycle.Max;
}

void CyclePost(inout Context ctx, uint li, in LayerParams ly, inout LayerVals vals) {
	switch (ly.LayType) {
	case RSalienceAChLayer:
		float lay1MaxAct = RSalAChLayMaxAct(ly.RSalACh.SrcLay1Idx);
		float lay2MaxAct = RSalAChLayMaxAct(ly.RSalACh.SrcLay2Idx);
		float lay3MaxAct = RSalAChLayMaxAct(ly.RSalACh.SrcLay3Idx);
		float lay4MaxAct = RSalAChLayMaxAct(ly.RSalACh.SrcLay4Idx);
		ly.CyclePostRSalAChLayer(ctx, vals, lay1MaxAct, lay2MaxAct, lay3MaxAct, lay4MaxAct);
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
void main(uint3 idx : SV_DispatchThreadID) { // over Layers
	if (idx.x == 0) {
		for (int li = 0; li < Ctx[0].NLayers; li++) {
			CyclePost(Ctx[0], li, Layers[li], LayVals[li]);
		}
	}
}

