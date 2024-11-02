// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// performs the DWt function on all sending projections

#include "context.hlsl"
#include "layerparams.hlsl"
#include "pathparams.hlsl"

// Set 0: uniform layer params -- could not have paths also be uniform..
@group(0) @binding(0)
var<storage, read_write> Layers: array<LayerParams>;
@group(0) @binding(1)
var<storage, read_write> Paths: array<PathParams>;

// Set 1: effectively uniform indexes and path params as structured buffers in storage
@group(1) @binding(0)
var<storage, read_write> NeuronIxs: array<u32>; // [Neurons][Indexes]
@group(1) @binding(1)
var<storage, read_write> SynapseIxs: array<u32>; // [Layer][SendPaths][SendNeurons][Syns]
@group(1) @binding(2)
var<storage, read_write> SendCon: array<StartN>; // [Layer][SendPaths][SendNeurons]
@group(1) @binding(3)
var<storage, read_write> RecvPathIndexes: array<u32>; // [Layer][RecvPaths]
@group(1) @binding(4)
var<storage, read_write> RecvCon: array<StartN>; // [Layer][RecvPaths][RecvNeurons]
@group(1) @binding(5)
var<storage, read_write> RecvSynIndexes: array<u32>; // [Layer][RecvPaths][RecvNeurons][Syns]

// Set 2: main network structs and vals -- all are writable
@group(2) @binding(0)
var<storage, read_write> Ctx: array<Context>; // [0]
@group(2) @binding(1)
var<storage, read_write> Neurons: array<f32>; // [Neurons][Vars][Data]
@group(2) @binding(2)
var<storage, read_write> NeuronAvgs: array<f32>; // [Neurons][Vars]
@group(2) @binding(3)
var<storage, read_write> Pools: array<f32>; // [Layer][Pools][Data]
@group(2) @binding(4)
var<storage, read_write> LayValues: array<LayerValues>; // [Layer][Data]
@group(2) @binding(5)
var<storage, read_write> Globals: array<f32>; // [NGlobals]
@group(2) @binding(6)
var<storage, read_write> Exts: array<f32>; // [In / Out Layers][Neurons][Data]

// There might be a limit of 8 buffers per set -- can't remember..

// Set 3: synapse vars
@group(3) @binding(0)
var<storage, read_write> GBuf: array<i32>; // [Layer][RecvPaths][RecvNeurons][MaxDel+1][Data]
@group(3) @binding(1)
var<storage, read_write> GSyns: array<f32>; // [Layer][RecvPaths][RecvNeurons][Data]
@group(3) @binding(2)
var<storage, read_write> Synapses: array<f32>; // [Layer][SendPaths][SendNeurons][Syns]

// todo: future expansion to add more tranches of Synapses

// Set 4: SynCa -- can only access in 2^31 chunks
@group(4) @binding(0)
var<storage, read_write> SynapseCas: array<f32>; // [Layer][SendPaths][SendNeurons][Syns][Data]
@group(4) @binding(1)
var<storage, read_write> SynapseCas1: array<f32>;
@group(4) @binding(2)
var<storage, read_write> SynapseCas2: array<f32>;
@group(4) @binding(3)
var<storage, read_write> SynapseCas3: array<f32>;
@group(4) @binding(4)
var<storage, read_write> SynapseCas4: array<f32>;
@group(4) @binding(5)
var<storage, read_write> SynapseCas5: array<f32>;
@group(4) @binding(6)
var<storage, read_write> SynapseCas6: array<f32>;
@group(4) @binding(7)
var<storage, read_write> SynapseCas7: array<f32>;


void DWtSyn2(in Context ctx, in LayerParams rlay, in PathParams pj, uint syni, uint di, uint si, uint ri) {
	if(pj.Learn.Learn == 0) {
		return;
	}
	bool isTarget = (rlay.Acts.Clamp.IsTarget == 1);
	uint pi = NrnI(ctx, ri, NrnSubPool);

	pj.DWtSyn(ctx, syni, si, ri, di, Pools[rlay.Indexes.PoolIndex(0, di)], Pools[rlay.Indexes.PoolIndex(pi, di)], isTarget);
}

void DWtSyn(in Context ctx, uint syni, uint di) {
	uint pi = SynI(ctx, syni, SynPathIndex);
	uint si = SynI(ctx, syni, SynSendIndex);
	uint ri = SynI(ctx, syni, SynRecvIndex);
	DWtSyn2(ctx, Layers[Paths[pi].Indexes.RecvLayer], Paths[pi], syni, di, si, ri);
}


[numthreads(64, 1, 1)]
void main(uint3 idx : SV_DispatchThreadID) { // over Synapses * Data
	uint syni = Ctx[0].NetIndexes.ItemIndex(idx.x);
	if (!Ctx[0].NetIndexes.SynIndexIsValid(syni)) {
		return;
	}
	uint di = Ctx[0].NetIndexes.DataIndex(idx.x);
	if (!Ctx[0].NetIndexes.DataIndexIsValid(di)) {
		return;
	}
	DWtSyn(Ctx[0], syni, di);
}



