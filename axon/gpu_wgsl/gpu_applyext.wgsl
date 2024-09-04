// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// calls ApplyExt on neurons

#include "context.wgsl"
#include "layerparams.wgsl"

// Set 0: uniform layer params -- could not have paths also be uniform..
@group(0) @binding(0)
var<storage, read_write> Layers: array<LayerParams>;
// @group(0) @binding(1)
// var<storage, read_write> Paths: array<PathParams>;

// Set 1: effectively uniform indexes and path params as structured buffers in storage
@group(1) @binding(0)
var<storage, read_write> NeuronIxs: array<u32>; // [Neurons][Indexes]

@group(1) @binding(1)
var<storage, read_write> SynapseIxs: array<u32>; // [Layer][SendPaths][SendNeurons][Syns]
// @group(1) @binding(2)
// var<storage, read_write> SendCon: array<StartN>; // [Layer][SendPaths][SendNeurons]
@group(1) @binding(3)
var<storage, read_write> RecvPathIndexes: array<u32>; // [Layer][RecvPaths]
// @group(1) @binding(4)
// var<storage, read_write> RecvCon: array<StartN>; // [Layer][RecvPaths][RecvNeurons]
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

fn ApplyExt2(ctx: ptr<function,Context>, ly: ptr<function,LayerParams>, ni: u32, di: u32) {
	let lni = ni - (*ly).Indexes.NeurSt; // layer-based 
	LayerParams_InitExt(ly, ctx, ni, di);
	if (IsExtLayerType((*ly).LayType)) {
		let ei = LayerIndexes_ExtIndex((*ly).Indexes, lni, di) + (*ly).Indexes.ExtsSt;
		LayerParams_ApplyExtValue(ly, ctx, ni, di, Exts[ei]);
	}
}

fn ApplyExt(ctx: ptr<function,Context>, ni: u32, di: u32) {
	ApplyExt2(ctx, Layers[NrnI(ctx, ni, NrnLayIndex)], ni, di);
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) { // over Neurons x Data
	var ctx = Ctx[0];
	let ni = NetIndexes_ItemIndex(ctx, idx.x);
	if (!NetIndexes_NeurIndexIsValid(ctx, ni)) {
		return;
	}
	let di = NetIndexes_DataIndex(ctx, idx.x);
	if (!NetIndexes_DataIndexIsValid(ctx, di)) {
		return;
	}
	ApplyExt(ctx, ni, di);
}

