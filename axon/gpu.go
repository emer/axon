// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//go:generate gosl --keep -exclude=Update,UpdateParams,Defaults,AllParams github.com/goki/mat32/fastexp.go github.com/emer/etable/minmax ../chans/chans.go ../chans ../kinase ../fsfffb/inhib.go ../fsfffb github.com/emer/emergent/etime github.com/emer/emergent/ringidx time.go neuron.go synapse.go pool.go act.go inhib.go learn.go layerparams.go prjnparams.go prjnvals.go gpu_exttopool.hlsl gpu_laygi.hlsl gpu_poolgi.hlsl gpu_cycle.hlsl gpu_dwt.hlsl

// Overarching rules:
// * Everything must be stored in top-level arrays of structs
//    these are only variable length data structures
// * Efficient access requires Start, N indexes in to data structs
//    and there must be a contiguous layout for each different way
//    of iterating over the data -- this means both recv and send

// cycle update:

// PrjnGatherSpikes:
// gpu_gfmspikes.hlsl [Prjns]

// GiFmSpikes:
// gpu_exttopool.hlsl [Neurons] // todo: can this be done in send spike??
// gpu_laygi.hlsl [Layers]
// gpu_poolgi.hlsl [Pools]

// CycleNeuron:
// gpu_cycle.hlsl [Neurons]
