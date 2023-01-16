// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//go:generate gosl --keep -exclude=Update,UpdateParams,Defaults,AllParams github.com/goki/mat32/fastexp.go github.com/emer/etable/minmax ../chans/chans.go ../chans ../kinase ../fsfffb/inhib.go ../fsfffb github.com/emer/emergent/etime time.go neuron.go synapse.go pool.go act.go inhib.go learn.go layerparams.go prjnparams.go gpu_cycle.hlsl gpu_dwt.hlsl

// Overarching rules:
// * Everything must be stored in top-level arrays of structs
//    these are only variable length data structures
// * Efficient access requires Start, N indexes in to data structs
//    and there must be a contiguous layout for each different way
//    of iterating over the data -- this means both recv and send
