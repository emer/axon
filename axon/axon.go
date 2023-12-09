// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/emer"
)

// AxonNetwork defines the essential algorithmic API for Axon, at the network level.
// These are the methods that the user calls in their Sim code:
// * NewState
// * Cycle
// * NewPhase
// * DWt
// * WtFmDwt
// Because we don't want to have to force the user to use the interface cast in calling
// these methods, we provide Impl versions here that are the implementations
// which the user-facing method calls through the interface cast.
// Specialized algorithms should thus only change the Impl version, which is what
// is exposed here in this interface.
//
// There is now a strong constraint that all Cycle level computation takes place
// in one pass at the Layer level, which greatly improves threading efficiency.
//
// All of the structural API is in emer.Network, which this interface also inherits for
// convenience.
type AxonNetwork interface {
	emer.Network

	// AsAxon returns this network as a axon.Network -- so that the
	// AxonNetwork interface does not need to include accessors
	// to all the basic stuff
	AsAxon() *Network
}

// AxonLayer defines the essential algorithmic API for Axon, at the layer level.
// These are the methods that the axon.Network calls on its layers at each step
// of processing.  Other Layer types can selectively re-implement (override) these methods
// to modify the computation, while inheriting the basic behavior for non-overridden methods.
//
// All of the structural API is in emer.Layer, which this interface also inherits for
// convenience.
type AxonLayer interface {
	emer.Layer

	// AsAxon returns this layer as a axon.Layer -- so that the AxonLayer
	// interface does not need to include accessors to all the basic stuff
	AsAxon() *Layer

	// PostBuild performs special post-Build() configuration steps for specific algorithms,
	// using configuration data set in BuildConfig during the ConfigNet process.
	PostBuild()
}

// AxonPrjn defines the essential algorithmic API for Axon, at the projection level.
// These are the methods that the axon.Layer calls on its prjns at each step
// of processing.  Other Prjn types can selectively re-implement (override) these methods
// to modify the computation, while inheriting the basic behavior for non-overridden methods.
//
// All of the structural API is in emer.Prjn, which this interface also inherits for
// convenience.
type AxonPrjn interface {
	emer.Prjn

	// AsAxon returns this prjn as a axon.Prjn -- so that the AxonPrjn
	// interface does not need to include accessors to all the basic stuff.
	AsAxon() *Prjn
}

type AxonPrjns []*Prjn

func (ap *AxonPrjns) Add(pj *Prjn) {
	*ap = append(*ap, pj)
}
