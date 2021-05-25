// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/etensor"
)

// AxonNetwork defines the essential algorithmic API for Axon, at the network level.
// These are the methods that the user calls in their Sim code:
// * AlphaCycInit
// * Cycle
// * QuarterFinal
// * DWt
// * WtFmDwt
// Because we don't want to have to force the user to use the interface cast in calling
// these methods, we provide Impl versions here that are the implementations
// which the user-facing method calls.
//
// Typically most changes in algorithm can be accomplished directly in the Layer
// or Prjn level, but sometimes (e.g., in deep) additional full-network passes
// are required.
//
// All of the structural API is in emer.Network, which this interface also inherits for
// convenience.
type AxonNetwork interface {
	emer.Network

	// AsAxon returns this network as a axon.Network -- so that the
	// AxonNetwork interface does not need to include accessors
	// to all the basic stuff
	AsAxon() *Network

	// AlphaCycInitImpl handles all initialization at start of new input pattern, including computing
	// input scaling from running average activation etc.
	AlphaCycInitImpl()

	// CycleImpl runs one cycle of activation updating:
	// * Sends Ge increments from sending to receiving layers
	// * Average and Max Ge stats
	// * Inhibition based on Ge stats and Act Stats (computed at end of Cycle)
	// * Activation from Ge, Gi, and Gl
	// * Average and Max Act stats
	// This basic version doesn't use the time info, but more specialized types do, and we
	// want to keep a consistent API for end-user code.
	CycleImpl(ltime *Time)

	// CyclePostImpl is called after the standard Cycle update, and calls CyclePost
	// on Layers -- this is reserved for any kind of special ad-hoc types that
	// need to do something special after Act is finally computed.
	// For example, sending a neuromodulatory signal such as dopamine.
	CyclePostImpl(ltime *Time)

	// QuarterFinalImpl does updating after end of a quarter
	QuarterFinalImpl(ltime *Time)

	// DWtImpl computes the weight change (learning) based on current
	// running-average activation values
	DWtImpl()

	// WtFmDWtImpl updates the weights from delta-weight changes.
	// Also calls SynScale every Interval times
	WtFmDWtImpl()

	// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
	// GScale conductance scaling, and adapting inhibition
	SlowAdapt()
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

	// InitWts initializes the weight values in the network, i.e., resetting learning
	// Also calls InitActs
	InitWts()

	// InitActAvg initializes the running-average activation values that drive learning.
	InitActAvg()

	// InitActs fully initializes activation state -- only called automatically during InitWts
	InitActs()

	// InitWtsSym initializes the weight symmetry -- higher layers copy weights from lower layers
	InitWtSym()

	// InitExt initializes external input state -- called prior to apply ext
	InitExt()

	// ApplyExt applies external input in the form of an etensor.Tensor
	// If the layer is a Target or Compare layer type, then it goes in Targ
	// otherwise it goes in Ext.
	ApplyExt(ext etensor.Tensor)

	// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
	// If the layer is a Target or Compare layer type, then it goes in Targ
	// otherwise it goes in Ext
	ApplyExt1D(ext []float64)

	// UpdateExtFlags updates the neuron flags for external input based on current
	// layer Type field -- call this if the Type has changed since the last
	// ApplyExt* method call.
	UpdateExtFlags()

	// IsTarget returns true if this layer is a Target layer.
	// By default, returns true for layers of Type == emer.Target
	// Other Target layers include the TRCLayer in deep predictive learning.
	// It is also used in SynScale to not apply it to target layers.
	// In both cases, Target layers are purely error-driven.
	IsTarget() bool

	// IsInput returns true if this layer is an Input layer.
	// By default, returns true for layers of Type == emer.Input
	// Used to prevent adapting of inhibition or TrgAvg values.
	IsInput() bool

	// AlphaCycInit handles all initialization at start of new input pattern,
	// including computing netinput scaling from running average activation etc.
	// should already have presented the external input to the network at this point.
	AlphaCycInit()

	// InitGScale computes the initial scaling factor for synaptic input conductances G,
	// stored in GScale.Scale, based on sending layer initial activation.
	InitGScale()

	// GenNoise generates random noise for all neurons
	GenNoise()

	// DecayState decays activation state by given proportion (default is on ly.Act.Init.Decay)
	DecayState(decay float32)

	// RateClamp hard-clamps the activations in the layer -- called during AlphaCycInit
	// for hard-clamped Input layers
	RateClamp()

	//////////////////////////////////////////////////////////////////////////////////////
	//  Cycle Methods

	// SendSpike sends spike to receivers
	SendSpike(ltime *Time)

	// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta
	GFmInc(ltime *Time)

	// AvgMaxGe computes the average and max Ge stats, used in inhibition
	AvgMaxGe(ltime *Time)

	// InhibiFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
	InhibFmGeAct(ltime *Time)

	// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
	// and updates learning running-average activations from that Act
	ActFmG(ltime *Time)

	// AvgMaxAct computes the average and max Act stats, used in inhibition
	AvgMaxAct(ltime *Time)

	// CyclePost is called after the standard Cycle update, as a separate
	// network layer loop.
	// This is reserved for any kind of special ad-hoc types that
	// need to do something special after Act is finally computed.
	// For example, sending a neuromodulatory signal such as dopamine.
	CyclePost(ltime *Time)

	//////////////////////////////////////////////////////////////////////////////////////
	//  Quarter Methods

	// QuarterFinal does updating after end of a quarter
	QuarterFinal(ltime *Time)

	// CosDiffFmActs computes the cosine difference in activation state
	// between minus and plus phases.
	CosDiffFmActs()

	// DWt computes the weight change (learning) -- calls DWt method on sending projections
	DWt()

	// WtFmDWt updates the weights from delta-weight changes.
	// Computed from receiver perspective, does SubMean.
	WtFmDWt()

	// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
	// GScale conductance scaling, SWt updating, and adapting inhibition
	SlowAdapt()

	// LrateMult sets the new Lrate parameter for Prjns to LrateInit * mult.
	// Useful for implementing learning rate schedules.
	LrateMult(mult float32)
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

	// InitWts initializes weight values according to Learn.WtInit params
	InitWts()

	// InitWtSym initializes weight symmetry -- is given the reciprocal projection where
	// the Send and Recv layers are reversed.
	InitWtSym(rpj AxonPrjn)

	// InitGbuf initializes the per-projection synaptic conductance buffers.
	// This is not typically needed (called during InitWts, InitActs)
	// but can be called when needed.
	InitGbuf()

	// SendSpike sends a spike from sending neuron index si,
	// to add to buffer on receivers.
	SendSpike(si int)

	// RecvGInc increments the receiver's synaptic conductances from those of all the projections.
	RecvGInc(ltime *Time)

	// DWt computes the weight change (learning) -- on sending projections
	DWt()

	// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
	WtFmDWt()

	// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
	// GScale conductance scaling, and adapting inhibition
	SlowAdapt()

	// LrateMult sets the new Lrate parameter for Prjns to LrateInit * mult.
	// Useful for implementing learning rate schedules.
	LrateMult(mult float32)
}
