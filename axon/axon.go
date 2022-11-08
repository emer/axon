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

	// NewStateImpl handles all initialization at start of new input pattern, including computing
	// input scaling from running average activation etc.
	NewStateImpl()

	// Cycle handles entire update for one cycle (msec) of neuron activity state.
	CycleImpl(ctime *Time)

	// MinusPhaseImpl does updating after minus phase
	MinusPhaseImpl(ctime *Time)

	// PlusPhaseImpl does updating after plus phase
	PlusPhaseImpl(ctime *Time)

	// DWtImpl computes the weight change (learning) based on current
	// running-average activation values
	DWtImpl(ctime *Time)

	// WtFmDWtImpl updates the weights from delta-weight changes.
	// Also calls SynScale every Interval times
	WtFmDWtImpl(ctime *Time)

	// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
	// GScale conductance scaling, and adapting inhibition
	SlowAdapt(ctime *Time)
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

	// NeurStartIdx is the starting index in global network slice of neurons for
	// neurons in this layer
	NeurStartIdx() int

	// InitWts initializes the weight values in the network, i.e., resetting learning
	// Also calls InitActs
	InitWts()

	// InitActAvg initializes the running-average activation values that drive learning.
	InitActAvg()

	// InitActs fully initializes activation state -- only called automatically during InitWts
	InitActs()

	// InitWtsSym initializes the weight symmetry -- higher layers copy weights from lower layers
	InitWtSym()

	// InitGScale computes the initial scaling factor for synaptic input conductances G,
	// stored in GScale.Scale, based on sending layer initial activation.
	InitGScale()

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

	// NewState handles all initialization at start of new input pattern,
	// including computing Ge scaling from running average activation etc.
	// should already have presented the external input to the network at this point.
	NewState()

	// DecayState decays activation state by given proportion (default is on ly.Act.Init.Decay)
	DecayState(decay, glong float32)

	//////////////////////////////////////////////////////////////////////////////////////
	//  Cycle Methods

	// GiFmSpikes integrates new inhibitory conductances from Spikes
	// at the layer and pool level
	GiFmSpikes(ctime *Time)

	// CycleNeuron does one cycle (msec) of updating at the neuron level
	// calls the following via this AxonLay interface:
	// * Ginteg
	// * SpikeFmG
	// * PostAct
	// * SendSpike
	CycleNeuron(ni int, nrn *Neuron, ctime *Time)

	// GInteg integrates conductances G over time (Ge, NMDA, etc).
	// reads pool Gi values
	GInteg(ni int, nrn *Neuron, ctime *Time)

	// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
	SpikeFmG(ni int, nrn *Neuron, ctime *Time)

	// PostAct does updates at neuron level after activation (spiking)
	// updated for all neurons.
	// It is a hook for specialized algorithms -- empty at Axon base level
	PostAct(ni int, nrn *Neuron, ctime *Time)

	// SendSpike sends spike to receivers -- last step in Cycle, integrated
	// the next time around.
	// Writes to sending projections for this neuron.
	SendSpike(ni int, nrn *Neuron, ctime *Time)

	// CyclePost is called after the standard Cycle update, as a separate
	// network layer loop.
	// This is reserved for any kind of special ad-hoc types that
	// need to do something special after Act is finally computed.
	// For example, sending a neuromodulatory signal such as dopamine.
	CyclePost(ctime *Time)

	// MinusPhase does updating after end of minus phase
	MinusPhase(ctime *Time)

	// PlusPhase does updating after end of plus phase
	PlusPhase(ctime *Time)

	// SpkSt1 saves current activations into SpkSt1
	SpkSt1(ctime *Time)

	// SpkSt2 saves current activations into SpkSt2
	SpkSt2(ctime *Time)

	// CorSimFmActs computes the correlation similarity
	// (centered cosine aka normalized dot product)
	// in activation state between minus and plus phases
	// (1 = identical, 0 = uncorrelated).
	CorSimFmActs()

	// DWtLayer does weight change at the layer level.
	// does NOT call main projection-level DWt method.
	// in base, only calls DTrgAvgFmErr
	DWtLayer(ctime *Time)

	// WtFmDWtLayer does weight update at the layer level.
	// does NOT call main projection-level WtFmDWt method.
	// in base, only calls TrgAvgFmD
	WtFmDWtLayer(ctime *Time)

	// SlowAdapt is the layer-level slow adaptation functions.
	// Calls AdaptInhib and AvgDifFmTrgAvg for Synaptic Scaling.
	// Does NOT call projection-level methods.
	SlowAdapt(ctime *Time)

	// SynFail updates synaptic weight failure only -- normally done as part of DWt
	// and WtFmDWt, but this call can be used during testing to update failing synapses.
	SynFail(ctime *Time)
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

	// InitGBuffs initializes the per-projection synaptic conductance buffers.
	// This is not typically needed (called during InitWts, InitActs)
	// but can be called when needed.  Must be called to completely initialize
	// prior activity, e.g., full Glong clearing.
	InitGBuffs()

	// SendSpike sends a spike from sending neuron index si,
	// to add to buffer on receivers.
	SendSpike(si int)

	// GFmSpikes increments synaptic conductances from Spikes
	// including pooled aggregation of spikes into Pools for FS-FFFB inhib.
	GFmSpikes(ctime *Time)

	// SendSynCa updates synaptic calcium based on spiking, for SynSpkTheta mode.
	// Optimized version only updates at point of spiking.
	// This pass goes through in sending order, filtering on sending spike.
	SendSynCa(ctime *Time)

	// RecvSynCa updates synaptic calcium based on spiking, for SynSpkTheta mode.
	// Optimized version only updates at point of spiking.
	// This pass goes through in recv order, filtering on recv spike.
	RecvSynCa(ctime *Time)

	// DWt computes the weight change (learning) -- on sending projections.
	DWt(ctime *Time)

	// DWtSubMean subtracts the mean from any projections that have SubMean > 0.
	// This is called on *receiving* projections, prior to WtFmDwt.
	DWtSubMean(ctime *Time)

	// WtFmDWt updates the synaptic weight values from delta-weight changes,
	// on sending projections
	WtFmDWt(ctime *Time)

	// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
	// GScale conductance scaling, and adapting inhibition
	SlowAdapt(ctime *Time)

	// SynFail updates synaptic weight failure only -- normally done as part of DWt
	// and WtFmDWt, but this call can be used during testing to update failing synapses.
	SynFail(ctime *Time)
}
