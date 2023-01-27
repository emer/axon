// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/etensor"
	"github.com/goki/gosl/sltype"
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
	NewStateImpl(ctx *Context)

	// Cycle handles entire update for one cycle (msec) of neuron activity state.
	CycleImpl(ctx *Context)

	// MinusPhaseImpl does updating after minus phase
	MinusPhaseImpl(ctx *Context)

	// PlusPhaseImpl does updating after plus phase
	PlusPhaseImpl(ctx *Context)

	// DWtImpl computes the weight change (learning) based on current
	// running-average activation values
	DWtImpl(ctx *Context)

	// WtFmDWtImpl updates the weights from delta-weight changes.
	// Also calls SynScale every Interval times
	WtFmDWtImpl(ctx *Context)

	// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
	// GScale conductance scaling, and adapting inhibition
	SlowAdapt(ctx *Context)
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

	// LayerType returns the axon-specific LayerTypes type
	LayerType() LayerTypes

	// NeurStartIdx is the starting index in global network slice of neurons for
	// neurons in this layer -- convenience interface method for threading dispatch.
	NeurStartIdx() int

	// SetBuildConfig sets named configuration parameter to given string value
	// to be used in the PostBuild stage -- mainly for layer names that need to be
	// looked up and turned into indexes, after entire network is built.
	SetBuildConfig(param, val string)

	// PostBuild performs special post-Build() configuration steps for specific algorithms,
	// using configuration data from SetBuildConfig during the ConfigNet process.
	PostBuild()

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
	// If the layer is a Target or Compare layer type, then it goes in Target
	// otherwise it goes in Ext.
	ApplyExt(ext etensor.Tensor)

	// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
	// If the layer is a Target or Compare layer type, then it goes in Target
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

	// RecvPrjns returns the slice of receiving projections for this layer
	RecvPrjns() *AxonPrjns

	// SendPrjns returns the slice of sending projections for this layer
	SendPrjns() *AxonPrjns

	//////////////////////////////////////////////////////////////////////////////////////
	//  Cycle Methods

	// GatherSpikes integrates G*Raw and G*Syn values for given neuron
	// while integrating the Prjn-level GSyn integrated values.
	// ni is layer-specific index of neuron within its layer.
	GatherSpikes(ctx *Context, ni uint32, nrn *Neuron)

	// GiFmSpikes integrates new inhibitory conductances from Spikes
	// at the layer and pool level
	GiFmSpikes(ctx *Context)

	// CycleNeuron does one cycle (msec) of updating at the neuron level
	// calls the following via this AxonLay interface:
	// * GInteg
	// * SpikeFmG
	// * PostAct
	CycleNeuron(ctx *Context, ni uint32, nrn *Neuron)

	// GInteg integrates conductances G over time (Ge, NMDA, etc).
	// reads pool Gi values.
	GInteg(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, vals *LayerVals, randctr *sltype.Uint2)

	// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
	SpikeFmG(ctx *Context, ni uint32, nrn *Neuron)

	// PostSpike does updates at neuron level after spiking has been computed.
	// This is where special layer types add extra code.
	PostSpike(ctx *Context, ni uint32, nrn *Neuron)

	// SendSpike sends spike to receivers -- last step in Cycle, integrated
	// the next time around.
	// Writes to sending projections for this neuron.
	SendSpike(ctx *Context)

	// CyclePost is called after the standard Cycle update, as a separate
	// network layer loop.
	// This is reserved for any kind of special ad-hoc types that
	// need to do something special after Spiking is finally computed and Sent.
	// It ONLY runs on the CPU, not the GPU -- should update global values
	// in the Context state which are re-sync'd back to GPU,
	// and values in other layers MUST come from LayerVals because
	// this is the only data that is sync'd back from the GPU each cycle.
	// For example, updating a neuromodulatory signal such as dopamine.
	CyclePost(ctx *Context)

	// NewState handles all initialization at start of new input pattern,
	// including computing Ge scaling from running average activation etc.
	// should already have presented the external input to the network at this point.
	NewState(ctx *Context)

	// DecayState decays activation state by given proportion (default is on ly.Params.Act.Init.Decay)
	DecayState(ctx *Context, decay, glong float32)

	//////////////////////////////////////////////////////////////////////////////////////
	//  Phase Methods

	// MinusPhase does updating after end of minus phase
	MinusPhase(ctx *Context)

	// PlusPhase does updating after end of plus phase
	PlusPhase(ctx *Context)

	// SpkSt1 saves current activations into SpkSt1
	SpkSt1(ctx *Context)

	// SpkSt2 saves current activations into SpkSt2
	SpkSt2(ctx *Context)

	// CorSimFmActs computes the correlation similarity
	// (centered cosine aka normalized dot product)
	// in activation state between minus and plus phases
	// (1 = identical, 0 = uncorrelated).
	CorSimFmActs()

	//////////////////////////////////////////////////////////////////////////////////////
	//  Learn Methods

	// DWtLayer does weight change at the layer level.
	// does NOT call main projection-level DWt method.
	// in base, only calls DTrgAvgFmErr
	DWtLayer(ctx *Context)

	// WtFmDWtLayer does weight update at the layer level.
	// does NOT call main projection-level WtFmDWt method.
	// in base, only calls TrgAvgFmD
	WtFmDWtLayer(ctx *Context)

	// SlowAdapt is the layer-level slow adaptation functions.
	// Calls AdaptInhib and AvgDifFmTrgAvg for Synaptic Scaling.
	// Does NOT call projection-level methods.
	SlowAdapt(ctx *Context)

	// SynFail updates synaptic weight failure only -- normally done as part of DWt
	// and WtFmDWt, but this call can be used during testing to update failing synapses.
	SynFail(ctx *Context)

	// SendCtxtGe sends activation (CaSpkP) over CTCtxtPrjn projections to integrate
	// CtxtGe excitatory conductance on CT layers.
	// This should be called at the end of the Plus (5IB Burst) phase via Network.CTCtxt
	SendCtxtGe(ctx *Context)

	// CtxtFmGe integrates new CtxtGe excitatory conductance from projections, and computes
	// overall Ctxt value, only on CT layers.
	// This should be called at the end of the Plus (5IB Bursting) phase via Network.CTCtxt
	CtxtFmGe(ctx *Context)
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

	// PrjnType returns the axon-specific PrjnTypes type
	PrjnType() PrjnTypes

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
	SendSpike(ctx *Context, sendIdx int)

	// RecvSpikes receives spikes from the sending neurons at index sendIdx
	// into the GBuf buffer on the receiver side. The buffer on the receiver side
	// is a ring buffer, which is used for modelling the time delay between
	// sending and receiving spikes.
	RecvSpikes(ctx *Context, recvIdx int)

	// SendSynCa updates synaptic calcium based on spiking, for SynSpkTheta mode.
	// Optimized version only updates at point of spiking.
	// This pass goes through in sending order, filtering on sending spike.
	SendSynCa(ctx *Context)

	// RecvSynCa updates synaptic calcium based on spiking, for SynSpkTheta mode.
	// Optimized version only updates at point of spiking.
	// This pass goes through in recv order, filtering on recv spike.
	RecvSynCa(ctx *Context)

	// DWt computes the weight change (learning) -- on sending projections.
	DWt(ctx *Context)

	// DWtSubMean subtracts the mean from any projections that have SubMean > 0.
	// This is called on *receiving* projections, prior to WtFmDwt.
	DWtSubMean(ctx *Context)

	// WtFmDWt updates the synaptic weight values from delta-weight changes,
	// on sending projections
	WtFmDWt(ctx *Context)

	// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
	// GScale conductance scaling, and adapting inhibition
	SlowAdapt(ctx *Context)

	// SynFail updates synaptic weight failure only -- normally done as part of DWt
	// and WtFmDWt, but this call can be used during testing to update failing synapses.
	SynFail(ctx *Context)
}

type AxonPrjns []AxonPrjn

func (pl *AxonPrjns) Add(p AxonPrjn) {
	(*pl) = append(*pl, p)
}
