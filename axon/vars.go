// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "cogentcore.org/lab/tensor"

//go:generate gosl -exclude=Update,UpdateParams,Defaults,AllParams,ShouldDisplay

// CurrentNetwork is set in [Network.SetAsCurrent] method, which sets all
// global variables to point to the current network to be processed.
// These global vars are necessary for GPU kernel computation.
var CurrentNetwork *Network

//gosl:start

// vars are all the global vars for axon GPU / CPU computation.
//
//gosl:vars
var (
	//////// Params

	// Layers are all the layer parameters.
	//gosl:group Params
	//gosl:read-only
	Layers []LayerParams

	// Paths are all the path parameters.
	//gosl:read-only
	Paths []PathParams

	//////// Indexes

	// NetworkIxs have indexes and sizes for entire network (one only).
	//gosl:group Indexes
	//gosl:read-only
	NetworkIxs []NetworkIndexes

	// PoolIxs have index values for each Pool.
	// [Layer * Pools][PoolIndexVars]
	//gosl:read-only
	//gosl:dims 2
	PoolIxs *tensor.Uint32

	// NeuronIxs have index values for each neuron: index into layer, pools.
	// [Neurons][Indexes]
	//gosl:read-only
	//gosl:dims 2
	NeuronIxs *tensor.Uint32

	// SynapseIxs have index values for each synapse:
	// providing index into recv, send neurons, path.
	// [Indexes][NSyns]; NSyns = [Layer][SendPaths][SendNeurons][Syns]
	//gosl:read-only
	//gosl:dims 2
	SynapseIxs *tensor.Uint32

	// PathSendCon are starting offset and N cons for each sending neuron,
	// for indexing into the Syns synapses, which are organized sender-based.
	// [NSendCon][StartNN]; NSendCon = [Layer][SendPaths][SendNeurons]
	//gosl:read-only
	//gosl:dims 2
	PathSendCon *tensor.Uint32

	// RecvPathIxs indexes into Paths (organized by SendPath) organized
	// by recv pathways. needed for iterating through recv paths efficiently on GPU.
	// [NRecvPaths] = [Layer][RecvPaths]
	//gosl:read-only
	//gosl:dims 1
	RecvPathIxs *tensor.Uint32

	// PathRecvCon are the receiving path starting index and number of connections.
	// [NRecvCon][StartNN]; NRecvCon = [Layer][RecvPaths][RecvNeurons]
	//gosl:read-only
	//gosl:dims 2
	PathRecvCon *tensor.Uint32

	// RecvSynIxs are the indexes into Synapses for each recv neuron, organized
	// into blocks according to PathRecvCon, for receiver-based access.
	// [NSyns] = [Layer][RecvPaths][RecvNeurons][Syns]
	//gosl:read-only
	//gosl:dims 1
	RecvSynIxs *tensor.Uint32

	//////// Neuron+ State

	// Ctx is the current context state (one only). This is read-only except in
	// specific kernels.
	//gosl:group Neurons
	//gosl:read-or-write
	Ctx []Context

	// Neurons are all the neuron state variables.
	// [Neurons][Data][NeuronVarsN+NCaBins]
	//gosl:dims 3
	Neurons *tensor.Float32

	// NeuronAvgs are variables with averages over the
	// Data parallel dimension for each neuron.
	// [Neurons][Vars]
	//gosl:dims 2
	NeuronAvgs *tensor.Float32

	// LayerStates holds layer-level state values, with variables defined in
	// [LayerVars], for each layer and Data parallel index.
	// [Layer][Data][LayerVarsN]
	//gosl:dims 3
	LayerStates *tensor.Float32

	// GlobalScalars are the global scalar state variables.
	// [GlobalScalarsN + 2*NCaBins][Data]
	//gosl:dims 2
	GlobalScalars *tensor.Float32

	// GlobalVectors are the global vector state variables.
	// [GlobalVectorsN][MaxVecN][Data]
	//gosl:dims 3
	GlobalVectors *tensor.Float32

	// Exts are external input values for all Input / Target / Compare layers
	// in the network. The ApplyExt methods write to this per layer,
	// and it is then actually applied in one consistent method.
	// [NExts][Data]; NExts = [In / Out Layers][Neurons]
	//gosl:dims 2
	Exts *tensor.Float32

	// Pools are the [PoolVars] float32 state values for layer and sub-pool inhibition,
	// Including the float32 AvgMax values by Phase and variable: use [AvgMaxVarIndex].
	// [Layer * Pools][Data][PoolVars+AvgMax]
	//gosl:dims 3
	Pools *tensor.Float32

	// PoolsInt are the [PoolIntVars] int32 state values for layer and sub-pool
	// inhibition, AvgMax atomic integration, and other vars: use [AvgMaxIntVarIndex]
	// [Layer * Pools][Data][PoolIntVars+AvgMax]
	//gosl:dims 3
	PoolsInt *tensor.Int32

	//////// Synapse State

	// PathGBuf is the conductance buffer for accumulating spikes.
	// Subslices are allocated to each pathway.
	// Uses int-encoded values for faster GPU atomic integration.
	// [NPathNeur][Data][MaxDel+1]; NPathNeur = [Layer][RecvPaths][RecvNeurons]
	//gosl:group Synapse
	//gosl:dims 3
	PathGBuf *tensor.Int32

	// PathGSyns are synaptic conductance integrated over time per pathway
	// per recv neurons. spikes come in via PathBuf.
	// subslices are allocated to each pathway.
	// [NPathNeur][Data]
	//gosl:dims 2
	PathGSyns *tensor.Float32

	//	Synapses are the synapse level variables (weights etc).
	// These do not depend on the data parallel index, unlike [SynapseTraces].
	// [NSyns][Vars]; NSyns = [Layer][SendPaths][SendNeurons][Syns]
	//gosl:dims 2
	Synapses *tensor.Float32

	// SynapseTraces are synaptic variables that depend on the data
	// parallel index, for accumulating learning traces and weight changes per data.
	// This is the largest data size, so nbuffs multiple instances are used
	// to handle larger networks.
	// [NSyns][Data][Vars]; NSyns = [Layer][SendPaths][SendNeurons][Syns]
	//gosl:dims 3
	//gosl:nbuffs 7
	SynapseTraces *tensor.Float32
)

//gosl:end
