// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "cogentcore.org/core/tensor"

// nogo:generate gosl -exclude=Update,UpdateParams,Defaults,AllParams,ShouldDisplay

// vars are all the global vars for axon GPU / CPU computation.
//
//gosl:vars
var (
	//////////////////// Params

	// LayParams are all the layer parameters.
	//gosl:group Params
	//gosl:read-only
	LayParams []LayerParams

	// Paths are all the path parameters.
	//gosl:read-only
	Paths []PathParams

	//////////////////// Indexes

	// NeuronIxs have index values for each neuron: index into layer, pools.
	// [Indexes][Neurons]
	//gosl:group Indexes
	//gosl:read-only
	NeuronIxs *tensor.Uint32

	// SynapseIxs have index values for each synapse:
	// providing index into recv, send neurons, path.
	// [Indexes][NSyns]; NSyns = [Layer][SendPaths][SendNeurons][Syns]
	//gosl:read-only
	SynapseIxs *tensor.Uint32

	// PathSendCon are starting offset and N cons for each sending neuron,
	// for indexing into the Syns synapses, which are organized sender-based.
	// [NSendCon][StartNN]; NSendCon = [Layer][SendPaths][SendNeurons]
	//gosl:read-only
	PathSendCon *tensor.Uint32

	// RecvPathIxs indexes into Paths (organized by SendPath) organized
	// by recv pathways. needed for iterating through recv paths efficiently on GPU.
	// [NRecvPaths] = [Layer][RecvPaths]
	//gosl:read-only
	RecvPathIxs *tensor.Uint32

	// PathRecvCon are the receiving path starting index and number of connections.
	// [NRecvCon][StartNN]; NRecvCon = [Layer][RecvPaths][RecvNeurons]
	//gosl:read-only
	PathRecvCon *tensor.Uint32

	// RecvSynIxs are the indexes into Synapses for each recv neuron, organized
	// into blocks according to PathRecvCon, for receiver-based access.
	// [NSyns] = [Layer][RecvPaths][RecvNeurons][Syns]
	//gosl:read-only
	RecvSynIxs *tensor.Uint32

	//////////////////// Neuron State

	// Ctx is the current context state (one).
	//gosl:group Neurons
	Ctx []Context

	// Neurons are all the neuron state variables.
	// [Vars][Neurons][Data]
	Neurons *tensor.Float32

	// NeuronAvgs are variables with averages over the
	// Data parallel dimension for each neuron.
	// [Vars][Neurons]
	NeuronAvgs *tensor.Float32

	// Pools are the inhibitory pool variables for each layer and pool, and Data.
	// [Layer][Pools][Data]
	Pools []Pool

	// LayValues are the [LayerValues] for each layer and Data.
	// [Layer][Data]
	LayValues []LayerValues

	// GlobalScalars are the global scalar state variables.
	// [GlobalScalarsN][Data]
	GlobalScalars *tensor.Float32

	// GlobalVectors are the global vector state variables.
	// [GlobalVectorsN][MaxVecN][Data]
	GlobalVectors *tensor.Float32

	// Exts are external input values for all Input / Target / Compare layers
	// in the network. The ApplyExt methods write to this per layer,
	// and it is then actually applied in one consistent method.
	// [NExts][Data]; NExts = [In / Out Layers][Neurons]
	Exts *tensor.Float32

	//////////////////// Synapse State

	// PathGBuf is the conductance buffer for accumulating spikes.
	// subslices are allocated to each pathway.
	// uses int-encoded float values for faster GPU atomic integration.
	// [NPathNeur][MaxDel+1][Data]; NPathNeur = [Layer][RecvPaths][RecvNeurons]
	//gosl:group Synapse
	PathGBuf *tensor.Int32

	// PathGSyns are synaptic conductance integrated over time per pathway
	// per recv neurons. spikes come in via PathBuf.
	// subslices are allocated to each pathway.
	// [NPathNeur][Data]
	PathGSyns *tensor.Float32

	//	Synapses are the synapse level variables (weights etc).
	// [Vars][NSyns]; NSyns = [Layer][SendPaths][SendNeurons][Syns]
	Synapses *tensor.Float32

	//////////////////// SynapseTraces

	// SynapseTraces are data parallel synaptic variables,
	// for accumulating learning traces and weight changes per data.
	// This is the largest data size, so multiple instances are used
	// to handle larger networks.
	// [Vars][NSyns][Data]; NSyns = [Layer][SendPaths][SendNeurons][Syns]
	//gosl:group SynapseTraces
	SynapseTraces *tensor.Float32

	// SynapseTraces1 is an overflow buffer fro SynapseTraces.
	SynapseTraces1 *tensor.Float32
)
