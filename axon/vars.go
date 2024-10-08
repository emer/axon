// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//go:generate gosl -exclude=Update,UpdateParams,Defaults,AllParams,ShouldDisplay 

// vars are all the global vars for axon GPU / CPU computation.
//gosl:vars
var (
	// Layers are all the layer parameters.
	//gosl:group Params
	//gosl:read-only
	Layers []LayerParams>
	
	// Paths are all the path parameters.
	//gosl:read-only
	Paths []PathParams>

	// NeuronIxs have index values for each neuron: index into layer, pools.
	//gosl:group Indexes
	//gosl:read-only
	NeuronIxs tensor.Uint32 // [Indexes][Neurons]

	// SynapseIxs have index values for each synapse:
	// providing index into recv, send neurons, path.
	//gosl:read-only
	SynapseIxs	tensor.Uint32 // [Layer][SendPaths][SendNeurons][Syns]

	// PathSendCon are starting offset and N cons for each sending neuron,
	// for indexing into the Syns synapses, which are organized sender-based.
	//gosl:read-only
	PathSendCon tensor.Uint32 // [Layer][SendPaths][SendNeurons][StartNN]

	// RecvPathIndexes indexes into Paths (organized by SendPath) organized
	// by recv pathways. needed for iterating through recv
	// paths efficiently on GPU.
	//gosl:read-only
	RecvPathIndexes tensor.Uint32 // [Layer][RecvPaths]
	
	// RecvCon are the receiving path starting index and number of connections.
	//gosl:read-only
	RecvCon tensor.Uint32  // [Layer][RecvPaths][RecvNeurons]

	// RecvSynIndexes are the indexes into Synapses for each recv neuron, organized
	// into blocks according to PathRecvCon, for receiver-based access.
	//gosl:read-only
	RecvSynIndexes tensor.Uint32 // [Layer][RecvPaths][RecvNeurons][Syns]

	// Ctx is the current context state.
	//gosl:group Neurons
	Ctx	[]Context> // [0]

	// Neurons are all the neuron state variables.
	Neurons tensor.Float32 // [Vars][Neurons][Data]
	
	// NeuronAvgs are variables with averages over the
	// Data parallel dimension for each neuron.
	NeuronAvgs tensor.Float32 // [Vars][Neurons]

	// Pools are the inhibitory pool variables for each layer and pool, and Data.
	Pools tensor.Float32 // [Layer][Pools][Data]
	
	// LayValues are the [LayerValues] for each layer and Data.
	LayValues []LayerValues // [Layer][Data]

	// Globals are the global state variables.
	Globals tensor.Float32 // [NGlobals]

	// Exts are external input values for all Input / Target / Compare layers
	// in the network. The ApplyExt methods write to this per layer,
	// and it is then actually applied in one consistent method.
	Exts tensor.Float32 // [In / Out Layers][Neurons][Data]

	// PathGBuf is the conductance buffer for accumulating spikes.
	// subslices are allocated to each pathway.
	// uses int-encoded float values for faster GPU atomic integration.
	//gosl:group Synapse
	PathGBuf tensor.Int32 // [Layer][RecvPaths][RecvNeurons][MaxDel+1][Data]

	// PathGSyns are synaptic conductance integrated over time per pathway
	// per recv neurons. spikes come in via PathBuf.
	// subslices are allocated to each pathway.
	PathGSyns tensor.Float32 // [Layer][RecvPaths][RecvNeurons][Data]

	//	Synapses are the synapse level variables (weights etc).
	// Not data parallel.
	Synapses tensor.Float32 // [Vars][Layer][SendPaths][SendNeurons][Syns]

	// todo: future expansion to add more tranches of Synapses

	// SynapseTraces are data parallel synaptic variables,
	// for accumulating learning traces and weight changes per data.
	// This is the largest data size, so multiple instances are used
	// to handle larger networks.
	//gosl:group SynapseTraces
	SynapseTraces tensor.Float32 // [Vars][Layer][SendPaths][SendNeurons][Syns][Data]

	// SynapseTraces1 is an overflow buffer fro SynapseTraces.
	SynapseTraces1 tensor.Float32 

	// SynapseTraces2 is an overflow buffer fro SynapseTraces.
	SynapseTraces2 tensor.Float32 
)

