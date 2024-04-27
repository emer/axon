# Reinforcement Learning and Dopamine

The rl_{[net.go](axon/rl_net.go), [layers.go](axon/rl_layers.go), [prjns.go](axon/rl_prjns.go)} files in axon provide core infrastructure for dopamine neuromodulation and reinforcement learning, including the Rescorla-Wagner learning algorithm (RW) and Temporal Differences (TD) learning.

In the new GPU-based code, the `Context` type, `NeuroModValues` field is used to communicate global neuromodulatory signals around the network: broadcasting new modulatory signals and communicating computed values between specialized layer types.

You can just directly set the `Context.NeuroMod` values to arbitrarily clamp them, and use a standard `InputLayer` to display the clamped value.

Here's the info from `layertypes.go` -- see `examples/rl` for example usage, and use the Network `AddRWLayers` or `AddTDLayers` methods to get all the relevant layers configured properly.

	// RewLayer represents positive or negative reward values across 2 units,
	// showing spiking rates for each, and Act always represents signed value.
	RewLayer

	// RSalienceAChLayer reads Max layer activity from specified source layer(s)
	// and optionally the global Context.NeuroMod.Rew or RewPred state variables,
	// and updates the global ACh = Max of all as the positively rectified,
	// non-prediction-discounted reward salience signal.
	// Acetylcholine (ACh) is known to represent these values.
	RSalienceAChLayer

	// RWPredLayer computes reward prediction for a simple Rescorla-Wagner
	// learning dynamic (i.e., PV learning in the PVLV framework).
	// Activity is computed as linear function of excitatory conductance
	// (which can be negative -- there are no constraints).
	// Use with RWPrjn which does simple delta-rule learning on minus-plus.
	RWPredLayer

	// RWDaLayer computes a dopamine (DA) signal based on a simple Rescorla-Wagner
	// learning dynamic (i.e., PV learning in the PVLV framework).
	// It computes difference between r(t) and RWPred values.
	// r(t) is accessed directly from a Rew layer -- if no external input then no
	// DA is computed -- critical for effective use of RW only for PV cases.
	// RWPred prediction is also accessed directly from Rew layer to avoid any issues.
	RWDaLayer

	// TDPredLayer is the temporal differences reward prediction layer.
	// It represents estimated value V(t) in the minus phase, and computes
	// estimated V(t+1) based on its learned weights in plus phase,
	// using the TDPredPrjn projection type for DA modulated learning.
	TDPredLayer

	// TDIntegLayer is the temporal differences reward integration layer.
	// It represents estimated value V(t) from prior time step in the minus phase,
	// and estimated discount * V(t+1) + r(t) in the plus phase.
	// It gets Rew, PrevPred from Context.NeuroMod, and Special
	// LayerValues from TDPredLayer.
	TDIntegLayer

	// TDDaLayer computes a dopamine (DA) signal as the temporal difference (TD)
	// between the TDIntegLayer activations in the minus and plus phase.
	// These are retrieved from Special LayerValues.
	TDDaLayer

Some important considerations:    
    
* The RW and TD DA layers use the `CyclePost` layer-level method to send the DA to other layers, at end of each cycle, after activation is updated.  Thus, DA lags by 1 cycle, which typically should not be a problem. 

* See [Rubicon](Rubicon.md) for the full biologically based PVLV model of phasic dopamine.

* To encode positive and negative values using spiking, 2 units are used, one for positive and the other for negative.  The `Act` value always represents the (signed) computed value, not the spike rate, where applicable.
