# Kinase Equations

This standalone simulation is used to explore the synapse-level updating of calcium-based signals that drive the more abstract forms of Kinase learning rules, based on recv and send spiking impulses, passing through a cascade of exponential integrations with different time constants.

The Leabra and initial Axon equations used these cascading updated variables on each neuron (recv and send) separately, with the final values multiplied for the CHL-like DWt function.

* `CaM` = first level integration: `CaM += (g * Spike - CaM) / MTau` -- spike drives up toward a max per-spike amount with gain factor (g = 8 std).  The g multiplier gives a faster effective rate constant for going up vs. decay back down -- this just sets the overall scale of the values.  This represents an abstract version of CaM (calmodulin) which is activated by calcium influx from NMDA receptors.  This is `AvgSS` in Leabra.

* `CaP` = LTP, plus phase faster integration of CaM, reflecting CaMKII biologically.  Faster time constant makes this reflect the plus phase relative to the `CaD` LTD, minus-phase variable.  This is `AvgS` in Leabra.

* `CaD` = LTD, minus phase slower integration of CaP (cascaded one further step).  This is `AvgM` in Leabra.

The CHL product-based learning rule is a function of the difference between the plus-phase vs. minus-phase factors:

* `PDWt = SCaP * RCaP - SCaD * RCaD`

For the rate-code activations in Leabra, the product of these averages is likely to be similar to the average of the products at a synapse level, and computing neuron-level values is *much* faster computationally than integrating the products at the synapse level.  Indeed, experiments (a long time ago) showed no advantages to doing the synapse-level integration in Leabra.

# Synapse-level integration of spikes

For spiking, the relative timing of pre-post spiking has been an obsession since the discovery of STDP.

However, at a computational level, capturing these pre-post timing interactions clearly requires computationally-expensive synapse-level integration.  Thus, a major technical issue we address is how to more efficiently integrate this synapse-level signal at the theoretically most efficient level which is when either the sender or receiver spikes, with the subsequent integration computed based on time passed instead of incrementally updating.

The first issue is how the pre-post spikes interact in computing the Ca and CaM first-level integrations of synaptic activity.  Here are some options:

* "Or" rule: `SynSpk = SSpk || RSpk` -- either spike counts, but there is no specific interaction -- this is the least product-like.  This SynSpk value then drives the same cascade of time integrations.  

* 

