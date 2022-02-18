# Kinase Equations

See: https://github.com/ccnlab/kinase/tree/main/sims/kinase for a parallel exploration based on a biophysically detailed model, building up from the Urakubo et al (2008) model.

This standalone simulation is used to explore the synapse-level updating of calcium-based signals that drive the more abstract forms of Kinase learning rules, based on recv and send spiking impulses, passing through a cascade of exponential integrations with different time constants.

The Leabra and initial Axon equations used these cascading updated variables on each neuron (recv and send) separately, with the final values multiplied for the CHL-like DWt function.

* `CaM` = first level integration: `CaM += (g * Spike - CaM) / MTau` -- spike drives up toward a max per-spike amount with gain factor (g = 8 std).  The g multiplier gives a faster effective rate constant for going up vs. decay back down -- this just sets the overall scale of the values.  This represents an abstract version of CaM (calmodulin) which is activated by calcium influx from NMDA receptors.  This is `AvgSS` in Leabra.

* `CaP` = LTP, plus phase faster integration of CaM, reflecting CaMKII biologically.  Faster time constant makes this reflect the plus phase relative to the `CaD` LTD, minus-phase variable.  This is `AvgS` in Leabra.

* `CaD` = LTD, minus phase slower integration of CaP (cascaded one further step), reflecting DAPK1 biologically.  This is `AvgM` in Leabra.

The CHL product-based learning rule is a function of the difference between the plus-phase vs. minus-phase products of factors for S = Sender and R = Receiver:

* `PDWt = SCaP * RCaP - SCaD * RCaD`

This is the `NeurSpkCa` algorithm -- separate neuron-level Ca integration.

For the rate-code activations in Leabra, the product of these averages is likely to be similar to the average of the products at a synapse level, and computing neuron-level values is *much* faster computationally than integrating the products at the synapse level.  Indeed, experiments (a long time ago) showed no advantages to doing the synapse-level integration in Leabra.

# Synapse-level integration of spikes

For spiking, the relative timing of pre-post spiking has been an obsession since the discovery of STDP.

However, at a computational level, capturing these pre-post timing interactions clearly requires computationally-expensive synapse-level integration.  Thus, a major technical issue we address is how to more efficiently integrate this synapse-level signal at the theoretically most efficient level which is when either the sender or receiver spikes, with the subsequent integration computed based on time passed instead of incrementally updating.

The first issue is how the pre-post spikes interact in computing the Ca and CaM first-level integrations of synaptic activity.  The simplest, most straightforward case is:

* `SynSpkCa` "OR" rule: `SynSpk = SSpk || RSpk` -- either spike counts, and there is no explicit interaction at the level of the Ca influx per spike, or anything distinctive about a pre vs. postsynaptic spike.

Counterintuitively, under typical conditions in dendritic spines as explored in the biophysical [kinase](https://github.com/ccnlab/kinase/tree/main/sims/kinase) model, this is actually a reasonable first-order approximation of calcium influx from pre-post spikes.  The postsynaptic Vm is often reasonably depolarized in active neurons in awake behaving conditions, so the Glu release from pre-spikes has a direct impact, while there is often enough residual channel open state from prior Glu release for backpropagating action potentials to drive an extra burst of Ca influx.

In this context, it is critical to consider more naturalistic conditions relative to the relatively sterile pairs of isolated spikes in the standard STDP paradigm, which is highly unrepresentative and thus can give rise to incorrect assumptions and intuitions about the dynamics in more naturalistic conditions. 

Also, the `SynSpk` value then drives the CaM, CaP, CaD cascade of time integrations, and this integration process is remarkably sensitive to the ordering of spikes as they come in: the integration dynamics produce strong interactions across the different timescales, as became more clear when implementing the computational optimization described below: a huge 3 factor lookup table is needed to capture the interactions between the 3 timescales.  Thus, even if the spikes themselves don't interact per-se, the individual synapse will end up with a unique integrated signal, relative to what would have otherwise happened at each send, recv neuron independently.

![20hz SynSpkCa integration](results/fig_synspk_mpd_optimized_fit_20hz_res01.png?raw=true "SynSpkCa for  20hz of both pre-post firing")

It is also notable that the `CaD` value in both of these cases reaches its asymptotic value right around the 200 msec point


## SynSpkCa DWt is much less variable than `NeurSpkCa`

One of the most salient results from this model is that integrating spiking at the synapse level ("average of products") produces a much less variable DWt signal than doing it at the neuron level ("product of averages").  The intuition is that for the latter (aka `NeurSpkCa`), a great deal depends on the precise timing of the *last few spikes* at the end of the sequence, and the product operation itself introduces further variance (each term has multiplicative "leverage" over the other).

Here are results for different matched levels of firing, showing final DWt values after each of 10000 different 200 msec trials of learning, with the indicated level of firing on both send and recv, with no change over time.  The expected DWt value should be 0 as there is no minus-plus phase difference.

100 Hz:

![SynSpkCa vs. NeurSpkCa at 100hz](results/fig_synspk_vs_neurspk_100hz.png?raw=true "SynSpkCa vs. NeurSpkCa at 100hz")

50 Hz:

![SynSpkCa vs. NeurSpkCa at 50hz](results/fig_synspk_vs_neurspk_50hz.png?raw=true "SynSpkCa vs. NeurSpkCa at 50z")

20 Hz:

![SynSpkCa vs. NeurSpkCa at 20hz](results/fig_synspk_vs_neurspk_20hz.png?raw=true "SynSpkCa vs. NeurSpkCa at 20hz")

The differences are greatest when the firing rate is high.  Lower firing rates allow for greater variance in precise timing, and thus have greater variance in the Syn mechanism.

# Optimized SynSpkCa integration only at Spikes

For the simple `SynSpkCa` rule, it is possible to only update values at each spike event, because the intervening dynamics are fully determined therefrom.  However, due to the cascading nature of the integration, it is highly non-linear and requires a massive lookup table to accurately capture the dynamics, as a function of the exact CaM, CaP, and CaD levels at the moment a new spike comes in.

This is now implemented in the `kinase` package, using the `Funts` function tables.

With a resolution of .01, the results are quite accurate.  Compare the `SynC` continuously-integrated values vs. the `SynO` optimized values in the following figures:

100Hz:

![Optimized integration, 100hz](results/fig_synspk_mpd_optimized_fit_100hz_res01.png?raw=true "Optimized integration at 100hz")

20Hz:

![Optimized integration, 20hz](results/fig_synspk_mpd_optimized_fit_20hz_res01.png?raw=true "Optimized integration at 20hz")

20Hz with Res = .02, which significantly reduces the size of the lookup table needed for the CaD level integration, from 887 to 112 million entries:

![Optimized integration, 20hz](results/fig_synspk_mpd_optimized_fit_20hz_res01.png?raw=true "Optimized integration at 20hz")

The differences in the optimized vs. continuously-integrated cases is now evident, but the margin of error is relatively small and does not notably increase over time, as the errors tend to cancel out over time.


