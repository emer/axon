# FS (Fast & Slow) FFFB Inhibition

[FFFB](https://github.com/emer/axon/tree/master/fffb) is the feedforward (FF) and feedback (FB) inhibition mechanism, originally developed for the Leabra model.  It applies inhibition as a function of the average `Ge` (excitatory conductance) of units in the layer (FF = reflecting all the excitatory input to a layer) and average `Act` rate-code-like activation within the layer (FB = reflecting the activity level within the layer itself), which is slowly integrated over time as a function of the ISI (inter-spike-interval).

This new FS -- fast and slow -- version of FFFB starts directly with spike input signals instead of using "internal" variables like Ge and Act, and uses time dynamics based on an emerging consensus about the differences between three major classes of inhibitory interneurons, and their functional properties, e.g., [Cardin, 2018](#references).

* **PV:** fast-spiking basket cells that target the cell bodies of excitatory neurons and coexpress the calcium-binding protein parvalbumin (PV). These are the "first responders", and are also rapidly depressing -- they provide quick control of activity, responding to FF new input and FB drive, allowing the first spiking pyramidal neurons to quickly shut off other competitors.

* **SST:** more slowly-responding, low-threshold spiking cells that target the distal dendrites of excitatory neurons and coexpress the peptide somatostatin (SST). These require repetitive, facilitating, afferent input to be activated, and may regulate the dendritic integration of synaptic inputs over a longer timescale. The dependence in the original FFFB of FB on the slower integrated Act variable, which only comes on after the first spike (in order to compute the ISI), is consistent with these slower SST dynamics.

* **VIP:** sparse dendrite-targeting cells that synapse onto SST interneurons and the dendrites of pyramidal neurons, and coexpress vasoactive intestinal peptide (VIP). VIP interneurons are a subset of the larger 5HT3aR-expressing interneuron class. These can provide disinhibition of SST inhibition.  These are targeted by thalamic projections into layer 1 of cortex, and may be responsible for arousal and gating-like modulation from the thalamus.  We do not directly implement them in axon, but do indirectly capture their effects in the gating dynamics of the `pcore` model.

Remarkably, the parameters that work to enable successful error-driven learning in axon using the new FS-FFFB equations end up very closely reproducing the net inhibition produced by the original FFFB model, as shown in the following figure.

![FS-FFFB vs original FFFB](fig_fs_vs_orig_fffb_layer2.png?raw=true "FS-FFFB vs original FFFB")

**Figure 1:** Comparison of original FFFB inhibition (`OGi`) vs. new FS-FFFB inhibition (`SGi`) from the [inhib](https://github.com/emer/axon/tree/examples/inhib) example simulation, showing that the FS-FFFB parameters that enable successful learning produce nearly identical overall levels of inhibition compared to the original.  `Act.Avg` shows the time-averaged activity in the lower layer that feeds into the one shown, and resembles the slow SST response, while the jagged ups and downs are due to the fast PV component.

# PV is fast and highly chaotic

The dramatic swings in Gi levels as shown in the above figure are all due to the PV Fast component, which increases directly as a function of incoming FF and FB spikes, and decays with a fast time constant of 6 msec (default):

```Go
	FSi += (FFs + FB*FBs) - FSi/FSTau
```

where:
* `FSi` = fast PV contribution to inhibition, time-integrated.
* `FFs` = normalized sum of incoming FF spikes.
* `FBs` = normalized sum of incoming FB spikes from neurons in the same pool being inhibited.
* `FB` = weighting factor for FB spikes, which defaults to 1 but needs to be smaller for smaller networks (0.5) and larger for larger ones (e.g., 4).
* `FSTau`  = time constant for decaying FSi (6 msec default).


# SST is a slow time-average

The slow SST contribution slowly tracks overall spiking activity in the pool, roughly as the `Act.Avg` green line in the above figure, based on the following equations:

```Go
	SSi += (SSf*FBs - SSi) / SSiTau
	SSf += FBs*(1-SSf) - SSf / SSfTau
```    

where:
* `SSi` = slow SST contribution to inhibition, time-integrated.
* `SSf` = synaptic facilitation component for SS, which increases as a function of spiking activity as shown in the 2nd equation.
* `FBs` = normalized sum of incoming FB spikes from neurons in the same pool being inhibited.
* `SSiTau` = integration time constant for SSi, which is 50 msec by default (slow).
* `SSfTau` = time constant for SSf, which is 20 msec by default.


# Combined Gi

The combined overall inhibitory conductance `Gi` is based on a thresholded version of the FSi component plus a weighted contribution of the `SSi` level, which tends to be very weak due to the long time integral:

```Go
    Gi = |FSi > FS0|_+ + SS * SSi
```

where:
* `Gi` = overall inhibitory conductance.
* `FSi` = fast-spiking inhibition per above.
* `SSi` = slow-spiking inhibition per above.
* `FS0` = threshold for FSi, default .1 as in the original FFFB, below which it contributes 0.  This factor is important for filtering out small levels of incoming spikes and produces an observed nonlinearity in the Gi response.
* `SS` = multiplier for SSi, which is 30 by default: SSi is relatively weak so this needs to be a strong multiplier to get into the range of FSi.

# References

* Cardin, J. A. (2018). Inhibitory interneurons regulate temporal precision and correlations in cortical circuits. Trends in Neurosciences, 41(10), 689–700. https://doi.org/10.1016/j.tins.2018.07.015


