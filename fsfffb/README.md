# FS (Fast & Slow) FFFB Inhibition

FFFB is the feedforward (FF) and feedback (FB) inhibition mechanism, originally developed for the Leabra model.

FS fast and slow adds additional time dynamics that are already somewhat accidentally present in the Axon version of FFFB, based on an emerging consensus about the differences between three major classes of inhibitory interneurons, and their functional properties, e.g., **Cardin18**:

* **PV:** fast-spiking basket cells that target the cell bodies of excitatory neurons and coexpress the calcium-binding protein parvalbumin (PV). These are the "first responders", and are also rapidly depressing -- they provide quick control of activity, responding to FF new input and FB drive, allowing the first spiking pyramidal neurons to quickly shut off other competitors.

* **SST:** low-threshold spiking cells that target the distal dendrites of excitatory neurons and coexpress the peptide somatostatin (SST). These require repetitive, facilitating, afferent input to be activated, and may regulate the dendritic integration of synaptic inputs over a longer timescale. The current dependence of FB inhib on the slower integrated Act variable, which only comes on after the first spike (in order to compute the ISI), may reflect the SST dynamics.

* **VIP:** sparse dendrite-targeting cells that synapse onto SST interneurons and the dendrites of pyramidal neurons, and coexpress vasoactive intestinal peptide (VIP). VIP interneurons are a subset of the larger 5HT3aR-expressing interneuron class. These can provide disinhibition of SST inhibition.


