# gabab

Explores the GABA-B dynamics from Sanders et al (2013) and Thomson & Destexhe, 1999.

GABA-B is an inhibitory channel activated by the usual GABA inhibitory neurotransmitter, which is coupled to the GIRK *G-protein coupled inwardly rectifying potassium (K) channel*.  It is ubiquitous in the brain, and is likely essential for basic neural function (especially in spiking networks from a computational perspective).  The inward rectification is caused by a Mg+ ion block *from the inside* of the neuron, which means that these channels are most open when the neuron is hyperpolarized (inactive), and thus it serves to *keep inactive neurons inactive*.

Based on Fig 15 of TD99, using a double-exponential function with 60 msec time to peak, and roughly 200 msec time to return to baseline, along with sigmoidal function of spiking for peak conductance, and the Mg inverse rectification curve.

Double exponential can't quite fit the right time-course, but 45 rise and 50 decay gives pretty reasonable looking overall distribution with peak at 47, and about .2 left after 200 msec. 35 / 40 has peak around 37 and .1 left after 100.

* Sanders, H., Berends, M., Major, G., Goldman, M. S., & Lisman, J. E. (2013). NMDA and GABAB (KIR) Conductances: The “Perfect Couple” for Bistability. Journal of Neuroscience, 33(2), 424–429. https://doi.org/10.1523/JNEUROSCI.1854-12.2013

* Thomson AM, Destexhe A (1999) Dual intracellular recordings and computational models of slow inhibitory postsynaptic potentials in rat neocortical and hippocampal slices. Neuroscience 92:1193–1215.

