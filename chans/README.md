# Chans: Channels of all types

`chans.go` contains `Chans` which defines the basic `E, L, I, K` channels.

Other channels are described below.  There are various `_plot` directories with simple programs to plot the various functions and explore parameters.  The README files in these dirs contain more detailed info about the implementation of each channel type, etc.

This package will eventually contain the whole vocabulary of channels known to exist in neurons. Not all are likely to be incorporated into the `axon` neuron types -- we are gradually including channels as needed based on well-defined functional understanding of their role.

The implementation of several of these channels comes from standard biophysically detailed models such as [Migliore et al. (1999)](#references) and [Poirazi et al. (2003)](#references), which were used in the [Urakubo et al (2008)](#references) model.  See also [Brette et al, 2007](#references) and [NEST model directory](https://nest-simulator.readthedocs.io/en/stable/models/index.html) for documented examples, including: [AdEx](https://nest-simulator.readthedocs.io/en/stable/models/aeif_cond_exp.html), [Traub HH](https://nest-simulator.readthedocs.io/en/stable/models/hh_cond_exp_traub.html).  The [Brian Examples](https://brian2.readthedocs.io/en/stable/examples/index.html) contain full easy-to-read equations for various standard models, including [Brunel & Wang, 2001](https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html). Also see [Wikipedia: Biological neuron model](https://en.wikipedia.org/wiki/Biological_neuron_model) for a nice overview.

See [ModelDB Currents](https://senselab.med.yale.edu/NeuronDB/NeuronalCurrents) and [ModelDB Current Search](https://senselab.med.yale.edu/ModelDB/FindByCurrent) for a standardized list of currents included in biophysical models made in NEURON and related software.  By far the most numerous category are the K+ potassium channels, which can be gated by voltage, sodium, calcium, and other factors, and modulate the excitability of the neuron.

# Basic E, L, I channels

`chans.go` defines these basic maximal conductances and reversal potentials for the core channels in any biophysical conductance-based model:

* E = Excitatory synaptic conductance of Na+ through AMPA channels
* L = Leak from always-open K+ channels
* I = Inhibition from GABA-A Cl channels

`axon` does not include the classical Hodgkin-Huxley Na+ and K+ channels that drive the spike, and instead uses the AdEx mechanism with an exponential approximation and adapting dynamics from K+ channels at different time constants, as described below.

# Longer time-scale conductances from GABA-B / GIRK and NMDA receptors

This package implements two complementary conductances, GABA-B / GIRK and NMDA, that have relatively long time constants (on the order of 100-200 msec) and thus drive these longer-scale dynamics in neurons, beyond the basic AMPA and GABA-A channels with short ~10 msec time constants.  In addition to their longer time constants, these neurons have interesting mirror-image rectification properties (voltage dependencies), inward and outward, which are a critical part of their function [(Sanders et al, 2013)](#references).

## GABA-B / GIRK

GABA-B is an inhibitory channel activated by the usual GABA inhibitory neurotransmitter, which is coupled to the GIRK *G-protein coupled inwardly rectifying potassium (K) channel*.  It is ubiquitous in the brain, and is likely essential for basic neural function (especially in spiking networks from a computational perspective).  The inward rectification is caused by a Mg+ ion block *from the inside* of the neuron, which means that these channels are most open when the neuron is hyperpolarized (inactive), and thus it serves to *keep inactive neurons inactive*.

In standard Leabra rate-code neurons using FFFB inhibition, the continuous nature of the GABA-A type inhibition serves this function already, so these GABA-B channels have not been as important, but whenever a discrete spiking function has been used along with FFFB inhibition or direct interneuron inhibition, there is a strong tendency for every neuron to fire at some point, in a rolling fashion, because neurons that are initially inhibited during the first round of firing can just pop back up once that initial wave of associated GABA-A inhibition passes.  This is especially problematic for untrained networks where excitatory connections are not well differentiated, and neurons are receiving very similar levels of excitatory input.  In this case, learning does not have the ability to further differentiate the neurons, and does not work effectively.

## NMDA

NMDA is predominantly important for learning in most parts of the brain, but in frontal areas especially, a particular type of NMDA receptor drives sustained active maintenance, by driving sustained excitatory voltage-gated conductances (Goldman-Rakic, 1995; Lisman et al, 1998; Brunel & Wang, 2001).  The well-known external Mg+ ion block of NMDA channels, which is essential for its Hebbian character in learning, also has the beneficial effect of *keeping active neurons active* by virtue of causing an outward rectification that is the mirror-image of the GIRK inward rectification, with the complementary functional effect.  The Brunel & Wang (2001) model is widely used in many other active maintenance / working memory models.

Sanders et al, 2013 emphasize the "perfect couple" nature of the complementary bistability of GABA-B and NMDA, and show that it produces a more robust active maintenance dynamic in frontal cortical circuits.

We are using both NMDA and GABA-B to support robust active maintenance but also the ability to clear out previously-maintained representations, as a result of the GABA-B sustained inhibition.

## Implementation

### Spiking effects on Vm: Backpropagating Action Potentials

Spikes in the soma of a neuron also drive backpropagating action potentials back into the dendrites, which increases the overall membrane potential there above what is produced by the raw synaptic inputs.  In our rate code models, we simulate these effects by adding a proportion of the rate code activation value to the `Vm` value.  Also note that this `Vm` value is an equilibrium Vm that is not reset back to rest after each spike, so it is itself perhaps a bit elevated as a result.

### GABA-B

The GABA-B / GIRK dynamics are based on a combination of Sanders et al, 2013 and Thomson & Destexhe, 1999 (which was major source for Sanders et al, 2013).

T&D found a sigmoidal logistic function described the maximum conductance as a function of GABA spikes, which saturates after about 10 spikes, and a conductance time course that is fit by a 4th-order function similar to the Hodgkin Huxley equations, with a peak at about 60 msec and a total duration envelope of around 200 msec.  Sanders et al implemented these dynamics directly and describe the result as such:

> These equations give a latency of GABAB-activated KIR current of 􏰅10 ms, time to peak of 􏰅50 ms, and a decay time constant of 􏰅80 ms as described by Thomson and Destexhe (1999). A slow time constant for GABA removal was chosen because much of the GABAB response is mediated by extrasynaptic receptors (Kohl and Paulsen, 2010) and because the time constant for GABA decline at extracellular sites appears to be much slower than at synaptic sites.

We take a simpler approach which is to just use a bi-exponential function to simulate the time course, using 45 and 50 msec time constants.

See `gabab_plot` subdirectory for program that generates these plots:

![GABA-B V-G](gabab_plot/fig_sanders_et_al_13_kir_v_g.png)

**V-G plot for GABA-B / GIRK** from Sanders et al, 2013.

![GABA-B S-G](gabab_plot/fig_thomson_destexhe99_s_g_sigmoid.png)

**Spike-G sigmoid plot for GABA-B / GIRK** from Thomson & Destexhe (1999)

![GABA-B Time](gabab_plot/fig_thomson_destexhe99_g_time.png)

**Bi-exponential time course with 45msec rise and 50msec decay** fitting to results from Thomson & Destexhe (1999)

### NMDA

The NMDA code uses the exponential function from Brunel & Wang (2001), which is very similar in shape to the Sanders et al, 2013 curve, to compute the voltage dependency of the current:

```Go
	vbio := v*100 - 100
	return 1 / (1 + 0.28*math32.Exp(-0.062*vbio))
```   

where v is the normalized Vm value from the Leabra rate code equations -- `vbio` converts it into biological units.

See the `nmda_plot` directory for a program that plots this function, which looks like this (in terms of net conductance also taking into account the driving potential, which adds an `(E - v)` term in the numerator).

![GABA-B V-G](nmda_plot/fig_brunelwang01.png)

**V-G plot for NMDA** from Brunel & Wang (2001)

# VGCC: Voltage-gated Calcium Channels

There are a large number of VGCC's (Dolphin, 2018; Cain & Snutch, 2012) denoted by letters in descending order of the voltage threshold for activation: L, PQ, N, R, T, which have corresponding Ca_v names: Ca_v1.1, 1.2, 1.3. 1.4 are all L type, 2.1, 2.2, 2.3 are PQ, N, and R, respectively, and T type (low threshold) comprise 3.1, 3.2, and 3.3.  Each channel is characterized by the voltage dependency and inactivation functions.  The table here summarizes the different types:

| Letter | Ca_v    | V Threshold  | Inactivation | Location | Function              |
| ------ | ------- | ------------ | ------------ | -------- | --------------------- |
|  L     | 1.1-1.4 | high (-40mV) | fast         | Cortex + | closely tracks spikes |
|  PQ    | 2.1     | high         | ?            | Cerebellum (Purk, Gran) | ?      |
|  N     | 2.2     | high         | ?            | everywhere? | ?                  |
|  R     | 2.3     | med          | ?            | Cerebellum Gran | ?              | 
|  T     | 3.1-.3  | low          | ?            | 5IB, subcortical  | low-freq osc |


* L type is the classic "VGCC" in dendritic spines in pyramidal cells, implemented in `VGCCParams` in `vgcc.go`.  See [vgcc_plot](https://github.com/emer/axon/tree/master/chans/vgcc_plot) for more info.  It is now incorporated into the base axon Neuron type, where it plays a critical role in driving Ca from backpropagated spikes, to mix in with NMDA-Ca for the Ge-based "trace" learning rule.

* PQ and R are specific to cerebellum.

* The T type is the most important for low frequency oscillations, and is absent in pyramidal neurons outside of the 5IB layer 5 neurons, which are the primary bursting type.  It is most important for subcortical neurons, such as in TRN.  See [Destexhe et al, 1998 model in BRIAN](https://brian2.readthedocs.io/en/stable/examples/frompapers.Destexhe_et_al_1998.html) for an implementation.

# AK: A-type voltage-gated potassium channel

AK (in `ak.go`) is voltage gated with maximal activation around -37 mV.  It is particularly important for counteracting the excitatory effects of VGCC L-type channels which can otherwise drive runaway excitatory currents (i.e., think of it as an "emergency brake" and is needed for this reason whenever adding VGCC to a model), and is co-localized with them in pyramidal cell dendrites.  It is included in the basic `axon` Neuron.

It has two state variables, M (v-gated opening) and H (v-gated closing), which integrate with fast and slow time constants, respectively. H relatively quickly hits an asymptotic level of inactivation for sustained activity patterns. See AKsParams for a much simpler version that works fine when full AP-like spikes are not simulated, as in our standard axon models.

# K+ channels that drive adaptation: KNa, CaK, sAHP

There are multiple types of K+ channels that contribute to *adaptation* -- slowing of the rate of spiking over time for a constant excitatory input [(Dwivedi & Bhalla, 2021)](#references).  This is a critical property of neurons, to make them responsive to changes --- constants are filtered out.  Somehow, the computational modeling community, and perhaps the broader neuroscience world as well, has focused on calcium-gated K channels, and not the sodium-gated ones.  However, the Na+ gated ones are much simpler to implement, and have been clearly demonstrated to underlie a significant proportion of the observed adaptation dynamic, so they are the primary form of adaptation implemented in the axon base neuron.

Dwivedi & Bhalla (2021) define three broad timescales for AHP (afterhyperpolarization) currents:

* fast = fAHP: activated within 1-5 ms
* medium = mAHP: activated within 10-300 ms
* slow = sAHP: activated between .5 - multiple seconds

## KNa

See [Bhattacharjee & Kaczmarek (2005)](#references) and [Kaczmarek (2013)](#references) for reviews of the literature on various time-scales of KNa channels.  The dynamics are simple: the conductance rises with every spike-driven influx of Na, and it decays with a time-constant.

```Go
	if spike {
		gKNa += Rise * (Max - gKNa)
	} else {
		gKNa -= gKNa / Tau
	}
```

The different types are:

| Channel Type     | Tau (ms) | Rise  |  Max  |
|------------------|----------|-------|-------|
| Fast (pseudo-M-type) | 50       | 0.05  | 0.1   |
| Medium (Slick)   | 200      | 0.02  | 0.1   |
| Slow (Slack)     | 1000     | 0.001 | 1.0   |


## Calcium-gated Potassium Channels: SK and BK

There are two major types of Ca-gated K channels: "small" K (SK, SKCa) and "big" K (BK, BKCa).  These channels are more complicated to simulate relative to KNa, because they depend on Ca dynamics which are much more complicated than just tracking spiking.  

The SK channel (in `scka.go`) is based on the implementation by [Fujita et al (2012)](#references), in turn based on [Gunay et al (2008)](#references), using a simple Hill equation which takes the form of $X / (X + C_{50})$ where $C_{50}$ is the concentration at which the value is at 50%.  A different logistic exponential equation was given in [Gillies & Willshaw, 2006](#references), in a model of the subthalamic nucleus (STN) cell.  Dwivedi & Bhalla (2021) give an activation time constant of 5-15 ms and decay constant of around 30 ms for the SKCa.

The SKCa channel is used in the basal ganglia `pcore` STN neuron, using the more slowly integrated `CaD` calcium signal (which also drives Ca-based learning).  It plays a critical role in pausing neural activity after a brief bit of activity triggered by a new PFC input representation.

## M-type (KCNQ, Kv7): AcH modulated

The M-type (muscarinic, mAHP) channel is voltage sensitive, but opens fully at low voltages (-60 mV), and can be closed by acetylcholine (AcH) and many other things [(Greene & Hoshi, 2017)](#references).  There are many subtypes due to different constituents.  In general it takes a while to activate, around 100 msec or more, and deactivates on that same timescale.  Thus, it is an important contributor to the mAHP that can be modulated by various neuromodulators.  https://neuronaldynamics.epfl.ch/online/Ch2.S3.html describe it as having a higher activation potential (-40mV) and faster decay rate (50 ms) and thus being similar to KNa in being only activated by spikes and decaying at the faster rate, as shown above.  Thus, we have subsumed this "basic" version of M-type in the KNa dynamics.

The original characterization of the M-type current in most models derives from [Gutfreund et al (1995)](#references), as implemented in NEURON by [Mainen & Sejnowski (1996)](#references) -- see https://icg.neurotheory.ox.ac.uk/viewer/?family=1&channel=1706 for the geneology of this code!

There is a voltage gating factor *n* (often labeled *m* for other channels) which has an asymptotic drive value as an exponential logistic function of Vm, and a variable tau that is also a function of Vm.


*
https://senselab.med.yale.edu/ModelDB/ShowModel?model=181967&file=/CutsuridisPoirazi2015/km.mod#tabs-2
* https://senselab.med.yale.edu/ModelDB/ShowModel?model=231185&file=/orientation_preference/mod.files/km.mod#tabs-2
* https://senselab.med.yale.edu/ModelDB/ShowModel?model=266901&file=/TomkoEtAl2021/Mods/kmb.mod#tabs-2

## sAHP: slow afterhyperpolarization

[Larsson (2013)](#references) provides a nice narrative about the difficulty in tracking down the origin of a very slow, long-lasting sAHP current that has been observed in hippocampal and other neurons.  It appears to be yet another modulator on the M-type channels, that is driven by calcium sensor pathways that have longer time constants.  There is more work to be done here, but we can safely use a mechanism that takes a long time to build up before activating the K+ channels, and then takes a long time to decay as well.  This will provide appropriate dynamics for the CT neurons.

# HCN channels: I_h

HCN = hyperpolarization-activated cyclic nucleotide-gated nonselective cation channels.  Differentiate some layer V PFC neurons: DembrowChitwoodJohnston10.

Magee98: Overall, Ih acts to dampen dendritic excitability, but its largest impact is on the subthreshold range of membrane potentials where the integration of inhibitory and excitatory synaptic inputs takes place.

* The rate of current activation was voltage-dependent such that the activation time constant decreased with increasing hyperpolarization (from 50 msec at -75 mV to 16 msec at -125 mV.

* Tail current decay time constants decreased with increasing depolarization (30 msec at -70 mV to 7 msec at -30 mV)

* From the mean reversal potentials a Na􏰁/K􏰁 permeability ratio of 0.35 could be deter- mined with the Goldman–Hodgkin–Katz equation

* HarnettMageeWilliams15: At distal apical dendritic trunk and tuft sites, we find that HCN channels have predominately inhibitory actions, controlling the initiation and propagation of dendritic spikes. In contrast, at proximal apical dendritic and somatic sites, HCN channels exert excitatory influences by decreasing the threshold excitatory input required to evoke action potential output. 


# References

* Bhattacharjee, A., & Kaczmarek, L. K. (2005). For K+ channels, Na+ is the new Ca2+. Trends in Neurosciences, 28(8), 422–428. https://doi.org/10.1016/j.tins.2005.06.003

* Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., Diesmann, M., Morrison, A., Goodman, P. H., Harris, F. C., & Others. (2007). Simulation of networks of spiking neurons: A review of tools and strategies. Journal of Computational Neuroscience, 23(3), 349–398. http://www.ncbi.nlm.nih.gov/pubmed/17629781

* Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1994). Synthesis of models for excitable membranes, synaptic transmission and neuromodulation using a common kinetic formalism. Journal of Computational Neuroscience, 1(3), 195–230. https://doi.org/10.1007/BF00961734

* Dwivedi, D., & Bhalla, U. S. (2021). Physiology and Therapeutic Potential of SK, H, and M Medium AfterHyperPolarization Ion Channels. Frontiers in Molecular Neuroscience, 14. https://www.frontiersin.org/articles/10.3389/fnmol.2021.658435

* Fujita, T., Fukai, T., & Kitano, K. (2012). Influences of membrane properties on phase response curve and synchronization stability in a model globus pallidus neuron. Journal of Computational Neuroscience, 32(3), 539–553. https://doi.org/10.1007/s10827-011-0368-2

* Gillies, A., & Willshaw, D. (2006). Membrane Channel Interactions Underlying Rat Subthalamic Projection Neuron Rhythmic and Bursting Activity. Journal of Neurophysiology, 95(4), 2352–2365. https://doi.org/10.1152/jn.00525.2005

* Greene, D. L., & Hoshi, N. (2017). Modulation of Kv7 channels and excitability in the brain. Cellular and Molecular Life Sciences : CMLS, 74(3), 495–508. https://doi.org/10.1007/s00018-016-2359-y

* Günay, C., Edgerton, J. R., & Jaeger, D. (2008). Channel Density Distributions Explain Spiking Variability in the Globus Pallidus: A Combined Physiology and Computer Simulation Database Approach. Journal of Neuroscience, 28(30), 7476–7491. https://doi.org/10.1523/JNEUROSCI.4198-07.2008

* Gutfreund, Y., Yarom, Y., & Segev, I. (1995). Subthreshold oscillations and resonant frequency in guinea-pig cortical neurons: Physiology and modelling. The Journal of Physiology, 483(3), 621–640. https://doi.org/10.1113/jphysiol.1995.sp020611

* Kaczmarek, L. K. (2013). Slack, Slick, and Sodium-Activated Potassium Channels. ISRN Neuroscience, 2013. https://doi.org/10.1155/2013/354262

* Larsson, H. P. (2013). What Determines the Kinetics of the Slow Afterhyperpolarization (sAHP) in Neurons? Biophysical Journal, 104(2), 281–283. https://doi.org/10.1016/j.bpj.2012.11.3832

* Mainen, Z. F., & Sejnowski, T. J. (1996). Influence of dendritic structure on firing pattern in model neocortical neurons. Nature, 382, 363. http://www.ncbi.nlm.nih.gov/pubmed/8684467

* Migliore, M., Hoffman, D. A., Magee, J. C., & Johnston, D. (1999). Role of an A-Type K+ Conductance in the Back-Propagation of Action Potentials in the Dendrites of Hippocampal Pyramidal Neurons. *Journal of Computational Neuroscience, 7(1),* 5–15. https://doi.org/10.1023/A:1008906225285

* Poirazi, P., Brannon, T., & Mel, B. W. (2003). Arithmetic of Subthreshold Synaptic Summation in a Model CA1 Pyramidal Cell. *Neuron, 37(6),* 977–987. https://doi.org/10.1016/S0896-6273(03)00148-X

* Sanders, H., Berends, M., Major, G., Goldman, M. S., & Lisman, J. E. (2013). NMDA and GABAB (KIR) Conductances: The “Perfect Couple” for Bistability. Journal of Neuroscience, 33(2), 424–429. https://doi.org/10.1523/JNEUROSCI.1854-12.2013

* Urakubo, H., Honda, M., Froemke, R. C., & Kuroda, S. (2008). Requirement of an allosteric kinetics of NMDA receptors for spike timing-dependent plasticity. *The Journal of Neuroscience, 28(13),* 3310–3323. http://www.ncbi.nlm.nih.gov/pubmed/18367598


