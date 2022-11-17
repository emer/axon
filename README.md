# Axon in Go emergent

[![Go Report Card](https://goreportcard.com/badge/github.com/emer/axon)](https://goreportcard.com/report/github.com/emer/axon)
[![GoDoc](https://godoc.org/github.com/emer/axon?status.svg)](https://godoc.org/github.com/emer/axon)
[![codecov](https://codecov.io/gh/emer/axon/branch/master/graph/badge.svg)](https://codecov.io/gh/codecov/emer/axon)
[![Travis](https://travis-ci.com/emer/axon.svg?branch=master)](https://travis-ci.com/emer/axon)

This is the Go implementation of the Axon algorithm for spiking, biologically-based models of cognition, based on the [Go emergent](https://github.com/emer/emergent) framework (with optional Python interface), and the  [leabra](https://github.com/emer/leabra) framework for rate-code models.

Axon is the spiking version of [Leabra](https://github.com/emer/leabra), with several advances.  As an acronym, it stands for *Adaptive eXcitation Of Noise*, reflecting the ability to learn using the power of error-backpropagation in the context of noisy spiking activation.  The spiking function of the axon is what was previously missing from Leabra.  See [Axor](https://github.com/emer/axor) for a rate-code version of axon.

See [Wiki Install](https://github.com/emer/emergent/wiki/Install) for installation instructions, and the [Wiki Rationale](https://github.com/emer/emergent/wiki/Rationale) and [History](https://github.com/emer/emergent/wiki/History) pages for a more detailed rationale for the new version of emergent, and a history of emergent (and its predecessors).

See the [ra25 example](https://github.com/emer/axon/blob/master/examples/ra25/README.md) for a complete working example (intended to be a good starting point for creating your own models), and any of the 26 models in the [Comp Cog Neuro sims](https://github.com/CompCogNeuro/axon) repository which also provide good starting points.  See the [etable wiki](https://github.com/emer/etable/wiki) for docs and example code for the widely-used etable data table structure, and the `family_trees` example in the CCN textbook sims which has good examples of many standard network representation analysis techniques (PCA, cluster plots, RSA).

See [python README](https://github.com/emer/axon/blob/master/python/README.md) and [Python Wiki](https://github.com/emer/emergent/wiki/Python) for info on using Python to run models.

# Current Status / News

* Feb 2022: Major development process to explore Kinase-based learning rules.

* May-July 2021: Initial implementation and significant experimentation.  The fully spiking-based Axon framework is capable of learning to categorize rendered 3D object images based on the deep, bidirectionally-connected LVis model originally reported in O'Reilly et al. (2013).  Given the noisy, complex nature of the spiking dynamics, getting this level of functionality out of a large, deep network architecture was not easy, and it drove a number of critical additional mechanisms that are necessary for this model to work.

# Converting Existing Leabra Models

In the name of progress and making things sensible going forward, various names have been changed to better fit the way that Axon works.  This will require some renaming for existing Leabra models.

## leabra -> axon

The first step is to do a global replace of `leabra` -> `axon` and `Leabra` -> `Axon`

## Alpha Cycle -> Theta Cycle

The notion of the AlphaCycle as a basic unit of processing in Leabra does not quite work in Axon: it typically takes 200 msec (cycles) for a full learning cycle, corresponding to a ThetaCycle.  This was actually the original idea for predictive learning, where the full processing of a given input state was supposed to take two alpha cycles = 1 theta cycle, which corresponds to the strongly-peaked duration of a single fixation in humans.  In Axon, the processing of time has been generalized, so it isn't tied so specifically to a particular cycle.

The concept of a `Quarter` is no longer as sensible, so it has been removed.  In the `TimeScales` enum, it is replaced with `GammaCycle`. There are now two more generic `SpkSt1` and `SpkSt2` activation state variables that can be captured at any point by calling methods of the corresponding name.  `ActPrv` is the new name for `ActQ0` and it is captured in the `NewState` method.

* `NewState` methods are called when starting to process a new "state", which corresponds to a new fixation, a new sniff, and generally a new "trial" in standard neural network terminology.
    + `AlphaCycleInit` -> `NewState`
    + `Time.AlphaCycStart` -> `NewState`

* The standard `AlphaCyc` method in your simulation should be replaced with something like `ThetaCyc` from the `ra25` example, which contains all the new timing logic.  In general, see the `ra25` example for all the relevant updates.

## Params

* `Prjn.WtInit` -> `Prjn.SWt.Init` -- the spine / slow / structural wt (SWt) params control weight initialization now.

* `Prjn.WtScale` -> `Prjn.PrjnScale` -- overall projection-level scaling has been renamed `PrjnScale` which better reflects its nature.

* `Prjn.Learn.Lrate` -> `Prjn.Learn.Lrate.Base` -- added separate params for `.Mod` and `.Sched`

* Removed: all the BCM-related `AvgL` mechanisms -- only using the core error-driven learning components.

## Stats

SSE not used anymore -- relevant call is now `PctUnitErr` which returns proportion (0-1) of units that have an error (activity on wrong side of target).  This is not normalized for expected activity, so in general scales with that and will tend to be rather low.

## Deep

Must call `ss.Net.CTCtxt()` explicitly to update the CT context reps -- generally call after plus phase, but can also experiment with other timings.

# Design

* `axon` sub-package provides a clean, well-organized implementation of core Axon algorithms and Network structures. More specialized modifications such as `DeepAxon` or `PBWM` or `PVLV` are all (going to be) implemented as additional specialized code that builds on / replaces elements of the basic version.  The goal is to make all of the code simpler, more transparent, and more easily modified by end users.  You should not have to dig through deep chains of C++ inheritance to find out what is going on.  Nevertheless, the basic tradeoffs of code re-use dictate that not everything should be in-line in one massive blob of code, so there is still some inevitable tracking down of function calls etc.  The algorithm overview below should be helpful in finding everything.

* `ActParams` (in [act.go](https://github.com/emer/axon/blob/master/axon/act.go)), `InhibParams` (in [inhib.go](https://github.com/emer/axon/blob/master/axon/inhib.go)), and `LearnNeurParams` / `LearnSynParams` (in [learn.go](https://github.com/emer/axon/blob/master/axon/learn.go)) provide the core parameters and functions used, including the X-over-X-plus-1 activation function, FFFB inhibition, and the XCal BCM-like learning rule, etc.  This function-based organization should be clearer than the purely structural organization used in C++ emergent.

* There are 3 main levels of structure: `Network`, `Layer` and `Prjn` (projection).  The network calls methods on its Layers, and Layers iterate over both `Neuron` data structures (which have only a minimal set of methods) and the `Prjn`s, to implement the relevant computations.  The `Prjn` fully manages everything about a projection of connectivity between two layers, including the full list of `Syanpse` elements in the connection.  There is no "ConGroup" or "ConState" level as was used in C++, which greatly simplifies many things.  The Layer also has a set of `Pool` elements, one for each level at which inhibition is computed (there is always one for the Layer, and then optionally one for each Sub-Pool of units (*Pool* is the new simpler term for "Unit Group" from C++ emergent).

* The `NetworkBase` and `LayerBase` structs manage all the core structural aspects of things (data structures etc), and then the algorithm-specific versions (e.g., `axon.Network`) use Go's anonymous embedding (akin to inheritance in C++) to transparently get all that functionality, while then directly implementing the algorithm code.  Almost every step of computation has an associated method in `axon.Layer`, so look first in [layer.go](https://github.com/emer/axon/blob/master/axon/layer.go) to see how something is implemented.

* Each structural element directly has all the parameters controlling its behavior -- e.g., the `Layer` contains an `ActParams` field (named `Act`), etc, instead of using a separate `Spec` structure as in C++ emergent.  The Spec-like ability to share parameter settings across multiple layers etc is instead achieved through a **styling**-based paradigm -- you apply parameter "styles" to relevant layers instead of assigning different specs to them.  This paradigm should be less confusing and less likely to result in accidental or poorly-understood parameter applications.  We adopt the CSS (cascading-style-sheets) standard where parameters can be specifed in terms of the Name of an object (e.g., `#Hidden`), the *Class* of an object (e.g., `.TopDown` -- where the class name TopDown is manually assigned to relevant elements), and the *Type* of an object (e.g., `Layer` applies to all layers).  Multiple space-separated classes can be assigned to any given element, enabling a powerful combinatorial styling strategy to be used.

* Go uses `interfaces` to represent abstract collections of functionality (i.e., sets of methods).  The `emer` package provides a set of interfaces for each structural level (e.g., `emer.Layer` etc) -- any given specific layer must implement all of these methods, and the structural containers (e.g., the list of layers in a network) are lists of these interfaces.  An interface is implicitly a *pointer* to an actual concrete object that implements the interface.  Thus, we typically need to convert this interface into the pointer to the actual concrete type, as in:

```Go
func (nt *Network) InitActs() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(*Layer).InitActs() // ly is the emer.Layer interface -- (*Layer) converts to axon.Layer
	}
}
```

* The emer interfaces are designed to support generic access to network state, e.g., for the 3D network viewer, but specifically avoid anything algorithmic.  Thus, they should allow viewing of any kind of network, including PyTorch backprop nets.

* There is also a `axon.AxonLayer` and `axon.AxonPrjn` interface, defined in [axon.go](https://github.com/emer/axon/blob/master/axon/axon.go), which provides a virtual interface for the Axon-specific algorithm functions at the basic level.  These interfaces are used in the base axon code, so that any more specialized version that embeds the basic axon types will be called instead.  See `deep` sub-package for implemented example that does DeepAxon on top of the basic `axon` foundation.

* Layers have a `Shape` property, using the `etensor.Shape` type, which specifies their n-dimensional (tensor) shape.  Standard layers are expected to use a 2D Y*X shape (note: dimension order is now outer-to-inner or *RowMajor* now), and a 4D shape then enables `Pools` ("unit groups") as hypercolumn-like structures within a layer that can have their own local level of inihbition, and are also used extensively for organizing patterns of connectivity.

# Naming Conventions

There are several changes from the original C++ emergent implementation for how things are named now:
* `Pool <- Unit_Group` -- A group of Neurons that share pooled inhibition.  Can be entire layer and / or sub-pools within a layer.
* `AlphaCyc <- Trial` -- We are now distinguishing more clearly between network-level timing units (e.g., the 100 msec alpha cycle over which learning operates within posterior cortex) and environmental or experimental timing units, e.g., the `Trial` etc. Please see the [TimeScales](https://godoc.org/github.com/emer/axon/axon#TimeScales) type for an attempt to standardize the different units of time along these different dimensions.  The `examples/ra25` example uses trials and epochs for controlling the "environment" (such as it is), while the algorithm-specific code refers to AlphaCyc, Quarter, and Cycle, which are the only time scales that are specifically coded within the algorithm -- everything else is up to the specific model code.

# The Axon Algorithm

Axon is the spiking version of [leabra](https://github.com/emer/leabra), which uses rate-code neurons.  The newer mechanisms in Axon will be incorporated into `Axor` -- the rate-code version of axon: [axor](https://github.com/emer/axor).

Like Leabra, Axon is intended to capture a middle ground between neuroscience, computation, and cognition, providing a computationally effective framework based directly on the biology, to understand how cognitive function emerges from the brain.  See [Computational Cognitive Neuroscience](https://CompCogNeuro.org) for a full textbook on the principles and many implemented models.

## Core Leabra-based Features

* **Spiking AdEx Neurons:** Axon uses the full conductance-based *AdEx* (adapting exponential) discrete spiking model of Gerstner and colleagues [Scholarpedia article on AdEx](https://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model), using normalized units as shown here: [google sheet](https://docs.google.com/spreadsheets/d/1jn-NcXY4-y3pOw6inFOgPYlaQodrGIjcsAWkiD9f1FQ/edit?usp=sharing).  Parameterizable synaptic communication delays are also supported, with biologically-based defaults.  Adaptation is implemented using three time scales of sodium-gated potassium channels: `KNa` (Kaczmarek, 2013), which are critical for learning (in the rate code Leabra model, they were problematic for learning).  Leabra implemented a close rate-code approximation to this form of spiking.

* **Auto-normalized, relatively scaled Excitatory Conductances:** As in Leabra, the excitatory synaptic input conductance (`Ge` in the code, known as *net input* in artificial neural networks) is computed as an average, not a sum, over connections, based on normalized weight values, which are subject to scaling on a projection level to alter relative contributions.  Automatic scaling is performed to compensate for differences in expected activity level in the different projections.  See section on [Input Scaling](#input-scaling) for details.

* **FFFB computed inhibition:** Also as in Leabra, inhibition is computed using a feed-forward (FF) and feed-back (FB) inhibition function (*FFFB*) that closely approximates the behavior of inhibitory interneurons in the neocortex.  FF is based on a multiplicative factor applied to the average excitatory conductance coming into a layer, and FB is based on a multiplicative factor applied to the average activation within the layer.  These simple linear functions do an excellent job of controlling the overall activation levels in bidirectionally connected networks, producing behavior very similar to the more abstract computational implementation of kWTA dynamics implemented in previous versions.  See the `examples/inhib` model (from the CCN textbook originally) for an exploration of the basic excitatory and inhibitory dynamics in these models, comparing interneurons with FFFB.

TODO: new Kinase mechanism.

* **Temporal-difference activation states drive approximate backpropagation learning:** Axon also uses the same learning equation as in Leabra, derived from a highly detailed model of spike timing dependent plasticity (STDP) by Urakubo, Honda, Froemke, et al (2008), that produces a combination of Hebbian associative and error-driven learning.  For historical reasons, we call this the *XCAL* equation (*eXtended Contrastive Attractor Learning*), and it is functionally very similar to the *BCM* learning rule developed by Bienenstock, Cooper, and Munro (1982).  The essential learning dynamic involves a Hebbian-like co-product of sending neuron activation times receiving neuron activation, which biologically reflects the amount of calcium entering through NMDA channels, and this co-product is then compared against a floating threshold value.

    Unlike in Leabra, we do not need the BCM Hebbian mode of learning in the Axon spiking framework, because the considerable variability in activity states driving learning results in significant sampling of the entire distribution around any given point in the network activation space.  Thus, the synaptic weights from purely error-driven learning resemble those from Hebbian learning in the rate-code model.  To produce error-driven learning, the floating threshold is based on a faster running average of activation co-products (`AvgM`), which reflects an expectation or prediction, against which the instantaneous, later outcome is compared.

    The GeneRec algorithm (O'Reilly, 1996) shows how this simple contrastive-hebbian-learning (CHL) error-driven learning term, in the context of bidirectional excitatory connectivity, approximates the error backpropagation algorithm.  In short, the top-down excitatory connections in the outcome or *plus* phase convey a target signal (filtered appropriately through the weights reflecting a given neuron's contribution to the output activity), which when contrasted against the expectation or *minus* phase, yields the error gradient for that neuron.  More recent work shows how predictive error-driven learning could be supported by specific features of the thalamocortical circuitry between the neocortex and higher-order thalamic nuclei such as the pulvinar and MD nucleus (O'Reilly et al, 2021).

* **Soft weight bounding and contrast enhancement:**  Weights are subject to a contrast enhancement function, which compensates for the soft (exponential) weight bounding that keeps weights within the normalized 0-1 range.  Contrast enhancement is important for enhancing the selectivity of learning, and generally results in faster learning with better overall results.  Learning operates on the underlying internal linear weight value, `LWt`, while the effective overall `Wt` driving excitatory conductances is the contrast-enhanced version.  Biologically, we associate the underlying linear weight value with internal synaptic factors such as actin scaffolding, CaMKII phosphorlation level, etc, while the contrast enhancement operates at the level of AMPA receptor expression.

## New Axon Mechanisms

The following mechanisms were essential for getting the spiking-based model to function effectively.  Several of these mechanisms were critical for combating the positive feedback loops that plague networks with bidirectional excitatory connections, producing "hog units" that were a major problem in Leabra.

* **Longer-acting, bistable NMDA and GABA-B currents:** An essential step for enabling spiking neurons to form suitably stable, selective representations for learning was the inclusion of both NMDA and GABA-B channels, which are voltage dependent in a complementary manner as captured in the [Sanders et al, 2013](#References) model (which provided the basis for the implementation here).  These channels have long time constants and the voltage dependence causes them to promote a bistable activation state, with a smaller subset of neurons that have extra excitatory drive from the NMDA and avoid extra inhibition from GABA-B, while a majority of neurons have the opposite profile: extra inhibition from GABA-B and no additional excitation from NMDA.  With stronger conductance levels, these channels can produce robust active maintenance dynamics characteristic of layer 3 in the prefrontal cortex (PFC), but for posterior cortex, we use lower values that produce a weaker, but still essential, form of bistability.  Without these channels, neurons all just "take turns" firing at different points in time, and there is no sense in which a small subset are engaged to represent a specific input pattern -- that had been a blocking failure in all prior attempts to use spiking in Leabra models.

* **Zero-sum weight changes:** Based on the cogent analysis of Schraudolph (1998), and application to ResNets (He et al, 2016), it is important to keep neurons in their sensitive dynamic range, and prevent any *DC* or *main effect* component of the learning signal to blot out meaningful *residual* patterns of sensitivity.  This can be done by various forms of "centering", i.e., subtracting the mean.  In Axon, we enforce zero-sum synaptic weight changes, such that the mean synaptic weight change across the synapses in a given neuron's receiving projection is subtracted from all such computed weight changes.  The `Prjn.Learn.XCal.SubMean` parameter controls how much of the mean `DWt` value is subtracted: 1 = full zero sum, which works the best.  There is significant biological data in support of such a zero-sum nature to synaptic plasticity (cites).  Functionally, it is a key weapon against the positive-feedback loop "hog unit" problems.

* **Target intrinsic activity levels:** Schraudolph notes that a *bias weight* is ideal for absorbing all of the DC or main effect signal for a neuron.  Biologically, Gina Turrigiano has discovered extensive evidence for *synaptic scaling* mechanisms that dynamically adjust for changes in overall excitability or mean activity of a neuron across longer time scales (e.g., as a result of monocular depravation experiments).  Furthermore, she found that there is a range of different target activity levels across neurons, and that each neuron has a tendency to maintain its own target level.   In Axon, we implemented a simple form of this mechanism by: 1. giving each neuron its own target average activity level (`TrgAvg`); 2. monitoring its activity over longer time scales (`ActAvg` and `AvgPct` as a normalized value); and 3. scaling the overall synaptic weights (`LWt`) to maintain this level by adapting in a soft-bounded way as a function of `TrgAvg - AvgPct`.

    This is similar to a major function performed by the BCM learning algorithm in the Leabra framework, but we found that by moving this mechanism into a longer time-scale outer-loop mechanism (consistent with Turigiano's data), it worked much more effectively.  By contrast, the BCM learning ended up interfering with the error-driven learning signal, and required relatively quick time-constants to adapt responsively as a neuron's activity started to change.  To allow this target activity signal to adapt in response to the DC component of a neuron's error signal, the local plus - minus phase difference (`ActDiff`) drives (in a zero-sum manner) changes in `TrgAvg`, subject also to Min / Max bounds.  These params are on the `Layer` in `Learn.SynScale`.

* **SWt: structural, slowly-adapting weights:** Biologically, the overall synaptic efficacy is determined by two major factors: the number of excitatory AMPA receptors (which adapt rapidly in learning), and size, number, and shape of the dendritic spines (which adapt more slowly, requiring protein synthesis, typically during sleep).  It was useful to introduce a slowly-adapting structural component to the weight, `SWt`, to complement the rapidly adapting `LWt` value (reflecting AMPA receptor expression), which has a multiplicative relationship with `LWt` in determining the effective `Wt` value.  The `SWt` value is typically initialized with all of the initial random variance across synapses, and it learns on the `SlowInterval` (100 iterations of faster learning) on the zero-sum accumulated `DWt` changes since the last such update, with a slow learning rate (e.g., 0.001) on top of the existing faster learning rate applied to each accumulated `DWt` value.  This slowly-changing SWt value is critical for preserving a background level of synaptic variability, even as faster weights change.  This synaptic variability is essential for prevent degenerate hog-unit representations from taking over, while learning slowly accumulates effective new weight configurations.  In addition, a small additional element of random variability, colorfully named `DreamVar`, can be injected into the active `LWt` values after each `SWt` update, which further preserves this critical resource of variability driving exploration of the space during learning.

* **Dopamine-like neuromodulation of learning rate:** The plus phase pattern can be fairly different from minus phase, even when there is no error.  This erodes the signal-to-noise-ratio of learning.  Dopamine-like neuromodulatory control over the learning rate can compensate, by ramping up learning when an overall behavioral error is present, while ramping it down when it is not.  The mechanism is simple: there is a baseline learning rate, on top of which the error-modulated portion is added.

* **Shortcut connections.**  These are ubiquitous in the brain, and in deep spiking networks, they are essential for moderating the tendency of the network to have wildly oscillatory waves of activity and silence in response to new inputs.  Weaker, random short-cuts allow a kind warning signal for impending inputs, diffusely ramping up excitatory and inhibitory activity.  It is possible that the *Claustrum* and other nonspecific thalamic areas provide some of this effect in the brain, by driving diffuse broad activation in the awake state, to overcome excessive oscillatory entrainment.

## Minor but Important

There are a number of other more minor but still quite important details that make the models work.

* *Current clamping the inputs.* Initially, inputs were driven with a uniform Poisson spike train based on their external clamped activity level.  However, this prevents the emergence of more natural timing dynamics including the critical "time to first spike" for new inputs, which is proportional to the excitatory drive, and gives an important first-wave signal through the network.  Thus, especially for complex distributed activity patterns, using the `GeClamp` where the input directly drives Ge in the input neurons, is best.

* *Direct adaptation of projection scaling.*  This is very technical, but important, to maintain neural signaling in the sensitive, functional range, despite major changes taking place over the course of learning.  In Leabra, each projection was scaled by the average activity of the sending layer, to achieve a uniform contribution of each pathway (subject to further parameterized re-weighting), and these scalings were updated continuously *as a function of running-average activity in the layers*.  However, the mapping between average activity and scaling is not perfect, and it works much better to just directly adapt the scaling factors themselves.  There is a target `Act.GTarg.GeMax` value on the layer, and excitatory projections are scaled to keep the actual max Ge excitation in that target range, with tolerances for being under or over.  In general, activity is low early in learning, and adapting scales to make this input stronger is not beneficial.  Thus, the default low-side tolerance is quite wide, while the upper-side is tight and any deviation above the target results in reducing the overall scaling factor.  Note that this is quite distinct from the synaptic scaling described above, because it operates across the entire projection, not per-neuron, and operates on a very slow time scale, akin to developmental time in an organism.

* *Adaptation of layer inhibition.*  This is similar to projection scaling, but for overall layer inhibition.  TODO: revisit again in context of scaling adaptation.

## Important Stats

The only way to manage the complexity of large spiking nets is to develop advanced statistics that reveal what is going on, especially when things go wrong.  These include:

* Basic "activation health": proper function depends on neurons remaining in a sensitive range of excitatory and inhibitory inputs, so these are monitored.  Each layer has `ActAvg` with `AvgMaxGeM` reporting average maximum minus-phase Ge values -- these are what is regulated relative to `Act.GTarg.GeMax`, but also must be examined early in training to ensure that initial excitation is not too weak.  The layer `Inhib.ActAvg.Init` can be set to adjust -- and unlike in Leabra, there is a separate `Target` value that controls adaptation of layer-level inhibition.

* Hogging and the flip-side: dead units.

* PCA of overall representational complexity.  Even if individual neurons are not hogging the space, the overall compelxity of representations can be reduced through a more distributed form of hogging.  PCA provides a good measure of that.

## Not important

* Extra bursting for changed plus-phase neurons

# Functional Advantages of Spikes

One particular advantage of spikes in this context is that they produce a looser form of coupling between neurons in general: with rate code neurons, once a pattern of activity gets established, it is very hard for something different to break through.  The active neurons are constantly firing, constantly dominating the "conversation", and don't allow anyone else to get a word in edgewise.  By contrast, spiking neurons are only periodically communicating, leaving lots of gaps where new signals can assert themselves, and their *attractor* states are thus much more *porous* and flexible.  Furthermore, spikes famously allow for more fine-grained timing dynamics to significantly affect communication efficacy: synchronous firing over a population of neurons results in *temporal summation* of excitatory currents, whereas that same spiking pattern distributed more uniformly across time has much less effect.  Thus, bursting and synchrony dynamics allow for a much deeper effective dynamic range of signaling.  This is particularly important in enabling plus-phase activation signals carrying outcome signals to penetrate effectively into deep networks.

Despite these advantages, the spiking networks still exhibit strong hog-unit tendencies, but mechanisms inspired directly from known biology, along with longstanding computational principles, have provided effective ways of managing them.  Spiking networks obviously are also just much more noisy than rate code models, so we can't reasonably expect perfect, deterministic performance from them.  Thus, it is essential to use more robust error measures, and more lenient expectations for overall performance.

# Neural data for parameters

* Brunel00: https://nest-simulator.readthedocs.io/en/nest-2.20.1/auto_examples/brunel_alpha_nest.html
* SahHestrinNicoll90 -- actual data for AMPA, NMDA
* XiangHuguenardPrince98a -- actual data for GABAa

Axonal conduction delays:

* http://www.scholarpedia.org/article/Axonal_conduction_delay
    + thalamocortical is very fast: 1.2 ms
    + corticocortical axonal conduction delays in monkey average 2.3 ms (.5 to 8 range)
    + corpus callosum (long distance) average around 10 ms
    + Brunel00: 1.5 ms

AMPA rise, decay times:

* SHN90: rise times 1-3ms -- recorded at soma -- reflect conductance delays
* SHN90: decay times 4-8ms mostly
* Brunel00: 0.5 ms -- too short!

GABAa rise, decay times:

* XHP98a: 0.5ms rise, 6-7ms decay

## Temporal and Spatial Dynamics of Dendritic Integration

A series of models published around the year 2000 investigated the role of active dendritic channels on signal integration across different dendritic compartments (Migliore et al, 1999; Poirazi et al, 2003; Jarsky et al, 2005) -- see Spruston (2008) and Poirazi & Papoutsi (2020) for reviews.  The Urakubo et al (2008) model uses the Migliore and Poirazi models as a starting point.  A common conclusion was that the A-type K channel can be inactivated as a result of elevated Vm in the dendrite, driving a nonlinear gating-like interaction between dendritic inputs: when enough input comes in (e.g., from 2 different projections), then the rest of the inputs are all integrated more-or-less linearly, but below this critical threshold, inputs are much more damped by the active A-type K channels.  There are also other complications associated with VGCC L-type and T-type voltage-gated Ca channels which can drive Ca spikes, to amplify regular AMPA conductances, relative weakness and attenuation of active HH Na spiking channels in dendrites, and issues of where inhibition comes in, etc.  See following figure from Spruston (2008) for a summary of some key traces:

![Differences between Soma vs. Dendrites](fig_dendrite_coinc_spruston08_fig5.png?raw=true "Figure 5 from Spruston (2008), showing mutual dependence on two dendritic compartments for synaptic integration in panel a, and also significant differences in temporal duration of elevated Vm in dendrites vs. soma.")

The key question here is: to what extent do we need to account for these additional degrees of freedom in the model?  First, a key premise of the AdEx model is that anything taking place during the action potential (AP) is too fast and stereotyped to worry about.  Further, AdEx primarily accounts for soma-level spiking dynamics, and may not do a good job of reflecting what happens in the more distal dendrites -- as we can see in the above figure, membrane potential in the dendrites updates more slowly and is not reset as significantly after spiking.  This is captured in the use of separate Vm variables: `VmDend` for dendrites, and plain `Vm` for the somatic spiking potential.  The `VmDend` is what drives the NMDA and GABAB channels, and thus its ability to sustain a higher overall level of activation is critical for the stabilizing effects of these channels.

Here are some specific considerations and changes to address these issues:

* The `Kdr` delayed rectifier channel, part of the classical HH model, resets the membrane potential back to resting after a spike -- according to detailed traces from the Urakubo model with this channel, and the above figures, this is not quite an instantaneous process, with a time constant somewhere between 1-2 msec.  This is not important for overall spiking behavior, but it is important when Vm is used for more realistic Ca-based learning (as in the Urakubo model).  This more realistic `VmR` reset behavior is now captured as of v1.2.95 via the `RTau` time constant, which decays `Vm` back to `VmR` within the `Tr` refractory period of 3 msec, which fits well with the Urakubo traces for isolated spikes.

* In the simplest model used initially, `VmDend` did not get the Exp rise of the spike, nor did it experience the `VmR` reset, and it has a slower time constant of integration.  As of v1.2.95, the new `Dend` params specify a `GbarExp` parameter that applies a fraction of the Exp slope to `VmDend`, and a `GbarR` param that injects a proportional amount of leak current during the spike reset (`Tr`) window to bring the Vm back down a bit, reflecting the weaker amount of `Kdr` out in the dendrites.  This produces traces that resemble the above figure, as shown in the following run of the `examples/neuron` model, comparing VmDend with Vm.  Preliminary indications suggest this has a significant benefit on model performance overall (on ra25 and fsa so far), presumably by engaging NMDA and GABAB channels better.

![VmDend vs Vm in neuron example](fig_axon_neuron_vmdend.png?raw=true "VmDend now has dynamics that reflect weaker trace of soma spiking -- initial results suggest this improves performance by better engaging the NMDA / GABAB channels.")

* As for the broader question of more coincidence-driven dynamics in the dendrites, or an AND-like mutual interdependence among inputs to different branches, driven by A-type K channels, it is likely that in the awake behaving context (*in activo*) as compared to the slices where these original studies were done, there is always a reasonable background level of synaptic input such that these channels are largely inactivated anyway.  This corresponds to the important differences between upstate / downstate that also largely disappear in awake behaving vs. anesthetized or slice preps.  Nevertheless, it is worth continuing to investigate this issue and explore the potential implications of these mechanisms in actual running models.  TODO: create atype channels in glong (rename to something else, maybe just `chans` for channels)

## Urakubo

See the `examples/urakubo` directory for a replication of the Urakubo et al (2008) Ca-driven signaling mechanisms involving CaMKII, PKA, CaN, PP1, and AMPAR trafficking under their influence.  This model replicates experimental results from a range of STDP paradigms, and can be used as a exploratory lab for investigating different learning possible mechanisms beyond the simple XCAL equation derived from the original Urakubo et al model.

# Extensions: BG / PFC / DA

There are various extensions to the algorithm that implement special neural mechanisms associated with the prefrontal cortex and basal ganglia [PBWM](#pbwm), dopamine systems [PVLV](#pvlv), the [Hippocampus](#hippocampus), and predictive learning and temporal integration dynamics associated with the thalamocortical circuits [DeepAxon](#deepaxon).  All of these are (will be) implemented as additional modifications of the core, simple `axon` implementation, instead of having everything rolled into one giant hairball as in the original C++ implementation.

# Implementation strategy

* Prjn integrates conductances for each recv prjn
* Send spike adds wt to ring buff at given offset
* For all the spikes that arrive that timestep, increment G += wts
* build in psyn with probability of failure -- needs to be sensitive to wt
* then decay with tau
* integrate all the G's in to cell body
* support background balanced G, I current on cell body to diminish sensitivity! Init

* Detailed model of dendritic dynamics relative to cell body, with data -- dendrites a bit slower, don't reset after spiking?  GaoGrahamZhouEtAl20


# Pseudocode as a LaTeX doc for Paper Appendix

You can copy the mediawiki source of the following section into a file, and run [pandoc](https://pandoc.org/) on it to convert to LaTeX (or other formats) for inclusion in a paper.  As this wiki page is always kept updated, it is best to regenerate from this source -- very easy:

```bash
curl "https://raw.githubusercontent.com/emer/axon/master/README.md" -o appendix.md
pandoc appendix.md -f gfm -t latex -o appendix.tex
```

You can then edit the resulting .tex file to only include the parts you want, etc.

# Axon Algorithm Equations

The pseudocode for Axon is given here, showing exactly how the pieces of the algorithm fit together, using the equations and variables from the actual code.  Compared to the original C++ emergent implementation, this Go version of emergent is much more readable, while also not being too much slower overall.

There are also other implementations of Axon available:
* [axon7](https://github.com/PrincetonUniversity/axon7) Python implementation of the version 7 of Axon, by Daniel Greenidge and Ken Norman at Princeton.
* [Matlab](https://github.com/emer/cemer/blob/master/Matlab/) (link into the cemer C++ emergent source tree) -- a complete implementation of these equations in Matlab, coded by Sergio Verduzco-Flores.
* [Python](https://github.com/benureau/axon) implementation by Fabien Benureau.
* [R](https://github.com/johannes-titz/axon) implementation by Johannes Titz.

This repository contains specialized additions to the core algorithm described here:
* [deep](https://github.com/emer/axon/blob/master/deep) has the DeepAxon mechanisms for simulating the deep neocortical <-> thalamus pathways (wherein basic Axon represents purely superficial-layer processing)
* [pbwm](https://github.com/emer/axon/blob/master/rl) has basic reinforcement learning models such as Rescorla-Wagner and TD (temporal differences).
* [pbwm](https://github.com/emer/axon/blob/master/pbwm1) has the prefrontal-cortex basal ganglia working memory model (PBWM).
* [hip](https://github.com/emer/axon/blob/master/hip) has the hippocampus specific learning mechanisms.

## Timing

Axon is organized around the following timing, based on an internally-generated alpha-frequency (10 Hz, 100 msec periods) cycle of expectation followed by outcome, supported by neocortical circuitry in the deep layers and the thalamus, as hypothesized in the [DeepAxon](#deepaxon) extension to standard Axon:

* A **Trial** lasts 100 msec (10 Hz, alpha frequency), and comprises one sequence of expectation -- outcome learning, organized into 4 quarters.
    + Biologically, the deep neocortical layers (layers 5, 6) and the thalamus have a natural oscillatory rhythm at the alpha frequency.  Specific dynamics in these layers organize the cycle of expectation vs. outcome within the alpha cycle.

* A **Quarter** lasts 25 msec (40 Hz, gamma frequency) -- the first 3 quarters (75 msec) form the expectation / minus phase, and the final quarter are the outcome / plus phase.
    + Biologically, the superficial neocortical layers (layers 2, 3) have a gamma frequency oscillation, supporting the quarter-level organization.

* A **Cycle** represents 1 msec of processing, where each neuron updates its membrane potential etc according to the above equations.

## Variables

The `axon.Neuron` struct contains all the neuron (unit) level variables, and the `axon.Layer` contains a simple Go slice of these variables.  Optionally, there can be `axon.Pool` pools of subsets of neurons that correspond to hypercolumns, and support more local inhibitory dynamics (these used to be called UnitGroups in the C++ version).

* `Act`   = overall rate coded activation value -- what is sent to other neurons -- typically in range 0-1
* `Ge` = total excitatory synaptic conductance -- the net excitatory input to the neuron -- does *not* include Gbar.E
* `Gi` = total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I
* `Inet` = net current produced by all channels -- drives update of Vm
* `Vm` = membrane potential -- integrates Inet current over time
* `Target` = target value: drives learning to produce this activation value
* `Ext` = external input: drives activation of unit from outside influences (e.g., sensory input)
* `AvgSS` = super-short time-scale activation average -- provides the lowest-level time integration -- for spiking this integrates over spikes before subsequent averaging, and it is also useful for rate-code to provide a longer time integral overall
* `AvgS` = short time-scale activation average -- tracks the most recent activation states (integrates over AvgSS values), and represents the plus phase for learning in XCAL algorithms
* `AvgM` = medium time-scale activation average -- integrates over AvgS values, and represents the minus phase for learning in XCAL algorithms
* `AvgL` = long time-scale average of medium-time scale (trial level) activation, used for the BCM-style floating threshold in XCAL
* `AvgLLrn` = how much to learn based on the long-term floating threshold (AvgL) for BCM-style Hebbian learning -- is modulated by level of AvgL itself (stronger Hebbian as average activation goes higher) and optionally the average amount of error experienced in the layer (to retain a common proportionality with the level of error-driven learning across layers)
* `AvgSLrn` = short time-scale activation average that is actually used for learning -- typically includes a small contribution from AvgM in addition to mostly AvgS, as determined by `LrnActAvgParams.LrnM` -- important to ensure that when unit turns off in plus phase (short time scale), enough medium-phase trace remains so that learning signal doesn't just go all the way to 0, at which point no learning would take place
* `ActM` = records the traditional posterior-cortical minus phase activation, as activation after third quarter of current alpha cycle
* `ActP` = records the traditional posterior-cortical plus_phase activation, as activation at end of current alpha cycle
* `ActDiff` = ActP - ActM -- difference between plus and minus phase acts -- reflects the individual error gradient for this neuron in standard error-driven learning terms
* `ActDel` delta activation: change in Act from one cycle to next -- can be useful to track where changes are taking place
* `ActAvg` = average activation (of final plus phase activation state) over long time intervals (time constant = DtPars.AvgTau -- typically 200) -- useful for finding hog units and seeing overall distribution of activation
* `Noise` = noise value added to unit (`ActNoiseParams` determines distribution, and when / where it is added)
* `GiSyn` = aggregated synaptic inhibition (from Inhib projections) -- time integral of GiRaw -- this is added with computed FFFB inhibition to get the full inhibition in Gi
* `GiSelf` = total amount of self-inhibition -- time-integrated to avoid oscillations

The following are more implementation-level variables used in integrating synaptic inputs:

* `ActSent` = last activation value sent (only send when diff is over threshold)
* `GeRaw` = raw excitatory conductance (net input) received from sending units (send delta's are added to this value)
* `GeInc` = delta increment in GeRaw sent using SendGeDelta
* `GiRaw` = raw inhibitory conductance (net input) received from sending units (send delta's are added to this value)
* `GiInc` = delta increment in GiRaw sent using SendGeDelta

Neurons are connected via synapses parameterized with the following variables, contained in the `axon.Synapse` struct.  The `axon.Prjn` contains all of the synaptic connections for all the neurons across a given layer -- there are no Neuron-level data structures in the Go version.

* `Wt` = synaptic weight value -- sigmoid contrast-enhanced
* `LWt` = linear (underlying) weight value -- learns according to the lrate specified in the connection spec -- this is converted into the effective weight value, Wt, via sigmoidal contrast enhancement (see `WtSigParams`)
* `DWt` = change in synaptic weight, from learning
* `Norm` = DWt normalization factor -- reset to max of abs value of DWt, decays slowly down over time -- serves as an estimate of variance in weight changes over time
* `Moment` = momentum -- time-integrated DWt changes, to accumulate a consistent direction of weight change and cancel out dithering contradictory changes

## Activation Update Cycle (every 1 msec): Ge, Gi, Act

The `axon.Network` `Cycle` method in `axon/network.go` looks like this:

```Go
// Cycle runs one cycle of activation updating:
// * Sends Ge increments from sending to receiving layers
// * Average and Max Ge stats
// * Inhibition based on Ge stats and Act Stats (computed at end of Cycle)
// * Activation from Ge, Gi, and Gl
// * PostAct: Average and Max Act stats, Synaptic Ca updates
// This basic version doesn't use the time info, but more specialized types do, and we
// want to keep a consistent API for end-user code.
func (nt *Network) Cycle(ctime *Time) {
	nt.SendGDelta(ctime) // also does integ
	nt.AvgMaxGe(ctime)
	nt.InhibFmGeAct(ctime)
	nt.SpikeFmG(ctime)
	nt.PostAct(ctime)
}
```

For every cycle of activation updating, we compute the excitatory input conductance `Ge`, then compute inhibition `Gi` based on average `Ge` and `Act` (from previous cycle), then compute the `Act` based on those conductances.  The equations below are not shown in computational order but rather conceptual order for greater clarity.  All of the relevant parameters are in the `axon.Layer.Act` and `Inhib` fields, which are of type `ActParams` and `InhibParams` -- in this Go version, the parameters have been organized functionally, not structurally, into three categories.

* `Ge` excitatory conductance is actually computed using a highly efficient delta-sender-activation based algorithm, which only does the expensive multiplication of activations * weights when the sending activation changes by a given amount (`OptThreshParams.Delta`).  However, conceptually, the conductance is given by this equation:
    + `GeRaw += Sum_(recv) Prjn.GScale * Send.Act * Wt`
        + `Prjn.GScale` is the [Input Scaling](#input-scaling) factor that includes 1/N to compute an average, and the `WtScaleParams` `Abs` absolute scaling and `Rel` relative scaling, which allow one to easily modulate the overall strength of different input projections.
    + `Ge += DtParams.Integ * (1/ DtParams.GTau) * (GeRaw - Ge)`
        + This does a time integration of excitatory conductance, `GTau = 1.4` default, and global integration time constant, `Integ = 1` for 1 msec default.

* `Gi` inhibtory conductance combines computed and synaptic-level inhibition (if present) -- most of code is in `axon/inhib.go`
    + `ffNetin = avgGe + FFFBParams.MaxVsAvg * (maxGe - avgGe)`
    + `ffi = FFFBParams.FF * MAX(ffNetin - FFBParams.FF0, 0)`
        + feedforward component of inhibition with FF multiplier (1 by default) -- has FF0 offset and can't be negative (that's what the MAX(.. ,0) part does).
        + `avgGe` is average of Ge variable across relevant Pool of neurons, depending on what level this is being computed at, and `maxGe` is max of Ge across Pool
    + `fbi += (1 / FFFBParams.FBTau) * (FFFBParams.FB * avgAct - fbi`
        + feedback component of inhibition with FB multiplier (1 by default) -- requires time integration to dampen oscillations that otherwise occur -- FBTau = 1.4 default.
    + `Gi = FFFBParams.Gi * (ffi + fbi)`
        + total inhibitory conductance, with global Gi multiplier -- default of 1.8 typically produces good sparse distributed representations in reasonably large layers (25 units or more).

* `Act` activation from Ge, Gi, Gl (most of code is in `axon/act.go`, e.g., `ActParams.SpikeFmG` method).  When neurons are above thresholds in subsequent condition, they obey the "geLin" function which is linear in Ge:
    + `geThr = (Gi * (Erev.I - Thr) + Gbar.L * (Erev.L - Thr) / (Thr - Erev.E)`
    + `nwAct = NoisyXX1(Ge * Gbar.E - geThr)`
        + geThr = amount of excitatory conductance required to put the neuron exactly at the firing threshold, `XX1Params.Thr` = .5 default, and NoisyXX1 is the x / (x+1) function convolved with gaussian noise kernel, where x = `XX1Parms.Gain` * Ge - geThr) and Gain is 100 by default
    + `if Act < XX1Params.VmActThr && Vm <= X11Params.Thr: nwAct = NoisyXX1(Vm - Thr)`
        + it is important that the time to first "spike" (above-threshold activation) be governed by membrane potential Vm integration dynamics, but after that point, it is essential that activation drive directly from the excitatory conductance Ge relative to the geThr threshold.
    + `Act += (1 / DTParams.VmTau) * (nwAct - Act)`
        + time-integration of the activation, using same time constant as Vm integration (VmTau = 3.3 default)
    + `Vm += (1 / DTParams.VmTau) * Inet`
    + `Inet = Ge * (Erev.E - Vm) + Gbar.L * (Erev.L - Vm) + Gi * (Erev.I - Vm) + Noise`
        + Membrane potential computed from net current via standard RC model of membrane potential integration.  In practice we use normalized Erev reversal potentials and Gbar max conductances, derived from biophysical values: Erev.E = 1, .L = 0.3, .I = 0.25, Gbar's are all 1 except Gbar.L = .2 default.

## Learning

![XCAL DWt Function](fig_xcal_dwt_fun.png?raw=true "The XCAL dWt function, showing direction and magnitude of synaptic weight changes dWt as a function of the short-term average activity of the sending neuron *x* times the receiving neuron *y*.  This quantity is a simple mathematical approximation to the level of postsynaptic Ca++, reflecting the dependence of the NMDA channel on both sending and receiving neural activity.  This function was extracted directly from the detailed biophysical Urakubo et al. 2008 model, by fitting a piecewise linear function to the synaptic weight change behavior that emerges from it as a function of a wide range of sending and receiving spiking patterns.")

Learning is based on running-averages of activation variables, parameterized in the `axon.Layer.Learn` `LearnParams` field, mostly implemented in the `axon/learn.go` file.

* **Running averages** computed continuously every cycle, and note the compounding form.  Tau params in `LrnActAvgParams`:
    + `AvgSS += (1 / SSTau) * (Act - AvgSS)`
        + super-short time scale running average, SSTau = 2 default -- this was introduced to smooth out discrete spiking signal, but is also useful for rate code.
    + `AvgS += (1 / STau) * (AvgSS - AvgS)`
        + short time scale running average, STau = 2 default -- this represents the *plus phase* or actual outcome signal in comparison to AvgM
    + `AvgM += (1 / MTau) * (AvgS - AvgM)`
        + medium time-scale running average, MTau = 10 -- this represents the *minus phase* or expectation signal in comparison to AvgS
    + `AvgL += (1 / Tau) * (Gain * AvgM - AvgL); AvgL = MAX(AvgL, Min)`
        + long-term running average -- this is computed just once per learning trial, *not every cycle* like the ones above -- params on `AvgLParams`: Tau = 10, Gain = 2.5 (this is a key param -- best value can be lower or higher) Min = .2
    + `AvgLLrn = ((Max - Min) / (Gain - Min)) * (AvgL - Min)`
        + learning strength factor for how much to learn based on AvgL floating threshold -- this is dynamically modulated by strength of AvgL itself, and this turns out to be critical -- the amount of this learning increases as units are more consistently active all the time (i.e., "hog" units).  Params on `AvgLParams`, Min = 0.0001, Max = 0.5. Note that this depends on having a clear max to AvgL, which is an advantage of the exponential running-average form above.
    + `AvgLLrn *= MAX(1 - layCosDiffAvg, ModMin)`
        + also modulate by time-averaged cosine (normalized dot product) between minus and plus phase activation states in given receiving layer (layCosDiffAvg), (time constant 100) -- if error signals are small in a given layer, then Hebbian learning should also be relatively weak so that it doesn't overpower it -- and conversely, layers with higher levels of error signals can handle (and benefit from) more Hebbian learning.  The MAX(ModMin) (ModMin = .01) factor ensures that there is a minimum level of .01 Hebbian (multiplying the previously-computed factor above).  The .01 * .05 factors give an upper-level value of .0005 to use for a fixed constant AvgLLrn value -- just slightly less than this (.0004) seems to work best if not using these adaptive factors.
    + `AvgSLrn = (1-LrnM) * AvgS + LrnM * AvgM`
        + mix in some of the medium-term factor into the short-term factor -- this is important for ensuring that when neuron turns off in the plus phase (short term), that enough trace of earlier minus-phase activation remains to drive it into the LTD weight decrease region -- LrnM = .1 default.

* **Learning equation**:
    + `srs = Send.AvgSLrn * Recv.AvgSLrn`
    + `srm = Send.AvgM * Recv.AvgM`
    + `dwt = XCAL(srs, srm) + Recv.AvgLLrn * XCAL(srs, Recv.AvgL)`
        + weight change is sum of two factors: error-driven based on medium-term threshold (srm), and BCM Hebbian based on long-term threshold of the recv unit (Recv.AvgL)
    + XCAL is the "check mark" linearized BCM-style learning function (see figure) that was derived from the Urakubo Et Al (2008) STDP model, as described in more detail in the [CCN textbook](https://CompCogNeuro.org)
        + `XCAL(x, th) = (x < DThr) ? 0 : (x > th * DRev) ? (x - th) : (-x * ((1-DRev)/DRev))`
        + DThr = 0.0001, DRev = 0.1 defaults, and x ? y : z terminology is C syntax for: if x is true, then y, else z

    + **DWtNorm** -- normalizing the DWt weight changes is standard in current backprop, using the AdamMax version of the original RMS normalization idea, and benefits Axon as well, and is On by default, params on `DwtNormParams`:
        + `Norm = MAX((1 - (1 / DecayTau)) * Norm, ABS(dwt))`
            + increment the Norm normalization using abs (L1 norm) instead of squaring (L2 norm), and with a small amount of decay: DecayTau = 1000.
        + `dwt *= LrComp / MAX(Norm, NormMin)`
            + normalize dwt weight change by the normalization factor, but with a minimum to prevent dividing by 0 -- LrComp compensates overall learning rate for this normalization (.15 default) so a consistent learning rate can be used, and NormMin = .001 default.
    + **Momentum** -- momentum is turned On by default, and has significant benefits for preventing hog units by driving more rapid specialization and convergence on promising error gradients.  Parameters on `MomentumParams`:
        + `Moment = (1 - (1 / MTau)) * Moment + dwt`
        + `dwt = LrComp * Moment`
            + increment momentum from new weight change, MTau = 10, corresponding to standard .9 momentum factor (sometimes 20 = .95 is better), with LrComp = .1 comp compensating for increased effective learning rate.
    + `DWt = Lrate * dwt`
        + final effective weight change includes overall learning rate multiplier.  For learning rate schedules, just directly manipulate the learning rate parameter -- not using any kind of builtin schedule mechanism.

* **Weight Balance** -- this option (off by default but recommended for larger models) attempts to maintain more balanced weights across units, to prevent some units from hogging the representational space, by changing the rates of weight increase and decrease in the soft weight bounding function, as a function of the average receiving weights.  All params in `WtBalParams`:
    + `if (Wb.Avg < LoThr): Wb.Fact = LoGain * (LoThr - MAX(Wb.Avg, AvgThr)); Wb.Dec = 1 / (1 + Wb.Fact); Wb.Inc = 2 - Wb.Dec`
    + `else: Wb.Fact = HiGain * (Wb.Avg - HiThr); Wb.Inc = 1 / (1 + Wb.Fact); Wb.Dec = 2 - Wb.Inc`
        + `Wb` is the `WtBalRecvPrjn` structure stored on the `axon.Prjn`, per each Recv neuron.  `Wb.Avg` = average of recv weights (computed separately and only every N = 10 weight updates, to minimize computational cost).  If this average is relatively low (compared to LoThr = .4) then there is a bias to increase more than decrease, in proportion to how much below this threshold they are (LoGain = 6).  If the average is relatively high (compared to HiThr = .4), then decreases are stronger than increases, HiGain = 4.
    + A key feature of this mechanism is that it does not change the sign of any weight changes, including not causing weights to change that are otherwise not changing due to the learning rule.  This is not true of an alternative mechanism that has been used in various models, which normalizes the total weight value by subtracting the average.  Overall this weight balance mechanism is important for larger networks on harder tasks, where the hogging problem can be a significant problem.

* **Weight update equation**
    + The `LWt` value is the linear, non-contrast enhanced version of the weight value, and `Wt` is the sigmoidal contrast-enhanced version, which is used for sending netinput to other neurons.  One can compute LWt from Wt and vice-versa, but numerical errors can accumulate in going back-and forth more than necessary, and it is generally faster to just store these two weight values.
    + `DWt *= (DWt > 0) ? Wb.Inc * (1-LWt) : Wb.Dec * LWt`
        + soft weight bounding -- weight increases exponentially decelerate toward upper bound of 1, and decreases toward lower bound of 0, based on linear, non-contrast enhanced LWt weights.  The `Wb` factors are how the weight balance term shift the overall magnitude of weight increases and decreases.
    + `LWt += DWt`
        + increment the linear weights with the bounded DWt term
    + `Wt = SIG(LWt)`
        + new weight value is sigmoidal contrast enhanced version of linear weight
        + `SIG(w) = 1 / (1 + (Off * (1-w)/w)^Gain)`
    + `DWt = 0`
        + reset weight changes now that they have been applied.

## Input Scaling

The `Ge` and `Gi` synaptic conductances computed from a given projection from one layer to the next reflect the number of receptors currently open and capable of passing current, which is a function of the activity of the sending layer, and total number of synapses.  We use a set of equations to automatically normalize (rescale) these factors across different projections, so that each projection has roughly an equal influence on the receiving neuron, by default.

The most important factor to be mindful of for this automatic rescaling process is the expected activity level in a given sending layer.  This is set initially to `Layer.Inhib.ActAvg.Init`, and adapted from there by the various other parameters in that `Inhib.ActAvg` struct.  It is a good idea in general to set that `Init` value to a reasonable estimate of the proportion of activity you expect in the layer, and in very small networks, it is typically much better to just set the `Fixed` flag and keep this `Init` value as such, as otherwise the automatically computed averages can fluctuate significantly and thus create corresponding changes in input scaling.  The default `UseFirst` flag tries to avoid the dependence on the `Init` values but sometimes the first value may not be very representative, so it is better to set `Init` and turn off `UseFirst` for more reliable performance.

Furthermore, we add two tunable parameters that further scale the overall conductance received from a given projection (one in a *relative* way compared to other projections, and the other a simple *absolute* multiplicative scaling factor).  These are some of the most important parameters to configure in the model -- in particular the strength of top-down "back" projections typically must be relatively weak compared to bottom-up forward projections (e.g., a relative scaling factor of 0.1 or 0.2 relative to the forward projections).

The scaling contributions of these two factors are:

* `GScale = WtScale.Abs * (WtScale.Rel / Sum(all WtScale.Rel))`

Thus, all the `Rel` factors contribute in proportion to their relative value compared to the sum of all such factors across all receiving projections into a layer, while `Abs` just multiplies directly.

In general, you want to adjust the `Rel` factors, to keep the total `Ge` and `Gi` levels relatively constant, while just shifting the relative contributions.  In the relatively rare case where the overall `Ge` levels are too high or too low, you should adjust the `Abs` values to compensate.

Typically the `Ge` value should be between .5 and 1, to maintain a reasonably responsive neural response, and avoid numerical integration instabilities and saturation that can arise if the values get too high.  You can record the `Layer.Pools[0].Inhib.Ge.Avg` and `.Max` values at the epoch level to see how these are looking -- this is especially important in large networks, and those with unusual, complex patterns of connectivity, where things might get out of whack.

### Automatic Rescaling

Here are the relevant factors that are used to compute the automatic rescaling to take into account the expected activity level on the sending layer, and the number of connections in the projection.  The actual code is in `axon/layer.go: GScaleFmAvgAct()` and `axon/act.go SLayActScale`

* `savg` = sending layer average activation
* `snu` = sending layer number of units
* `ncon` = number of connections
* `slayActN = int(Round(savg * snu))` -- must be at least 1
* `sc` = scaling factor, which is roughly 1 / expected number of active sending connections.
* `if ncon == snu:` -- full connectivity
    + `sc = 1 / slayActN`
* `else:`           -- partial connectivity -- trickier
    + `avgActN = int(Round(savg * ncon))` -- avg proportion of connections
    + `expActN = avgActN + 2`  -- add an extra 2 variance around expected value
    + `maxActN = MIN(ncon, sLayActN)`  -- can't be more than number active
    + `expActN = MIN(expActN, maxActN)`  -- constrain
    + `sc = 1 / expActN`

This `sc` factor multiplies the `GScale` factor as computed above.

# References

* Jarsky, T., Roxin, A., Kath, W. L., & Spruston, N. (2005). Conditional dendritic spike propagation following distal synaptic activation of hippocampal CA1 pyramidal neurons. Nat Neurosci, 8, 1667–1676. http://dx.doi.org/10.1038/nn1599

* Migliore, M., Hoffman, D. A., Magee, J. C., & Johnston, D. (1999). Role of an A-Type K+ Conductance in the Back-Propagation of Action Potentials in the Dendrites of Hippocampal Pyramidal Neurons. Journal of Computational Neuroscience, 7(1), 5–15. https://doi.org/10.1023/A:1008906225285

* Poirazi, P., Brannon, T., & Mel, B. W. (2003). Arithmetic of Subthreshold Synaptic Summation in a Model CA1 Pyramidal Cell. Neuron, 37(6), 977–987. https://doi.org/10.1016/S0896-6273(03)00148-X

* Poirazi, P., & Papoutsi, A. (2020). Illuminating dendritic function with computational models. Nature Reviews Neuroscience, 21(6), 303–321. https://doi.org/10.1038/s41583-020-0301-7

* Sanders, H., Berends, M., Major, G., Goldman, M. S., & Lisman, J. E. (2013). NMDA and GABAB (KIR) Conductances: The "Perfect Couple" for Bistability. Journal of Neuroscience, 33(2), 424–429. https://doi.org/10.1523/JNEUROSCI.1854-12.2013

* Spruston, N. (2008). Pyramidal neurons: Dendritic structure and synaptic integration. Nature Reviews. Neuroscience, 9(3), 201–221. http://www.ncbi.nlm.nih.gov/pubmed/18270515

* Urakubo, H., Honda, M., Froemke, R. C., & Kuroda, S. (2008). Requirement of an allosteric kinetics of NMDA receptors for spike timing-dependent plasticity. The Journal of Neuroscience, 28(13), 3310–3323. http://www.ncbi.nlm.nih.gov/pubmed/18367598
