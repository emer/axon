# Axon in Go emergent

[![Go Report Card](https://goreportcard.com/badge/github.com/emer/axon)](https://goreportcard.com/report/github.com/emer/axon)
[![GoDoc](https://godoc.org/github.com/emer/axon?status.svg)](https://godoc.org/github.com/emer/axon)
[![codecov](https://codecov.io/gh/emer/axon/branch/master/graph/badge.svg)](https://codecov.io/gh/emer/axon)
[![Travis](https://travis-ci.com/emer/axon.svg?branch=master)](https://travis-ci.com/emer/axon)

This is the Go implementation of the Axon algorithm for spiking, biologically-based models of cognition, based on the [Go emergent](https://github.com/emer/emergent) framework (with optional Python interface), and the  [leabra](https://github.com/emer/leabra) framework for rate-code models.

Axon is the spiking version of [Leabra](https://github.com/emer/leabra), with several advances.  As an acronym, *axon* could stand for *Adaptive eXcitation Of Noise*, reflecting the ability to learn using the power of error-backpropagation in the context of noisy spiking activation.  The spiking function of the axon is what was previously missing from Leabra.

See [Wiki Install](https://github.com/emer/axon/wiki/Install) for installation instructions.

See the [ra25 example](https://github.com/emer/axon/blob/master/examples/ra25/README.md) for a complete working example (intended to be a good starting point for creating your own models), and any of the 26 models in the [Comp Cog Neuro sims](https://github.com/CompCogNeuro/axon) repository which also provide good starting points.  See the [etable wiki](https://github.com/emer/etable/wiki) for docs and example code for the widely-used etable data table structure, and the `family_trees` example in the CCN textbook sims which has good examples of many standard network representation analysis techniques (PCA, cluster plots, RSA).

The [Wiki Convert From Leabra](https://github.com/emer/axon/wiki/Convert-From-Leabra) page has information for converting existing Go leabra models.

See [python README](https://github.com/emer/axon/blob/master/python/README.md) and [Python Wiki](https://github.com/emer/emergent/wiki/Python) for info on using Python to run models.  NOTE: not yet updated.

# Current Status / News

* Dec 2022: **v1.6.12** represents the start of an anticipated stable plateau in development, with this README fully updated to describe the current algorithm, and well-tested and biologically-based implementations of all the major elements of the core algorithm, and initial steps on specialized PFC / BG / RL algorithms as integrated in the `examples/boa` model.

* May-July 2021: Initial implementation and significant experimentation.  The fully spiking-based Axon framework is capable of learning to categorize rendered 3D object images based on the deep, bidirectionally-connected LVis model originally reported in O'Reilly et al. (2013).  Given the noisy, complex nature of the spiking dynamics, getting this level of functionality out of a large, deep network architecture was not easy, and it drove a number of critical additional mechanisms that are necessary for this model to work.

# Design and Organization

* `ActParams` (in [act.go](https://github.com/emer/axon/blob/master/axon/act.go)), `InhibParams` (in [inhib.go](https://github.com/emer/axon/blob/master/axon/inhib.go)), and `LearnNeurParams` / `LearnSynParams` (in [learn.go](https://github.com/emer/axon/blob/master/axon/learn.go)) provide the core parameters and functions used.

* There are 3 main levels of structure: `Network`, `Layer` and `Prjn` (projection).  The network calls methods on its Layers, and Layers iterate over both `Neuron` data structures (which have only a minimal set of methods) and the `Prjn`s, to implement the relevant computations.  The `Prjn` fully manages everything about a projection of connectivity between two layers, including the full list of `Synapse` elements in the connection.  The Layer also has a set of `Pool` elements, one for each level at which inhibition is computed (there is always one for the Layer, and then optionally one for each Sub-Pool of units.

* The `NetworkBase` and `LayerBase` structs manage all the core structural aspects of things (data structures etc), and then the algorithm-specific versions (e.g., `axon.Network`) use Go's anonymous embedding (akin to inheritance in C++) to transparently get all that functionality, while then directly implementing the algorithm code.  Almost every step of computation has an associated method in `axon.Layer`, so look first in [layer.go](https://github.com/emer/axon/blob/master/axon/layer.go) to see how something is implemented.

* Each structural element directly has all the parameters controlling its behavior -- e.g., the `Layer` contains an `ActParams` field (named `Act`), etc.  The ability to share parameter settings across multiple layers etc is achieved through a **styling**-based paradigm -- you apply parameter "styles" to relevant layers -- see [Params](https://github.com/emer/emergent/wiki/Params) for more info.  We adopt the CSS (cascading-style-sheets) standard where parameters can be specifed in terms of the Name of an object (e.g., `#Hidden`), the *Class* of an object (e.g., `.TopDown` -- where the class name TopDown is manually assigned to relevant elements), and the *Type* of an object (e.g., `Layer` applies to all layers).  Multiple space-separated classes can be assigned to any given element, enabling a powerful combinatorial styling strategy to be used.

* Go uses `interface`s to represent abstract collections of functionality (i.e., sets of methods).  The `emer` package provides a set of interfaces for each structural level (e.g., `emer.Layer` etc) -- any given specific layer must implement all of these methods, and the structural containers (e.g., the list of layers in a network) are lists of these interfaces.  An interface is implicitly a *pointer* to an actual concrete object that implements the interface.  Thus, we typically need to convert this interface into the pointer to the actual concrete type, as in:

```Go
func (nt *Network) InitActs() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(AxonLayer).InitActs()
	}
}
```

* The `emer` interfaces are designed to support generic access to network state, e.g., for the 3D network viewer, but specifically avoid anything algorithmic.  Thus, they allow viewing of any kind of network, including PyTorch backprop nets.

* The `axon.AxonLayer` and `axon.AxonPrjn` interfaces, defined in [axon.go](https://github.com/emer/axon/blob/master/axon/axon.go), extend the `emer` interfaces to virtualize the Axon-specific algorithm functions at the basic level.  These interfaces are used in the base axon code, so that any more specialized version that embeds the basic axon types will be called instead.  See `deep` sub-package for implemented example that does DeepAxon on top of the basic `axon` foundation.

* Layers have a `Shape` property, using the `etensor.Shape` type, which specifies their n-dimensional (tensor) shape.  Standard layers are expected to use a 2D Y*X shape (note: dimension order is now outer-to-inner or *RowMajor* now), and a 4D shape then enables `Pools` as hypercolumn-like structures within a layer that can have their own local level of inihbition, and are also used extensively for organizing patterns of connectivity.

# Pseudocode as a LaTeX doc for Paper Appendix

You can copy the markdown source of this README into a file, and run [pandoc](https://pandoc.org/) on it to convert to LaTeX (or other formats) for inclusion in a paper.  As this page is always kept updated, it is best to regenerate from this source -- very easy:

```bash
curl "https://raw.githubusercontent.com/emer/axon/master/README.md" -o appendix.md
pandoc appendix.md -f gfm -t latex -o appendix.tex
```

You can then edit the resulting .tex file to only include the parts you want, etc.

# Overview of the Axon Algorithm

Axon is the spiking version of [leabra](https://github.com/emer/leabra), which uses rate-code neurons instead of spiking.  Like Leabra, Axon is intended to capture a middle ground between neuroscience, computation, and cognition, providing a computationally effective framework based directly on the biology, to understand how cognitive function emerges from the brain.  See [Computational Cognitive Neuroscience](https://CompCogNeuro.org) for a full textbook on the principles and many implemented models.

First we present a brief narrative overview of axon, followed by a detailed list of all the equations and associated parameters.

## Functional Advantages of Spikes

Aside from biological fidelity, does discrete spiking actually afford any computational / functional advantages over the kind of rate code activation used in Leabra and most abstract neural networks?  Perhaps surprisingly, there isn't a particularly strong answer to this question in the broader field, and more work needs to be done to specifically test and document the differences below (we plan to create an *Axoff* package which is as identical as possible, just with rate codes instead of spikes, to help address this question here).

Meanwhile, based on extensive experience with Axon and Leabra, here are some likely advantages of spikes:

1. **Graded behavior with fast initial responding:** Spiking networks can exhibit both very fast initial communication of information *and* finely graded, proportional responses.  In Leabra, if you turn down the activation gain factor, you can get finely graded responses, but the network is slow to respond as activation takes a while to build and propagate, *or* you can get fast initial responses with high gain, but then it tends to exhibit strongly bimodal, non-graded behavior (especially in the context of bidirectional attractor dynamics).  Simon Thorpe and colleagues have emphasized this point about the significant information value carried in the timing of the first wave of spikes [(Thorpe et al., 1996)](#references).  After this first wave, reasonable rates of subsequent spiking (max around 100 Hz or every 10 msec) can send a graded rate-code-like signal over time.

2. **Graded, permeable attractors:** In rate-code networks, each neuron is continuously broadcasting its graded activation value on every time step, creating a "wall of activation" that is relatively impermeable to new signals.  By contrast, in spiking networks, there are always gaps in the signal, which can allow new information to more easily penetrate and shape the ongoing "conversation" among neurons.  One manifestation of this difference is that Leabra models typically require significant amounts of decay between trials, to allow new inputs to shape the network response, while Axon models do not.  This difference is particularly notable in large, deep networks.

3. **Stochastic sampling:** The thresholded nature of spiking creates natural variability where small differences in timing can be magnified chaos-like into larger differences in which neurons end up spiking (the first to spike generally inhibits other neurons), contributing to an observed high level of variability in spiking network responding approximated by the Poisson distribution.  This is a "natural" form of stochasticity without requiring actual added random noise, and it may be that it ends up retaining more of the relevant underlying signal as a result.  Specifically, attempts to add noise in Leabra never ended up improving network performance, presumably because it directly diluted the signal with noise.  By contrast, the natural stochasticity in Axon networks appears to do a good job of probabilistically sampling noise distributions [(McKee et al, 2021)](#references), and still allows high levels of performance (although consistently perfect responding is generally not observed; nor is it in people or other animals).

4. **Time is a real additional dimension:** Spiking networks exhibit a number of additional time-dependent effects that are absent in rate-code models with their continuous communication, including synchrony, temporal summation, bursting vs. pausing, etc.  A number of theories have speculated about the additional signaling capabilities of synchrony or bursting, e.g., as an additional attentional or binding factor.  In Axon, we don't specifically build in any such mechanisms, and the relevant learning mechanisms required to leverage this additional dimension are not necessarily obviously present in the biology (see the [kinase](https://github.com/ccnlab/kinase/tree/main/sims/kinase) model and discussion). Nevertheless it is likely that there are important emergent temporal dynamics, and the version of the learning rule that *does* afford at least some sensitivity to coincident neural firing does actually work better in practice, so more work needs to be done to understand these issues in the context of the Axon model.

## Activation: AdEx Conductance-based Spiking

* **Spiking AdEx Neurons:** Axon uses the full conductance-based *AdEx* (adapting exponential) discrete spiking model of Gerstner and colleagues ([Wikipedia on  AdEx](https://en.wikipedia.org/wiki/Exponential_integrate-and-fire)), using normalized units as shown here: [google sheet](https://docs.google.com/spreadsheets/d/1jn-NcXY4-y3pOw6inFOgPYlaQodrGIjcsAWkiD9f1FQ/edit?usp=sharing).  Parameterizable synaptic communication delays are also supported, with biologically-based defaults.  Adaptation is implemented in a more realistic manner compared to standard AdEx, using the M-type medium time-scale m-AHP (afterhyperpolarizing) channel, and two longer time scales of sodium-gated potassium channels: `KNa` [(Kaczmarek, 2013)](#references).  Leabra implemented a close rate-code approximation to AdEx.

    AdEx elides the very fast sodium-potassium channels that drive the action potential spiking, as captured in the seminal [Hodgkin & Huxley (1952)](#references) (HH) equations, which are 4th order polynomials, and thus require a very fine-grained time scale below 1 msec and / or more computationally expensive integration methods.  The qualitative properties of these dynamics are instead captured using an exponential function in AdEx, which can be updated at the time scale of 1 msec.
    
    Despite this simplification, AdEx supports neurophysiologically-based conductance equations so that any number of standard channel types can be added to the model, each with their own conductance function.  See [chans](https://github.com/emer/axon/tree/master/chans) for a description of the channel types supported, and the code implementing them, including NMDA and GABA-B as described next.

* **Longer-acting, bistable NMDA and GABA-B currents:** An essential step for enabling spiking neurons to form suitably stable, selective representations for learning was the inclusion of both NMDA and GABA-B channels, which are voltage dependent in a complementary manner as captured in the [Sanders et al, 2013](#References) model (which provided the basis for the implementation here).  These channels have long time constants and the voltage dependence causes them to promote a bistable activation state, with a smaller subset of neurons that have extra excitatory drive from the NMDA and avoid extra inhibition from GABA-B, while a majority of neurons have the opposite profile: extra inhibition from GABA-B and no additional excitation from NMDA.  With stronger conductance levels, these channels can produce robust active maintenance dynamics characteristic of layer 3 in the prefrontal cortex (PFC), but for posterior cortex, we use lower values that produce a weaker, but still essential, form of bistability.  Without these channels, neurons all just "take turns" firing at different points in time, and there is no sense in which a small subset are engaged to represent a specific input pattern -- that had been a blocking failure in all prior attempts to use spiking in Leabra models.    
    
* **Auto-normalized, relatively scaled Excitatory Conductances:** As in Leabra, the excitatory synaptic input conductance (`Ge` in the code, known as *net input* in artificial neural networks) is computed as an average, not a sum, over connections, based on normalized weight values, which are subject to scaling on a projection level to alter relative contributions.  Automatic scaling is performed to compensate for differences in expected activity level in the different projections.  See section on [Input Scaling](#input-scaling) for details.  All of this makes it much easier to create models of different sizes and configurations with minimal (though still non-zero) need for additional parameter tweaking.

## Temporal and Spatial Dynamics of Dendritic Integration

A key dividing line in biological realism of neural models concerns the inclusion of separate dynamics for dendrites versus the soma, with a considerable literature arguing that significant computational functionality arises from nonlinear dynamics in the dendritic integration process.  AdEx is a single compartment "point neuron" model (soma only), and obviously there is a major tradeoff in computational cost associated with modeling dendritic dynamics within individual neurons in any detail.  In Axon, we have taken a middle ground (as usual), by including a separate dendritic membrane potential `VmDend` that better reflects the dynamics of depolarization in the dendrites, relative to the standard AdEx `Vm` which reflects full integration in the soma.  Voltage-gated channels localized in the dendrites, including NMDA and GABA-B, are driven by this VmDend, and doing so results in significantly better performance vs. using the somatic Vm.

Furthermore, synaptic inputs are integrated first by separate projections, and then integrated into the full somatic conductances, and thus it is possible to implement nonlinear interactions among the different dendritic branches where these different projections may be organized.  This is done specifically in the MSN (medium spiny neurons) of the basal ganglia in the `pcore` algorithm, and in the `PT` (pyramidal tract, layer 5IB intrinsic bursting) neurons also implemented in `pcore`.  As noted above, each projection is also subject to different scaling factors, which while still linear, is critical for enabling models to function properly (e.g., top-down projections must in general be significantly weaker than bottom-up projections, to keep the models from hallucinating).  These ways of capturing dendritic dynamics probably capture a reasonable proportion of the relevant functional properties in the biology, but more work with direct comparisons with fully detailed compartmental models is necessary to understand these issues better.  See the [Appendix: Dendritic Dynamics](appendix:-dendritic-dynamics) for more discussion.

## Inhibitory Competition Function Simulating Effects of Interneurons

The pyramidal cells of the neocortex that are the main target of axon models only send excitatory glutamatergic signals via positive-only discrete spiking communication, and are bidirectionally connected.  With all this excitation, it is essential to have pooled inhibition to balance things out and prevent runaway excitatory feedback loops.  Inhibitory competition provides many computational benefits for reducing the dimensionality of the neural representations (i.e., *sparse* distributed representations) and restricting learning to only a small subset of neurons, as discussed extensively in the [Comp Cog Neuro textbook](https://CompCogNeuro.org).  It is likely that the combination of positive-only weights and spiking activations, along with inhibitory competition, is *essential* for enabling axon to learn in large, deep networks, where more abstract, unconstrained algorithms like the Boltzmann machine fail to scale (paper TBD).

Inhibition is provided in the neocortex primarily by the fast-spiking parvalbumin positive (PV+) and slower-acting somatostatin positive (SST+) inhibitory interneurons in the cortex [Cardin, 2018](#references).  Instead of explicitly simulating these neurons, a key simplification in Leabra that eliminated many difficult-to-tune parameters and made the models much more robust overall was the use of a summary inhibitory function.  This function directly computes a pooled inhibitory conductance `Gi` as a function of the feedforward (FF) excitation coming into a Pool of neurons, along with feedback (FB) from the activity level within the pool.  Fortuitously, this same [FFFB](https://github.com/emer/axon/tree/master/fffb) function works well with spiking as well as rate code activations, but it has some biologically implausible properties, and also at a computational level requires multiple layer-level `for` loops that interfere with full parallelization of the code.

Thus, we are now using the [FS-FFFB](https://github.com/emer/axon/tree/master/fsfffb) *fast & slow* FFFB function that more explicitly captures the contributions of the PV+ and SST+ interneurons, and is based directly on FF and FB spikes, without requiring access to the internal Ge and Act rate-code variables in each neuron.  See above link for more info.  This function works even better overall than the original FFFB, in addition to providing a much more direct mapping onto the underlying biology.

See the `examples/inhib` model (from the CCN textbook originally) for an exploration of the basic excitatory and inhibitory dynamics in these models, comparing interneurons with FS-FFFB.

## Kinase-based, Trace-enabled Error-backpropagation Learning

A defining feature of Leabra, and Axon, is that learning is **error driven**, using a *temporal difference* to represent the error signal as a difference in two states of activity over time: *minus* (prediction) then *plus* (outcome).  This form of error-driven learning is biologically plausible, by virtue of the use of bidirectional connectivity to convey error signals throughout the network.  Any temporal difference arising anywhere in the network can propagate differences throughout the network -- mathematically approximating error backpropagation error gradients [(O'Reilly, 1996)](#references).

In the original Leabra (and early Axon) formulation, a version of the simple, elegant contrastive hebbian learning (CHL) rule was used:

$$ dW = (x^+ y^+) - (x^- y^-) $$

where the weight change $dW$ is proportional to *difference* of two hebbian-like products of sending (*x*) and receiving (*y*) activity, between the plus and minus phase.

Axon now uses a different formulation of error-driven learning, that accomplishes three major objectives relative to CHL:
1. Enabling greater sensitivity to small error gradients that can accumulate over time, by computing the error in part based on *linear* net-input terms instead of the highly non-linear activation terms in CHL (representing something like a spiking rate).
2. Supporting a temporally-extended *eligibility trace* factor that provides a biologically-plausible way of approximating the computationally-powerful backprop-through-time (BPTT) algorithm [(Bellec et al, 2020)](#references).
3. More directly connecting to the underlying biochemical mechanisms that drive synaptic changes, in terms of calcium-activated *kinases*, as explored in the more biophysically detailed [kinase](https://github.com/ccnlab/kinase/tree/main/sims/kinase) model.

The detailed derivation of this *kinase trace* learning mechanism is provided below: [Appendix: Kinase-Trace Learning Rule Derivation](#appendix:-kinase-trace-learning-rule-derivation), and summarized here.

Also, as originally developed in Leabra, the `deep` package implements the extra anatomically-motivated mechanisms for *predictive* error-driven learning [OReilly et al., 2021](https://ccnlab.org/papers/OReillyRussinZolfagharEtAl21.pdf), where the minus phase represents a prediction and the plus phase represents the actual outcome.  Biologically, we hypothesize that the two pathways of connectivity into the Pulvinar nucleus of the thalamus convey a top-down prediction and a bottom-up ground-truth outcome, providing an abundant source of error signals without requiring an explicit teacher.

### Error Gradient and Credit Assignment

Mathematically, error-driven learning has two components: the **error gradient**, which reflects the contribution of the receiving neuron to the overall network error, and the **credit assignment** factor that determines how much *credit / blame* for this error gradient to assign to each sending neuron:

```
dW = Error * Credit
```

In the simplest form of error-driven learning, the *delta rule*, these two terms are:

$$ dW = (y^+ - y^-) x $$

where $y^+$ is the *target* activity of the receiving neuron in the plus phase vs. its *actual* activity $y^-$ in the minus phase (this difference representing the *Err* gradient), and $x$ is the sending neuron activity, which serves as the credit assignment.  Thus, more active senders get more of the credit / blame (and completely inactive neurons escape any).

In the mathematics of backpropagation, with a rearrangement of terms as in [(O'Reilly, 1996)](#references), the Error gradient factor in a bidirectionally-connected network can be computed as a function of the difference between *net input* like terms in the plus and minus phases, which are the dot product of sending activations times the weights:

$$ g = \sum_i w_i x_i $$

as:

$$ Error = (g^+ - g^-) $$

while the credit assignment factor is the sending activation times the derivative of the receiving activation:

$$ Credit = x_i y' $$

### Error Gradient: Linear Net-Input and Nonlinear Activation

The actual form of error-gradient term used in Axon includes a contribution of the receiving unit activity in addition to the net-input term:

$$ Error = (g^+ + \gamma y^+) - (g^- + \gamma y^-) $$

where $\gamma$ (about .3 in effect by default) weights the contribution of receiving activity to the error term, relative to the net input.

The inclusion of the receiving activation, in deviation from the standard error backpropagation equations, is necessary for learning to work effectively in the context of inhibitory competition and sparse distributed representations, where neurons have a highly nonlinear activation dynamic, and most are not active at all, despite having significant excitatory net input.  In particular, it is often the case that a given neuron will be out-competed (inhibited) in the plus phase by other neurons, despite having relatively consistent levels of excitatory input across both phases.  By including the activations in the error signal factor, this change in receiving activity level will show up in the error term, and cause the weight changes to reflect the fact that this neuron may not be as useful for the current trial.  On the other hand, using purely activation-based terms in the error signal as in the CHL version above makes the learning *too* nonlinear and loses the beneficial graded nature of the linear net-input based gradient, where small error gradients can accumulate over time to continually drive learning.  In models using CHL, learning would often just get "stuck" at a given intermediate level.  Thus, as is often the case, a balanced approach combining both factors works best.

It is notable that the standard backpropagation equations do *not* include a contribution of the receiving activity in the error signal factor, and thus drive every unit to learn based on linearly re-weighted versions of the same overall error gradient signal.  However, a critical difference between backprop and Axon is that backprop nets rely extensively on negative synaptic weight values, which can thus change the sign of the error-gradient factor.  By contrast, the positive-only weights in Axon mean that the net-input factor is monotonically and positively related to the strength of the sending activations, resulting in much more homogenous error signals across the layer.  Thus, the inclusion of the activation term in the error-signal computation can also be seen as a way of producing greater neural differentiation in the context of this positive-only weight constraint.

From a biological perspective, the combination of net-input and receiving activity terms captures the two main sources of Ca in the receiving neuron's dendritic spines: NMDA and VGCCs (voltage-gated calcium channels).  NMDA channels are driven by sending-neuron glutamate release as in the net-input factor, and VGCCs are driven exclusively by receiving neuron spiking activity.  Although the NMDA channel also reflects receiving neuron membrane depolarization due to the need for Mg (magnesium) ion unblocking, in practice there is often sufficient depolarization in the dendritic compartments, such that Ca influx is mostly a function of sending activity.  Furthermore, the receiving depolarization factor in NMDA is also captured by the receiving activity term, which thus reflects both VGCC and the receiving activity dependent aspect of the NMDA channel.  See below for more details, along with the [kinase](https://github.com/ccnlab/kinase/tree/main/sims/kinase) info.

### Credit Assignment: Temporal Eligibility Trace

The extra mathematical steps taken in [(O'Reilly, 1996)](#references) to get from backpropagation to the CHL algorithm end up eliminating the factorization of the learning rule into clear Error vs. Credit terms.  While this produces a nice simple equation, it makes it essentially impossible to apply the results of [Bellec et al., (2020)](#references), who showed that backprop-through-time can be approximated through the use of a credit-assignment term that serves as a kind of temporal *eligibility trace*, integrating sender-times-receiver activity over a window of prior time steps.

By adopting the above form of error gradient term, we can now also adopt this trace-based credit assignment term as:

$$ Credit = < x y >_t y' $$

where the angle-bracket expression indicates an exponential running-average time-integration of the sender and receiver activation product over time.

The most computationally-effective form of learning goes one step further in computing this credit-assignment trace factor, by integrating spike-driven activity traces (representing calcium currents) within a given theta cycle of activity, in addition to integrating across multiple such theta-cycle "trials" of activity for the eligibility trace.  This is significantly more computationally-expensive, as it requires synapse-level integration of the calcium currents on the millisecond-level timescale (a highly optimized form of computation is used so updates occur only at the time of spiking, resulting in a roughly 2x increase in computational cost overall).

The final term in the credit assignment factor is the derivative of the receiving activation function, $y'$, which would be rather difficult to compute exactly for the actual AdEx spiking dynamics used in Axon.  Fortunately, using the derivative of a sigmoid-shaped logistic function works very well in practice, and captures the essential functional logic for this derivative: learning should be maximized when the receiving neuron is in its most sensitive part of its activation function (i.e., when the derivative is the highest), and minimized when it is basically "pinned" against either the upper or lower extremes.  Specifically the derivative of the logistic is:

$$ y' = y (1-y) $$

which is maximal at y = .5 and zero at either 0 or 1.  In Axon, this is computed using a time-integrated spike-driven Ca-like term (`CaSpkD`), with the max value across the layer used instead of the fixed 1 constant.  In addition, it is useful to use an additional factor that reflects the normalized difference in receiving spiking across the minus and plus phase, which can be thought of as an empirical measure of the sensitivity of the receiving neuron to changes over time:

$$ y' = y (1-y) \frac{y^+ - y^-}{MAX(y^+, y^-)} $$

## Stabilization and Rescaling Mechanisms

A collection of biologically-motivated mechanisms are used to provide a stronger "backbone" or "spine" for the otherwise somewhat "squishy" learning that emerges from the above error-driven learning mechanisms, serving to stabilize learning over longer time scales, and prevent parasitic positive feedback loops that otherwise plague these bidirectionally-connected networks.  These positive feedback loops emerge because the networks tend to settle into stable attractor states due to the bidirectional, generally symmetric connectivity, and there is a tendency for a few such states to get broader and broader, capturing more and more of the "representational space".  The credit assignment process, which is based on activation, contributes to this "rich get richer" dynamic where the most active neurons experience the greatest weight changes.  We colloquially refer to this as the "hog unit" problem, where a small number of units start to hog the representational space, and it represents a major practical barrier to effective learning if not managed properly.  Note that this problem does not arise in the vast majority of purely feedforward networks used in the broader neural network field, which do not exhibit attractor dynamics.  However, this kind of phenomenon is problematic in other frameworks with the potential for such positive feedback loops, such as on-policy reinforcement learning or generative adversarial networks. 

Metaphorically, various forms of equalizing taxation and wealth redistribution are required to level the playing field.  The set of stabilizing, anti-hog mechanisms in Axon include:

1. **SWt:** structural, slowly-adapting weights.  In addition to the usual learning weights driven by the above equations, we introduce a much more slowly-adapting, multiplicative `SWt` that represents the biological properties of the dendritic *spine* -- these SWts "literally" give the model a spine!  Spines are structural complexes where all the synaptic machinery is organized, and they slowly grow and shrink via genetically-controlled, activity-dependent protein remodeling processes, primarily involving the *actin* fibers also found in muscles.  The SWt is multiplicative in the sense that larger vs. smaller spines provide more or less room for the AMPA receptors that constitute the adaptive weight value.  The net effect is that the more rapid trial-by-trial weight changes are constrained by this more slowly-adapting multiplicative factor, preventing more extreme changes.  Furthermore, the SWt values are constrained by a zero-sum dynamic relative to the set of receiving connections into a given neuron, preventing the neuron from increasing all of its weights higher and hogging the space.

2. **Target activity levels:** There is extensive evidence from Gina Turrigiano and collaborators, among others, that synapses are homeostatically rescaled to maintain target levels of overall activity, which vary across individual neurons (e.g., [Torrado Pacheco et al., 2021](#references)).  Axon simulates this process, at the same slower timescale as updating the SWts, which are also involved in the rescaling process.  The target activity levels can also slowly adapt over time, but this adaptation is typically subject to a zero-sum constraint, so any increase in activity in one neuron must be compensated for by reductions elsewhere.

3. **Zero-sum weight changes:** In some cases it can also be useful to constrain the faster error-driven weight changes to be zero-sum, which is supported by an optional parameter.  This zero-sum logic was nicely articulated by [Schraudolph (1998)](#references), and is implemented in the widely-used ResNet models.

4. **Soft bounding and contrast enhancement:** To keep individual weight magnitudes bounded, we use a standard exponential-approach "soft bounding" dynamic (increases are multiplied by $1-w$; decreases by $w$).  In addition, as developed in the Leabra model, it is useful to add a *contrast enhancement* mechanism to counteract the compressive effects of this soft bounding, so that effective weights span the full range of weight values.

# Implementation strategy

* Prjn integrates conductances for each recv prjn
* Send spike adds wt to ring buff at given offset
* For all the spikes that arrive that timestep, increment G += wts
* build in psyn with probability of failure -- needs to be sensitive to wt
* then decay with tau
* integrate all the G's in to cell body
* support background balanced G, I current on cell body to diminish sensitivity! Init

* Detailed model of dendritic dynamics relative to cell body, with data -- dendrites a bit slower, don't reset after spiking?  GaoGrahamZhouEtAl20


# Axon Algorithm Equations

The pseudocode for Axon is given here, showing exactly how the pieces of the algorithm fit together, using the equations and variables from the actual code.

## Timing

Axon is organized around a 200 msec *theta* cycle (5 Hz), which is perhaps not coincidently the modal peak for the duration of an eye fixation, and can be thought of as two 100 msec *alpha* cycles, which together comprise the minimal unit for predictive error driven learning according to the [deep](https://github.com/emer/axon/blob/master/deep) predictive learning framework.  Note that Leabra worked well with just a 100 msec alpha cycle, but it has not been possible to get the temporal difference error-driven learning mechanism to work at that scale, while it works very well at 200 msec.  The following terms are used:

* A **Theta Cycle** or **Trial** or lasts 200 msec (5 Hz, theta frequency), and comprises one sequence of expectation -- outcome learning, with the final 50 msec comprising the *plus* phase when the outcome is active, while the preceding 150 msec is the *minus* phase when the network generates its own prediction or expectation.

* A **Cycle** represents 1 msec of processing, where each neuron is fully updated including all the conductances, integrated into the `Vm` membrane potential, which can then drive a `Spike` if it gets over threshold.

## Variables

The [`axon.Neuron`](https://github.com/emer/axon/blob/master/axon/neuron.go) struct contains all the neuron (unit) level variables, and the [`axon.Layer`](https://github.com/emer/axon/blob/master/axon/layer.go) contains a simple Go slice of these variables.  Optionally, there can be [`axon.Pool`](https://github.com/emer/axon/blob/master/axon/pool.go) pools of subsets of neurons that correspond to hypercolumns, and support more local inhibitory dynamics.

* `Spike` = whether neuron has spiked or not on this cycle (0 or 1)
* `Spiked` = 1 if neuron has spiked within the last 10 cycles (msecs), corresponding to a nominal max spiking rate of 100 Hz, 0 otherwise -- useful for visualization and computing activity levels in terms of average spiked levels
* `Act` = rate-coded activation value reflecting instantaneous estimated rate of spiking, based on 1 / ISIAvg.  This drives feedback inhibition in the FFFB function (todo: this will change when better inhibition is implemented), and is integrated over time for ActInt which is then used for performance statistics and layer average activations, etc.  Should not be used for learning or other computations.
* `ActInt` = integrated running-average activation value computed from Act to produce a longer-term integrated value reflecting the overall activation state across a reasonable time scale to reflect overall response of network to current input state -- this is copied to ActM and ActP at the ends of the minus and plus phases, respectively, and used in computing performance-level statistics (which are typically based on ActM).  Should not be used for learning or other computations.
* `ActM` = ActInt activation state at end of third quarter, representing the posterior-cortical minus phase activation -- used for statistics and monitoring network performance. Should not be used for learning or other computations.
* `ActP` = ActInt activation state at end of fourth quarter, representing the posterior-cortical plus_phase activation -- used for statistics and monitoring network performance.  Should not be used for learning or other computations.
* `Ext` = external input: drives activation of unit from outside influences (e.g., sensory input).
* `Target` = target value: drives learning to produce this activation value.
* `GeSyn` = time-integrated total excitatory synaptic conductance, with an instantaneous rise time from each spike (in GeRaw) and exponential decay with Dt.GeTau, aggregated over projections -- does *not* include Gbar.E.
* `Ge` = total excitatory conductance, including all forms of excitation (e.g., NMDA) -- does *not* include Gbar.E.
* `GiSyn` = time-integrated total inhibitory synaptic conductance, with an instantaneous rise time from each spike (in GiRaw) and exponential decay with Dt.GiTau, aggregated over projections -- does *not* include Gbar.I.  This is added with computed FFFB inhibition to get the full inhibition in Gi.
* `Gi` = total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I.
* `Gk` = total potassium conductance, typically reflecting sodium-gated potassium currents involved in adaptation effects -- does *not* include Gbar.K.
* `Inet` = net current produced by all channels -- drives update of Vm.
* `Vm` = membrane potential -- integrates Inet current over time.
* `VmDend` = dendritic membrane potential -- has a slower time constant, is not subject to the VmR reset after spiking.
* `CaSyn` = spike-driven calcium trace for synapse-level Ca-driven learning: exponential integration of SpikeG * Spike at SynTau time constant (typically 30).  Synapses integrate send.CaSyn * recv.CaSyn across M, P, D time integrals for the synaptic trace driving credit assignment in learning. Time constant reflects binding time of Glu to NMDA and Ca buffering postsynaptically, and determines time window where pre * post spiking must overlap to drive learning.
* `CaSpkM` = spike-driven calcium trace used as a neuron-level proxy for synpatic credit assignment factor based on time-integrated spiking: exponential integration of SpikeG * Spike at MTau time constant (typically 5).  Simulates a calmodulin (CaM) like signal at the most abstract level.
* `CaSpkP` = cascaded integration of CaSpkM at PTau time constant (typically 40), representing neuron-level purely spiking version of plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act.
* `CaSpkD` = cascaded integration CaSpkP at DTau time constant (typically 40), representing neuron-level purely spiking version of minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule. Used for specialized learning and computational functions, statistics, instead of Act.
* `CaSpkPM`= minus-phase snapshot of the CaSpkP value -- similar to ActM but using a more directly spike-integrated value.
* `CaLrn` = recv neuron calcium signal used to drive temporal error difference component of standard learning rule, combining NMDA (NmdaCa) and spiking-driven VGCC (VgccCaInt) calcium sources (vs. CaSpk* which only reflects spiking component).  This is integrated into CaM, CaP, CaD, and temporal derivative is CaP - CaD (CaMKII - DAPK1).  This approximates the backprop error derivative on net input, but VGCC component adds a proportion of recv activation delta as well -- a balance of both works best.  The synaptic-level trace multiplier provides the credit assignment factor, reflecting coincident activity and potentially integrated over longer multi-trial timescales.
* `CaM` = integrated CaLrn at MTau timescale (typically 5), simulating a calmodulin (CaM) like signal, which then drives CaP, CaD for delta signal driving error-driven learning.
* `CaP` = cascaded integration of CaM at PTau time constant (typically 40), representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule.
* `CaD` = cascaded integratoin of CaP at DTau time constant (typically 40), representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule.
* `CaDiff` = difference between CaP - CaD -- this is the error signal that drives error-driven learning.
* `SpkMaxCa` = Ca integrated like CaSpkP but only starting at MacCycStart cycle, to prevent inclusion of carryover spiking from prior theta cycle trial -- the PTau time constant otherwise results in significant carryover.
* `SpkMax` = maximum CaSpkP across one theta cycle time window -- used for specialized algorithms that have more phasic behavior within a single trial, e.g., BG Matrix layer gating.  Also useful for visualization of peak activity of neurons.
* `SpkPrv` = final CaSpkD activation state at end of previous theta cycle.  used for specialized learning mechanisms that operate on delayed sending activations.
* `SpkSt1` = the activation state at specific time point within current state processing window (e.g., 50 msec for beta cycle within standard theta cycle), as saved by SpkSt1() function.  Used for example in hippocampus for CA3, CA1 learning.
* `SpkSt2` = the activation state at specific time point within current state processing window (e.g., 100 msec for beta cycle within standard theta cycle), as saved by SpkSt2() function.  Used for example in hippocampus for CA3, CA1 learning.
* `RLrate` = recv-unit based learning rate multiplier, reflecting the sigmoid derivative computed from the CaSpkD of recv unit, and the normalized difference CaSpkP - CaSpkD / MAX(CaSpkP - CaSpkD).
* `ActAvg` = average activation (of minus phase activation state) over long time intervals (time constant = Dt.LongAvgTau) -- useful for finding hog units and seeing overall distribution of activation.
* `AvgPct`= ActAvg as a proportion of overall layer activation -- this is used for synaptic scaling to match TrgAvg activation -- updated at SlowInterval intervals
* `TrgAvg` = neuron's target average activation as a proportion of overall layer activation, assigned during weight initialization, driving synaptic scaling relative to AvgPct.
* `DTrgAvg` = change in neuron's target average activation as a result of unit-wise error gradient -- acts like a bias weight.  MPI needs to share these across processors.
* `AvgDif` = AvgPct - TrgAvg -- i.e., the error in overall activity level relative to set point for this neuron, which drives synaptic scaling -- updated at SlowInterval intervals.
* `Attn` = Attentional modulation factor, which can be set by special layers such as the TRC -- multiplies Ge.
* `ISI` = current inter-spike-interval -- counts up since last spike.  Starts at -1 when initialized.
* `ISIAvg` = average inter-spike-interval -- average time interval between spikes, integrated with ISITau rate constant (relatively fast) to capture something close to an instantaneous spiking rate.  Starts at -1 when initialized, and goes to -2 after first spike, and is only valid after the second spike post-initialization.
* `GeNoiseP` = accumulating poisson probability factor for driving excitatory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda.
* `GeNoise` = integrated noise excitatory conductance, added into Ge.
* `GiNoiseP` = accumulating poisson probability factor for driving inhibitory noise spiking -- multiply times uniform random deviate at each time step, until it gets below the target threshold based on lambda.
* `GiNoise` = integrated noise inhibotyr conductance, added into Gi.
* `GeM` = time-averaged Ge value over the minus phase -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive.
* `GiM` = time-averaged GiSyn value over the minus phase -- useful for stats to set strength of connections etc to get neurons into right range of overall excitatory drive.
* `MahpN` = accumulating voltage-gated gating value for the medium time scale AHP.
* `SahpCa` = slowly accumulating calcium value that drives the slow AHP.
* `SahpN` = sAHP gating value.
* `GknaMed` = conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick) -- produces accommodation / adaptation of firing.
* `GknaSlow` = conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack) -- produces accommodation / adaptation of firing.
* `GnmdaSyn` = integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant.
* `Gnmda` = net postsynaptic (recv) NMDA conductance, after Mg V-gating and Gbar -- added directly to Ge as it has the same reversal potential.
* `GnmdaLrn` = learning version of integrated NMDA recv synaptic current -- adds GeRaw and decays with time constant -- drives NmdaCa that then drives CaM for learning.
* `NmdaCa` = NMDA calcium computed from GnmdaLrn, drives learning via CaM.
* `SnmdaO` = Sender-based number of open NMDA channels based on spiking activity and consequent glutamate release for all sending synapses -- this is the presynaptic component of NMDA activation that can be used for computing Ca levels for learning -- increases by (1-SnmdaI)*(1-SnmdaO) with spiking and decays otherwise.
* `SnmdaI` = Sender-based inhibitory factor on NMDA as a function of sending (presynaptic) spiking history, capturing the allosteric dynamics from Urakubo et al (2008) model.  Increases to 1 with every spike, and decays back to 0 with its own longer decay rate.
* `GgabaB` = net GABA-B conductance, after Vm gating and Gbar + Gbase -- applies to Gk, not Gi, for GIRK, with .1 reversal potential.
* `GABAB` = GABA-B / GIRK activation -- time-integrated value with rise and decay time constants.
* `GABABx` = GABA-B / GIRK internal drive variable -- gets the raw activation and decays.
* `Gvgcc` = conductance (via Ca) for VGCC voltage gated calcium channels.
* `VgccM` = activation gate of VGCC channels.
* `VgccH` = inactivation gate of VGCC channels.
* `VgccCa` = instantaneous VGCC calcium flux -- can be driven by spiking or directly from Gvgcc.
* `VgccCaInt` = time-integrated VGCC calcium flux -- this is actually what drives learning.
* `GeExt` = extra excitatory conductance added to Ge -- from Ext input, deep.GeCtxt etc.
* `GeRaw` = raw excitatory conductance (net input) received from senders = current raw spiking drive.
* `GeBase` = baseline level of Ge, added to GeRaw, for intrinsic excitability.
* `GiRaw` = raw inhibitory conductance (net input) received from senders  = current raw spiking drive.
* `GiBase` = baseline level of Gi, added to GiRaw, for intrinsic excitability.
* `SSGi` = SST+ somatostatin positive slow spiking inhibition.
* `SSGiDend` = amount of SST+ somatostatin positive slow spiking inhibition applied to dendritic Vm (VmDend).
* `Gak` = conductance of A-type K potassium channels.

Neurons are connected via synapses parameterized with the following variables, contained in the [`axon.Synapse`](https://github.com/emer/axon/blob/master/axon/synapse.go) struct.  The [`axon.Prjn`](https://github.com/emer/axon/blob/master/axon/prjn.go) contains all of the synaptic connections for all the neurons across a given layer -- there are no Neuron-level data structures in the Go version.

* `Wt` = effective synaptic weight value, determining how much conductance one spike drives on the receiving neuron.  Wt = SWt * WtSig(LWt), where WtSig produces values between 0-2 based on LWt, centered on 1
* `SWt` = slowly adapting structural weight value, which acts as a multiplicative scaling factor on synaptic efficacy: biologically represents the physical size and efficacy of the dendritic spine, while the LWt reflects the AMPA receptor efficacy and number.  SWt values adapt in an outer loop along with synaptic scaling, with constraints to prevent runaway positive feedback loops and maintain variance and further capacity to learn.  Initial variance is all in SWt, with LWt set to .5, and scaling absorbs some of LWt into SWt.
* `LWt` = rapidly learning, linear weight value -- learns according to the lrate specified in the connection spec.  Initially all LWt are .5, which gives 1 from WtSig function, 
* `DWt` = change in synaptic weight, from learning
* `DSWt` = change in SWt slow synaptic weight -- accumulates DWt
* `Ca` = Raw calcium singal for Kinase learning: SpikeG * (send.CaSyn * recv.CaSyn)
* `CaM` = first stage running average (mean) Ca calcium level (like CaM = calmodulin), feeds into CaP
* `CaP` = shorter timescale integrated CaM value, representing the plus, LTP direction of weight change and capturing the function of CaMKII in the Kinase learning rule
* `CaD` = longer timescale integrated CaP value, representing the minus, LTD direction of weight change and capturing the function of DAPK1 in the Kinase learning rule
* `Tr` = trace of synaptic activity over time -- used for credit assignment in learning.

## Activation Update Cycle (every 1 msec): Ge, Gi, Vm, Spike

The `axon.Network` `CycleImpl` method in [`axon/network.go`](https://github.com/emer/axon/blob/master/axon/network.go) calls the following functions in order:

* `GFmSpikes` on all `Prjn`s: integrates Raw and Syn conductances for each Prjn from spikes sent previously, into `GVals` organized by receiving neuron index, so they can then be integrated into the full somatic conductances in CycleNeuron.

* `GiFmSpikes` on all `Layer`s: computes inhibitory conductances based on total incoming FF and FB spikes into the layer, using the [FS-FFFB](https://github.com/emer/axon/tree/master/fsfffb) summary functions.

* `CycleNeuron` on all `Neuron`s: integrates the Ge and Gi conductances from above, updates all the other channel conductances as described in [chans](https://github.com/emer/axon/tree/master/chans), and then computes `Inet` as the net current from all these conductances, which then drives updates to `Vm` and `VmDend`.  If `Vm` exceeds threshold then `Spike` = 1.

* `SendSpike` on all `Neuron`s: for each neuron with `Spike` = 1, adds scaled synaptic weight value to `GBuf` ring buffer for efficiently delaying receipt of the spike per parametrized `Com.Delay` cycles.  This is what the `GFmSpikes` then integrates.  This is very expensive computationally because it goes through synapses on each msec Cycle.

* `CyclePost` on all `Layer`s: a hook for specialized algorithms to do something special.

* `SendSynCa` and `RecvSynCa` on all `Prjn`s: update synapse-level calcium (Ca) for any neurons that spiked (on either the send or recv side).  This is very expensive computationally because it goes through synapses on each msec Cycle.

### Cycle Equations

All of the relevant parameters are in the `axon.Layer.Act` and `Inhib` fields, which are of type `ActParams` and `InhibParams` -- in this Go version, the parameters have been organized functionally, not structurally, into three categories.

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

        


* **SWt: structural, slowly-adapting weights:** Biologically, the overall synaptic efficacy is determined by two major factors: the number of excitatory AMPA receptors (which adapt rapidly in learning), and size, number, and shape of the dendritic spines (which adapt more slowly, requiring protein synthesis, typically during sleep).  It was useful to introduce a slowly-adapting structural component to the weight, `SWt`, to complement the rapidly adapting `LWt` value (reflecting AMPA receptor expression), which has a multiplicative relationship with `LWt` in determining the effective `Wt` value.  The `SWt` value is typically initialized with all of the initial random variance across synapses, and it learns on the `SlowInterval` (100 iterations of faster learning) on the zero-sum accumulated `DWt` changes since the last such update, with a slow learning rate (e.g., 0.001) on top of the existing faster learning rate applied to each accumulated `DWt` value.  This slowly-changing SWt value is critical for preserving a background level of synaptic variability, even as faster weights change.  This synaptic variability is essential for prevent degenerate hog-unit representations from taking over, while learning slowly accumulates effective new weight configurations.  In addition, a small additional element of random variability, colorfully named `DreamVar`, can be injected into the active `LWt` values after each `SWt` update, which further preserves this critical resource of variability driving exploration of the space during learning.

* **Soft weight bounding and contrast enhancement:**  Weights are subject to a contrast enhancement function, which compensates for the soft (exponential) weight bounding that keeps weights within the normalized 0-1 range.  Contrast enhancement is important for enhancing the selectivity of learning, and generally results in faster learning with better overall results.  Learning operates on the underlying internal linear weight value, `LWt`, while the effective overall `Wt` driving excitatory conductances is the contrast-enhanced version.  Biologically, we associate the underlying linear weight value with internal synaptic factors such as actin scaffolding, CaMKII phosphorlation level, etc, while the contrast enhancement operates at the level of AMPA receptor expression.



* **Zero-sum weight changes:** In Axon, we enforce zero-sum synaptic weight changes, such that the mean synaptic weight change across the synapses in a given neuron's receiving projection is subtracted from all such computed weight changes.  The `Prjn.Learn.XCal.SubMean` parameter controls how much of the mean `DWt` value is subtracted: 1 = full zero sum, which works the best.  There is significant biological data in support of such a zero-sum nature to synaptic plasticity (cites).  Functionally, it is a key weapon against the positive-feedback loop "hog unit" problems.

* **Target intrinsic activity levels:** Schraudolph notes that a *bias weight* is ideal for absorbing all of the DC or main effect signal for a neuron.  Biologically, Gina Turrigiano has discovered extensive evidence for *synaptic scaling* mechanisms that dynamically adjust for changes in overall excitability or mean activity of a neuron across longer time scales (e.g., as a result of monocular depravation experiments).  Furthermore, she found that there is a range of different target activity levels across neurons, and that each neuron has a tendency to maintain its own target level.   In Axon, we implemented a simple form of this mechanism by: 1. giving each neuron its own target average activity level (`TrgAvg`); 2. monitoring its activity over longer time scales (`ActAvg` and `AvgPct` as a normalized value); and 3. scaling the overall synaptic weights (`LWt`) to maintain this level by adapting in a soft-bounded way as a function of `TrgAvg - AvgPct`.

    This is similar to a major function performed by the BCM learning algorithm in the Leabra framework, but we found that by moving this mechanism into a longer time-scale outer-loop mechanism (consistent with Turigiano's data), it worked much more effectively.  By contrast, the BCM learning ended up interfering with the error-driven learning signal, and required relatively quick time-constants to adapt responsively as a neuron's activity started to change.  To allow this target activity signal to adapt in response to the DC component of a neuron's error signal, the local plus - minus phase difference (`ActDiff`) drives (in a zero-sum manner) changes in `TrgAvg`, subject also to Min / Max bounds.  These params are on the `Layer` in `Learn.SynScale`.
        
        
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

# clippings




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


# Appendix: Specialized BG / PFC / DA / Etc Algorithms

There are various extensions to the algorithm that implement special neural mechanisms associated with the prefrontal cortex and basal ganglia [PBWM](#pbwm), dopamine systems [PVLV](#pvlv), the [Hippocampus](#hippocampus), and predictive learning and temporal integration dynamics associated with the thalamocortical circuits [DeepAxon](#deepaxon).  All of these are (will be) implemented as additional modifications of the core, simple `axon` implementation, instead of having everything rolled into one giant hairball as in the original C++ implementation.

This repository contains specialized additions to the core algorithm described here:
* [deep](https://github.com/emer/axon/blob/master/deep) has the DeepAxon mechanisms for simulating the deep neocortical <-> thalamus pathways (wherein basic Axon represents purely superficial-layer processing)
* [pbwm](https://github.com/emer/axon/blob/master/rl) has basic reinforcement learning models such as Rescorla-Wagner and TD (temporal differences).
* [pbwm](https://github.com/emer/axon/blob/master/pbwm1) has the prefrontal-cortex basal ganglia working memory model (PBWM).
* [hip](https://github.com/emer/axon/blob/master/hip) has the hippocampus specific learning mechanisms.



# Appendix: Kinase-Trace Learning Rule Derivation

To begin, the original *GeneRec* [(OReilly, 1996)](#references) derivation of CHL (contrastive hebbian learning) from error backpropagation goes like this:

$$ \frac{\partial E}{\partial w}=  \frac{\partial E}{\partial y} \frac{\partial y}{\partial g}  \frac{\partial g}{\partial w}$$

where *E* is overall error, *w* is the weight, *y* is recv unit activity, *g* is recv conductance (net input), and *x* is sending activity.  For a simple neural network:

$$ g = \sum x w $$

$$ y = f(g) $$

This chain rule turns into:

$$ dW = \frac{\partial E}{\partial w} =  \left[ \left( \sum_i x_i^+ - \sum_i x_i^- \right )w \right] y' x = (g^+ - g^-) y' x$$

Thus, the *Error* factor is $(g^+ - g^-)$ and $y' x$ is the *Credit* factor.  In words, the error signal is received by each unit in the form of their weighted net input from all other neurons -- the error is the temporal difference in this net input signal between the plus and minus phases.  And the credit assignment factor is the sending unit activity *x* times the derivative of activation function.

The presence of this derivative is critical -- and has many tradeoffs embedded within it, as discussed later (e.g., the ReLU eliminates the derivative by using a mostly linear function, and thereby eliminates the *vanishing gradient* problem that otherwise occurs in sigmoidal activation functions).

The original GeneRec derivation of CHL mixes these factors by approximating the derivative of the activation function using the discrete difference in receiving activation state, such that:

$$ (g^+ - g^-) y' \approx y^+ - y^- $$

In the GeneRec derivation, the approximate midpoint integration method, and symmetry preservation, cause the terms to get mixed together with the sending activations, producing the CHL algorithm.

To derive the new trace-enabling rule, we avoid this mixing, and explore learning using the more separable Error * Credit form.  In practice, the key issue is *on what variable is the temporal difference computed*: just using raw net input turns out to be too diffuse -- the units end up computing too similar of error gradients, and the credit assignment is not quite sufficient to separate them out.

In the Axon framework in particular, the weights are constrained to be positive, and especially at the start of learning, the net input terms are all fairly close in values across units.  The lateral inhibition provides the critical differentiation so that only a subset of neurons are active, and thus having some contribution of the actual receiving activity is critical for a learning rule that ends up having different neurons specializing on different aspects of the problem.  The relative lack of this kind of differential receiver-based credit assignment in backprop nets is a critical difference from the CHL learning rule -- in the GeneRec derivation, it arises from making the learning rule symmetric, so that the credit assignment factor includes both sides of the synapse.

In short, backprop is at one end of a continuum where the only credit assignment factor is presynaptic activity, and existing weights provide a "filter" through which the Error term is processed.  At the other end is the symmetric CHL equation where pre * post (*xy*) is the credit assignment factor in effect, and the "trace" equation is somewhere in between.

# Appendix: Neural data for parameters

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

# Appendix: Dendritic Dynamics

A series of models published around the year 2000 investigated the role of active dendritic channels on signal integration across different dendritic compartments [(Migliore et al, 1999; Poirazi et al, 2003; Jarsky et al, 2005)](#references) -- see [Spruston (2008), Poirazi & Papoutsi (2020)](#references) for reviews.  A common conclusion was that the A-type K channel can be inactivated as a result of elevated Vm in the dendrite, driving a nonlinear gating-like interaction between dendritic inputs: when enough input comes in (e.g., from 2 different projections), then the rest of the inputs are all integrated more-or-less linearly, but below this critical threshold, inputs are much more damped by the active A-type K channels.  There are also other complications associated with VGCC L-type and T-type voltage-gated Ca channels which can drive Ca spikes, to amplify regular AMPA conductances, relative weakness and attenuation of active HH Na spiking channels in dendrites, and issues of where inhibition comes in, etc.  See following figure from Spruston (2008) for a summary of some key traces:

![Differences between Soma vs. Dendrites](fig_dendrite_coinc_spruston08_fig5.png?raw=true "Figure 5 from Spruston (2008), showing mutual dependence on two dendritic compartments for synaptic integration in panel a, and also significant differences in temporal duration of elevated Vm in dendrites vs. soma.")

Here are some specific considerations and changes to capture some of these dynamics:

* The `Kdr` delayed rectifier channel, part of the classical HH model, resets the membrane potential back to resting after a spike -- according to detailed traces from the [Urakubo et al., 2008](https://github.com/ccnlab/kinase/tree/main/sims/urakubo) model with this channel, and the above figures, this is not quite an instantaneous process, with a time constant somewhere between 1-2 msec.  This is not important for overall spiking behavior, but it is important when Vm is used for more realistic Ca-based learning (as in the Urakubo model).  This more realistic `VmR` reset behavior is captured in Axon via the `RTau` time constant, which decays `Vm` back to `VmR` within the `Tr` refractory period of 3 msec, which fits well with the Urakubo traces for isolated spikes.

* The `Dend` params specify a `GbarExp` parameter that applies a fraction of the Exp slope to `VmDend`, and a `GbarR` param that injects a proportional amount of leak current during the spike reset (`Tr`) window to bring the Vm back down a bit, reflecting the weaker amount of `Kdr` out in the dendrites.  This produces traces that resemble the above figure, as shown in the following run of the `examples/neuron` model, comparing VmDend with Vm.  Preliminary indications suggest this has a significant benefit on model performance overall (on ra25 and fsa so far), presumably by engaging NMDA and GABAB channels better.

![VmDend vs Vm in neuron example](fig_axon_neuron_vmdend.png?raw=true "VmDend now has dynamics that reflect weaker trace of soma spiking -- initial results suggest this improves performance by better engaging the NMDA / GABAB channels.")

* As for the broader question of more coincidence-driven dynamics in the dendrites, or an AND-like mutual interdependence among inputs to different branches, driven by A-type K channels, it is likely that in the awake behaving context (*in activo*) as compared to the slices where these original studies were done, there is always a reasonable background level of synaptic input such that these channels are largely inactivated anyway.  This corresponds to the important differences between upstate / downstate that also largely disappear in awake behaving vs. anesthetized or slice preps.  Nevertheless, it is worth continuing to investigate this issue and explore the potential implications of these mechanisms in actual running models.  TODO: create atype channels in glong (rename to something else, maybe just `chans` for channels)


# References

* Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11(1), Article 1. https://doi.org/10.1038/s41467-020-17236-y

* Cardin, J. A. (2018). Inhibitory interneurons regulate temporal precision and correlations in cortical circuits. Trends in Neurosciences, 41(10), 689–700. https://doi.org/10.1016/j.tins.2018.07.015

* Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of Physiology, 117(4), 500–544. https://doi.org/10.1113/jphysiol.1952.sp004764

* Jarsky, T., Roxin, A., Kath, W. L., & Spruston, N. (2005). Conditional dendritic spike propagation following distal synaptic activation of hippocampal CA1 pyramidal neurons. Nat Neurosci, 8, 1667–1676. http://dx.doi.org/10.1038/nn1599

* Kaczmarek, L. K. (2013). Slack, Slick, and Sodium-Activated Potassium Channels. ISRN Neuroscience, 2013. https://doi.org/10.1155/2013/354262

* McKee, K. L., Crandell, I. C., Chaudhuri, R., & O’Reilly, R. C. (2021). Locally learned synaptic dropout for complete Bayesian inference. ArXiv:2111.09780 (q-Bio, Stat). http://arxiv.org/abs/2111.09780

* Migliore, M., Hoffman, D. A., Magee, J. C., & Johnston, D. (1999). Role of an A-Type K+ Conductance in the Back-Propagation of Action Potentials in the Dendrites of Hippocampal Pyramidal Neurons. Journal of Computational Neuroscience, 7(1), 5–15. https://doi.org/10.1023/A:1008906225285

* O’Reilly, R. C. (1996). Biologically plausible error-driven learning using local activation differences: The generalized recirculation algorithm. Neural Computation, 8(5), 895–938. https://doi.org/10.1162/neco.1996.8.5.895

* O’Reilly, R. C., Russin, J. L., Zolfaghar, M., & Rohrlich, J. (2020). Deep Predictive Learning in Neocortex and Pulvinar. ArXiv:2006.14800 (q-Bio). http://arxiv.org/abs/2006.14800

* Poirazi, P., Brannon, T., & Mel, B. W. (2003). Arithmetic of Subthreshold Synaptic Summation in a Model CA1 Pyramidal Cell. Neuron, 37(6), 977–987. https://doi.org/10.1016/S0896-6273(03)00148-X

* Poirazi, P., & Papoutsi, A. (2020). Illuminating dendritic function with computational models. Nature Reviews Neuroscience, 21(6), 303–321. https://doi.org/10.1038/s41583-020-0301-7

* Sanders, H., Berends, M., Major, G., Goldman, M. S., & Lisman, J. E. (2013). NMDA and GABAB (KIR) Conductances: The "Perfect Couple" for Bistability. Journal of Neuroscience, 33(2), 424–429. https://doi.org/10.1523/JNEUROSCI.1854-12.2013

* Schraudolph, N. N. (1998). Centering Neural Network Gradient Factors. In G. B. Orr & K.-R. Müller (Eds.), Neural Networks: Tricks of the Trade (pp. 207–226). Springer. https://doi.org/10.1007/3-540-49430-8_11

* Spruston, N. (2008). Pyramidal neurons: Dendritic structure and synaptic integration. Nature Reviews. Neuroscience, 9(3), 201–221. http://www.ncbi.nlm.nih.gov/pubmed/18270515

* Thorpe, S., Fize, D., & Marlot, C. (1996). Speed of Processing in the Human Visual System. Nature, 381(6582), 520–522.

* Torrado Pacheco, A., Bottorff, J., Gao, Y., & Turrigiano, G. G. (2021). Sleep Promotes Downward Firing Rate Homeostasis. Neuron, 109(3), 530-544.e6. https://doi.org/10.1016/j.neuron.2020.11.001

* Urakubo, H., Honda, M., Froemke, R. C., & Kuroda, S. (2008). Requirement of an allosteric kinetics of NMDA receptors for spike timing-dependent plasticity. The Journal of Neuroscience, 28(13), 3310–3323. http://www.ncbi.nlm.nih.gov/pubmed/18367598


