# Urakubo = Urakubo et al, 2008

This model re-implements the highly biophysically realistic model of spike-timing dependent plasticity from:

* Urakubo, H., Honda, M., Froemke, R. C., & Kuroda, S. (2008). Requirement of an allosteric kinetics of NMDA receptors for spike timing-dependent plasticity. The Journal of Neuroscience, 28(13), 3310–3323. http://www.ncbi.nlm.nih.gov/pubmed/18367598 -- [Main PDF](https://github.com/emer/axon/blob/master/examples/urakubo/UrakuboHondaFroemkeEtAl08.pdf) : [Supplemental PDF](https://github.com/emer/axon/blob/master/examples/urakubo/UrakuboHondaFroemkeEtAl08_suppl.pdf) : [Model Details](https://github.com/emer/axon/blob/master/examples/urakubo/UrakuboHondaFroemkeEtAl08_model_sub.pdf) (last one has all the key details but a few typos and the code is needed to get all the details right).

This model captures the complex postsynaptic chemical cascades that result in the changing levels of AMPA receptor numbers in the postsynaptic density (PSD) of a spine, as a function of the changing levels of calcium (Ca2+) entering the spine through NMDA and VGCC (voltage-gated calcium) channels.  The model mechanisms are strongly constrained bottom-up by the known kinetics of these chemical processes, and amazingly it captures many experimental results in the LTP / LTD experimental literature.  Thus, it provides an extremely valuable resource for understanding how plasticity works and how realistic but much simpler models can be constructed, which are actually usable in larger-scale simulations.

The complexity of this model of a single synaptic spine involves 115 different variables updated according to interacting differential equations requiring a time constant of 5e-5 (20 steps per the typical msec step used in axon).  In previous work with the original Genesis implementation of this model (see [CCN textbook](https://CompCogNeuro.org)), we were able to capture the net effect of this model on synaptic changes, at a rate-code level using random poisson spike trains, using the simple "checkmark" linearized version of the BCM learning rule (called "XCAL" for historical reasons), with an `r` value of around .9 (81% of the variance).  However, with the actual spiking present in axon, there is an important remaining question as to whether a more realistic (yet still simplified) learning mechanism might be able to take advantage of some aspects of the detailed spiking patterns to do "better" somehow than the simplified rate code model.

The original Genesis model is a good object lesson in how a generic modeling framework can obscure what is actually being computed.  The model shows up as a complex soup of crossing lines in the Genesis GUI, and the code is a massive dump of inscrutable numbers, with key bits of actual computation carefully hidden across many different bits of code, with seemingly endless flags and conditions making it very difficult to determine what code path is actually in effect.  In addition, Genesis is ancient software that requires an old gcc compiler to build it, and it runs under X windows.

By contrast, the present model is purpose-built in compact, well-documented highly readable Go code, using a few very simple types in the [emergent chem](https://github.com/emer/emergent/tree/master/chem) package, providing the basic `React` and `Enz` mechanisms used for simulating the chemical reactions.

# Classical STDP Replication

The classic Bi & Poo STDP curve emerges reliably with e.g., 100 reps with 50 secs final "burn in" of the weights -- maximum burn-in occurs after about 100 secs and after then things can start to decay. The curve gets stronger with more reps as well.

![STDPSweep](results/fig_urakubo22_stdp_100rep_final50.png?raw=true "Classic STDP with standard params")
