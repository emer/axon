# pcore_ds

This is a simple test of the [pcore](../../PCORE_BG.md) model of basal ganglia (BG) function, in the **Dorsal Striatum** (DS).  See [pcore_vs](../pcore_vs) for the Ventral Striatum (VS) model, which is optimized for making global Go vs. No decisions based on cost / benefit inputs (see also [BOA](../boa)).

The DS is the input layer for the primary motor control part of the basal ganglia, and this model learns to execute a sequence of motor actions through reinforcement  learning (RL), getting positive reinforcement for correct actions and lack of reinforcement for incorrect ones.  Critically, there is no omnicient "teacher" input: the model has to discover the correct action sequence purely through trial and error, "online" learning (i.e., it learns on a trial-by-trial basis as it acts).  This is the only biologically / ecologically realistic form of RL.

The model also has mechanisms to learn about the space of possible motor actions and their parameterization in the DS, which is driven by ascending projections from the brainstem and spinal motor system, via the deep cerebellar nuclei (DCN) to the CL (central lateral) nucleus of the thalamus.  This projection is not currently beneficial in the model, and will be revisited once a simple model of the cerebellum is implemented, in the context of more fine-grained parameterized motor control.  The descending projection from motor cortex also conveys useful motor signal information, and these cortical reps are directly shaped by the same ascending motor signals.

The model uses a simple dopamine (DA) "critic" system implemented in code, in `mseq_env.go`.  It just adapts a `RewPred` prediction whenever a reward is processed, using a simple learning rate (`RewPredLRate=0.01`).  The `RewPredMin=0.1` is critical to maintain negative DA signals for failures, so these cannot be fully predicted away.  Partial credit is given for partially-correct sequence outputs, which is important for the length-3 sequences but actually somewhat detrimental to the length-2 case.  Partial credit is computed as a *probability* of reward as a function of the number of correct actions: `p = NCorrect / SeqLen`, and the reward value if given is also equal to this `p` value.  If you just continuously give a constant `p` value partial credit, the model learns to expect this and fails to progress further.

These are ecologically-reasonable properties under the assumption that reward is discounted by effort, and random action choices will sometimes result in good outcomes, but generally with additional unnecessary steps.

To simplify the use of a consistent motor sequence across the parallel-data processing used on GPU (NData > 1), we just keep the target sequence as 0,1,2 etc, because the model doesn't know any better, and the random initial weights have no bias either.

# Network Overview

Once the dust settles, a summary of the biology and implementation in the model will be included here.  For now, see this [github Discussion](https://github.com/emer/axon/discussions/326).

# Results

As of 2024-02-29, the default parameters with 49 units (7x7) per layer result in:

* 22/25 learn on SeqLen=4, NActions=5, which has 5^4 = 625 total space to be searched
* 24/25 learn 10^3 = 1000 total space
* 25/25 learn on SeqLen=3, NActions=5 (5^3 = 125) with 25 units, very quickly and reliably (most in 3-5 epochs; ~5 avg).

The params have been relatively thoroughly "tweaked" at this point: see paramtweak.go for tweaking code.

The learned weights to the BG clearly show that it is disinhibiting the appropriate action at each step in the sequence.

# TODO:

* Set number of cycles per trial in terms of BG motor gating timing: constant offset from onset of VM gating timing, with a cutoff for "nothing happening" trials.

* "CL" not beneficial (implemented as direct MotorBS -> Matrix projections): rel weight of 0.002 is OK but starts to actually impair above that.  Likely that a functional cerebellum is needed to make this useful.  Also, investigate other modulatory inputs to CL that might alter its signal.  Key ref for diffs between CL and PF: LaceyBolamMagill07: C. J. Lacey, J. P. Bolam, P. J. Magill, Novel and distinct operational principles of intralaminar thalamic neurons and their striatal projections. J. Neurosci. 27, 4374–4384 (2007).

* Local patch (striosome) learning modulation.  The striosomes clearly provide a direct modulation of SNc dopamine on a "regional" basis.  They receive input preferentially from BOA "limbic" areas, but also potentially some of the same inputs as the local matrix MSNs.  There is also some PF-based circuitry and cell types of relevance.  It is possible that these circuits could provide something like a local partial credit signal, or DA in proportion to whether an action is expected or unexpected.  There is evidence for considerable ongoing micro-fluctuations in local DA, which is related to subsequent action selection: **MarkowitzGillisJayEtAI23**. 

* Learning in the other parts of the pcore circuit -- might help auto-adjust the parameters.  Need to figure out what logical learning rules would look like.


