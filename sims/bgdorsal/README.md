# BG Dorsal

This is a test of the [pcore](../../PCORE_BG.md) model of basal ganglia (BG) function, in the **Dorsal Striatum** (DS).  See [bgventral](../bgventral) for the Ventral Striatum (VS) model, which is optimized for making global Go vs. No decisions based on cost / benefit inputs (see also [Rubicon](../../Rubicon.md).

The DS is the input layer for the primary motor control part of the basal ganglia, and this model learns to execute a sequence of motor actions through reinforcement  learning (RL), getting positive reinforcement for correct actions and lack of reinforcement for incorrect ones.  Critically, there is no omnicient "teacher" input: the model has to discover the correct action sequence purely through trial and error, "online" learning (i.e., it learns on a trial-by-trial basis as it acts). This is the only biologically / ecologically realistic form of RL.

The model also has mechanisms to learn about the space of possible motor actions and their parameterization in the DS, which is driven by ascending pathways from the brainstem and spinal motor system, via the deep cerebellar nuclei (DCN) to the CL (central lateral) nucleus of the thalamus.  This pathway is not currently beneficial in the model, and will be revisited once a simple model of the cerebellum is implemented, in the context of more fine-grained parameterized motor control.  The descending pathway from motor cortex also conveys useful motor signal information, and these cortical reps are directly shaped by the same ascending motor signals.

The model uses a simple dopamine (DA) "critic" system implemented in code, in `mseq_env.go`.  It just adapts a `RewPred` prediction whenever a reward is processed, using a simple learning rate (`RewPredLRate=0.01`).  The `RewPredMin=0.1` is critical to maintain negative DA signals for failures, so these cannot be fully predicted away.  Partial credit is given for partially correct sequence outputs, which is important for the length-3 sequences but actually somewhat detrimental to the length-2 case.  Partial credit is computed as a *probability* of reward as a function of the number of correct actions: `p = NCorrect / SeqLen`, and the reward value if given is also equal to this `p` value.  If you just continuously give a constant `p` value partial credit, the model learns to expect this and fails to progress further.

These are ecologically reasonable properties under the assumption that reward is discounted by effort, and random action choices will sometimes result in good outcomes, but generally with additional unnecessary steps.

To simplify the use of a consistent motor sequence across the parallel-data processing used on GPU (NData > 1), we just keep the target sequence as 0,1,2 etc, because the model doesn't know any better, and the random initial weights have no bias either.

# Network Overview

Once the dust settles, a summary of the biology and implementation in the model will be included here.  For now, see this [github Discussion](https://github.com/emer/axon/discussions/326).

# Results

The params have been relatively thoroughly "tweaked" at this point: see paramtweak.go for tweaking code.

The learned weights to the BG clearly show that it is disinhibiting the appropriate action at each step in the sequence.


## Combinatorics

| Act ^ Len | N       | Notes          |
|-----------|---------|----------------|
| 10^3      |   1,000 | acts = harder  |
| 6^4       |   1,296 | easy           |
| 5^5       |   3,125 | easy, std test |
| 6^5       |   7,776 | still quick    |
| 7^5       |  16,807 | some fail      |
| 8^5       |  32,768 | ?              |
| 9^5       |  59,049 | ?              |
| 10^5      | 100,000 | ?              |
| 6^6       |  46,656 | ~50% learn     |
| 7^6       | 117,649 | ?              |

## Aug 6, 2025

| Act ^ Len | Fail | Mean  | SDev | Min  | Q1  | Median | Q3   | Max   | Job no |
|-----------|------|-------|------|------|-----|--------|------|-------|--------| 
| 5^5       |    0 |  7.6  |  5.6 | 2    | 4   | 5      | 10   | 28    | 504    |
| 6^5       |    0 | 13.8  | 13.6 | 3    | 5   | 8.5    | 15   | 59    | 510    |
| 7^5       |    8 | 43.7  | 70   | 3    | 6   | 12.5   | 30.7 | 200   | 519    |
| 10^3      |      |       |      |      |     |        |      |       |        |


# TODO:

* Set number of cycles per trial in terms of BG motor gating timing: constant offset from onset of VM gating timing, with a cutoff for "nothing happening" trials.
    * Attempted an impl but is difficult with CaBins -- tried shifting bins but awk..

* "CL" not beneficial (implemented as direct MotorBS -> Matrix pathways): rel weight of 0.002 is OK but starts to actually impair above that.  Likely that a functional cerebellum is needed to make this useful.  Also, investigate other modulatory inputs to CL that might alter its signal.  Key ref for diffs between CL and PF: LaceyBolamMagill07: C. J. Lacey, J. P. Bolam, P. J. Magill, Novel and distinct operational principles of intralaminar thalamic neurons and their striatal pathways. J. Neurosci. 27, 4374–4384 (2007).

* Local patch (striosome) learning modulation.  The striosomes clearly provide a direct modulation of SNc dopamine on a "regional" basis.  They receive input preferentially from BOA "limbic" areas, but also potentially some of the same inputs as the local matrix MSNs.  There is also some PF-based circuitry and cell types of relevance.  It is possible that these circuits could provide something like a local partial credit signal, or DA in proportion to whether an action is expected or unexpected.  There is evidence for considerable ongoing micro-fluctuations in local DA, which is related to subsequent action selection: **MarkowitzGillisJayEtAI23**. 

* Learning in the other parts of the pcore circuit -- might help auto-adjust the parameters.  Need to figure out what logical learning rules would look like.

# Param search notes

## 07/30/2025: patch

Key logic:

* PatchD1, D2 accumulates PF action output across sequence, learns final DA modulation.  At each point in time, signals the expected positive vs. negative.

* Patch should provide a useful signal to Matrix about what works -- use to modulate DA.

* 

* if PF == same stripe as Patch (i.e., selected), Patch discounts learning.
* if PF != same, then sign flips -- counterfactual. is this already built in?
* But it always biases learning in same direction as self.

Discount: 

* for trace, D1

## 01/14/2025: after fixes

* PF looks weaker in new vs. old; unclear why. Fixed PF weights not helpful (but not too bad either).

* most runs go up in performance but then often come back down: added an asymmetric learning rate for rewpred to try to deal with this -- works well!  `RewPredLRateUp` = 0.5 > 1.  Went from 11 fail to 3 fail on 3x8.

## 01/14/2025: Hebbian learning in MotorBS, with STN SKCa = 80 instead of 150, State layer size.

In general performance since switching to linear SynCa approximation has been significantly worse. Is it the Ca or something else that changed??

1. Discovered a critical parameter change is mentioned in rev `0d290fe7` on 4/19/2024, but didn't get baked into the defaults until a few refs later (definitely by 05/02/2024, `546ceada`), which improved pcore_vs and choice (need to revisit that!):
```Go
.DSTNLayer: ly.Acts.SKCa.CaRDecayTau = 150  // was 80 -- key diff!  
```

Previously, the value was 150, then it got changed to 80, which significantly impaired the models of that time -- restoring this parameter to 150 restored the very good performance reported above for 2024-02-09. Need to more systematically run models to detect these breakages!

With the 80 value, there was a regime of parameters that worked pretty well with SeqLen = 3, NActions = 5, but doesn't work _at all_ with Seq3x6, that was focused on the new `CaPScale` parameter changing the balance of CaP vs. CaD, producing a kind of hebbian learning effect. For projections to MotorBS and VLM1, turning this CaPScale up to 1.2, and adding DA1mod to MotorBS itself (0.03, dip gain .1), worked well. Also strangely turning down the learning rate on `#DGPiToMotorBS` (inhibitory gating from BG) to 0.0005 worked best. None of these parameters holds now that the STN, and thus overall BG, is back to functioning again. These params are captured in `b095ae18` 2025-01-13.

With STN SKCa = 150, get 100% success on Seq3x5.

2. The `State` layer `Nominal` act was set as if size = NActions but actual size should be SeqLen which had been fixed. Setting this back to NActions restored prior good performance (along with SKCa fix). But Nominal change alone (normalizing by SeqLen) is not as effective as sizing with NActions.. definitely need to investigate this!

