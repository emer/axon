# Rubicon Goal-Driven Motivated Behavior Model

This document describes the [Rubicon](../../Rubicon.md) model for goal-driven motivated behavior, which posits distinct  **goal-selection** vs. **goal-pursuit** (aka **goal-engaged**) states of the brain ([O'Reilly, 2020](https://ccnlab.org/papers/OReilly20.pdf); [Heckhausen & Gollwitzer, 1987](#references)).  In the goal selection phase, different options are considered and evaluated according to learned cost-benefit _utilities_ with the following brain areas protypically representing the following information:

* **Benefit** (expected positive _outcome_) in the **OFC** (orbital frontal cortex), **IL** (infralimbic cortex), and **aIC** (anterior insular cortex).

* **Cost** of actions in dorsal **ACC** (anterior cingulate cortex, or just Cg in rodent).

* **Utility** (Benefit vs Cost) in **PL** (prelimbic cortex).

* **Action plan** in the **dlPFC** (dorsolateral prefrontal cortex), which is  **ALM** (anterior lateral motor area) in rodents.

Each of these areas mutually informs the other through bidirectional _constraint satisfaction_ to propose a potential overall goal / plan state across these three areas, as discussed in [Herd et al., 2021](https://ccnlab.org/papers/HerdKruegerNairEtAl21.pdf).

If the proposed goal / plan is selected via BG (basal ganglia, implemented via [PCore](PCoreBG.md)) _gating_, it then drives stable active maintenance of this _goal state_ which is a distributed representation across these PFC areas.  This maintained goal state then drives coordinated behavior toward achieving the expected outcome through the selected action plan.

The maintenance of this consistent goal state is critical for allowing learning at the time when the goal is either achieved or abandoned, to update the representations that then drive goal selection behavior in the future, so that these choices can be informed by what actually happened on previous choices.  This then provides a solution to the _temporal credit assignment_ problem.

The time scale over which an active goal state is engaged is determined by the nature of the expected outcome, and in general there can be many factors operating across many different time scales that influence which goals are pursued at any given point.  However, it is assumed that there is always a _single current active goal_, which represents the "inner loop" that is actively shaping behavior right now.  Everything else serves as broader _context_ that shapes the goal selection process, according to longer timescale considerations.  The determination of which possible outcome is engaged for the current goal is a function of the learned predictability of such outcomes in relation to the action plan required to achieve it, because uncertainty represents a major cost factor in the goal selection phase.  Specifically, more predictable, and thus typically shorter-term, goals are favored, as long as they have reasonable outcome benefits.

The Rubicon model subsumes the PVLV [Mollick et al, 2020](#references) (Primary Value, Learned Value) framework for phasic dopamine firing in the Pavlovian, classical conditioning paradigm.  This provides the learning signals at the time of the outcome that drive learning in the BG.

Implementation files: rubicon_{[net.go](axon/rubicon_net.go), [layers.go](axon/rubicon_layers.go), [prjns.go](axon/rubicon_prjns.go)}.  Examples / test cases: [pvlv](examples/pvlv), [choose](examples/choose).

# Introduction and Overview

The integration of classical (passive, Pavlovian) conditioning with the mechanisms of the goal-driven system represents a useful way of understanding both of these mechanisms and literatures. In effect, classical conditioning leverages the core goal-driven mechanisms in a simplified manner, making it a useful paradigm for testing and validating these mechanisms.

In the goal-driven paradigm, the Pavlovian _conditioned stimulus_ (CS) is a signal for the opportunity to obtain a desired outcome (i.e., the _unconditioned stimulus_ or US).  Thus, a CS activates a goal state in the vmPFC (ventral and medial prefrontal cortex), specifically OFC, IL, and PL, in the Rubicon framework.  Notably, the passive nature of the Pavlovian paradigm means that no additional motor plan representations (e.g., in dlPFC, ALM) are needed.  This goal state is then maintained until the US outcome occurs, and helps establish the CS-US association at the core of classical conditioning.

The most relevant and challenging aspect of classical conditioning in this context is recognizing when an expected US is _not_ actually going to happen.  A non-reward trial is, superficially, indistinguishable from any other moment in time when nothing happens.  It is only because of the maintained internal expectation that it takes on significance, when this nothing happens instead of the expected something.  In the Rubicon framework, this expectation is synonymous with the maintained goal state, and thus a non-reward trial represents a goal failure state, and the proper processing of this goal failure requires a decision to [give up](#give-up) on the engaged goal.  The mechanics of this give-up decision and resulting dopamine and other learning effects are thus tested by simple partial reinforcement and extinction conditioning paradigms, in ways that were explored to some extent in the PVLV paper [Mollick et al, 2020](#references).

One of the most important conceptual advances achieved by integrating this goal-engaged framing with the PVLV model is in better understanding the central role of the LHb (lateral habenula), which is hypothesized to be responsible for integrating a variety of different signals to decide when to give up.  For example, the PVLV model incorporated the otherwise obscure distinction that the shunting of phasic DA bursts was driven directly by neurons in the VS, but dips are driven by the LHb.  This now makes complete sense: if a reward is actually delivered, then there is a direct bottom-up sensory input that directly drives DA, and there is no need for any complex decision making about this case: the VS can just shunt the DA drive in proportion to its learned expectations.  However, if the reward is omitted, the decision to recognize and process this as a sense of "disappointment" is a much more active one, requiring the integration of multiple factors, and thus a specialized brain system for doing so (the LHb).

The following sections provide a more detailed overview of the different components of the Rubicon model, followed by implementational details about learning equations and more specific dynamics.  Throughout this document, we focus on the computational and algorithmic interpretation of the known biological systems, without providing a detailed accounting of the relevant neuroscientific literature.  As such, everything stated here represents a crystalized hypothesis about the function of the relevant brain systems, without repeating that important caveat every time.

## Goal Representations in vmPFC

![Distributed Goal Representation](figs/fig_rubicon_loops_spiral_goals.png?raw=true "Distributed brain areas representing different aspect of a Goal")

**Figure 1:** A version of the classic [Alexander et al. (1986)](#references) 5 loops through the frontal / BG diagram (Striatum = input to BG, with different anatomical regions labeled: D = dorsal, M = medial, V = ventral; Thalamus = output of the BG; SMA = supplementary motor area under dlPFC), highlighting how the OFC, PL / ACC, and dlPFC form a distributed goal / plan representation.  The striatum disinhibits the mediodorsal (MD) thalamus at the time of goal selection, locking in a stable maintenance of the goal state until the goal is completed or abandoned.

Figure 1 shows the main distributed goal representation brain areas, interconnected with the BG and MD (mediodorsal) nucleus of the thalamus that is ultimately disinhibited by the BG to drive a goal engagement event.  Once the goal is engaged and maintained, it influences individual action choices in the lower-level motor areas (SMA).

![Rubicon Time Bridging Logic](figs/fig_rubicon_logic.png?raw=true "Overall Time Bridging Logic, to solve the temporal credit assignment problem")

**Figure 2:**  Diagram of key elements of the Rubicon model and how they bridge the goal engaged time window between CS trigger for goal engagement and US outcome.  The BLA (basolateral amygdala) helps bridge by mapping CSs into associated USs, and it drives representations in OFC to encode these predicted US outcomes, which are maintained in the goal state.

Figure 2 illustrates how the CS-driven goal selection drives active maintenance of the goal state across the duration of the goal execution, thereby bridging the temporal gap.  In addition to this active maintenance, there are other specialized "shortcut" pathways that help bridge the temporal gap, specifically the the basolateral amygdala (**BLA**).

The BLA learns to associate an initially neutral CS with the US that it is reliably associated with.  This learning provides a key step in enabling the system to later recognize the CS as a "trigger" opportunity to obtain its associated US -- if that US is consistent with the current Drive state (e.g., the CS is associated with food and the system is "hungry"), then it should engage a goal to obtain the US.  In the classical conditioning paradigm pioneered by Pavlov, the CS was a bell and the US was food.  The BLA learns the link between the bell and the food, effectively "translating" the meaning of the CS into pre-existing pathways in the brain that process different USs, thus causing Pavlov's dogs to salivate upon hearing the tone.

## Ventral BG Gating for Goal Selection and Maintenance

The ventral portion of the basal ganglia (VBG) plays the role of the final "Decider" in the Rubicon framework ([Herd et al., 2021](https://ccnlab.org/papers/HerdKruegerNairEtAl21.pdf)).  The input layer to the VBG is the _ventral striatum_ (**VS**) (also known as the _nucleus accumbens_ or NAc), which receives from the vmPFC goal areas (and other relevant sensory input areas), and projects to the _VP_ (ventral pallidum), which then projects to the **MD** thalamus, which is bidirectionally interconnected with the vmPFC goal areas (Figure 1).  If the VBG decides to select the currently proposed goal state represented in vmPFC, this results in a disinhibition of the MD thalamus, opening up the corticothalamic loop, which updates and locks in the goal representation.  This gating dynamic was originally captured in the PBWM (prefrontal-cortex basal-ganglia working memory) model ([O'Reilly & Frank, 2006; O'Reilly, 2006)](#references), and has recently been supported by detailed recording and optogenetic manipulation studies in the Svoboda lab [(Inagaki et al, 2022; 2018; Li et al., 2015; Guo et al., 2017; 2014)](#references).

The VBG is implemented using the [PCore](PCoreBG.md) model, which has **Go** (direct, D1) and **No** (indirect, D2) striatum input layers (specifically the _matrix_ = **Mtx** medium spiny neurons, MSNs).  These learn whether to gate in and maintain a goal state based on the current BLA state and associated learned information in the OFC, ACC, IL, and PL layers, trained by the phasic **DA** = dopamine signal (see [Dopamine System](#pvlv-phasic-dopamine) below).  The **GPe**, **GPi** and **STN** layers here are all described in the pcore link and enable the BG to exhibit strongly nonlinear decision dynamics.

An important feature of the VBG is its sensitivity to the *match* between the US outcome associated with the current input stimulus (CS) and the currently activated Drive(s), which is not present in the BLA signal itself (which just does the CS -> US translation).  This match computation works by setting the Drives as a *modulator* input to the BG, such that a given US / Drive specific pool in the BG can only fire if it is receiving a given Drive input.  Whether it actually does fire is determined by the BLA and inputs from OFC and ACC. The other unique feature of the BG is the ability to use dopamine-based learning to interpret the actual reward value of the OFC and ACC cortical representations, to make the final Go vs. NoGo decision for activating a proposed plan.

## Predictive Learning in vmPFC areas

![vmPFC Areas](figs/fig_rubicon_vmpfc_areas.png?raw=true "vmPFC areas across species")

**Figure 3:** Functional mappings across four species for the core vmPFC goal areas, based on [Roberts & Clarke (2019)](#references).  The rat PFC is essentially entirely these core goal-driven areas, which are then expanded and supplemented by extensive additional higher-level motor planning areas in the three primate species (Marmoset, Macaque, Human).

![vmPFC Areas](figs/fig_rubicon_anatomy_macaque_rat.png?raw=true "vmPFC areas in macaque and rat")

**Figure 4:** More detailed anatomy of the vmPFC areas in macaque and rat, showing the ventral and lateral surface (extending toward the bottom in the figure) that encodes more detailed outcome information, including in the anterior Insular cortex, aIC.

How do the various distributed goal state areas (shown in Figure 3 for three different species, and in greater detail in macaque and rat in Figure 4) actually learn to encode the information needed to make good decisions?  We hypothesize that they learn by predicting different outcomes, at two different time scales: a _continuous_ step-by-step prediction during the course of goal pursuit, and a _final_ prediction of the ultimate outcome, that is activated at the start and maintained throughout.

The continuous signal is important for tracking progress and accumulated costs toward the goal, while the final prediction is important to have at the start, so that the system has a reasonable encoding of the total estimated costs and benefits encoded in the vmPFC layers.  As we will see, these two time scales are differentially naturally available for costs and benefits, and more neuroscience research on the precise nature of these representations is needed.

### Deep PFC Layers for Predictive Learning and Goal Maintenance

Each vmPFC area has a set of neocortical layers that are specialized to support a different aspect of the overall tasks of representing the current state, predicting the future states, and maintaining the overall goal state, based on the [Deep predictive learning](Deep.md) framework [(O'Reilly et al, 2020)](#references).  These are as follows:

* `Super` (no suffix): implements the _superficial layer 2-3_ neurons, which function just like standard axon cortical neurons, and always represent the **current state** of things.  They learn continuously from predictive learning error signals, are widely bidirectionally interconnected with other cortical areas, and form the basis for the learned representations in the other layers.  The bidirectional interconnections support the key constraint satisfaction process that takes place to formulate a new proposed plan.

* `CT` (`CT` suffix): implements the layer 6 regular spiking CT _corticothalamic_ neurons that project into the thalamus, and are primarily responsible for **generating continuous predictions** over the thalamus based on temporally delayed _context_ signals, functioning computationally like the simple recurrent network (SRN) context [(Elman, 1990)](#references).

* `Pulvinar` (`P` suffix): implements the Pulvinar-like TRC (thalamic relay cell) neurons, upon which the prediction generated by CT layer is projected in the _minus phase_ (first part of the theta cycle) -- the `P` suffix thus also stands for **prediction**.  Then, in the _plus phase_ (second part of the theta cycle), the actual _outcome_ relative to that prediction is driven by strong driver inputs, originating from layer 5IB (intrinsic bursting) neurons from other neocortical areas, or from subcortical areas such as the cerebellum, motor brainstem, or superior colliculus.  The prediction error is thus the difference between these two phases of activity, and it is broadcast back up to the Super and CT layers to drive predictive learning that improves predictions over time.  Biologically, the pulvinar is the prototype for this predictive learning thalamus, for posterior cortical areas, but different parts of the **MD** likely play this role for PFC areas [(Rovo et al., 2012)](#references).  Also see [Root et al. (2015)](#references) for major review of VS -> MD pathways and associated recording, anatomy etc data.  

* `PTMaint` (`PT` suffix): implements a subset of _pyramidal tract_ (PT) layer 5b (deep) neurons that exhibit **robust active maintenance of the goal state**, and project extensively to subcortical areas, including the motor spinal cord for primary motor cortex areas.  These are gated by basal ganglia disinhibition of their corresponding **MD** thalamic layer, which modulates the NMDA-laden recurrent collaterals that sustain activity: this is the mechanism for BG gating and goal maintenance.   The final outcome predictions are driven by this PT layer activity, because it is stable throughout the goal state.  The [pfcmaint](examples/pfcmaint) example tests this mechanism specifically, including the `SMaint` self maintenance mechanism that simulates a larger population of interconnected neurons.

* `PTPred` (`PTp` suffix): implements a subset of PT neurons that are `CT` like in contributing to predictive learning over the thalamus, but receive input from the `PTMaint` cells and are thus only active during periods of active maintenance.  This layer provides the primary input to the `VSPatch` US-timing prediction layers, along with contributing to the "Pulvinar" predictions for other signals.

### Outcome in OFC, aIC, IL

OFC encodes predictions of the *outcome* of an action plan, i.e., the **US** = unconditioned stimulus (food, water, etc).  There are various subregions of OFC in primate (shown in Figure 4), with various levels of detail vs. abstraction and modality-level factors in representing outcomes.  The BLA is strongly interconnected with the OFC, and drives its outcome encoding, at an intermediate level of abstraction (e.g., the value and overall taste quality of food, but not the more detailed taste sensation, which would be in aIC).  There are both positive and negative US outcomes represented in the OFC -- we use a strong spatial segregation according to valence in the model, but this is not necessary in the biology (though it may be present to at least some extent).

The IL infralimbic cortex encodes outcomes at the level of abstract value (separately for positive and negative valence), independent of the more specific detailed sensations, by doing continuous prediction of the OFC layer states.  Thus, for example the `OFCposP` layer drives predictive learning in the `ILpos` layers, etc.

The final US outcome provides a natural representation of the final outcome state, but it is less clear what drives the continuous predictions for the OFC layers.  In many cases, there is a natural physical metric in terms of distance from a food source, which is perceptually available and could be incrementally predicted.  This could then be generalized for more abstract cases.  Given the functional importance of having a metric for goal proximity (which is clearly reflected in tonic dopamine levels, for example), it is likely that there are connectivity biases to support this distance-based prediction, and indeed it is clear that OFC neurons are strongly interconnected with hippocampal and other medial parietal spatial representation areas (e.g., retrosplenial cortex) (e.g., Boorman etc).

As noted above, the BLA also provides a synaptic learning mechanism for predicting the US based on a CS, but unlike the BLA, the OFC can actively maintain its US expectation (even if the CS later is occluded etc), and these maintained active neurons can *bias* processing in the rest of the system to organize appropriate behavior around this US.  Thus, the BLA and OFC work together to do CS -> US learning (BLA) and active maintenance and cognitive control of US expectations (OFC).

### ACC Motor Cost

The **ACC** encodes predictions of the cost of the motor actions necessary to accomplish the goal, by predicting brainstem level metabolic and time signals, continuously during goal performance and for the final accumulated levels.  The continuous availability of metabolic information naturally drives continuous prediction, but final prediction requires learning at the time of the outcome about the accumulated cost signals.

### PL Utility

By hierarchically predicting the IL and ACC layers, the PL layer learns an abstract representation that integrates positive outcomes, negative outcomes, and action costs, thereby encoding the overall utility of the goal.

### dlPFC / ALM Motor Plan

The **ALM** / **dlPFC** (dorsolateral prefrontal cortex) encodes an overall *policy* or *plan of action* for achieving the desired outcome, which is learned by predicting the sequence of actions performed.  dlPFC in the primate corresponds to ALM (anterior lateral motor area) in the rodent (not PL as is often suggested in the literature).

## PVLV Phasic Dopamine

<img src="figs/fig_pvlv_pv_lv_schultz_da.png" height="600">

**Figure 5:** PV and LV in the PVLV model: LV = Learned Value (Amygdala), which learns the value of conditioned stimuli (CSs). PV = Primary Value (Ventral Striatum, principally the Nucleus Accumbens Core, NAc), which learns to expect US (unconditioned stimuli, rewards) and shunt dopamine when they occur, or, when omitted, the LHb (lateral habenula) drives a dip / pause in DA firing.  Data from Schultz et al (1997) of VTA firing in a Pavlovian conditioning paradigm.

The VBG that drives goal selection learns under the influence of phasic dopamine neuromodulation released by the VTA (ventral tegmental area) in the midbrain.  The basal ganglia has the densest concentration of dopamine receptors, and, critically, reuptake mechanisms that allow it to be sensitive to rapid temporal transients, which most other brain areas lack.  There are many higher-level brain areas that drive this phasic dopamine release, and the PVLV (Primary Value, Learned Value) model [Mollick et al, 2020](#references) integrates contributions from the most important of these areas within a coherent overall computational framework.

Figure 5 shows the most important hypothesis of the PVLV model, which is that there are anatomically distinct brain areas involved in driving phasic dopamine responding to CSs (the LV system, principally involving the Amygdala), versus those involved in anticipating the US and shunting phasic dopamine firing that would otherwise occur then (in the VS), and driving phasic dopamine dips to omitted anticipated USs (in the LHb).  Thus, the main elements of PVLV are:

* The **Primary Value (PV)** (US outcome predicting) and **Learned Value (LV)** (CS predictive cue learning) systems.

* **PV = ventral striatum**, which learns to expect US outcomes and causes the phasic dopamine signal to reflect the difference between this expectation and the actual US outcome value experienced. This difference is termed a *reward prediction error* or *RPE*.

* **LV = amygdala**, which learns to associate CSs with US outcomes (rewards and punishments), thus acquiring new CS-outcome value associations (learned value).

This division of labor is consistent with a considerable amount of data [Hazy et al, 2010](#references). The 2020 PVLV model has a greatly elaborated representation of the amygdala and ventral striatal circuitry, including separate pathways for appetitive vs. aversive processing, as well as incorporating a central role for the *lateral habenula* (LHb) in driving pauses in dopamine cell firing (dipping) for worse than expected outcomes. Figure 1 provides a big-picture overview of the model.

![PV.6](figs/fig_bvpvlv_pv_lv_only.png?raw=true "PV.2")

**Figure 6:** Simplified diagram of major components of the PVLV model, with the LV Learned Value component in the Amygdala and PV Primary Value component in the Ventral Striatum (principally the Nucleus Accumbens Core, NAc).  LHb: Lateral Habenula, RMTg: RostroMedial Tegmentum, LDT: Laterodorsal Tegmentum, LHA: Lateral Hypothalamus, PBN: Parabrachial Nucleus. 

### A Central Challenge: Learning *Something* when *Nothing* happens

As noted above and articulated in the PVLV papers, a central challenge that any RL model must solve is to make learning dependent on expectations such that the *absence* of an expected outcome can serve as a learning event, with the appropriate effects.  This is critical for **extinction** learning, when an expected reward outcome no longer occurs, and the system must learn to no longer have this expectation of reward.  This issue is particularly challenging for PVLV because extinction learning involves different pathways than initial acquisition (e.g., BLA Ext vs. Acq layers, VSPatch, and the LHb dipping), so indirect effects of expectation are required.

The basic PVLV extinction dynamic involves the sustained goal engaged state, specifically in the OFC PT maintenance layer, driving learned `VSPatch` activity at the time of the expected US outcome.  The VSPatch then drives the LHb which in turn drives dipping (pausing) of VTA activity, which in turn causes widespread new learning to extinguish the CS -- US association.  As noted above, this seemingly over-elaborate chain of activity, and the need for a separate brain area involved in driving dipping of dopamine firing, makes more sense when we consider the further computations required in determining when goal failure should occur.

### LHb: The Brain's Extinguisher

The LHb (lateral habenula) has the following coordinated effects in the model:

* **DA dipping**: The phasic dip in DA activity shifts the balance from D1 to D2 in all DA-recipient neurons in the BG and BLA, causing learning to start to expect this absence (see below on BLA Ext pathway).  Note that this dopamine dip should represent not just the absence of an expected outcome, but also any accumulated effort costs incurred during goal pursuit.

* **OFC / goal gating off**: The LHb can also drive MD thalamic projections to the vmPFC goal areas, deactivating the maintained goal representations.  Implementationally, it happens simply by setting the `HasRew` flag in the `Context.NeuroMod` structure, which triggers decay of the relevant PFC areas (via the `Act.Decay.OnRew` flag).  This happens at the end of the Trial.

* **ACh signaling**: ACh (acetylcholine) is released for salient events, basically CS onset (via superior colliculus to LDT = laterodorsal tegmentum) and US onset, and it modulates learning and activity in the BLA and VS.  The LHb projections to the basal forebrain cholinergic system allow it to provide the key missing piece of ACh signaling for the absence of an expected US, so that a consistent framework of ACh neuromodulation can apply for all of these cases.

See the [Give up](#give-up) section below for more details on how the LHb computes when to give up on a maintained goal.

## Negative USs and Costs

There are two qualitatively different types of negative outcome values, which require distinct pathways within the model:

* `USneg`: Phasic, discrete "events", such as a shock or impact, that can be associated with a stimulus (CS) or action, and thereby potentially predicted and avoided in the future.  The BLA plays a well-established role here for establishing CS -- US associations (e.g., tone-shock in the widely used fear conditioning paradigm), in the same way as in the positive valence case.

* `Costs`: Continuous, inevitable costs, specifically Time and Effort, and lower-grade forms of pain, that animals are always seeking to minimize.  These are not associated with phasic CSs, but rather vary as a function of motor plans (e.g., climbing a steep hill vs. walking along a flat path).  These negative outcomes do not engage the BLA because they are not primarily CS-associated, and instead are predicted from motor-cingulate areas.

## Novelty-based Exploration

In order to learn about possible US associations of novel CSs, it is essential to approach them and determine if any kind of US outcome is associated with a given CS.  The Rubicon model has a default drive to explore novel stimuli, and there is a pathway converging on the BLA that is activated by novel CSs, which enables the model to engage a goal state organized specifically to explore novel CSs.  As experience accumulates, the standard BLA and other learning mechanisms take over and drive responding accordingly.

# Detailed Implementation

The following sections provide a more detailed description of each of the components of the Rubicon model.

# Timing: Trial-wise

First, we describe the overall timing dynamics.  In contrast to the minus-plus phase-based timing of cortical learning, the RL-based learning in Rubicon is generally organized on trial-wise boundaries, with some factors computed online within the trial.  Here is a schematic, for an intermediate amount of positive CS learning and VSPatch prediction of a positive US outcome, with an "Eat" action that drives the US:

| Trial Step:  |   0       |   1  |   2   |   3         |
| ------------ | --------- | ---- | ----- | ----------- |
| Event / Act  |  CS       |      | Eat   |  US         |
| SC -> ACh    |  +++      |      |       |             |
| BLA          |  ++       |      |  Rp   |  R          |
| BLA dw       | tr=S ⋅ ACh |      |       | R(R-Rp)tr   |
| OFC          |  BLA->    |  PT  |  PT   | reset PT    |
| VSPatch = VP |           |      | ++ Rp |             |
| VP dw        |           |      |       | Sp ⋅ Rp ⋅ DA |
| DA           | ++ (BLA)  |      |       | + (US-VPp)  |

* + = amount of activity, proportional to number of +'s.
* Rp = receiving activity on previous trial.
* PT = pyramidal tract neurons are active in OFC (and other vmPFC) due to CS-induced gating.
* DA at US is computed at start of trial in PVLV.NewState, based on VS D1 - D2 on prev trial.

# Basolateral Amygdala: BLA

The basolateral amygdala learns to associate an initially neutral CS with the US that it is reliably associated with.  The BLA learns at the time of the US, in response to dopamine and ACh (see below), so any stimulus or memory representation reliably present just before the onset of the US is then mapped onto the corresponding US-specific Pool in the BLA layer.  `BLAposAcqD1` is the positive-valence *acquisition* layer expressing more D1 dopamine receptors, and it is opposed (inhibited) by a corresponding `BLAposExtD2` D2-dominant *extinction* layer.  See the PVLV model [Mollick et al. (2020)](https://ccnlab.org/papers/MollickHazyKruegerEtAl20.pdf) for more details.

There are 2x2 BLA types: Positive or Negative valence US's with Acquisition vs. Extinction:

* `BLAposAcqD1` = Positive valence, Acquisition
* `BLAposExtD2` = Positive valence, Extinction
* `BLAnegAcqD2` = Negative valence, Acquisition
* `BLAnegAcqD1` = Negative valence, Extinction

The D1 / D2 flips the sign of the influence of DA on the activation and sign of learning of the BLA neuron (D1 = excitatory, D2 = inhibitory).

The learning rule uses a trace-based mechanism similar to the BG Matrix projection, which is critical for bridging the CS -- US time window and supporting both acquisition and extinction with the same equations.  The trace component `Tr` is established by CS-onset activity, and reflects the sending activation component `S`:

* `Tr = ACh * S`

At the time of a US, or when a "Give Up" effective US (with negative DA associated with "disappointment") occurs, then the receiver activity and the change in activity relative to the prior trial (`R - Rp`, to provide a natural limitation in learning) interact with the prior trace:

* `DWt = lr * Tr * R * (R - Rp)`

For acquisition, BLA neurons are initially only activated at the time of the US, and thus the delta `(R - Rp)` is large, but as they start getting activated at the time of the CS, this delta starts to decrease, limiting further learning.  The strength of this delta `(R - Rp)` factor can also be modulated by the `NegDeltaLRate` parameter when it goes negative, which is key for ensuring that the positive acquisition weights remain strong even during extinction (which primarily drives the ExtD2 neurons to become active and override the original acquisition signal).

For extinction, the negative dopamine and projections from the PTp as described in the next section work to drive weight strengthening or weakening as appopriate.
    
## BLA Extinction Learning

The `BLAposExtD2` extinction layer provides the "context specific override" of the acquisition learning in `BLAposAcqD1`, which itself only unlearns very slowly if at all.  It achieves this override in 3 different ways:

* **Direct inhibition** of corresponding BLA Acq neurons in the same US pool.  This is via a projection with the `BLAExtToAcq` class, and setting the `Prjn.PrjnScale.Abs` value of this projection controls the strength of this inhibition.

* **Opposing at CeM** -- the central nucleus layer, `CeMPos`, combines excitation from Acq and inhibition from Ext, and then drives output to the VTA -- so when Ext is active, CeM gets less active, reducing DA drive on VTA bursting.

* **VS NoGo** -- Ext projects to the ventral striatum / ventral pallidum (Vp) NoGo layer and can oppose gating directly there.

A key challenge here is to coordinate these two layers with respect to the US expectation that is being extinguished.  In the Leabra PVLV version a modulatory projection from Acq to Ext served this function.

In this model, we instead leverage the OFC PT active maintenance to select the US that is being expected, as a consequence of BLA Acq triggering active maintenance gating of the specific expected US outcome, which then projects into BLA Ext.  

To work with the Novelty case (below), and leverage the delta-based learning rule in acquisition, we make the `OFC PTpred` projection into the `Ext` layer modulated by ACh, and `Ext` is also inhibited by the US itself.  This means that it will only activate at a *goal failure* event.  It is also possible to use a `CtxtPrjn` projection to have BLA Acq drive Ext in a trial-delayed manner (this is also ACh gated), to avoid any dependence on PT active maintenance.

Importantly, while the learning of Ext is driven by the maintained PTpred, it must have a fast CS-driven activation in order to oppose the Acq pool.  Thus, this CS-driven pathway (which includes contextual inputs via the hippocampus and other areas) is the primary locus of learning, and it also is not ACh gated because it must remain active persistently to counter the Acq activation. 

# Novelty & Curiosity Drive

<img src="figs/fig_pvlv_bla_novelty.png" height="600">

The first pool in the BLAposAcq / PosExt layers is reserved for the "novelty" case (see above figure).  This pool is driven by persistent excitatory input and active by default unless inhibited by another layer -- i.e., a CS is novel until proven otherwise.  If the pursuit of the CS leads to no positive US outcome, then the corresponding extinction layer will learn to oppose the positive novelty activation.  If it leads to a US outcome, then that association will be learned, and the CS will then activate that US instead of novelty.

The delta learning rule for novelty extinction works via the same extinction logic as regular US pathways: if no US outcome occurs, and the maximum effort limit has been reached, then the LHb dip / reset dynamic is activated, triggering phasic ACh release as if an actual US had occurred.

## BLANovelCS

The first, novelty pool in the BLA requires a stimulus input to drive it when a novel CS comes on.  This is accomplished in a simple way in the model through a layer (`BLANovelCS`) with fixed (non-learning) synaptic weights from all CS inputs, such that any given CS will activate a particular activity pattern over this layer.  This layer projects into the first BLApos pools, and drives activity there for novel CSs.  When a specific CS acquires a specific US outcome representation in the BLA, that pathway should out-compete the "default" BLANovelCS pathway.

# Central Amygdala: CeM

For now, CeM is only serving as an integration of BLA `Acq` and `Ext` inputs, representing the balance between them.  This uses non-learning projections and mostly CeM is redundant with BLA anyway.  This also elides the function CeL.  Biologically, the CeM is capable of independent associative learning, but this is likely redundant with BLA learning in most cases.

# Superior Colliculus: SC -> LDT ACh

In the [Mollick et al. (2020)](#references) version, the PPTg (pedunculopontine tegmentum) directly computed a temporal derivative of amygdala inputs, to account for phasic DA firing only at the *onset* of a CS, when the BLA has sustained CS activity.  However, updated research shows that this is incorrect, and it is also causes problems functionally in the context of the BOA model.

In the new model, ACh (acetylcholine) from the LDT (laterodorsal tegmentum) provides a *disinhibitory* signal for VTA bursting, triggered by CS onset by the SC (superior colliculus), such that the sustained CeM -> VTA drive for a learned CS only results in bursting at CS onset, when LDT ACh is elevated from CS -> SC novel input.

* `SC` = superior colliculus, which shows strong *stimulus specific adaptation* (SSA) behavior, such that firing is strong at the onset of a new stimulus, and decreases by 60% or so thereafter [(Dutta & Gutfreund, 2014; Boehnke et al, 2011)](#references).  Thus, the temporal derivative onset filter is on the pure sensory side, not after going through the amygdala.

* `LDT` = laterodorsal tegmentum, which is the "limbic" portion of the mesopontine tegmentum, which also contains the pedunculopontine nucleus / tegmentum (PPN or PPT) [(Omelchenko & Sesack, 2005; Huerta-Ocampo et al. (2021)](#references).  The LDT receives primarily from the SC and OFC, ACC, and barely anything from the amygdala (same with PPN, which is more strongly connected to motor circuits and sends more output to SNc -- basically a motor version of LDT).  LDT contains glutamatergic, GABAergic, and ACh neurons (co-releasing at least GABA and ACh in some cases), and strongly modulates firing in the VTA, inducing bursting [(Dautan et al. (2016)](#references).  LDT also drives neurons in the nucleus basalis and CIN (cholinergic interneurons) in the BG, which all have consistent CS / US onset salience firing [(Sturgill et al., 2020)](#references).  LDT ACh has widespread projections to the BG (including SNr / GPi and thalamus), modulating gating to only occur at onset of novel stimuli.

Furthermore, the OFC & ACC input to LDT serves to inhibit ACh salience signals when there is already an engaged goal, to reduce distraction.  To enable this inhibition, it is critical to add this code in the `ApplyPVLV` function called at the start of each trial:

```Go
	pv.SetGoalMaintFromLayer(ctx, di, ss.Net, "PLposPT", 0.3) // PT layer is  maint
```

which sets the `GoalMaint` global variable, that then drives inhibition of ACh in LDT via its `MaintInhib` parameter.

The SC layer has a relatively strong trial scale adaptation current that causes activity to diminish over trials, using the sodium-driven potassium channel (KNa):
```
"Layer.Acts.KNa.Slow.Max":     "0.5", // 0.5 to 1 generally works
```

# VSPatch

The ventral striatum (VS) patch neurons functionally provide the discounting of primary reward outcomes based on learned predictions, comprising the PV = primary value component of the PVLV algorithm.

It is critical to have the D1 vs D2 opponent dynamic for this layer, so that the D2 can learn to inhibit responding for non-reward trials, while D1 learns about signals that predict upcoming reward (see subsection below).  For now, we are omitting the negative valence case because there is reasonable variability and uncertainty in the extent to which negative outcomes can be diminished by learned expectations, and especially the extent to which there is a "relief burst" when an expected negative outcome does not occur. 

The learning rule here is a standard "3 factor" dopamine-modulated learning, very similar to the BLA Ext case, and operates on the prior time step activations (`Sp`, `Rp`), based on the [timing](#timing) logic shown above, and to ensure that the learning is predicting rather than just directly reporting the actual current time step activations:

* `DWt = lr * DALr * Sp * Rp`

where `DAlr` is the dopamine-signed learning rate factor for D1 vs. D2, which is a function of US for the current trial (applied at start of a trial) minus VSPatch _from the prior time step_. Thus the prediction error in VSPatch relative to US reward drives learning, such that it will always adjust to reduce error, consistent with standard Rescorla-Wagner / TD learning rules.

## Non-reward Non-responding

A major challenge for VSPatch is extinguishing the prediction on non-reward trials leading up to an expected reward trial.  Various experiments that altered the learning rates etc impaired the ability to accurately match the target value, so now we are just using a threshold _that applies on the exported global DA value_ for small non-reward VSPatch values (i.e., if VSPatch is 0.1 or lower on a non-reward trial, the DA value is 0).  Critically, the VSPatch itself learns based on the global `VSPatchPosRPE` value, which is computed on the non-thresholded value, so it still drives learning to push the prediction downward, which does eventually happen over time.

Biologically, this would require a subpopulation of VTA neurons that project to the VSPatch neurons, that have less of a threshold in their response from LHb dipping inputs that the VSPatch drives.

# Give up

The decision to give up on an engaged goal is very difficult.  The value system is biased toward not giving up in general, because doing so requires accepting the accumulated costs as "disappointment" negative dopamine.  Giving up also updates the expected outcome estimates for such a goal / plan in the future, which we don't want to be inaccurately pessimistic.  On the other hand, we don't want to expend energy and opportunity cost if the desired outcome is unlikely to actually happen, and cutting ones losses as quickly as possible is advantageous here.  This tradeoff is captured in the often-subtle distinction between _perseverance_ (sticking with your goal and ultimately getting the payoff) and _perseveration_ (failing to give up on a hopeless goal).

Biologically, the LHb (lateral habenula) is hypothesized to integrate the relevant factors to make the final give-up decision.  These factors are as follows, listed in precedence order:

* **Utility:** Weighing the accumulated costs relative to the potential expected outcome.  At some point, even if the desired outcome might occur, it would not be worth the expended effort and opportunity costs (when you could have been doing something else of greater potential value).  This is related to the [marginal value theorem](https://en.wikipedia.org/wiki/Marginal_value_theorem) (wikipedia) for the specific case of foraging, which provides one way of computing the point of diminishing returns.  Costs are continuously available (and drive predictive learning), but the final outcome is only available as a learned estimate, and thus has uncertainty associated with it.  The estimated positive outcome value, driven by the `ILposPT` to `PVposFinal` pathway, is decoded along with the uncertainty in the neural signal to compute this Utility factor.

* **Timing:** The likelihood that the expected positive outcome is not actually going to happen, based on specific learned predictions in the VSPatch about the timing for the outcome.  VSPatch learns about specific discrete timing, in order to cancel the dopamine burst, and is penalized for anticipatory activity (that would generate a negative DA signal).  The timing function is based on integrated (summed) VSPatch value over time, and also the temporal variance (running average absolute value of change from one trial to the next), so that it does not come into effect until it has been relatively static recently, indicating that the timing window has likely past

* **Progress:** An estimate of the current rate of progress toward the goal, which can be dissociated from the specific timing of the actual outcome provided by the VSPatch.  Even if the "usual" expected Timing of the outcome has passed, if still making progress, then it can make sense to persevere (until it becomes perseveration).  The continuous estimate of progress toward the goal based on sensory distance is used for this factor.

These factors are integrated using weighting terms `Wgiveup` and `Wcontinue` that separately add to determine the numerator and denominator of an odds ratio representing the probability of giving up vs. continuing:

```go
    P(GiveUp) = 1 / (1 + (Wcontinue / Wgiveup))
```

When `Wgiveup == Wcontinue`, the resulting probability is 1/2, and as `Wgiveup` grows relative to `Wcontinue`, the overall probability approaches 1.

1. **Utility** has an overall weight factor `U` (which tends to be large) that multiplies the contributions of the costs and benefits to the corresponding weight factors, such that as the costs start to outweigh the benefits, the probability of giving up can increase significantly:

```go
Wgiveup = U * cost
Wcontinue = U * benefit
```

2. **Timing** uses a weight factor `T` multiplying a function of the normalized (relative to an expected Max) summed VSPatch outcome prediction that is penalized by the normalized temporal variance of that signal, such that when the value has stabilized (low variance), the full sum is applied:  

```go
Wgiveup = T * VSPatchSum * (1 - Var(VSPatchSum))
Wcontinue = T * (1 - VSPatchSum) * Var(VSPatchSum)
```

3. **Progress** uses weight factor `P` multiplying the current time-integrated rate of decrease in estimated distance to the goal for the continuing weight, and the opposite for giving up.  Thus, if the progress rate is positive, it favors continuing, and favors giving up if negative.  It is well known that animals (including humans) do not like to move away from the perceived goal, so perhaps this has something to do with that.

```go
Wgiveup = P * -ProgressRate
Wcontinue = P * ProgressRate
```

# Goal Dynamics

In this section, we discuss a range of important issues that future versions of the Rubicon model need to address concerning the dynamics and learning of goal representations.

## Motor Plan Updating and Online Control

Of all the elements of the Rubicon framework, the motor plan representation hypothesized for dlPFC areas is the most unclear in terms of what it specifies and how it might be learned.  By contrast, the outcome and cost representations in OFC, ACC, IL, and PL areas are all firmly grounded in brainstem-level mechanisms that have a clear evolutionary basis.  Thus, a major goal of the model is to understand how these motor plan representations emerge over the course of learning.  Intuitively, the main idea is that initially random patterns of neural activity are shaped by learning to predict the sequence of individual actions taken during goal pursuit, and the neural bias to maintain a consistent state over time.  This should potentially result in a stable overall activity pattern that drives appropriate actions in order to accomplish the goal outcome.

Within the context of an individual goal pursuit episode, it may be possible to update only the action plan component of the representation, while retaining the outcome and sensory context (CS) information from the originally selected goal state.  Thus, a shift in "strategy" may still be accommodated within one overall goal pursuit episode, as long as it doesn't represent too extreme a form of "mission creep".  Biologically, the action plan representation in dlPFC does have its own separate BG gating inputs, via VA thalamic pathways, so this "dual ported" nature of the dlPFC may support this kind of midstream updating.  Presumably, such updates drive additional learning of the overall plan representations, so that future goal selection episodes can operate on the updated motor plans.

The most important constraint on this motor plan updating is that it not violate the original cost-benefit tradeoff that supported the goal selection in the first place, which would be considered a problematic form of mission creep.  Due to having to pay the disappointment cost of giving up, it may be attractive to update instead of accepting these costs and starting the goal selection process anew.  Therefore, it is likely that any updated motor plan would have to be evaluated as not having significantly worse effort / negative outcome costs.  Presumably it would not be possible to reset the current accumulated costs, so in terms of the final accounting in the end, an updated action plan might end up being penalized somewhat by the accumulated costs under the previous action plan, but these wrinkles would likely wash out over time.

## Continuous vs. Phasic US Outcomes

The discrete partitioning of time into distinct _goal selection_ vs. _goal pursuit_ modes, which is central to the Rubicon framework, depends on there being a _point_ at which the goal is considered satisfied (or give-up is triggered).  How does that work in the many naturalistic environments where US consumption is a protracted, continuous process, for example in the case of foraging?  There may not be one simple answer here.  If the food items being foraged are at all spaced out, it would not be unreasonable to expect each one to require its own goal-selection dynamic, to evaluate the alternatives for exactly what next item to target.  However, if they are closely spaced and the harvesting is relatively automatic and easy, then a goal might encompass something like "get all the low-hanging, good-looking fruit from this bush".  As such, the individual positive US outcomes must be summed over time to decide when the goal has been accomplished.

Alternatively, when a human sits down to a prepared meal: the goal may be considered accomplished at the point of sitting down, and after that point, the active goals then focus on social dynamics over the meal.  Thus, everything depends on learned "schemas" for how predictable the different outcomes are: if sitting down to a familiar meal is highly likely to produce the known predictable outcome, the goal can be considered accomplished at the point of sitting down, but otherwise it might need to encompass the actual consumption of the meal.

These considerations highlight a critical issue about the core "accounting" requirements for the learning process to function properly.  If the determination of what counts as having satisfied the goal is at all subjective in nature, then what prevents the system from just "short circuiting" the process entirely and calling any old outcome "good enough", and thereby getting the desired positive reward without having to do any extra hard work?  The negative consequences of such a dynamic are aptly illustrated by the heroin addict, who neglects necessary bodily needs and often dies by overdose.  Thus, it is important that the US outcome evaluation is somehow strongly grounded in "reality" and not subject to too much fudging.

* initial state robustly maintained: matching process enforces accountability -- whatever you promised yourself at the outset, must match the outcome at the end.  and goal selection is very cautious...

* social enforcement: if you promise something and that motivates people to do things, they are very upset if you don't deliver!  strong punishment of cheaters.  you punish yourself for cheating yourself?

## Subgoals, Longer-term Goals

The above considerations reinforce the idea that it is fundamentally the OFC-level _outcome_ that anchors the goal expectations and dynamics: this outcome must be robustly maintained once selected, and once it is satisfied, then that goal state is deactivated and a new goal must be selected.  There is likely not a strong cost associated with having relatively rapid sequences of small goals, if that is what is appropriate based on the outcomes available, and if some larger time-scale of outcome is not reliably predictable.  This is consistent with the goal engaged state being the inner-loop of active, engaged behavior, operating within the context of other longer-term considerations and constraints that shape each goal selection process.

In this sense, the goal-engaged state might be considered a kind of "subgoal" relative to these longer-term factors.  There are two major issues that need further clarification.  1. What happens when you need to accomplish a subgoal that is _not_ associated with a biologically recognized outcome value?  2. Are there special mechanisms needed to maintain outer-loop contextual information, and apply relevant "accounting" to these outer-loop goals, in the same way that the core inner-loop goal dynamics obey the temporal credit assignment accounting between goal selection and subsequent outcomes? 

The arbitrary "perceptually defined subgoal" case (without a biologically grounded US outcome) could be considered a precise definition of "subgoal" vs. goal.  In the Rubicon framework, the simplest move is to treat such subgoals as markers of progress toward a proper US-grounded goal, which can affect the give-up dynamics and help maintain the goal-engaged state, but do not constitute a full goal outcome.  In short, a biologically grounded US is still required to drive the goal-selection process (i.e., you can't engage the goal system with purely perceptual "arbitrary" outcomes).

The phenomenon of _stimulus substitution_ is important here, because it allows otherwise arbitrary stimuli to take on the role of a full US, with _money_ being the prototypical example.  Many aspects of human behavior can be understood as being organized around obtaining money in some form.  Furthermore, the notion of _social currency_ is critical, because some of our strongest US motivations are social in nature (e.g., social approval vs. disaproval).  This social currency can drive various forms of stimulus substitution (e.g., accomplishing some kind of task that people generally regard as being difficult, and would thus be impressive), that can then drive people to do interesting things.

Critically, there needs to be a sufficiently reliable learned association between the previously arbitrary stimulus and a biologically grounded US, to enable it to act as a US.  This restricts the space of such USs so that the core accounting logic does not get undermined, while still allowing more open-ended behavior.  There are many relevant questions here about how _much_ of a task needs to be accomplished to amount to a full US, but in general the same issues of learning apply.

The second factor regarding the nature of the longer-term "contextual" factors that influence immediate goal-selection dynamics is very relevant here.  For example, if you happen to have a larger goal of getting a PhD because you think that will increase your social currency significantly, then that very long-term goal needs to be somehow translated into a large number of subtasks at finer temporal scales.  Somehow, it must boil down to "I will feel good if I write a paragraph of this paper", so that you can set that as an immediate goal.

One important principle is that each short-term goal-engaged step toward a larger, longer-term goal serves as a learning experience that shapes those longer-term goal representations, because they will get gated in and maintained along with everything else.  Thus, each step along a larger journey "carries the water" forward, updating and shaping the longer-term goals in important ways.  If these steps end up driving consistent negative learning outcomes, then that will weaken the longer-term goal representations, while positive outcomes will strengthen them.

In addition, a major advance in the human brain is the ability to organize our motivations around larger-scale abstractions where longer-term plans can be formulated in the absence of any actual experience (e.g., "go to the moon within 10 years").  This example also suggests that social and cultural factors play a huge role in shaping and sustaining an individual's longer-term goals. There are many unresolved questions here, but in any case, it is reasonable to at least start with the simple case of biologically grounded immediate US outcomes for now.

## Perceptual Expectations, Progress Tracking

To measure progress toward a goal, it is useful to have various forms of perceptual expectations, against which the current state can be compared.  Are there special vmPFC areas for maintaining such expectations, and how do they interact with relevant posterior cortical brain areas, and the rest of the goal system?

## Alternative Tracking, Goal Updating, and Hemispheric Specialization

A central hypothesis in the Rubicon model is that because goal representations are distributed across multiple brain areas each specializing on a different factor (OFC, ACC, IL, PL, dlPFC), _bidirectional constraint satisfaction_ dynamics between these brain areas is required to generate a good goal proposal that satisfies the current constraints [(Herd et al., 2021)](https://ccnlab.org/papers/HerdKruegerNairEtAl21.pdf).  This means that only _one_ such proposal can be evaluated at a time, because the binding problems associated with trying to multiplex multiple such representations across distributed brain areas are intractable (despite numerous attempts in the literature to propose otherwise).

In low-stakes situations (small costs, risks) with reasonable overall estimated utility, just taking the proposed goal will likely work well.  However, better decisions can often be made by considering multiple different alternatives before making the final choice, at the cost of taking more time and effort.  One simple way to accomplish these further consideration steps is to use the non-dominant hemisphere to represent a "current best" alternative, while the dominant hemisphere generates novel proposals relative to that.  If the novel proposal is sufficiently better than the current best alternative, it is selected and drives active behavior.  If it is better than the alternative but still not sufficiently above threshold, then it is gated into the non-dominant hemisphere, and otherwise another novel proposal is generated.  This enables a simple BG gating strategy to manage the overall decision-making process.

Furthermore, as the goal-pursuit proceeds, the non-dominant hemisphere could track whether the alternative proposal, or some updated variant thereof, might look better than the engaged, dominant one. If it does, then a goal update could occur, swapping the alternative in as the new active goal.  As long as this alternative goal state sufficiently captures the initial conditions present at the time of goal selection, it would still satisfy the critical _goal accounting_ criteria, such that the subsequent outcome will accurately inform future choices in similar situations.

This hemispheric specialization model for engaged vs. alternate goals is broadly consistent with lateralization data in people with ADHD (Hale et al., 2009; Silk et al., 2017) and also possible sex differences (e.g., Reber & Tranel, 2017; Thilers et al., 2007).  The widely cited "Extreme male brain theory of autism" (Baron-Cohen, 2002) also intuitively captures the idea that stereotypically male behavior is narrowly focused on a single task, similar to certain individuals in the autism spectrum, whereas females and people with ADHD are stereotypically better at switching among multiple tasks.  However, the relevant data on this is highly controversial and inconsistent (e.g., Lui et al, 2021).  Nevertheless, one way to reconcile the widespread popular belief that women are better at multitasking than men is to consider _motivational_ instead of raw performance differences: women may be more _willing_ to multitask than men, but in laboratory tasks where there is a strong implied performance demand, these motivational differences are not evident.

# Known Limitations and Issues

## Somewhat brittle mechanisms

There are often multiple ways of implementing a given component, that differ in overall robustness, and in general each component has already been improved in various ways over multiple iterations, but there is always more work to do here.

* BLAExt vs. Acq could be more robust -- Ext activity depends on PT -> Ext strength..

# TODO

* ACh and negative DA are out of sync for GiveUp: key point: LHb doesn't do any dipping *until* the point of give up!   then it sends in the full negative wallop.  don't just read out of VSPatch -- use LHb-mediated signal.  this also relates to NR -- if NR, no VSPatch!  done!  forget about the threshold!

* need threshold on VSPatch to accumulate into Sum for giveup -- ignore small.

* BLA activity is strongly DA modulated instead of ach -- that gives the delta for Ext learning.  Keep -da mod for context too, but NOT for regular CS inputs.  Key: need BLAExt to activate at CS with nothing else going on -- so no DA, ACh mod for direct CS pathway, but do need DA mod for ctxt pathway to activate on giveup.

* add a few more trials at end of pvlv to allow for give up to kick in.

* in general during acq (or acq of ext), BLA should NOT be active, then it is.

* BLAExt not extinguishing in pvlv -- need it to be stronger -- now that BLA Acq inhib is 2 instead of 2.2



# References

* Alexander, G. E., DeLong, M. R., & Strick, P. L. (1986). Parallel organization of functionally segregated circuits linking basal ganglia and cortex. Annual Review of Neuroscience, 9, 357–381. http://www.ncbi.nlm.nih.gov/pubmed/3085570

* Boehnke, S. E., Berg, D. J., Marino, R. A., Baldi, P. F., Itti, L., & Munoz, D. P. (2011). Visual adaptation and novelty responses in the superior colliculus. European Journal of Neuroscience, 34(5), 766–779. https://doi.org/10.1111/j.1460-9568.2011.07805.x

* Bouton, M. E. (2004). Context and behavioral processes in extinction. Learning & Memory, 11(5), 485–494. http://dx.doi.org/10.1101/lm.78804

* Brischoux, F., Chakraborty, S., Brierley, D. I., & Ungless, M. A. (2009). Phasic excitation of dopamine neurons in ventral {VTA} by noxious stimuli. Proceedings of the National Academy of Sciences USA, 106(12), 4894–4899. http://www.ncbi.nlm.nih.gov/pubmed/19261850

* Chernysheva, M., Sych, Y., Fomins, A., Warren, J. L. A., Lewis, C., Capdevila, L. S., Boehringer, R., Amadei, E. A., Grewe, B. F., O’Connor, E. C., Hall, B. J., & Helmchen, F. (2021). Striatum-projecting prefrontal cortex neurons support working memory maintenance (p. 2021.12.03.471159). bioRxiv. https://doi.org/10.1101/2021.12.03.471159

* Dautan, D., Souza, A. S., Huerta-Ocampo, I., Valencia, M., Assous, M., Witten, I. B., Deisseroth, K., Tepper, J. M., Bolam, J. P., Gerdjikov, T. V., & Mena-Segovia, J. (2016). Segregated cholinergic transmission modulates dopamine neurons integrated in distinct functional circuits. Nature Neuroscience, 19(8), Article 8. https://doi.org/10.1038/nn.4335

* Dutta, A., & Gutfreund, Y. (2014). Saliency mapping in the optic tectum and its relationship to habituation. Frontiers in Integrative Neuroscience, 8. https://www.frontiersin.org/articles/10.3389/fnint.2014.00001

* Elman, J. L. (1990). Finding structure in time. Cognitive Science, 14(2), 179–211.

* Gerfen, C. R., & Surmeier, D. J. (2011). Modulation of striatal projection systems by dopamine. Annual Review of Neuroscience, 34, 441–466. http://www.ncbi.nlm.nih.gov/pubmed/21469956

* Guo, K., Yamawaki, N., Svoboda, K., & Shepherd, G. M. G. (2018). Anterolateral motor cortex connects with a medial subdivision of ventromedial thalamus through cell type-specific circuits, forming an excitatory thalamo-cortico-thalamic loop via layer 1 apical tuft dendrites of layer 5b pyramidal tract type neurons. Journal of Neuroscience, 38(41), 8787–8797. https://doi.org/10.1523/JNEUROSCI.1333-18.2018

* Hazy, T. E., Frank, M. J., & O’Reilly, R. C. (2010). Neural mechanisms of acquired phasic dopamine responses in learning. Neuroscience and Biobehavioral Reviews, 34(5), 701–720. http://www.ncbi.nlm.nih.gov/pubmed/19944716 [PDF](https://ccnlab.org/papers/HazyFrankOReilly10.pdf)

* Heckhausen, H., & Gollwitzer, P. M. (1987). Thought contents and cognitive functioning in motivational versus volitional states of mind. Motivation and Emotion, 11(2), 101–120. https://doi.org/10.1007/BF00992338

* Herry, C., Ciocchi, S., Senn, V., Demmou, L., Müller, C., & Lüthi, A. (2008). Switching on and off fear by distinct neuronal circuits. Nature, 454(7204), 1–7. http://www.ncbi.nlm.nih.gov/pubmed/18615015

* Huerta-Ocampo, I., Dautan, D., Gut, N. K., Khan, B., & Mena-Segovia, J. (2021). Whole-brain mapping of monosynaptic inputs to midbrain cholinergic neurons. Scientific Reports, 11, 9055. https://doi.org/10.1038/s41598-021-88374-6

* Matsumoto, M., & Hikosaka, O. (2007). Lateral habenula as a source of negative reward signals in dopamine neurons. Nature, 447, 1111–1115. http://www.ncbi.nlm.nih.gov/pubmed/17522629

* Matsumoto, O., & Hikosaka, M. (2009). Representation of negative motivational value in the primate lateral habenula. Nature Neuroscience, 12(1), 77–84. http://www.citeulike.org/user/nishiokov/article/3823302

* McDannald, M. A., Lucantonio, F., Burke, K. A., Niv, Y., & Schoenbaum, G. (2011). Ventral striatum and orbitofrontal cortex are both required for model-based, but not model-free, reinforcement learning. The Journal of Neuroscience, 31(7), 2700–2705. https://doi.org/10.1523/JNEUROSCI.5499-10.2011

* Mollick, J. A., Hazy, T. E., Krueger, K. A., Nair, A., Mackie, P., Herd, S. A., & O'Reilly, R. C. (2020). A systems-neuroscience model of phasic dopamine. Psychological Review, 127(6), 972–1021. https://doi.org/10.1037/rev0000199.  [PDF](https://ccnlab.org/papers/MollickHazyKruegerEtAl20.pdf)

* Root, D. H., Melendez, R. I., Zaborszky, L., & Napier, T. C. (2015). The ventral pallidum: Subregion-specific functional anatomy and roles in motivated behaviors. Progress in Neurobiology, 130, 29–70. https://doi.org/10.1016/j.pneurobio.2015.03.005

* Omelchenko, N., & Sesack, S. R. (2005). Laterodorsal tegmental projections to identified cell populations in the rat ventral tegmental area. The Journal of Comparative Neurology, 483(2), 217–235. https://doi.org/10.1002/cne.20417

* O’Reilly, R. C. (2020). Unraveling the Mysteries of Motivation. Trends in Cognitive Sciences. https://doi.org/10.1016/j.tics.2020.03.001

* O’Reilly, R. C., Russin, J. L., Zolfaghar, M., & Rohrlich, J. (2020). Deep Predictive Learning in Neocortex and Pulvinar. ArXiv:2006.14800 [q-Bio]. http://arxiv.org/abs/2006.14800

* O’Reilly, R. C., Frank, M. J., Hazy, T. E., & Watz, B. (2007). PVLV: The primary value and learned value Pavlovian learning algorithm. Behavioral Neuroscience, 121(1), 31–49. http://www.ncbi.nlm.nih.gov/pubmed/17324049 [PDF](https://ccnlab.org/papers/OReillyFrankHazyEtAl07.pdf)

* Sturgill, J. F., Hegedus, P., Li, S. J., Chevy, Q., Siebels, A., Jing, M., Li, Y., Hangya, B., & Kepecs, A. (2020). Basal forebrain-derived acetylcholine encodes valence-free reinforcement prediction error (p. 2020.02.17.953141). bioRxiv. https://doi.org/10.1101/2020.02.17.953141

* Tobler, P. N., Dickinson, A., & Schultz, W. (2003). Coding of predicted reward omission by dopamine neurons in a conditioned inhibition paradigm. Journal of Neuroscience, 23, 10402–10410. http://www.ncbi.nlm.nih.gov/pubmed/14614099

* Waelti, P., Dickinson, A., & Schultz, W. (2001). Dopamine responses comply with basic assumptions of formal learning theory. Nature, 412, 43–48. http://www.ncbi.nlm.nih.gov/pubmed/11452299


