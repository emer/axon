# Rubicon Goal-Driven Motivated Behavior Model

This document describes the [Rubicon](../../Rubicon.md) model for goal-driven motivated behavior, which posits distinct  **goal-selection** vs. **goal-engaged** states of the brain ([O'Reilly, 2020](https://ccnlab.org/papers/OReilly20.pdf); [Heckhausen & Gollwitzer, 1987](#references)).  In the goal selection phase, different options are considered and evaluated according to learned cost-benefit _utilities_ with the following brain areas protypically representing the following information:

* **Benefit** (expected positive _outcome_) in the **OFC** (orbital frontal cortex), **IL** (infralimbic cortex), and **aIC** (anterior insular cortex).

* **Cost** of actions in dorsal **ACC** (anterior cingulate cortex, or just Cg in rodent).

* **Utility** (Benefit vs Cost) in **PL** (prelimbic cortex).

* **Action plan** in the **dlPFC** (dorsolateral prefrontal cortex), which is  **ALM** (anterior lateral motor area) in rodents.

Each of these areas mutually informs the other through bidirectional _constraint satisfaction_ to propose a potential overall goal / plan state across these three areas, as discussed in [Herd et al., 2021](https://ccnlab.org/papers/HerdKruegerNairEtAl21.pdf).

If the proposed goal / plan is selected via BG (basal ganglia, implemented via [PCore](PCoreBG.md)) _gating_, it then drives stable active maintenance of this _goal state_ which is a distributed representation across these PFC areas.  This maintained goal state then drives coordinated behavior toward achieving the expected outcome through the selected action plan.

The maintenance of this consistent goal state is critical for allowing learning at the time when the goal is either achieved or abandoned, to update the representations that then drive goal selection behavior in the future, so that these choices can be informed by what actually happened on previous choices.  This then provides a solution to the _temporal credit assignment_ problem.

The Rubicon model subsumes the PVLV [Mollick et al, 2020](#references) (Primary Value, Learned Value) framework for phasic dopamine firing in the Pavlovian, classical conditioning paradigm.  This provides the learning signals at the time of the outcome that drive learning in the BG.

Implementation files: rubicon_{[net.go](axon/rubicon_net.go), [layers.go](axon/rubicon_layers.go), [prjns.go](axon/rubicon_prjns.go)}.  Examples / test cases: [pvlv](examples/pvlv), [choose](examples/choose).

# Introduction and Overview

The integration of classical (passive, Pavlovian) conditioning with the mechanisms of the goal-driven system represents a useful way of understanding both of these mechanisms and literatures. In effect, classical conditioning leverages the core goal-driven mechanisms in a simplified manner, making it a useful paradigm for testing and validating these mechanisms.

In the goal-driven paradigm, the Pavlovian _conditioned stimulus_ (CS) is a signal for the opportunity to obtain a desired outcome (i.e., the _unconditioned stimulus_ or US).  Thus, a CS activates a goal state in the vmPFC (ventral and medial prefrontal cortex), specifically OFC, IL, and PL, in the Rubicon framework.  Notably, the passive nature of the Pavlovian paradigm means that no additional motor plan representations (e.g., in dlPFC, ALM) are needed.  This goal state is then maintained until the US outcome occurs, and helps establish the CS-US association at the core of classical conditioning.

The most relevant and challenging aspect of classical conditioning in this context is recognizing when an expected US is _not_ actually going to happen.  A non-reward trial is, superficially, indistinguishable from any other moment in time when nothing happens.  It is only because of the maintained internal expectation that it takes on significance, when this nothing happens instead of the expected something.  In the Rubicon framework, this expectation is synonymous with the maintained goal state, and thus a non-reward trial represents a goal failure state, and the proper processing of this goal failure requires a decision to [give up](#give-up) on the engaged goal.  The mechanics of this give-up decision and resulting dopamine and other learning effects are thus tested by simple partial reinforcement and extinction conditioning paradigms, in ways that were explored to some extent in the PVLV paper [Mollick et al, 2020](#references).

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

The ventral portion of the basal ganglia (VBG) plays the role of the final "Decider" in the Rubicon framework ([Herd et al., 2021](https://ccnlab.org/papers/HerdKruegerNairEtAl21.pdf)).  The input layer to the VBG is the _ventral striatum_ (**VS**) (also known as the _nucleus accumbens_ or NAcc), which receives from the vmPFC goal areas (and other relevant sensory input areas), and projects to the _VP_ (ventral pallidum), which then projects to the **MD** thalamus, which is bidirectionally interconnected with the vmPFC goal areas (Figure 1).  If the VBG decides to select the currently-proposed goal state represented in vmPFC, this results in a disinhibition of the MD thalamus, opening up the corticothalamic loop, which updates and locks in the goal representation.  This gating dynamic was originally captured in the PBWM (prefrontal-cortex basal-ganglia working memory) model ([O'Reilly & Frank, 2006; O'Reilly, 2006)](#references), and has recently been supported by detailed recording and optogenetic manipulation studies in the Svoboda lab [(Inagaki et al, 2022; 2018; Li et al., 2015; Guo et al., 2017; 2014)](#references).

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

* `CT` (`CT` suffix): implements the layer 6 regular spiking CT _corticothalamic_ neurons that project into the thalamus, and are primarily responsible for **generating continuous predictions** over the thalamus based on temporally-delayed _context_ signals, functioning computationally like the simple recurrent network (SRN) context [(Elman, 1990)](#references).

* `Pulvinar` (`P` suffix): implements the Pulvinar-like TRC (thalamic relay cell) neurons, upon which the prediction generated by CT layer is projected in the _minus phase_ (first part of the theta cycle) -- the `P` suffix thus also stands for **prediction**.  Then, in the _plus phase_ (second part of the theta cycle), the actual _outcome_ relative to that prediction is driven by strong driver inputs, originating from layer 5IB (intrinsic bursting) neurons from other neocortical areas, or from subcortical areas such as the cerebellum, motor brainstem, or superior colliculus.  The prediction error is thus the difference between these two phases of activity, and it is broadcast back up to the Super and CT layers to drive predictive learning that improves predictions over time.  Biologically, the pulvinar is the prototype for this predictive learning thalamus, for posterior cortical areas, but different parts of the **MD** likely play this role for PFC areas [(Rovo et al., 2012)](#references).  Also see [Root et al. (2015)](#references) for major review of VS -> MD pathways and associated recording, anatomy etc data.  

* `PTMaint` (`PT` suffix): implements a subset of _pyramidal tract_ (PT) layer 5b (deep) neurons that exhibit **robust active maintenance of the goal state**, and project extensively to subcortical areas, including the motor spinal cord for primary motor cortex areas.  These are gated by basal ganglia disinhibition of their corresponding **MD** thalamic layer, which modulates the NMDA-laden recurrent collaterals that sustain activity: this is the mechanism for BG gating and goal maintenance.   The final outcome predictions are driven by this PT layer activity, because it is stable throughout the goal state.

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

There are many brain areas involved in controlling the phasic firing of dopamine cells in the VTA (ventral tegmental area) and SNc (substantia nigra, pars reticulata). The PVLV model integrates contributions from the most important of these areas within a coherent overall computational framework including: 1) multiple sub-regions of the amygdala, an area long implicated in affective processing of both positive and negative emotion; 2) multiple pathways within the ventral striatum (VS, which includes the nucleus accumbens, NAc), also important in many aspects of emotional expression; and, 3) the lateral habenula (LHb) pathway, recently identified as the substrate responsible for the inhibitory pausing (dipping) of dopamine neuron activity [Matsumoto & Hikosaka, 2007; Matsumoto & Hikosaka, 2009](#references).

The basic functions of the model can be seen in Pavlovian conditioning:

* Initially neutral cues (**conditioned stimuli; CSs**) are paired with rewards or punishments (**unconditioned stimuli; USs**), resulting in the acquisition of conditioned associations between CS -> US.

* Critically, phasic dopamine responses that initially occur for unexpected USs come to occur at the time of the CS instead.

PVLV models the neurobiological mechanisms that cause this change in dopamine signaling to occur.  The overarching idea behind the PVLV model [OReilly et al, 2007](#references) is that there are two separate brain systems underlying two separate aspects of reward learning:

* The **Primary Value (PV)** (US outcome predicting) and **Learned Value (LV)** (CS predictive cue learning) systems.

* **PV = ventral striatum**, which learns to expect US outcomes and causes the phasic dopamine signal to reflect the difference between this expectation and the actual US outcome value experienced. This difference is termed a *reward prediction error* or *RPE*.

* **LV = amygdala**, which learns to associate CSs with US outcomes (rewards and punishments), thus acquiring new CS-outcome value associations (learned value).

This division of labor is consistent with a considerable amount of data [Hazy et al, 2010](#references). The 2020 PVLV model has a greatly elaborated representation of the amygdala and ventral striatal circuitry, including explicitly separate pathways for appetitive vs. aversive processing, as well as incorporating a central role for the *lateral habenula* (LHb) in driving pauses in dopamine cell firing (dipping) for worse than expected outcomes. Figure 1 provides a big-picture overview of the model.

![PV.2](figs/fig_bvpvlv_pv_lv_only.png?raw=true "PV.2")

**Figure 2:** Simplified diagram of major components of the PVLV model, with the LV Learned Value component in the Amygdala and PV Primary Value component in the Ventral Striatum (principally the Nucleus Accumbens Core, NAc).  LHb: Lateral Habenula, RMTg: RostroMedial Tegmentum, LDT: Laterodorsal Tegmentum, LHA: Lateral Hypothalamus, PBN: Parabrachial Nucleus. 

Note that we use anatomical labels for computationally-specified functions consistent with our theory, without continually reminding the reader that of course this is all a simplified theory for what these brain areas are actually doing.  If it is useful for you, just imagine it says "we hypothesize that the function of area X is.." everywhere.

# Timing: Trial-wise

In contrast to the minus-plus phase-based timing of cortical learning, the RL-based learning in PVLV is generally organized on trial-wise boundaries, with some factors computed online within the trial.  Here is a schematic, for an intermediate amount of positive CS learning and VSPatch prediction of a positive US outcome, with an "Eat" action that drives the US:

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

# A Central Challenge: Learning *Something* when *Nothing* happens

As articulated in the PVLV papers, a central challenge that any RL model must solve is to make learning dependent on expectations such that the *absence* of an expected outcome can serve as a learning event, with the appropriate effects.  This is critical for **extinction** learning, when an expected reward outcome no longer occurs, and the system must learn to no longer have this expectation of reward.  This issue is particularly challenging for PVLV because extinction learning involves different pathways than initial acquisition (e.g., BLA Ext vs. Acq layers, VSPatch, and the LHb dipping), so indirect effects of expectation are required.

The expectation in PVLV is carried by active maintenance in the OFC and other vmPFC areas, engaged by the CS onset, which then project to the ventral striatum neurons that drive shunting inhibition and dipping via the LHb pathways.

In the context of the full BG / OFC / ACC ([BOA](examples/boa/README.md)) goal-directed framework (i.e., the *Rubicon* model), this active maintenance is a full *goal* representation that includes an expected outcome (OFC) and an action plan (dlPFC) and expected utility (ACC, PL = prelimbic).  In this case, an extinction event is equivalent to *goal failure*, when the expected (desired) outcome did not occur.  Thus, it is associated with the deactivation of any maintained goal state, and any additional cost associated with failure (at least the effort expended so far).  In this context, it is also possible (likely?) that outcome expectations in a simple Pavlovian context could be dissociable from a full goal-engagement state, involving only the OFC-specific portion of the full goal-engaged activity state.

In the 2020 version of PVLV, activation of the BLA by a CS, and subsequent activity of the external USTime input layer, represented the expectation and provided the *modulatory* activity on BLAExt and VSPatch to enable extinction learning conditioned on expectations.  The model relied heavily on these modulatory and externally-generated representations -- the current Axon version takes the next step in implementing these in a more realistic and robust manner.

In the current version, the gated goal engaged state, corresponding to *OFC PT layer sustained activity*, is the key neural indicator of an active expectation.  Thus, we need to ensure that the model exhibits proper CS-driven BG (VP) gating of active maintenance in OFC PT, and that this maintained activity is available at the right time (and not the wrong time), and in the right form, to drive extinction learning in the right way.  Thus, the PT layers are the core "backbone" of the model, bridging between the LV and PV sides.

The key logic for extinction (in a positive valence context), and related RPE (reward prediction error) expectation-dependent dynamics of DA bursting, is that the `VSPatch` layer (called `PVi` = Primary Value inhibition in earlier PVLV models) learns to expect when a primary value (PV) US (unconditioned stimulus) outcome occurs, and it drives both shunting inhibition to the VTA (to reduce firing when the US does occur) and also dipping via the LHb when the US does not occur.

# LHb: The Brain's Extinguisher

The LHb (lateral habenula) turns an expected (via VSPatch inputs) but absent US non-event into a full-fledged active neural signal, which drives three distinct outputs:

* **DA dipping**: The combination of an active VSPatch input signaling the expectation of a US at the current point in time, with no bottom-up actual US receipt (e.g., via hypothalamic inputs), results in activity of a subset of LHb neurons, which then drive dipping (pausing) of DA tonic activity (via the inhibition provided by the RMTg).  This phasic dip in DA activity shifts the balance from D1 to D2 in all DA-recipient neurons in the BG and BLA, causing learning to start to expect this absence (see below on BLA Ext pathway).

* **OFC / goal gating off**: the LHb dip activation is summed over trials, and when it reaches a threshold level, the system effectively "gives up" on the current US expectation.  This amounts to deactivating any existing goal state (i.e., the OFC maintained activity in this model), and in the goal-driven learning framework (BOA), it also entails "paying the cost" of accumulated effort toward the goal (i.e., extra negative US / DA dipping -- gets applied to the first USneg value), which may be associated with the subjective sense of "disappointment".  Biologically, this is thought to occur via MD thalamic projections to OFC, ACC, dlPFC areas, which are the same pathways activated when an actual US is received and likewise deactivates these areas.  Implementationally, it happens simply by setting the `HasRew` flag in the `Context.NeuroMod` structure, which triggers decay of the relevant PFC areas (via the `Act.Decay.OnRew` flag).  This happens at the end of the Trial.

* **ACh signaling**: ACh (acetylcholine) is released for salient events, basically CS onset (via superior colliculus to LDT = laterodorsal tegmentum) and US onset, and it modulates learning and activity in the BLA and VS.  The LHb projections to the basal forebrain cholinergic system allow it to provide the key missing piece of ACh signaling for the absence of an expected US, so that a consistent framework of ACh neuromodulation can apply for all of these cases.

Taken together, these key functions provide a compelling computational role for why the brain has a separate neural system for recognizing the absence of an expected US.  In the mathematics of the temporal-difference (TD) equations, a negative TD signal associated with a missing expected reward is no different than the reduction associated with the prediction of a reward that does occur, but the brain treats these two very differently.  The VSPatch provides a shunting-only effect directly to the VTA to reduce dopamine firing for expected rewards (USs) that occur, but the LHb is a special system for the case where USs fail to occur.

See the [Give up](#give-up) section below for more details.

# PT (Pyramidal Tract) Sustained and Dynamic Goal Maintenance

Pyramidal tract (PT) neurons in layer 5 of each PFC area provide the robust active maintenance of goal state over time, serving as a sustained *bridge* between the initiation of the goal at the time of the CS to the time of the US.  See the `PTMaintLayer` (PT) and `PTPredLayer` (PTp) in the [deep](DEEP.md) deep predictive learning framework.

These PT layers are interconnected with the MD thalamus (and other thalamic areas such as VM = ventromedial, VA = ventral anterior, VL = ventral lateral) and are the primary output neurons of the cortex.  In M1 (primary motor cortex), these are the neurons that project to the spinal cord muscle outputs and actually drive motor behavior.  The thalamic connections are "side branches" of these descending subcortical pathways.  See [Guo et al. (2018)](#references).  There is a large diversity of PT neuron "subtypes" with different temporal and content-based response properties -- we are only implementing the subset of active maintenance (working memory) PT neurons.

The BG ventral striatum (VS) & ventral pallidum (VP) drives disinhibitory gating of active maintenance in the PT layers, consistent with the longstanding ideas in the PBWM (Prefrontal-cortex, Basal-ganglia Working Memory) framework and recent updates thereof.  The following are specific issues that need to be resolved in the implementation to make everything work properly in the PVLV context.

## Time of learning vs. gating issues

Gating happens within the CS-onset theta cycle, driven by direct BLA recognition of the CS and parallel superior colliculus (SC) phasic activity to CS stimulus onset, which drives LDT (laterodorsal tegmentum) ACh, which disinhibits the BG.  This results in an activated OFC PT layer by the end of the trial.  This is problematic in the PVLV context, because the active PT layer will drive activity and learning in the VSPatch layer consistent with the goal engaged state.  In the 2020 PVLV model, the USTime input was programmed to activate only on the post-CS trial.

## Time / context specificity of PT activity vs. Stable maintenance

There is a basic tension in PT maintained activity, which can be resolved by having two different PT layer types: `PTMaintLayer` (`PT` suffix) and `PTPredLayer` (`PTp` suffix).  On the one hand, PT needs to maintain a stable representation of the stimulus and other context present at the time of initial goal activation (gating), so that learning can properly bridge across the full time window between the initial gating and final outcome.  This is what `PTMaintLayer` achieves, using strong recurrent NMDA projections that stably maintain the gated activity state.  The [pfcmaint](examples/pfcmaint) example test project tests this mechanism specifically, including the `SMaint` self maintenance mechanism that simulates a larger population of interconnected neurons.

On the other hand, prediction of specific events within the goal-engaged window, and especially the prediction of when the goal will be achieved, requires a continuously-updating dynamic representation, like that provided by the CT layer, which is specifically driven by predictive learning of state as it updates across time.  However, the CT layer is always active, and thus does not strongly distinguish the critical difference between goal engaged vs. not.  Furthermore, anatomically, CT only projects to the thalamus and does not have the broad broadcasting ability of the PT layer.  Electrophysiologically, there is plenty of evidence for both sustained and dynamically updating PT activity.  As usual, these are not strongly segregated anatomically, but we use different layers in the model to make the connectivity and parameterization simpler.

The `PTPredLayer` represents the integration of PT stable maintenance and CT dynamic updating.  This layer type receives a temporally-delayed `CTCtxtPrjn` from the corresponding `PTMaintLayer`, which also solves the timing issue above, because the temporal delay prevents activity during the CS gating trial itself.  The PTPred layer is parameterized to not have strong active maintenance NMDA currents, and to track and help predict the relevant dynamic variables (time, effort etc).  See [deep](DEEP.md) for more info.

## Extinction learning and goal inactivation

As noted above, the LHb will drive deactivation of the PT active maintenance layer, signaling that the expected US outcome was not achieved.  Because the PT drives input into the VSPatch expectation layer, it then no longer signals the expectation.  Thus, it is important for both the goal-driven and pavlovian paradigms to deactivate the PT maintenance at this point.

# BLA: Basolateral Amygdala

The basolateral amygdala learns to associate an initially neutral CS with the US that it is reliably associated with.  This learning provides a key step in enabling the system to later recognize the CS as a "trigger" opportunity to obtain its associated US -- if that US is consistent with the current Drive state (e.g., the CS is associated with food and the system is "hungry"), then it should engage a goal to obtain the US.  In the classical conditioning paradigm pioneered by Pavlov, the CS was a bell and the US was food.  The BLA learns the link between the bell and the food, effectively "translating" the meaning of the CS into pre-existing pathways in the brain that process different USs, thus causing Pavlov's dogs to salivate upon hearing the tone. 

The BLA learns at the time of the US, in response to dopamine and ACh (see below), so any stimulus or memory representation reliably present just before the onset of the US is then mapped onto the corresponding US-specific Pool in the BLA layer.  `BLAposAcqD1` is the positive-valence *acquisition* layer expressing more D1 dopamine receptors, and it is opposed (inhibited) by a corresponding `BLAposExtD2` D2-dominant *extinction* layer.  The PVLV model of [Mollick et al. (2020)](https://ccnlab.org/papers/MollickHazyKruegerEtAl20.pdf) describes all of this in great detail. 

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
    
## Extinction learning

The `BLAposExtD2` extinction layer provides the "context specific override" of the acquisition learning in `BLAposAcqD1`, which itself only unlearns very slowly if at all.  It achieves this override in 3 different ways:

* **Direct inhibition** of corresponding BLA Acq neurons in the same US pool.  This is via a projection with the `BLAExtToAcq` class, and setting the `Prjn.PrjnScale.Abs` value of this projection controls the strength of this inhibition.

* **Opposing at CeM** -- the central nucleus layer, `CeMPos`, combines excitation from Acq and inhibition from Ext, and then drives output to the VTA -- so when Ext is active, CeM gets less active, reducing DA drive on VTA bursting.

* **VS NoGo** -- Ext projects to the ventral striatum / ventral pallidum (Vp) NoGo layer and can oppose gating directly there.

A key challenge here is to coordinate these two layers with respect to the US expectation that is being extinguished.  In the Leabra PVLV version a modulatory projection from Acq to Ext served this function.

In this model, we instead leverage the OFC PT active maintenance to select the US that is being expected, as a consequence of BLA Acq triggering active maintenance gating of the specific expected US outcome, which then projects into BLA Ext.  

To work with the Novelty case (below), and leverage the delta-based learning rule in acquisition, we make the `OFC PTpred` projection into the `Ext` layer modulated by ACh, and `Ext` is also inhibited by the US itself.  This means that it will only activate at a *goal failure* event.  It is also possible to use a `CtxtPrjn` projection to have BLA Acq drive Ext in a trial-delayed manner (this is also ACh gated), to avoid any dependence on PT active maintenance.

Importantly, while the learning of Ext is driven by the maintained PTpred, it must have a fast CS-driven activation in order to oppose the Acq pool.  Thus, this CS-driven pathway (which includes contextual inputs via the hippocampus and other areas) is the primary locus of learning, and it also is not ACh gated because it must remain active persistently to counter the Acq activation. 

## Novelty & Curiosity Drive

<img src="figs/fig_pvlv_bla_novelty.png" height="600">

The first pool in the BLAposAcq / PosExt layers is reserved for the "novelty" case (see above figure).  This pool is driven by persistent excitatory input and active by default unless inhibited by another layer -- i.e., a CS is novel until proven otherwise.  If the pursuit of the CS leads to no positive US outcome, then the corresponding extinction layer will learn to oppose the positive novelty activation.  If it leads to a US outcome, then that association will be learned, and the CS will then activate that US instead of novelty.

The delta learning rule for novelty extinction works via the same extinction logic as regular US pathways: if no US outcome occurs, and the maximum effort limit has been reached, then the LHb dip / reset dynamic is activated, triggering phasic ACh release as if an actual US had occurred.

### BLANovelCS

The first, novelty pool in the BLA requires a stimulus input to drive it when a novel CS comes on.  This is accomplished in a simple way in the model through a layer (`BLANovelCS`) with fixed (non-learning) synaptic weights from all CS inputs, such that any given CS will activate a particular activity pattern over this layer.  This layer projects into the first BLApos pools, and drives activity there for novel CSs.  When a specific CS acquires a specific US outcome representation in the BLA, that pathway should out-compete the "default" BLANovelCS pathway.

## CeM

For now, CeM is only serving as an integration of BLA `Acq` and `Ext` inputs, representing the balance between them.  This uses non-learning projections and mostly CeM is redundant with BLA anyway.  This also elides the function CeL.

## Central Amygdala: CeM

 **CeM -> PPTg -> ACh** This pathway drives acetylcholine (ACh) release in response to *changes* in BLA activity from one trial step to the next, so that ACh can provide a phasic signal reflecting the onset of a *new* CS or US, consistent with available data about firing of neurons in the nucleus basalis and CIN (cholinergic interneurons) in the BG [(Sturgill et al., 2020)](#references).  This ACh signal modulates activity in the BG, so gating is restricted to these time points.  The `CeM` (central nucleus of the amygdala) provides a summary readout of the BLA activity levels, as the difference between the `Acq - Ext` activity, representing the overall CS activity strength.  This goes to the `PPTg` (pedunculopontine tegmental nucleus) which computes a temporal derivative of its CeM input, which then drives phasic DA (dopamine, in VTA and SNc anatomically) and ACh, as described in the PVLV model [(Mollick et al., 2020)](#references).

# SC -> LDT ACh

In the [Mollick et al. (2020)](#references) version, the PPTg (pedunculopontine tegmentum) directly computed a temporal derivative of amygdala inputs, to account for phasic DA firing only at the *onset* of a CS, when the BLA has sustained CS activity.  However, updated research shows that this is incorrect, and it is also causes problems functionally in the context of the BOA model.

In the new model, ACh (acetylcholine) from the LDT (laterodorsal tegmentum) provides a *disinhibitory* signal for VTA bursting, triggered by CS onset by the SC (superior colliculus), such that the sustained CeM -> VTA drive for a learned CS only results in bursting at CS onset, when LDT ACh is elevated from CS -> SC novel input.

* `SC` = superior colliculus, which shows strong *stimulus specific adaptation* (SSA) behavior, such that firing is strong at the onset of a new stimulus, and decreases by 60% or so thereafter [(Dutta & Gutfreund, 2014; Boehnke et al, 2011)](#references).  Thus, the temporal derivative onset filter is on the pure sensory side, not after going through the amygdala.

* `LDT` = laterodorsal tegmentum, which is the "limbic" portion of the mesopontine tegmentum, which also contains the pedunculopontine nucleus / tegmentum (PPN or PPT) [(Omelchenko & Sesack, 2005; Huerta-Ocampo et al. (2021)](#references).  The LDT receives primarily from the SC and OFC, ACC, and barely anything from the amygdala (same with PPN, which is more strongly connected to motor circuits and sends more output to SNc -- basically a motor version of LDT).  LDT contains glutamatergic, GABAergic, and ACh neurons (co-releasing at least GABA and ACh in some cases), and strongly modulates firing in the VTA, inducing bursting [(Dautan et al. (2016)](#references).  LDT also drives neurons in the nucleus basalis and CIN (cholinergic interneurons) in the BG, which all have consistent CS / US onset salience firing [(Sturgill et al., 2020)](#references).  LDT ACh has widespread projections to the BG (including SNr / GPi and thalamus), modulating gating to only occur at onset of novel stimuli.

Furthermore, the OFC & ACC input to LDT serves to inhibit ACh salience signals when there is already an engaged goal, to reduce distraction.  To enable this inhibition, it is critical to add this code in the `ApplyPVLV` function called at the start of each trial:

```Go
	pv.SetGoalMaintFromLayer(ctx, di, ss.Net, "PLposPT", 0.3) // PT layer is  maint
```

which sets the `GoalMaint` global variable, that then drives inhibition of ACh in LDT via its `MaintInhib` parameter.

## SC

The SC layer has a relatively-strong trial scale adaptation current that causes activity to diminish over trials, using the sodium-driven potassium channel (KNa):
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

# Negative USs and Costs

There are two qualitatively-different types of negative outcome values, which require distinct pathways within the model:

* `USneg`: Phasic, discrete "events", such as a shock or impact, that can be associated with a stimulus (CS) or action, and thereby potentially predicted and avoided in the future.  The BLA plays a well-established role here for establishing CS -- US associations (e.g., tone-shock in the widely-used fear conditioning paradigm), in the same way as in the positive valence case.

* `Costs`: Continuous, inevitable costs, specifically Time and Effort, and lower-grade forms of pain, that animals are always seeking to minimize.  These are not associated with phasic CSs, but rather vary as a function of motor plans (e.g., climbing a steep hill vs. walking along a flat path).  These negative outcomes do not engage the BLA because they are not primarily CS-associated, and instead are predicted from motor-cingulate areas.

# Online and Final Estimates

Like the distinction between `PTMaint` and `PTPred`, we need both a stable representation of the expected final outcomes for positive and negative cost factors, and an online incrementing estimate of our sense of progress toward the positive outcome, and accumulated cost.

For costs, the online increment is entirely straightforward, because we get that directly bottom-up, by hypothesis.  However, there is no obvious representation of the final accumulated cost estimate.

For positive outcomes (USs), the BLA provides a concrete representation of the expected US, which can be directly used to estimate value of final outcome, and it directly influences OFC continuously, so this information is widely available.  However, it is not explicitly represented as _canonical_ value-based code, e.g., a population-coded value used for all different outcomes.  Furthermore, there is no obvious representation of the incremental accumulating value.

Resolving these remaining uncertainties in these representations is a critical path TODO item, and they strongly impact the GiveUp computation as described in the next section.  Various possibilities are enumerated below.

## Emergent PFC representations

The working hypothesis has been that the PTMaint layers learn to reliably encode the final outcome values, while the PTPred layers learn the online incrementing estimates.  However, this has not yet been the main focus of investigation and optimization.

At a minimum, we likely need some more focus on the learning that happens at the time of the outcome: this needs to shape the representations in PTMaint layers, which in turn are driven by the superficial PFC layers at the time of goal gating.  For the OFCpos case, given the BLA US encoding, this seems reasonable, but we likely need to modulate the learning rate at the time of US outcome to encourage the proper learning.  And for this case, the question of incremental learning is more uncertain.

For the ACCcost layer, it is unclear how it might work, given the lack of a final cost representation.

One specific question is whether we can usefully create a focused test project to examine this question specifically.  Probably.  That should be the next goal.

## Explicit final value, cost representations

One possible additional step would be to add layers that just reflect the final estimated positive and cost values.  At the very least, we need to re-add these as decoding layers.  But it would make sense to have a brain area that just represents that explicitly -- receives from PT, and learns only at US for each value respectively, and can then be used for estimates.  Will do this now.

## Incremental positive value progress representation

Likewise, it makes sense to add an explicit abstract value progress layer.  How does it work though?  One idea is that you just train with PTp -- basically pvPosP.  first step is to start reading out that value to see how it is going.

# Give up

The decision to give up on an engaged goal is very difficult.  The value system is biased toward not giving up in general, because doing so requires accepting the accumulated costs as "disappointment" negative dopamine, and updates the expected outcome estimates for such a goal / plan in the future, which we don't want to be inaccurately pessimistic.  On the other hand, we don't want to expend energy and opportunity cost if the desired outcome is unlikely to actually happen, and cutting ones losses as quickly as possible is advantageous here.  This tradeoff is captured in the often-subtle distinction between _perseverance_ (sticking with your goal and ultimately getting the payoff) and _perseveration_ (failing to give up on a hopeless goal).

Giving up involves three separable factors, listed in precedence order:

* `Utility:` Weighing the accumulated costs relative to the potential expected outcome.  At some point, even if the desired outcome might occur, it would not be worth the expended effort and opportunity costs (when you could have been doing something else of greater potential value).  Because costs are known but the outcome is not, there is uncertainty in this factor, and computing the expected outcome value is difficult.  The result is a probability function reflecting this uncertainty.

* `Timing:` The likelihood that the expected positive outcome is not actually going to happen, based on specific learned reward predictions in the VSPatch.  VSPatch learns about specific discrete timing, in order to cancel the dopamine burst, and is penalized for anticipatory activity (that would generate a negative DA signal).  This yields a probability-like factor going from `NotYet` (0) to `Past` (1), which is computed using a sigmoidal function.

* `Progress:` An estimate of the current rate of progress toward the goal, which can be dissociated from the specific timing of the actual outcome provided by the VSPatch.  Even if the "usual" expected point of reward has passed, if still making progress, then it can make sense to persevere (until it becomes perseveration).

Each of these factors is computed as described in the sections below, and are integrated with the following logic:

1. If `Utility` goes negative, we give up regardless.  If `Utility` is strongly positive, that influences the extent to which we consider the remaining factors.  Thus, there are two functions computed on utility: `P(GiveUp | Utility)` is sigmoidal with high gain centered around the point where utility goes negative.  `P(ConsiderOtherFactors | Utility)` is more of a linear function of `1-Utility` (for normalized Utility).

2. If `Timing` is `NotYet`, we don't give up, regardless.  In novel domains where our estimate of timing is poor, this naturally biases the system to fall back on the overriding `Utility` factor.
    - It _might_ make sense to also include the Progress factor here if we're getting nothing from the Timing signal, but it is difficult to estimate future Timing factors in advance of actually getting a signal from the VSPatch (i.e., it is an implicit, weight-based signal).  Additional activation-based, explicit factors could be added in the future.

3. If `Timing` is `Past`, then `Progress` becomes relevant: can put off giving up as long as progress continues.  Need sufficient time integrals here to compute progress dynamics reliably.

## Utility

We already assume both a metabolic and time-based integrator of effort and opportunity costs, which are also estimated online via the ACCcost layers.  

  The VSPatch layer provides the best learned estimate of precisely when the outcome is expected, so it is the core driver of this mechanism.

However, VSPatch can be imprecise, and the cost of prematurely giving up is significant, so it is also important to include an estimate of current progress and proximity to the goal outcome: if you are currently making progress, then don't give up now -- instead, you should try harder!  This is the key logic of the norepinepherine (NE) system according to the Aston-Jones & Cohen model: it mediates between trying harder and giving up. Also, from the ADHD literature, it is potentially important for noise and exploration strategy dynamics.

There are two contributors to the sense of continued progress:

* VSPatch temporal derivative: if VSPatch goes down after being elevated, then it is clear that the predicted outcome is no longer being predicted.

TODO: Key idea: when rew pred delta goes negative (threshold) then give up.  Don't give up when making progress!!

* boa is currently giving up prior to getting reward; and pvlv is not generating enough negative DA on 50% B due to dip not reflecting full RPE.

* also, what about a generic rep of proximity to reward -- VSPatch is too precise -- currently USposP is prediction but is US specific.  USrewP or something?  need to give US more things to do to dynamically update.  Then can use this as an additional factor in Give up, to replace below:

* TODO: current mechanism is not very general -- uses OFCposPT layer to set GvOFCposPTMaint in layer_compute.go:PlusPhasePost, then in pvlv.go:PVposEst it uses this to compute PVposEst -- if currently maintaining then it assumes PVpos estimate is high..  


# Progress tracking

* leverages sensory distance as the original example;  SC starts with this?


# TODO / Issues


* BLAExt vs. Acq could be more robust -- Ext activity depends on PT -> Ext strength..


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


