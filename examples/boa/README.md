# BOA = BG, OFC, ACC

This model implements the `Rubicon` model for goal-driven motivated behavior, which posits distinct  **goal-selection** vs. **goal-engaged** states of the brain ([O'Reilly, 2020](https://ccnlab.org/papers/OReilly20.pdf); Heckhausen & Gollwitzer, 1987).  In the goal selection phase, different options are considered (explored) and evaluated according to learned cost-benefit *utilities* (represented in the *ACC* = anterior cingulate cortex), expected *outcome* in the *OFC* = orbital frontal cortex, and proposed *plan of action* in the *dlPFC* = dorsolateral prefrontal cortex, each of which mutually informs the other through bidirectional *constraint satisfaction* to propose a potential overall goal / plan state across these three areas, as discussed in [Herd et al., 2021](https://ccnlab.org/papers/HerdKruegerNairEtAl21.pdf).  If the proposed goal / plan is selected via BG (basal ganglia, implemented via [pcore](https://github.com/emer/axon/tree/master/pcore)) *gating*, then it drives stable active maintenance of this *goal state* which is a distributed representation across these PFC areas.  This maintained goal state then drives coordinated behavior toward achieving the expected outcome through the selected action plan.  Learning happens primarily when the goal is either achieved or abandoned, updating the expected utility, outcome, and action plan as a function of what actually happened during the goal engaged state.

## CS Approach task

The task paradigm is a simple ecologically-inspired task (a simplified version of the map-nav Fworld flat-world model).  The target behavior is to orient L / R until a CS sensory cue appears that is consistent with current Drive, and then move Forward until the Distance = proximal, and then Consume the US, receiving an amount of reward proportional to the value of the US minus the time and effort required to obtain it.

The main layers are:

* `Drives` = different body states (hunger, thirst, etc), satisfied by a corresponding US (unconditioned stimulus) outcome (food, water, etc).  These are detected and managed primarily in the hypothalamus and other such brainstem nuclei (PBN etc) and represented cortically in the insula (in posterior medial frontal cortex) as a primary interoceptive sensory area. More anterior areas of medial frontal cortex going into OFC represent the "PFC" for interoceptive states (for higher level control and active maintenance).

* `CS` = conditioned stimulus -- represents initially arbitrary sensory cues that are conceptually located by each US (simplest case is `CSPerDRive` = 1 -- one-to-one mapping), presented on a "fovea" input layer reflecting where the agent is looking.

* `Pos` = which of different locations where agent is currently looking, each of which can hold a different CS sensory cue.  Current location is an input, and determines contents of the fovea.  wraps around.

* `Dist` = distance to currently foveated CS

* `Time` = incrementing representation of time from last US received or CS detected.  In the brain, it is hypothesized that VS/VP gating triggers a reset of the timer, which itself is likely an interaction between subcortical and vmPFC drifting representations.

* `Act` and `VL` = Actions: `Forward, Left, Right, Consume`.  Consume happens at Dist = 0, Dist stays at 0 for a trial while consuming happens and the US is presented.  VL is the ventral lateral thalamus which is interconnected with primary motor cortex **M1** -- it represents the network's generated motor action, and then receives a plus phase subcortical input representing the cleaned-up version of the action it generated or the instinct-driven action, which is always shown in the Act layer.

The `PctCortex` factor determines the proportion of full approach trials (entire sequence of CS -> US) driven by the heuristic **instinct** behavior.  This behavior is acquired by the motor cortex over learning.  It is very simple: turn Left or Right (consistently if already turning) until a CS is detected that matches the current Drive, at which point move Forward, then Consume the US if Dist = 0. Before the model has learned about CSs, the instinct randomly approaches CSs after N turns. 

NOTE: aspects of above instinct behavior is still TODO!

## BG, OFC, ACC etc Layers and Functions

![BOA Areas](figs/fig_bg_loops_spiral_goals.png?raw=true "BOA Brain areas representing different aspect of a Goal")

![BOA Bridging Logic](figs/fig_boa_rubicon_logic.png?raw=true "Overall Time Bridging Logic")

* **BLA** = basolateral amygdala learns to associate an initially neutral CS with the US that it is reliably associated with.  This learning provides a key step in enabling the system to later recognize the CS as a "trigger" opportunity to obtain its associated US -- if that US is consistent with the current Drive state (e.g., the CS is associated with food and the system is "hungry"), then it should engage a goal to obtain the US.  In the classical conditioning paradigm pioneered by Pavlov, the CS was a bell and the US was food.  The BLA learns the link between the bell and the food, effectively "translating" the meaning of the CS into pre-existing pathways in the brain that process different USs, thus causing Pavlov's dogs to salivate upon hearing the tone.  The BLA learns at the time of the US, in response to dopamine and ACh (see below), so any stimulus reliably present just before the onset of the US is then mapped onto the corresponding US-specific Pool in the BLA layer.  `BLAPosAcqD1` is the positive-valence *acquisition* layer expressing more D1 dopamine receptors, and it is opposed (inhibited) by a corresponding `BLAPosExtD2` D2-dominant *extinction* layer.  The PVLV model of [Mollick et al. (2020)](https://ccnlab.org/papers/MollickHazyKruegerEtAl20.pdf) describes all of this in great detail.  The simple BOA model is not currently testing extinction.

* **BG** = **Vp / VS** = basal ganglia areas interconnected with OFC and ACC: Vp = ventral pallidum, VS = ventral striatum (also known as the nucleus accumbens or NAcc).  The unique ability of the BG is to *match* an input stimulus (CS) with the currently activated Drive(s), to detect when the BLA-activated CS actually satisfies the current Drive state.  The BLA translates the CS into its associated US, but it does not do this Drive-based matching -- it just signals what US you would get from that CS.  It is up to the BG to determine if this US actually matches the current Drive state.  This match computation works by setting the Drives as a *modulator* input to the BG, such that a given US / Drive specific pool in the BG can only fire if it is receiving a given Drive input.  Whether it actually does fire is determined by the BLA and inputs from OFC and ACC.

    The BG is implemented using [pcore](https://github.com/emer/axon/tree/master/pcore) which has **Go** and **No** striatum (matrix = **Mtx**) input layers, that learn whether to gate in and maintain a goal state based on the current BLA state and associated learned information in the OFC and ACC layers, trained by the phasic **DA** = dopamine signal.  The **GPe**, **GPi** and **STN** layers here are all described in the pcore link and just make the gating work well.  The net effect of Go firing is to *inhibit* GPi which then *disinhibits* the MD thalamus layers, toggling active maintenance of the goal state in the PT layers of the OFC and ACC.

* **OFC** encodes predictions of the *outcome* of an action plan -- i.e., the **US** = unconditioned stimulus (food, water, etc).  There are various subregions of OFC in primate (mOFC, lOFC, area 13) vs. rat (VOLO, MO), with various levels of detail vs. abstraction and modality-level factors in representing outcomes -- here we only represent a basic US-specific system that predicts the US on the `USP` layer, where the `P` suffix represents Prediction and Pulvinar (the actual thalamus layer would be the pulvinar-like areas of the MD as described below).  It learns this prediction based on BLA inputs at the time of the US, and is strongly driven by the BLA in general, so when a CS next appears, it will activate the associated OFC representation of the associated US outcome.  Unlike the BLA, the OFC can actively maintain this US expectation (even if the CS later is occluded etc), and these maintained active neurons can *bias* processing in the rest of the system to organize appropriate behavior around this US.  Thus, the BLA are a "dynamic duo" working together to do CS -> US learning (BLA) and active maintenance and cognitive control of US expectations (OFC).

* **ACC** encodes predictions of the overall *utility* of an action plan: the benefits of obtaining the US minus the costs entailed in doing so, which is learned by predicting the time and effort involved in the action plan.  Anatomically, area 24 (posterior ACC, Cg = cingulum in the rodent) is more directly action-specific and area 32 (in primate this is subgenual ACC -- sgACC, corresponding to prelimbic **PL** in rodent) is more abstract and represents overall plan utility.

* **ALM** / **dlPFC** (dorsolateral prefrontal cortex) encodes an overall *policy* or *plan of action* for achieving the desired outcome, which is learned by predicting the sequence of actions performed.  dlPFC in the primate corresponds to ALM (anterior lateral motor area) in the rodent (not PL as is often suggested in the literature).

The three PFC areas that together comprise the distributed goal representation (OFC, ACC, dlPFC / ALM) each have distinct *lamina* (neocortical layers 1-6) with different functions, along with associated connections to the thalamus (MD) in thalamocortical loops:

* **MD** = mediodorsal thalamus is the part of the thalamus that interconnects with the OFC, ACC, and dlPFC / ALM and receives inhibition from the Vp / VS system.  See [Root et al (2015)](#references) for major review of Vp -> MD pathways and associated recording, anatomy etc data.  MD is disinhibited at the time of CS-driven gating, and it bidirectionally connects to the *PT* (pyramidal tract) layers, providing a burst of activity that toggles their NMDA channels resulting ultimately in the active maintenance of the goal state in each of these areas.

* **PT** = pyramidal tract neurons in layer 5 of each PFC area, which provide the robust active maintenance of goal state over time, serving as a sustained *bridge* between the initiation of the goal at the time of the CS to the time of the US.  These are interconnected with the MD thalamus (and other thalamic areas such as VM = ventromedial, VA = ventral anterior, VL = ventral lateral) and are the primary output neurons of the cortex.  In M1 (primary motor cortex), these are the neurons that project to the spinal cord muscle outputs and actually drive motor behavior.  The thalamic connections are "side branches" of these descending subcortical pathways.  See Guo et al (2018) and figure below for more info.  There is a large diversity of PT neuron "subtypes" with different temporal and content-based response properties -- we are only implementing the subset of active maintenance (working memory) PT neurons.

* **CT** = corticothalamic neurons in layer 6 of each PFC area.  These are interconnected in multiple loops through the thalamus, and are functionally important for *context* and *continuous time* updating of activity patterns over time, as shown in the figure below.  Computationally, they are a critical element of the deep predictive learning framework described in [OReilly et al., 2021](https://ccnlab.org/papers/OReillyRussinZolfagharEtAl21.pdf), and are functionally similar to context layers in the simple recurrent network (SRN) model of Elman et al (1991).  They generate predictions over pulvinar-like thalamic layers, which are characteristic of some of the MD, and provide a key driver for learning useful representations in the cortex.  In the BOA model, these layers enable the PFC to learn about sequential events over time, complementing the stable bridging activity in the PT neurons.  You can think of them as the "stream of consciousness" neurons -- always updating and reflecting what you're currently thinking about or doing.  As shown in the figure below, they also provide an "open loop" input to the thalamic loops that drive gating in the PT -- they are not themselves subject to the gating signal from these thalamic areas, but they provide useful context info into those PT loops.

* **Super** = superficial layers 2-3 neurons (without any suffix in the model, e.g., plain OFC) always represent the current state of the network / world, inflluenced by various inputs.  They are bidirectionally connected and are where the constraint satisfaction process takes place to formulate a new proposed plan.  Consistent with data, we assume a subset of PT neurons represent these super states, and convey this info to the BG to influence gating, and also to the MD to provide a drive to activate it when the BG disinhibits it.  Super info is also a key source of input to the CT layer, although with a temporal delay so that CT is forced to try to make predictions about the current state based on prior state info.

Guo et al., 2018 thalamocortical connectivity in rodent ALM:

![PT / CT / Thalamus Connectivity](figs/fig_guo_etal_2018_alm_pt_ct_loops.png?raw=true "Guo et al., 2018 thalamocortical connectivity in rodent ALM")

Chernysheva et al. (2021) sequential PFC activity over a delay period, characteristic of CT neurons -- though these are actually PT neurons:

![Continuously updating CT-like PFC activity](figs/fig_chernysheva_etal_2021_pfc_sequential_reps.png?raw=true "Chernysheva et al. (2021) sequential PFC activity over a delay period, characteristic of CT neurons -- though these are actually PT neurons")

See [O'Reilly, 2020](https://ccnlab.org/papers/OReilly20.pdf) for more info about data and theory, and [github Discussion #56](https://github.com/emer/axon/discussions/56) for more detailed notes and data consulted in the process of developing this model.

# Stats in the logs

* `AllGood` = summary stat representing the average of several of the following stats.  If this is around 1, then the model should be performing well, both behaviorally and in terms of what each of the key layers is doing.

* `ActMatch` = match between network's action and the instinct-driven "correct" action.

* `PctCortex` = % of approach trials (entire sequence of explore then approach) driven by the cortex instead of the instinct.

* `MaintFail*` are loss of active maintenance of goal reps in PT layer -- making sure those are working properly across time.

* `WrongCSGate` is gating to approach the wrong CS (one that does not satisfy the current drive) -- this can happen if the BG happens to get activated via OFC and ACC patterns even if the CS is not associated with the drive-relevant US.  It will be punished by negative DA and should not keep happening.


