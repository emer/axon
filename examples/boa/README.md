# BOA = BG, OFC, ACC

This model implements the `Rubicon` model for goal-driven motivated behavior, which posits distinct  **goal-selection** vs. **goal-engaged** states of the brain ([O'Reilly, 2020](https://ccnlab.org/papers/OReilly20.pdf); [Heckhausen & Gollwitzer, 1987](#references)).  In the goal selection phase, different options are considered (explored) and evaluated according to learned cost-benefit *utilities* (represented in the *ACC* = anterior cingulate cortex), expected *outcome* in the *OFC* = orbital frontal cortex, and proposed *plan of action* in the *dlPFC* = dorsolateral prefrontal cortex, each of which mutually informs the other through bidirectional *constraint satisfaction* to propose a potential overall goal / plan state across these three areas, as discussed in [Herd et al., 2021](https://ccnlab.org/papers/HerdKruegerNairEtAl21.pdf).  If the proposed goal / plan is selected via BG (basal ganglia, implemented via [pcore](https://github.com/emer/axon/tree/master/pcore)) *gating*, then it drives stable active maintenance of this *goal state* which is a distributed representation across these PFC areas.  This maintained goal state then drives coordinated behavior toward achieving the expected outcome through the selected action plan.  Learning happens primarily when the goal is either achieved or abandoned, updating the expected utility, outcome, and action plan as a function of what actually happened during the goal engaged state.

## CS Approach task

The task paradigm is a simple ecologically-inspired task (a simplified version of the map-nav Fworld flat-world model).  The target behavior is to orient Left / Right until a CS sensory cue appears that is consistent with a US that satisfies the current Drive, and then move Forward until the Distance = proximal, and then Consume the US, receiving an amount of reward proportional to the value of the US minus the time and effort required to obtain it.

The main layers are:

* `Drives` = different body states (hunger, thirst, etc), satisfied by a corresponding US (unconditioned stimulus) outcome (food, water, etc).  These are detected and managed primarily in the hypothalamus and other such brainstem nuclei (PBN etc) and represented cortically in the insula (in posterior medial frontal cortex) as a primary interoceptive sensory area. More anterior areas of medial frontal cortex going into OFC represent the "PFC" for interoceptive states (for higher level control and active maintenance).

* `CS` = conditioned stimulus -- represents initially arbitrary sensory cues that are conceptually located by each US (simplest case is `CSPerDrive` = 1 -- one-to-one mapping), presented on a "fovea" input layer reflecting where the agent is looking.

* `Pos` = which direction the agent is currently looking, each of which can hold a different CS sensory cue.  Pos is an input.  You can think of it as the contents of the fovea.  wraps around.

* `Dist` = distance to currently foveated CS

* `Effort` = incrementing representation of effort exerted since last US received or CS detected.  In the brain, it is hypothesized that VS/VP gating triggers a reset of the timer, which itself is likely an interaction between subcortical and vmPFC drifting representations.

* `Act` and `VL` = Actions: `Forward, Left, Right, Consume`.  Consume happens at Dist = 0, Dist stays at 0 for a trial while consuming happens and the US is presented.  VL is the ventral lateral thalamus which is interconnected with primary motor cortex **M1** -- it represents the network's generated motor action, and then receives a plus phase subcortical input representing the cleaned-up version of the action it generated or the instinct-driven action, which is always shown in the Act layer.

The `PctCortex` factor determines the proportion of full approach trials (entire sequence of CS -> US) driven by the heuristic **instinct** behavior.  This behavior is acquired by the motor cortex over learning.  It is very simple: turn Left or Right (consistently if already turning) until a CS is detected that matches the current Drive, at which point move Forward, then Consume the US if Dist = 0. Before the model has learned about CSs, the instinct randomly approaches CSs after N turns. 

NOTE: aspects of above instinct behavior is still TODO!

## BG, OFC, ACC etc Layers and Functions

![BOA Areas](figs/fig_bg_loops_spiral_goals.png?raw=true "BOA Brain areas representing different aspect of a Goal")

**Figure 1:** A version of the classic [Alexander et al. (1986)](#references) 5 loops through the frontal / BG diagram (Striatum = input to BG, with different anatomical regions labeled: D = dorsal, M = medial, V = ventral; Thalamus = output of the BG; SMA = supplementary motor area under dlPFC), highlighting how the OFC, ACC, and dlPFC form a distributed goal / plan representation.

![BOA Bridging Logic](figs/fig_boa_rubicon_logic.png?raw=true "Overall Time Bridging Logic")

**Figure 2:**  Diagram of key elements of the BOA model and how they bridge the goal engaged time window between CS trigger for goal engagement and US outcome.

* **BLA** = basolateral amygdala learns to associate an initially neutral CS with the US that it is reliably associated with.  This learning provides a key step in enabling the system to later recognize the CS as a "trigger" opportunity to obtain its associated US -- if that US is consistent with the current Drive state (e.g., the CS is associated with food and the system is "hungry"), then it should engage a goal to obtain the US.  In the classical conditioning paradigm pioneered by Pavlov, the CS was a bell and the US was food.  The BLA learns the link between the bell and the food, effectively "translating" the meaning of the CS into pre-existing pathways in the brain that process different USs, thus causing Pavlov's dogs to salivate upon hearing the tone.  The BLA learns at the time of the US, in response to dopamine and ACh (see below), so any stimulus reliably present just before the onset of the US is then mapped onto the corresponding US-specific Pool in the BLA layer.  `BLAPosAcqD1` is the positive-valence *acquisition* layer expressing more D1 dopamine receptors, and it is opposed (inhibited) by a corresponding `BLAPosExtD2` D2-dominant *extinction* layer.  The PVLV model of [Mollick et al. (2020)](https://ccnlab.org/papers/MollickHazyKruegerEtAl20.pdf) describes all of this in great detail.  The simple BOA model is not currently testing extinction.

    **CeM -> PPTg -> ACh** This pathway drives acetylcholine (ACh) release in response to *changes* in BLA activity from one trial step to the next, so that ACh can provide a phasic signal reflecting the onset of a *new* CS or US, consistent with available data about firing of neurons in the nucleus basalis and CIN (cholinergic interneurons) in the BG [(Sturgill et al., 2020)](#references).  This ACh signal modulates activity in the BG, so gating is restricted to these time points.  The `CeM` (central nucleus of the amygdala) provides a summary readout of the BLA activity levels, as the difference between the `Acq - Ext` activity, representing the overall CS activity strength.  This goes to the `PPTg` (pedunculopontine tegmental nucleus) which computes a temporal derivative of its CeM input, which then drives phasic DA (dopamine, in VTA and SNc anatomically) and ACh, as described in the PVLV model [(Mollick et al., 2020)](#references).

* **BG** = **VS / VP** = basal ganglia areas interconnect with OFC and ACC: VS = ventral striatum (also known as the nucleus accumbens or NAcc), VP = ventral pallidum downstream of VS.  The unique ability of the BG is to *match* an input stimulus (CS) with the currently activated Drive(s), to detect when the BLA-activated CS actually satisfies the current Drive state.  The BLA translates the CS into its associated US, but it does not do this Drive-based matching -- it just signals what US you would get from that CS.  It is up to the BG to determine if this US actually matches the current Drive state.  This match computation works by setting the Drives as a *modulator* input to the BG, such that a given US / Drive specific pool in the BG can only fire if it is receiving a given Drive input.  Whether it actually does fire is determined by the BLA and inputs from OFC and ACC. The other unique feature of the BG is the ability to use dopamine-based learning to interpret the actual reward value of the OFC and ACC cortical representations, to make the final Go vs. NoGo decision for activating a proposed plan.

    The BG is implemented using [pcore](https://github.com/emer/axon/tree/master/pcore) which has **Go** and **No** striatum (matrix = **Mtx**) input layers, that learn whether to gate in and maintain a goal state based on the current BLA state and associated learned information in the OFC and ACC layers, trained by the phasic **DA** = dopamine signal (see [Dopamine System](#dopamine-system) below).  The **GPe**, **GPi** and **STN** layers here are all described in the pcore link and just make the gating work well.  The net effect of Go firing is to *inhibit* GPi which then *disinhibits* the MD thalamus layers, toggling active maintenance of the goal state in the PT layers of the OFC and ACC.  The gating happening at the time of the CS is reinforced by dopamine at the time of the US, by virtue of a synaptic trace of the original gating activity.

* **OFC** encodes predictions of the *outcome* of an action plan -- i.e., the **US** = unconditioned stimulus (food, water, etc).  There are various subregions of OFC in primate (mOFC, lOFC, area 13) vs. rat (VOLO, MO), with various levels of detail vs. abstraction and modality-level factors in representing outcomes -- here we only represent a basic US-specific system that predicts the US on the `USP` layer, where the `P` suffix represents Prediction and Pulvinar (the actual thalamus layer would be the pulvinar-like areas of the MD as described below).  It learns this prediction based on BLA inputs at the time of the US, and is strongly driven by the BLA in general, so when a CS next appears, it will activate the associated OFC representation of the associated US outcome.  Unlike the BLA, the OFC can actively maintain this US expectation (even if the CS later is occluded etc), and these maintained active neurons can *bias* processing in the rest of the system to organize appropriate behavior around this US.  Thus, the BLA and OFC are a "dynamic duo" working together to do CS -> US learning (BLA) and active maintenance and cognitive control of US expectations (OFC).

* **ACC** encodes predictions of the overall *utility* of an action plan: the benefits of obtaining the US minus the costs entailed in doing so, which is learned by predicting the time and effort involved in the action plan.  Anatomically, area 24 (posterior ACC, Cg = cingulum in the rodent) is more directly action-specific and area 32 (in primate this is subgenual ACC -- sgACC, corresponding to prelimbic **PL** in rodent) is more abstract and represents overall plan utility.

* **ALM** / **dlPFC** (dorsolateral prefrontal cortex) encodes an overall *policy* or *plan of action* for achieving the desired outcome, which is learned by predicting the sequence of actions performed.  dlPFC in the primate corresponds to ALM (anterior lateral motor area) in the rodent (not PL as is often suggested in the literature).

The three PFC areas that together comprise the distributed goal representation (OFC, ACC, dlPFC / ALM) each have distinct *lamina* (neocortical layers 1-6) with different functions, along with associated connections to the thalamus (MD) in thalamocortical loops:

* **MD** = mediodorsal thalamus is the part of the thalamus that interconnects with the OFC, ACC, and dlPFC / ALM and receives inhibition from the VS / VP system.  See [Root et al. (2015)](#references) for major review of VS -> MD pathways and associated recording, anatomy etc data.  MD is disinhibited at the time of CS-driven gating, and it bidirectionally connects to the *PT* (pyramidal tract) layers, providing a burst of activity that toggles their NMDA channels resulting ultimately in the active maintenance of the goal state in each of these areas.

* **PT** = pyramidal tract neurons in layer 5 of each PFC area, which provide the robust active maintenance of goal state over time, serving as a sustained *bridge* between the initiation of the goal at the time of the CS to the time of the US.  These are interconnected with the MD thalamus (and other thalamic areas such as VM = ventromedial, VA = ventral anterior, VL = ventral lateral) and are the primary output neurons of the cortex.  In M1 (primary motor cortex), these are the neurons that project to the spinal cord muscle outputs and actually drive motor behavior.  The thalamic connections are "side branches" of these descending subcortical pathways.  See Guo et al. (2018) and Figure 3 below for more info.  There is a large diversity of PT neuron "subtypes" with different temporal and content-based response properties -- we are only implementing the subset of active maintenance (working memory) PT neurons.

* **PTPred** = PT predictive learning neurons: has dynamically changing states that learn to predict everything that happens during a given period of persistent active goal maintenance.  Unlike the above PT (`PTMaintLayer`) layers which are relatively stable over time, the dynamic changing state in the Pred layers is useful for driving predictions of when US outcomes might occur, it the `VSPatch` layer.

* **CT** = corticothalamic neurons in layer 6 of each PFC area.  These are interconnected in multiple loops through the thalamus, and are functionally important for *context* and *continuous time* updating of activity patterns over time, as shown in Figure 4 below.  Computationally, they are a critical element of the deep predictive learning framework described in [OReilly et al., 2021](https://ccnlab.org/papers/OReillyRussinZolfagharEtAl21.pdf), and are functionally similar to context layers in the simple recurrent network (SRN) model of [Elman (1990)](#references).  They generate predictions over pulvinar-like thalamic layers, which are characteristic of some of the MD, and provide a key driver for learning useful representations in the cortex.  In the BOA model, these layers enable the PFC to learn about sequential events over time, complementing the stable bridging activity in the PT neurons.  You can think of them as the "stream of consciousness" neurons -- always updating and reflecting what you're currently thinking about or doing.  As shown in Figure 3 below, they also provide an "open loop" input to the thalamic loops that drive gating in the PT -- they are not themselves subject to the gating signal from these thalamic areas, but they provide useful context info into those PT loops.

* **Super** = superficial layers 2-3 neurons (without any suffix in the model, e.g., plain OFC) always represent the current state of the network / world, influenced by various inputs.  They are bidirectionally connected and are where the constraint satisfaction process takes place to formulate a new proposed plan.  Consistent with data, we assume a subset of PT neurons represent these super states, and convey this info to the BG to influence gating, and also to the MD to provide a drive to activate it when the BG disinhibits it.  Super info is also a key source of input to the CT layer, although with a temporal delay so that CT is forced to try to make predictions about the current state based on prior state info.

![PT / CT / Thalamus Connectivity](figs/fig_guo_etal_2018_alm_pt_ct_loops.png?raw=true "Guo et al., 2018 thalamocortical connectivity in rodent ALM")

**Figure 3:** Guo et al., 2018 thalamocortical connectivity in rodent ALM.

![Continuously updating CT-like PFC activity](figs/fig_chernysheva_etal_2021_pfc_sequential_reps.png?raw=true "Chernysheva et al. (2021) sequential PFC activity over a delay period, characteristic of CT neurons -- though these are actually PT neurons")

**Figure 4:** Chernysheva et al. (2021) sequential PFC activity over a delay period, characteristic of CT neurons -- though these are actually PT neurons.

See [O'Reilly, 2020](https://ccnlab.org/papers/OReilly20.pdf) for more info about data and theory, and [github Discussion #56](https://github.com/emer/axon/discussions/56) for more detailed notes and data consulted in the process of developing this model.

## Dopamine System and Reinforcement Learning

The dopamine system is implemented via the [PVLV](../../PVLV.md) framework (see [(Mollick et al. 2020)](#references)), including:

* **VTA** represents *dopamine*, which drives learning in the BG as a function of differences between expected and actual reward values, currently only at the time of US consumption.  Positive DA (biologically encoded as firing levels above a tonic baseline) reinforces prior Go firing, driving more goal engagement in similar circumstances in the future.  Negative DA (pauses in firing below baseline) reinforces prior NoGo firing, reducing goal engagement for worse-than-expected outcomes.  

* **USpos** positive-valence US (unconditioned stimulus, i.e., reward) outcomes are generated by the environment when the agent does a Consume action at the right place.  The resulting reward reflects the "subjective" value of the US in relation to the Drive state, modulated by the Effort expended to get it.

# ACh Reward Salience

The ACh neuromodulatory signal plays an essential role in the model, because it determines the time windows for BG gating and learning.  ACh = time, DA = value.  If fires at the wrong times, or doesn't fire at salient events, then the model will fail.

Via the PPTg temporal difference mechanism, ACh should respond to the onset of ANY novel CS, and at the time of the US.

TODO: If PFC is already goal-engaged, then ACh should be inhibited!  don't get distracted!  Top-down inhibitory connections to brainstem.  Implement as inhibitory connections to PPTg from PT?  Interacts with Gate layer replacement.

TODO: add stats for this!

# Matrix learning rule

The learning rule should be a function of both layer pool gated and local pool gated, and the ACh value indicating the opportunity to gate.

At time of CS, if layer gated:
* If local pool gated:
    + Tr = Send * Recv, later DA: Go up, No down  <- normal credit assignment
* If local NOT gated:
    + nothing -- we're good!

If layer NOT gated:  <- exploration opportunity cost
* In proportion to ACh, Go up, No down -- if no ACh, no learn!
    + This should NOT be in the final trace because it didn't do anything.
    + should be weaker and exploratory. -- .005 works and is needed in pcore.

# Stats in the logs

* `AllGood` = summary stat representing the average of several of the following stats.  If this is around 1, then the model should be performing well, both behaviorally and in terms of what each of the key layers is doing.

* `ActMatch` = match between network's action and the instinct-driven "correct" action.  Usually the `LeftCor` action -- tracking % of time it correctly does the Left action (in zoolander mode where it only goes left to search) -- is the most indicative

* `PctCortex` = % of approach trials (entire sequence of explore then approach) driven by the cortex instead of the instinct.

* `MaintFail*` are loss of active maintenance of goal reps in PT layer -- making sure those are working properly across time.

* `MaintEarly` means the PT layer is getting active prior to BG gating -- need to turn down `.SuperToPT` -- see below.

* `WrongCSGate` is gating to approach the wrong CS (one that does not satisfy the current drive) -- this can happen if the BG happens to get activated via OFC and ACC patterns even if the CS is not associated with the drive-relevant US.  It will be punished by negative DA and should not keep happening.

# Parameter tuning

## PT Tuning

The PT layer is somewhat sensitive in parameter tuning because it has several potentially conflicting demands: It must exhibit robust active maintenance (requiring strong excitation over time), while also being sensitive to the BG gating signal via MD modulatory projections.  This typically means that `.PTSelfMaint` `Abs` is strong enough to prevent `MaintFail`, but not too strong to cause `MaintEarly` to start happening.  `.SuperToPT` must be relatively weak (use .`Abs`) so super -> PT alone does not cause gating, but not too weak that it fails to gate when MD gets active.  Finally, `PTMaintLayer` `Dend.ModGain` is key for setting the strength of the MD thalamus modulatory projections into the PT -- these target the NMDA channels specifically, and help to "ignite" the active maintenance.

## MD Tuning

MD thalamus likewise has multiple conflicting demands, similar to PT: must not get active spontaneously, but needs to activate quickly when disinhibited from the BG.  The `Gated*` stats track MD activity (in addition to BG), similar to how `Maint*` tracks PT tuning.  `.SuperToThal` provides key `Abs` tuning for strength from cortex -- also `.CTToThal` controls CT input to MD -- this is not critical for basic tasks but may be important for more interesting predictive learning cases. 

# References 

* Alexander, G. E., DeLong, M. R., & Strick, P. L. (1986). Parallel organization of functionally segregated circuits linking basal ganglia and cortex. Annual Review of Neuroscience, 9, 357–381. http://www.ncbi.nlm.nih.gov/pubmed/3085570

* Chernysheva, M., Sych, Y., Fomins, A., Warren, J. L. A., Lewis, C., Capdevila, L. S., Boehringer, R., Amadei, E. A., Grewe, B. F., O’Connor, E. C., Hall, B. J., & Helmchen, F. (2021). Striatum-projecting prefrontal cortex neurons support working memory maintenance (p. 2021.12.03.471159). bioRxiv. https://doi.org/10.1101/2021.12.03.471159

* Elman, J. L. (1990). Finding structure in time. Cognitive Science, 14(2), 179–211.

* Guo, K., Yamawaki, N., Svoboda, K., & Shepherd, G. M. G. (2018). Anterolateral motor cortex connects with a medial subdivision of ventromedial thalamus through cell type-specific circuits, forming an excitatory thalamo-cortico-thalamic loop via layer 1 apical tuft dendrites of layer 5b pyramidal tract type neurons. Journal of Neuroscience, 38(41), 8787–8797. https://doi.org/10.1523/JNEUROSCI.1333-18.2018

* Heckhausen, H., & Gollwitzer, P. M. (1987). Thought contents and cognitive functioning in motivational versus volitional states of mind. Motivation and Emotion, 11(2), 101–120. https://doi.org/10.1007/BF00992338

* Mollick, J. A., Hazy, T. E., Krueger, K. A., Nair, A., Mackie, P., Herd, S. A., & O’Reilly, R. C. (2020). A systems-neuroscience model of phasic dopamine. Psychological Review, 127, 972–1021. https://doi.org/10.1037/rev0000199

* O’Reilly, R. C. (2020). Unraveling the Mysteries of Motivation. Trends in Cognitive Sciences. https://doi.org/10.1016/j.tics.2020.03.001

* O’Reilly, R. C., Russin, J. L., Zolfaghar, M., & Rohrlich, J. (2020). Deep Predictive Learning in Neocortex and Pulvinar. ArXiv:2006.14800 [q-Bio]. http://arxiv.org/abs/2006.14800

* Root, D. H., Melendez, R. I., Zaborszky, L., & Napier, T. C. (2015). The ventral pallidum: Subregion-specific functional anatomy and roles in motivated behaviors. Progress in Neurobiology, 130, 29–70. https://doi.org/10.1016/j.pneurobio.2015.03.005

* Sturgill, J. F., Hegedus, P., Li, S. J., Chevy, Q., Siebels, A., Jing, M., Li, Y., Hangya, B., & Kepecs, A. (2020). Basal forebrain-derived acetylcholine encodes valence-free reinforcement prediction error (p. 2020.02.17.953141). bioRxiv. https://doi.org/10.1101/2020.02.17.953141

