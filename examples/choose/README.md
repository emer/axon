# BOA = BG, OFC, ACC

This model implements the [Rubicon](../../Rubicon.md) model for goal-driven motivated behavior, in a decision-making task that requires choosing among options with different cost-benefit tradeoffs.  This exercises the core cost and benefit representations and goal selection and goal maintenance components of the Rubicon model

* TODO: compute us prob at start -- otherwise keeps doing again and again

## Arm Maze Bandit task

The task paradigm is an N-arm bandit task, implemented as a physical maze-like environment where the simulated rodent must walk down an arm to receive the reward outcome (_US_ = unconditioned stimulus) signalled by the stimulus (_CS_ = conditioned stimulus) visible from the start of the arm.  The arms can vary in length and effort (e.g., an elevated hill) to manipulate the cost, and the US varies in value according to internal _Drive_ states (e.g., hunger, thirst) and qualities of the US itself (e.g., how tasty it is).  Thus, the decision to enter an arm requires balancing the cost vs. benefit tradeoff or net utility, and all of this is learned through the course of exploring the maze over repeated trials.

The main layers are:

* `Drives` = different body states (hunger, thirst, etc), satisfied by a corresponding US (unconditioned stimulus) outcome (food, water, etc).  These are detected and managed primarily in the hypothalamus and other such brainstem nuclei (PBN etc) and represented cortically in the insula (in posterior medial frontal cortex) as a primary interoceptive sensory area. More anterior areas of medial frontal cortex going into OFC represent the "PFC" for interoceptive states (for higher level control and active maintenance).

* `CS` = conditioned stimulus -- represents initially arbitrary sensory cues that are conceptually located by each US (simplest case is `CSPerDrive` = 1 -- one-to-one mapping), presented on a "fovea" input layer reflecting where the agent is looking.

* `Pos` = which direction the agent is currently looking, each of which can hold a different CS sensory cue.  Pos is an input.  You can think of it as the contents of the fovea.  wraps around.

* `Dist` = distance to currently foveated CS

* `Effort` = incrementing representation of effort exerted since last US received or CS detected.  In the brain, it is hypothesized that VS/VP gating triggers a reset of the timer, which itself is likely an interaction between subcortical and vmPFC drifting representations.

* `Act` and `VL` = Actions: `Forward, Left, Right, Consume`.  Consume happens at Dist = 0, Dist stays at 0 for a trial while consuming happens and the US is presented.  VL is the ventral lateral thalamus which is interconnected with primary motor cortex **M1** -- it represents the network's generated motor action, and then receives a plus phase subcortical input representing the cleaned-up version of the action it generated or the instinct-driven action, which is always shown in the Act layer.

The `PctCortex` factor determines the proportion of full approach trials (entire sequence of CS -> US) driven by the heuristic **instinct** behavior.  This behavior is acquired by the motor cortex over learning.  It is very simple: turn Left or Right (consistently if already turning) until a CS is detected that matches the current Drive, at which point move Forward, then Consume the US if Dist = 0. Before the model has learned about CSs, the instinct randomly approaches CSs after N turns. 

NOTE: aspects of above instinct behavior is still TODO!


![PT / CT / Thalamus Connectivity](figs/fig_guo_etal_2018_alm_pt_ct_loops.png?raw=true "Guo et al., 2018 thalamocortical connectivity in rodent ALM")

**Figure 3:** Guo et al., 2018 thalamocortical connectivity in rodent ALM.

![Continuously updating CT-like PFC activity](figs/fig_chernysheva_etal_2021_pfc_sequential_reps.png?raw=true "Chernysheva et al. (2021) sequential PFC activity over a delay period, characteristic of CT neurons -- though these are actually PT neurons")

**Figure 4:** Chernysheva et al. (2021) sequential PFC activity over a delay period, characteristic of CT neurons -- though these are actually PT neurons.

See [O'Reilly, 2020](https://ccnlab.org/papers/OReilly20.pdf) for more info about data and theory, and [github Discussion #56](https://github.com/emer/axon/discussions/56) for more detailed notes and data consulted in the process of developing this model.

## Dopamine System and Reinforcement Learning

The dopamine system is implemented via the PVLV framework (see [(Mollick et al. 2020)](#references)), including:

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

The PT layer is somewhat sensitive in parameter tuning because it has several potentially conflicting demands: It must exhibit robust active maintenance (requiring strong excitation over time), while also being sensitive to the BG gating signal via MD modulatory pathways.  This typically means that `.PTSelfMaint` `Abs` is strong enough to prevent `MaintFail`, but not too strong to cause `MaintEarly` to start happening.  `.SuperToPT` must be relatively weak (use .`Abs`) so super -> PT alone does not cause gating, but not too weak that it fails to gate when MD gets active.  Finally, `PTMaintLayer` `Dend.ModGain` is key for setting the strength of the MD thalamus modulatory pathways into the PT -- these target the NMDA channels specifically, and help to "ignite" the active maintenance.

## MD Tuning

MD thalamus likewise has multiple conflicting demands, similar to PT: must not get active spontaneously, but needs to activate quickly when disinhibited from the BG.  The `Gated*` stats track MD activity (in addition to BG), similar to how `Maint*` tracks PT tuning.  `.SuperToThal` provides key `Abs` tuning for strength from cortex -- also `.CTToThal` controls CT input to MD -- this is not critical for basic tasks but may be important for more interesting predictive learning cases. 


