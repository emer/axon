# BOA = BG, OFC, ACC

This model implements the `Rubicon` model for goal-driven motivated behavior, which posits distinct  **goal-selection** vs. **goal-engaged** states of the brain [O'Reilly, 2020](https://ccnlab.org/papers/OReilly20.pdf); Heckhausen & Gollwitzer, 1987.  In the goal selection phase, different options are considered (explored) and evaluated according to learned cost-benefit *utilities* (represented in the *ACC* = anterior cingulate cortex), and one is selected via BG (basal ganglia, implemented via [pcore](https://github.com/Astera-org/axon/tree/master/pcore)) *gating* that drives stable active maintenance of a *goal state* which is a distributed representation across the *ACC*, *OFC* = orbital frontal cortex and *dlPFC* = dorsolateral prefrontal cortex).

![BOA Areas](figs/fig_bg_loops_spiral_goals.png?raw=true "BOA Brain areas representing different aspect of a Goal")

* **OFC** encodes predictions of the *outcome* of an action plan -- i.e., the **US** = unconditioned stimulus (food, water, etc).

* **ACC** encodes predictions of the overall *utility* of an action plan: benefits of obtaining the US minus the costs entailed in doing so, which is learned by predicting the time and effort involved in the action plan.

* **dlPFC** encodes an overall *policy* or *plan of action* for achieving the desired outcome, which is learned by predicting the sequence of actions performed.

See [O'Reilly, 2020](https://ccnlab.org/papers/OReilly20.pdf) for more info about data and theory.

The task paradigm is a simple ecologically-inspired task (a simplified version of the map-nav Fworld flat-world model), where there are:

* `Drives` = different body states (hunger, thirst, etc), satisfied by a corresponding US outcome.  These are detected and managed primarily in the hypothalamus and other such brainstem nuclei (PBN etc) and represented cortically in the insula (in posterior medial frontal cortex) as a primary interoceptive sensory area, with more anterior areas of medial frontal cortex going into OFC representing the "PFC" for interoceptive states (for higher level control and active maintenance).

* `CS` = different initially arbitrary sensory cues that are located by each US (simplest case is `CSPerDRive` = 1 -- one-to-one mapping), presented on a "fovea" input layer reflecting where the agent is looking.

* `Pos` = which of different locations where agent is currently looking, each of which can hold a different CS sensory cue.  Current location is an input, and determines contents of the fovea.  wraps around.

* `Dist` = distance to currently foveated CS

* `Time` = incrementing representation of time from last US received.

* Actions are: `Forward, Left, Right, Consume`.  Consume happens at Dist = 0, Dist stays at 0 for a trial while consuming happens and the US is presented.

The target behavior is to orient L / R until a CS sensory cue appears that is consistent with current Drive, and then move Forward until the Distance = proximal, and you then Consume.

# Stats

* There is a subcortical instinct, which is just heuristic code for action policy: explore then approach once desired CS is seen

* `ActMatch` = match between network's action and the instinct-driven "correct" action

* `PctCortex` = % of approach trials (entire sequence of explore then approach) driven by the cortex instead of the instinct.

* `MaintFail*` are loss of active maintenance of goal reps in PT layer (pyramidal tract, layer 5)

* `WrongCSGate` is gating to approach the wrong CS (one that does not satisfy the current drive)


# Overview of Model

![BOA Bridging Logic](figs/fig_boa_rubicon_logic.png?raw=true "Overall Time Bridging Logic")

TODO: describe, update figure

Anatomically, there are distinct circuits connecting through OFC, ACC and dlPFC (ADS '86):

* OFC -> VM Str / NAcc -> VP / mdm-GP -> mVA, MDmc
* ACC -> VM Str -> VP / rl-GP -> pm-MD (no VA?)
* SMA -> Putamen / DL Str -> vl-GP -> VL thal
* dlPFC -> dm Caudate -> mdm-GP -> VApc, MDpc

Thus, in principle each area can gate separately presumably, although our simplified model assumes that bidirectional cortical coordination causes them to typically all gate together.  OFC level gating may make parallel go / no choices about which outcome to pursue?

This gets into PVLV-level model issues about how these circuits are also involved in DA regulation...

The SMA action correlate is effectively "approach then consume" -- without an additional gating step required at the point of consumption -- use instinct to learn this sequence under guidance of SMAOut and current pos etc.

The default non-gated state is exploration.  You could have a goal-engaged version of exploration, but the default is to explore -- again instinct grounds this for the simple version..  some further issues to deal with later..

Need a clear signal for when you are looking at a good CS: this is the activation of US in OFC based on BLA input?

# Specific model properties

## Vs / Vp Drive * CS match detection

See https://github.com/Astera-org/axon/discussions/56#discussioncomment-3939045 for rationale for how it computes alignment between drive and CS.

Note that BLA does not have to do the same thing -- it should follow the CS wherever it goes.

# TODO:

## current limitations:

* Only goes left
* explicit gating input layer

## After CS gating, what happens for US gating?

If BG gates to wrong CS, then US reinforces negative DA for *right* CS.

In general, US gating should update CS-level gating but not continue to reinforce US?  Errors and blame are all about the CS-time gating event, so US-time is just confusing the matter.  Should be pure clear / toggle basically.

# DONE:

## Why is WrongCS gating happening?

OFC is active for drive, other CS is active.  OFC+drive is enough.  remove drive -> OFC -- ofc needs to be a clean representation of the CS-side of things, not the drive side -- can't contaminate "desires" with "reality"!



