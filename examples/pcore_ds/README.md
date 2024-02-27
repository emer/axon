# pcore_ds

This is a simple test of the [pcore](../../PCORE_BG.md) model of basal ganglia (BG) function, in the **Dorsal Striatum** (DS).  See [pcore_vs](../pcore_vs) for the Ventral Striatum (VS) model, which is optimized for making global Go vs. No decisions based on cost / benefit inputs (see also [BOA](../boa)).

The DS is the input layer for the primary motor control part of the basal ganglia, and this model learns to execute a sequence of motor actions through reinforcement  learning (RL), getting positive reinforcement for correct actions and lack of reinforcement for incorrect ones.  Critically, there is no omnicient "teacher" input: the model has to discover the correct action sequence purely through trial and error, "online" learning (i.e., it learns on a trial-by-trial basis as it acts).  This is the only biologically / ecologically realistic form of RL.

The model also has mechanisms to learn about the space of possible motor actions and their parameterization in the DS, which is 

This test model has all of the standard PFC layers, which are kept busy by predicting a sequence of input values on the `In` layer, via the [deep](../../DEEP.md) predictive learning mechanism (`InP` is the pulvinar layer representing the prediction of `In`).  This prediction task is completely orthogonal from the gating decision made by the BG, which is driven by the `ACCPos` and `ACCNeg` layers.

These `ACC` layers have `PopCode` representations of values, and the BG gating is trained to gate when Pos > Neg.  Typically you do `TrainRun` or `Step` with the step set to `Run`, and then `TestRun` which will run through all combinations of `ACCPos` (outer loop) and `ACCNeg` (inner loop), with 25 samples of each value to get statistics (it takes a while).  Click on `TestTrialStats Plot` to see the results -- you can click on `ACCPos` and `ACCNeg` to see those inputs, and then compare `Gated` with `Should` to see how the network performed.

# results

<img src="results/fig_pcore_train.png" width="800">

Training data shows close match between Gated and Should (high Match proportion).

<img src="results/fig_pcore_test_learned.png" width="800">

Testing data over ACC Pos (outer loop) and ACC Neg (inner loop) shows increasing probability of gating as Pos increases, and reduced firing, and slower RT, as Neg increases, closely matching the target `Should` behavior.  25 samples of each case are performed, so intermediate levels indicate probability of gating.  Model shows appropriate probabilistic behavior on the marginal cases.

# TODO:

* logical issue: how can the BG DA learning contribute to correct performance if all it does is timing.   the "wait for another option to be suggested by someone else" strategy is just not sufficiently robust (or concretely realizable in simple circuits).

* 26k SNr neurons is still plenty to code for specific actions.

* DA, learning etc
* PT_lower, PT_upper etc -- need to replace / rethink PTPred basically?
* GPU mode doesn't currently make any sense
