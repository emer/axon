# VSPatch

This is a test for representing graded numerical values in terms of the overall population activity, which is essential for the striatum neurons used in the [PCore](../../PCoreBG.md) and [Rubicon](../../Rubicon.md) models.  The specific test case here is the `VSPatch` neurons that predict and discount graded rewards.

The model has opponent D1 vs D2 `VSPatch` layers that are trained to predict the amount of reward for N different "conditions", with T theta trial steps leading up to the reward.  It needs to _not_ respond for the first T-1 time steps, and then accurately predict the graded reward value at T. The opponent D1 vs. D2 dynamic within the VSPatch layer is critical for enabling this dichotomous behavior to be learned.

The `Train/Stats` plot shows all the stats for each condition after each epoch of learning, where `NR` = the non-reward trial activity, and otherwise it shows the final reward trial values.  These conditions are also plotted in the `TrainEpoch` plot, after an initial epoch or more of learning.

