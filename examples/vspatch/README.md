# VSPatch

This is a test for representing graded numerical values in terms of the overall population activity, which is essential for the striatum neurons used in the [PCore](../../PCoreBG.md) and [Rubicon](../../Rubicon.md) models.  The specific test case here is the `VSPatch` neurons that predict and discount graded rewards.

The model only has a `VSPatch` layer which is trained to predict the amount of reward for N different "conditions", each of which unfolds over T time steps of neural activity, culminating in the reward.  It needs to _not_ respond for the first T-1 time steps, and then accurately predict the graded reward value at T.  The opponent D1 vs. D2 dynamic within the VSPatch layer is critical for enabling this dichotomous behavior to be learned.

The `CondStats` plot shows all the stats for each condition over learning, where `NR` = the non-reward trial activity, and otherwise it shows the final reward trial values.  These conditions are also plotted in the `TrainEpoch` plot.



