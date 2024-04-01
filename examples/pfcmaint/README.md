# VSPatch

This is a test for representing graded numerical values in terms of the overall population activity, which is essential for the striatum neurons used in the [pcore](../../PCORE_BG.md) and [pvlv](../../PVLV.md) models.  The specific test case here is the `VSPatch` neurons that predict and discount graded rewards.

The model only has a `VSPatch` layer which is trained to predict the amount of reward for N different "conditions", each of which has T time steps of neural activity, culminating in the reward.  It needs to _not_ respond for the first T-1 time steps, and then accurately predict the graded reward value at T.


