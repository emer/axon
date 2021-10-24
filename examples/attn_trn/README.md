# attn_trn

This is a test of TRN-based attention in a basic V1, V2, LIP localist network with gabor inputs.

Tests for effects of contextual normalization in Reynolds & Heeger, 2009 framework.
In terms of differential sizes of attentional spotlight vs. stimulus size.

See stims.go for stimulus input sets -- generate gabor gaussians with different levels of contrast in different input positions, along with top-down LIP attentional spotlights.

# Reynolds & Heeger, 2009 Test

Here's the key pattern demonstrating the critical effect of pooled inhibition:

![RT](figs/fig_reynolds_heeger_09_fig2.png?raw=true "Reynolds & Heeger, 2009, figure 2")

* Left panel (A): when the attentional spotlight is relatively wide, the effects are relatively small and only in a sensitive part of the curve because the broad top-down activation drives corresponding broad pooled inhibition and basically cancels it out.

* Right panel (B): small attentional spotlight is not canceled out and has increasingly large effects.  

![RT](figs/fig_attn_trn_reynolds_heeger_09_small_big_attn.png?raw=true "Model's behavior")

The model always shows the key qualitative effect of "contrast gain" vs. "response gain", but getting larger effects is somewhat difficult due to the extent to which the localist activations and connectivity quickly compound and saturate and don't linearly communicate a graded effect across layers.  It is likely in learned weights and more realistic distributed representations the dynamics would be more graded.

