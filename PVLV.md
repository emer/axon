# PVLV: Primary Value, Learned Value

This is a ground-up rewrite of PVLV [Mollick et al, 2020](#references) for axon, designed to capture the essential properties of the [Go leabra version](https://github.com/emer/leabra/tree/master/pvlv) in a yet simpler and cleaner way.  Each layer type is implemented in a more self-contained manner using the axon trace-based learning rule, which has better natural affordances for DA modulation.  Thus, these can be used in a more mix-and-match manner (e.g., the BLA can be used to train OFC, independent of its phasic dopamine role).

Files: pvlv_{[net.go](axon/pvlv_net.go), [layers.go](axon/pvlv_layers.go), [prjns.go](axon/pvlv_prjns.go)}.

# A Central Challenge: Learning *Something* when *Nothing* happens

As articulated in the PVLV papers, a central challenge that any RL model must solve is to make learning dependent on expectations such that the *absence* of an expected outcome can serve as a learning event, with the appropriate effects.  This is critical for extinction learning, when an expected reward outcome no longer occurs, and the system must learn to no longer have this expectation of reward.  This issue is particularly challenging for PVLV because extinction learning involves different pathways than initial acquisition (e.g., BLA Ext vs. Acq layers), so some kind of indirect effects of expectation are required.

In the 2020 version of the model, activation of the BLA by a CS, and subsequent activity of the USTime layer, represented the expectation and provided the *modulatory* activity on BLAExt and VSPatch to enable extinction learning conditioned on expectations.

In the current version, the gated goal engaged state, corresponding to *OFC PT layer sustained activity*, is the key neural indicator of an active expectation.  Thus, we need to ensure that the model exhibits proper CS-driven BG (VP) gating of active maintenance in OFC PT, and that this maintained activity is available at the right time to drive extinction learning in the right way.

## Time of learning vs. gating issues

Gating happens within the CS onset theta cycle, resulting in an activated OFC PT layer by the end of the trial, when learning happens.  Thus, barring some additional mechanism, learning from PT -> BLA and VSPatch will happen when it shouldn't.  To prevent this, we could do one of the following:

* Set a flag in context when gating happens, and exclude extinction learning specifically on such trials.

* Add a Maint layer that represents whether system is actively maintaining, updated at transition to next trial.. But this must still be used to condition learning b/c activity in PT is still there regardless..




# BLA: Basolateral Amygdala

The basolateral amygdala learns to associate an initially neutral CS with the US that it is reliably associated with.  This learning provides a key step in enabling the system to later recognize the CS as a "trigger" opportunity to obtain its associated US -- if that US is consistent with the current Drive state (e.g., the CS is associated with food and the system is "hungry"), then it should engage a goal to obtain the US.  In the classical conditioning paradigm pioneered by Pavlov, the CS was a bell and the US was food.  The BLA learns the link between the bell and the food, effectively "translating" the meaning of the CS into pre-existing pathways in the brain that process different USs, thus causing Pavlov's dogs to salivate upon hearing the tone.  The BLA learns at the time of the US, in response to dopamine and ACh (see below), so any stimulus reliably present just before the onset of the US is then mapped onto the corresponding US-specific Pool in the BLA layer.  `BLAPosAcqD1` is the positive-valence *acquisition* layer expressing more D1 dopamine receptors, and it is opposed (inhibited) by a corresponding `BLAPosExtD2` D2-dominant *extinction* layer.  The PVLV model of [Mollick et al. (2020)](https://ccnlab.org/papers/MollickHazyKruegerEtAl20.pdf) describes all of this in great detail. 

There are 2x2 BLA types: Positive or Negative valence US's with Acquisition vs. Extinction:

* BLAPosD1 = Pos / Acq
* BLAPosD2 = Pos / Ext
* BLANegD2 = Neg / Acq
* BLANegD1 = Neg / Ext

The D1 / D2 flips the sign of the influence of DA on the plus-phase activation of the BLA neuron (D1 = excitatory, D2 = inhibitory).

A major simplification and improvement in the axon version is that the extinction neurons receive from the OFC neurons that are activated by the corresponding acquisition neurons, thus solving the "learn from disappointment" problem in a much better way: when we are OFC-expecting a given US, and we give up on that and suck up the negative DA, then the corresponding BLA ext neurons get punished.

The (new) learning rule based on the axon trace code is:

* DWt = lr * (1 + g * abs(DA)) * abs(CaSpkP - SpkPrv) * Tr_prv * (CaSpkP - SpkPrv)
    + CaSpkP: current trial plus phase Ca -- has DA modulation reflected in a D1 / D2 direction
    + SpkPrv: CaSpkP from the previous ThetaCycle (trial).  Thus, as in the Leabra PVLV, the outcome / US is compared to the prior t-1 trial.
    + Tr_prv: is S * R trace from *previous* time (i.e., not yet reflecting new Tr update -- as in CTPrjn)
    + The DA modulation of learning rate is implemented via RLrate factor, in NeuroMod field.  This also includes the `Diff` lrate factor as in standard axon, which is likewise based on the between-trial diff as opposed to the plus - minus phase differences.

# CeM -> PPTg -> ACh

This pathway drives acetylcholine (ACh) release in response to *changes* in BLA activity from one trial step to the next, so that ACh can provide a phasic signal reflecting the onset of a *new* CS or US, consistent with available data about firing of neurons in the nucleus basalis and CIN (cholinergic interneurons) in the BG [(Sturgill et al., 2020)](#references).  This ACh signal modulates activity in the BG, so gating is restricted to these time points.  The `CeM` (central nucleus of the amygdala) provides a summary readout of the BLA activity levels, as the difference between the `Acq - Ext` activity, representing the overall CS activity strength.  This goes to the `PPTg` (pedunculopontine tegmental nucleus) which computes a temporal derivative of its CeM input, which then drives phasic DA (dopamine, in VTA and SNc anatomically) and ACh, as described in the PVLV model [(Mollick et al., 2020)](#references).

## CeM

For now, CeM is only serving as an integration of BLA `Acq` and `Ext` inputs, representing the balance between them.  This uses non-learning projections and mostly CeM is redundant with BLA anyway.  This also elides the function CeL (TODO: make sure this is OK!).
    
## PPTg

The PPTg layer computes the temporal derivative of the CeM inputs, based on the strength of feedforward inhibition on the previous ThetaCycle (trial) vs. the current one.  The strength of this factor is in this parameter:
```
"Layer.Inhib.Pool.FFPrv":     "10",
```

# VSPatch

The ventral striatum (VS) patch neurons functionally provide the discounting of primary reward outcomes based on learned predictions, comprising the PV = primary value component of the PVLV algorithm.


# References

* Mollick, J. A., Hazy, T. E., Krueger, K. A., Nair, A., Mackie, P., Herd, S. A., & O'Reilly, R. C. (2020). A systems-neuroscience model of phasic dopamine. Psychological Review, 127(6), 972–1021. https://doi.org/10.1037/rev0000199.  [PDF]((https://ccnlab.org/papers/MollickHazyKruegerEtAl20.pdf)

