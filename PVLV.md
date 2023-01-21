# PVLV: Primary Value, Learned Value

[![GoDoc](https://godoc.org/github.com/emer/axon/pvlv?status.svg)](https://godoc.org/github.com/emer/axon/pvlv)

This is a ground-up rewrite of PVLV [Mollick et al, 2020](#references) for axon, designed to capture the essential properties of the [Go leabra version](https://github.com/emer/leabra/tree/master/pvlv) in a yet simpler and cleaner way.  Each layer type is implemented in a more self-contained manner using the axon trace-based learning rule, which has better natural affordances for DA modulation.  Thus, these can be used in a more mix-and-match manner (e.g., the BLA can be used to train OFC, independent of its phasic dopamine role).

This is an incremental work-in-progress, documented as it goes along.

# BLA

There are 2x2 BLA types: Positive or Negative valence US's with Acquisition vs. Extinction:

* BLAPosD1 = Pos / Acq
* BLAPosD2 = Pos / Ext
* BLANegD2 = Neg / Acq
* BLANegD1 = Neg / Ext

The D1 / D2 flips the sign of the influence of DA on the plus-phase activation of the BLA neuron (D1 = excitatory, D2 = inhibitory).

A major simplification and improvement in the axon version is that the extinction neurons receive from the OFC neurons that are activated by the corresponding acquisition neurons, thus solving the "learn from disappointment" problem in a much better way: when we are OFC-expecting a given US, and we give up on that and suck up the negative DA, then the corresponding BLA ext neurons get punished.

The (new) learning rule based on the axon trace code is:

* DWt = lr * Tr_prv * (1 + abs(DA)) * (CaP - CaD)
    + CaP: plus phase Ca -- has DA modulation reflected in a D1 / D2 direction
    + CaD: minus phase Ca -- prior to DA -- note in original this was t-1 -- axon uses longer theta cycles and plus phase is when DA is delivered by default.
    + Tr_prv: is S * R trace from *previous* time (i.e., not yet reflecting new Tr update -- as in CTPrjn)
    + DA also drives lrate, included in RLRate which also does the small-delta filtering included in prior impl


# References

* Mollick, J. A., Hazy, T. E., Krueger, K. A., Nair, A., Mackie, P., Herd, S. A., & O'Reilly, R. C. (2020). A systems-neuroscience model of phasic dopamine. Psychological Review, 127(6), 972–1021. https://doi.org/10.1037/rev0000199
