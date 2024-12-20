# PFCMaint

This is a test for prefrontal cortex (PFC) active maintenance mechanisms, which are a key part of the [Deep](../../Deep.md) layer model in Axon, and are essential for the [Rubicon](../../Rubicon.md) goal-driven motivation models.

The task requires encoding and maintenance of a stimulus (10 different stimuli by default, each a different random pattern) over N time steps (10 by default), with an error score on the final stimulus representation.  It is also scored on predicting a simple localist time step representation that increases over the maintenance delay.

The `SMaint` self-maintenance mechanism for the `PTMaintLayer` is used by default, which simulates a population of interconnected PFC layer 5 neurons that mutually sustain each other via NMDA-gated recurrent excitation.  The Config has an option to instead use direct recurrent connections within the layer, which requires a much stronger level of NMDA current for small-sized layers.  Thus, the SMaint mechanism is an optimization allowing for smaller networks to be used.


