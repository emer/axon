# OFC / ACC

This model builds on the basic [pcore](https://github.com/emer/axon/tree/master/pcore) mechanisms to include temporal integration and prediction of salient outcomes in the OFC and ACC.

The paradigm is a simple ecologically-inspired task (a simplified version of the map-nav Fworld flat-world model), where there are:

* B different body states (hunger, thirst, etc), satisfied by a corresponding US outcome.

* C_B different CS sensory cues associated with each US (simplest case is C_B = 1 -- one-to-one mapping), presented on a "fovea" input layer.

* L different locations where you can be currently looking, each of which can hold a different CS sensory cue.  Current location is an input, and determines contents of the fovea.  wraps around.

* Distance layer with distance from locations where stuff actually is.  This can be represented as a popcode.  Start at random distances.

* Actions are: Forward, Left, Right, Consume.

Target behavior is to orient L / R until a CS sensory cue appears that is consistent with current body state, and then move Forward until the Distance = proximal, and you then Consume.

Mechanistically, this means that once it identifies a suitable CS, a *single* gating event locks in OFC, ACC, SMA deep layers until an outcome occurs -- need to get agate running for this!  Mnt -> Out -- see ccnsims/pfc/agate/cpt model for reference.  Good news: simpler, just one BG still.

The SMA action correlate is effectively "approach then consume" -- without an additional gating step required at the point of consumption -- use instinct to learn this sequence under guidance of SMAOut and current pos etc.

The default non-gated state is exploration.  You could have a goal-engaged version of exploration, but the default is to explore -- again instinct grounds this for the simple version..  some further issues to deal with later..

Need a clear signal for when you are looking at a good CS: this is the activation of US in OFC based on BLA input?


# TODO

Steps:

1. add BLA, OFC with pool-based drive separations -- and OFC has high inhibition, requiring both drive and BLA input to get over threshold..

2. VThal gating drives all *Out deep layers.


* train a pure non-gated M1 -- how high does act match go?  Does OFC learn to represent match of US and CS? First focus on getting that aspect of task working b/c nothing else works unless it does.

* maintain last action if not gated afresh -- how?  SMAd recurrents?  nmda higher?  M1?

* add act context for prediction?  m1p ok?

* modulate OFC / ACC learning mainly at end / DA mod


# Connecting CS and US modulated by Drive

The narrative story is that you have Drive state (bottom-up), learned CS -> US, and you need to decide whether the current CS is something to approach or not.

* First, BLA learns CS -> US and the output is US -> OFC US

* OFC US reps require an AND on BLA and Drive inputs -- matching in general is an AND operation -- do we need a subset of OFC with higher inhibitory threshold to drive this AND, or is regular learning enough?  Maybe start with that to get it working.

* During goal selection gating, BG looks for strong OFC US activation of any sort -- this could be an STN-level filter on enabling gating to take place in the first place -- when the OFC has a new strong US activation it triggers STNp -- otherwise not..


# Note: could use topo STN / Thal pathways to select scope of gating

don't need this now but could use later..

