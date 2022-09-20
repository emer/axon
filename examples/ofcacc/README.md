# OFC / ACC

This model builds on the basic [pcore](https://github.com/emer/axon/tree/master/pcore) mechanisms to include temporal integration and prediction of salient outcomes in the OFC and ACC.

The paradigm is a simple ecologically-inspired task (a simplified version of the map-nav Fworld flat-world model), where there are:

* B different body states (hunger, thirst, etc), satisfied by a corresponding US outcome.

* C_B different CS sensory cues associated with each US (simplest case is C_B = 1 -- one-to-one mapping), presented on a "fovea" input layer.

* L different locations where you can be currently looking, each of which can hold a different CS sensory cue.  Current location is an input, and determines contents of the fovea.  wraps around.

* Distance layer with distance from locations where stuff actually is.  This can be represented as a popcode.  Start at random distances.

* Actions are: Forward, Left, Right, Consume.

Target behavior is to orient L / R until a CS sensory cue appears that is consistent with current body state, and then move Forward until the Distance = proximal, and you then Consume.

# TODO

* check acts on all layers

* add act context for prediction?  m1p ok?


