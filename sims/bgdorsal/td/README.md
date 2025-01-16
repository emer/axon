# TD 

TD provides a simple TD Q learning solution to the simple motor sequence learning problem, to get a sense of its overall learning complexity.

It is a very simple problem so it is expected to be solved easily: the question is how fast it ends up being learned, given the amount of state space exploration that is required.

The trial / epoch setup is the same as the bgdorsal sim, so epochs are comparable units (128 trials per epoch).

# Results

## Seq4x6

* Epsilon Min = 0.01 seems pretty optimal -- any higher and it fails to converge
* Epsilon Decay = 0.2 or 0.5 work well -- helps converge more quickly.
* LRate Decay = 0.0001 or 0.00001 seem best: variance vs speed tradeoff overall

```
./td -runs 50 -epochs 1000 -env-seq-len 4 -env-n-actions 6 -td-l-rate-decay 0.0001 -td-epsilon-decay 0.5 -td-epsilon-min 0.01
```

The fastest times here are about 10 epochs, and you also get values in the 400+ epochs as well. 

## Seq3x10

Similar param issues, and in general get the same performance with less variability overall:

```
./td -runs 50 -epochs 1000 -env-seq-len 3 -env-n-actions 10 -td-l-rate-decay 0.0001 -td-epsilon-decay 0.5 -td-epsilon-min 0.01
```

## Exploring larger spaces

* 4x10 is more difficult for sure: have to reduce epsilon decay to 0.1

In general it seems sensibly related to the size of the space being searched.


