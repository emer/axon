# Napkin math

Back-of-the-envelope calculations for all the major parts of Axon.

Network:
- bench.go
- 2048 neurons
- 5 layers
- 10 epochs
- 1 thread

## NeuronFun
This function updates the neuron state.
It's called once for every millisecond.

Total time:  21s out of 70s total

- Size of neuron struct: ~80 parameters * 32bit = 320B
- 5 layers * 2048 neuronsPerLayer * 320B = 3.2MB for the complete Neuron[] slice
- 10 epochs * 10 patterns * 4 quarters * 50 cycles = 20K calls to NeuronFun
- Each call to NeuronFun iterates over the whole Neuron[] slice, updating the parameters of each neuron
- Effective Bandwidth (just Neuron[] slice!): 20K * 3.2MB = 64GB of memory access, 64GB/21s = 3GB/s

### Conclusions / Thoughts (for NeuronFun)
- At the current size, most of the neuron slice remains cached in L2/L3 cache. Even if the whole slice was in RAM, we should still be able to hit 10x faster perf if we multithreaded this (assuming the access is sequential)
- Memory for the Neuron slice only grows linearly. Memory access is predictable, and can be made sequential.
- This function is easily parallelizable, and (on an abstract level) very amendable to converting to GPU.
- We're far away from saturating the memory bandwidth. Why is it still sort-of slow?
  - Could be that we're doing a lot of computation
  - Our memory access is sort-of random (most of the struct members are fairly spread out)

