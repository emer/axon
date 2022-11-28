# Napkin math

Network:
- bench.go
- 2048 neurons
- 5 layers
- 10 epochs
- 1 thread

## NeuronFun
Total time:  21s out of 70s total

- Size of neuron struct: ~80 parameters * 32bit = 320B
- 5 layers * 2048 neuronsPerLayer * 320B = 3.2MB for the complete Neuron[] slice
- 10 epochs * 10 patterns * 4 quarters * 50 cycles = 20K calls to NeuronFun
- Effective Bandwidth (just Neuron[] slice!): 20K * 3.2MB = 64GB of memory access, 64GB/21s = 3GB/s

### Conclusions / Thoughs
- Most of the neuron slice will remain cached in L2/L3 cache.
- We're far away from saturating the memory bandwidth. Why is that?
  - Could be that we're doing a lot of computation
  - Our memory access is sort-of random (most of the struct members are fairly spread out)