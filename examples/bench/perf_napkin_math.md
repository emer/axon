# Napkin math

Back-of-the-envelope calculations for all the major parts of Axon.

Network:
- bench.go
- 2048 neurons
- 5 layers
- 10 epochs
- 1 thread

## Notes
- How generic are the patterns actually? Can we special-case eg the all-to-all connection? One-to-one
  - looks like Boa only uses 3 types of patterns: Full, OneToOne, PoolOneToOne

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
- The Neuron slice is very small, and the effective bandwidth that we're hitting is far away from what is possible -> not bandwidth bound
- The Neuron slice is not being cached, as it get's evicted from the cache by the SynapseFun that comes after.
- Memory for the Neuron slice only grows linearly. Memory access is predictable, and can be made sequential.
- This function is easily parallelizable, and (on an abstract level) very amendable to converting to GPU.
  - It seems like the current parallelization effort works well, giving us linear scaling as long as the problem size is big enough.
- We're far away from saturating the memory bandwidth. Why is it still sort-of slow?
  - Could be that we're doing a lot of computation
  - Our memory access is sort-of random (most of the struct members are fairly spread out) -> from the benchmarks, doesn't look like it explains more than 2x.

## SendSpikeFun
This function sends a spike to receiving projections, if the neuron has spiked.
It's called once for every millisecond, just like NeuronFun.

Total time: 26s out of 70s total, almost all of it spent in Axon.SendSpike

- Projection GBuf (conductance buffer) size is `#RecvNeurons*(timeDelay+1)`. It's a ring buffer, with this layout: `nrn0time0 | nrn0time1 | nrn0time2 | nrn1time0 | nrn1time1 | ...`
- How many of these entries do we access during each round? Just delay=0 and delay=maxDelay?
- Could also be represented using maxDelay-many pointers, each to a `#RecvNeuron`-sized array. Then we exchange the pointers during the timestep update (similar to multiple buffering).
- SendSpike performs a matrix product for each projection: `y=Wx`, where x is a binary vector, indicating whether the Neuron has spiked or not. How sparse is this vector?
- FLOPs: `2 * #outputs * #inputs`, where W is a `#outputs x #inputs` matrix
- Memory demand: ideally, loading only W + x, and storing y.
- So -> 2n^2 FLOPs, n^2 memory loads -> Constant operational intensity.

- Synapses: ~11 parameters * 32bit = 44B
- Projection: `#inNrn*#outNrn` * 44B (the actual Synapse structs) + (4B + 4B + 4B) * `#inNrn*#outNrn` (indexing based on Pattern)
- For the bench.go, we have: 7 Proj times
	- Syn[] struct: 2025*2025 Synapses * 44B
	- GBuf: 3 max delay * 2025 (neurons) * 4B (so small in size compared to the others that it doesn't really matter)
	- RecvConIdx: 2025*2025 * 4B
	- RecvSynIdx: 2025*2025 * 4B
	- SendConIdx: 2025*2025 * 4B

