# Reinforcement Learning and Dopamine

[![GoDoc](https://godoc.org/github.com/Astera-org/axon/rl?status.svg)](https://godoc.org/github.com/Astera-org/axon/rl)

The `rl` package provides core infrastructure for dopamine neuromodulation and reinforcement learning, including the Rescorla-Wagner learning algorithm (RW) and Temporal Differences (TD) learning, and a minimal `ClampDaLayer` that can be used to send an arbitrary DA signal.

* `da.go` defines a simple `DALayer` interface for getting and setting dopamine values, and a `SendDA` list of layer names that has convenience methods, and ability to send dopamine to any layer that implements the DALayer interface.

* The RW and TD DA layers use the `CyclePost` layer-level method to send the DA to other layers, at end of each cycle, after activation is updated.  Thus, DA lags by 1 cycle, which typically should not be a problem. 

* See the separate `pvlv` package for the full biologically-based pvlv model on top of this basic DA infrastructure.

* To encode positive and negative values using spiking, 2 units are used, one for positive and the other for negative.  The `Act` value always represents the (signed) computed value, not the spike rate, where applicable.
