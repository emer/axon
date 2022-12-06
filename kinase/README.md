# Kinase Learning Implementation

This implements central elements of the Kinase learning rule, including variables with associated time constants used for integrating calcium signals through the cascade of progressively longer time-integrals, from `Ca` -> `CaM` calmodulin (at `MTau`) -> CaMKII (`CaP` at `PTau` for LTP role) -> DAPK1 (`CaD` at `DTau` for LTD role).

See [kinaseq example](https://github.com/emer/axon/tree/master/examples/kinaseq) for an exploration of the implemented equations, and [kinase repository](https://github.com/ccnlab/kinase/tree/main/sims/kinase) for documentation and simulations about the biophysical basis of the equations.

# Time constants and Variables

* `MTau` (2 or 5) = calmodulin (`CaM`) time constant in cycles (msec) -- for synaptic-level integration this integrates on top of Ca signal from send->CaSyn * recv->CaSyn, each of which are typically integrated with a 30 msec Tau.

* `PTau` (40) = LTP spike-driven Ca factor (`CaP`) time constant in cycles (msec), simulating CaMKII in the Kinase framework, with 40 on top of MTau roughly tracking the biophysical rise time.  Computationally, CaP represents the plus phase learning signal that reflects the most recent past information.

* `DTau` (40) = LTD spike-driven Ca factor (`CaD`) time constant in cycles (msec), simulating DAPK1 in Kinase framework.  Computationally, CaD represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome).

