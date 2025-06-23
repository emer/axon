// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
This is the Go implementation of the Axon algorithm for spiking
biologically based models of cognition, based on the
emergent framework. Development of Axon is supported by the
Obelisk project at Astera.org and by collaborations with scientists
at the University of California Davis, and other institutions around
he world.

Axon is the spiking version of Leabra, with several advances.
As a backcronym, _Axon_ could stand for Adaptive eXcitation Of Noise,
reflecting the ability to learn using the power of error-backpropagation
in the context of noisy spiking activation. The spiking function of the
axon is what was previously missing from Leabra.

Axon is used to develop large-scale systems-neuroscience models
of the brain, and full documentation is available at CompCogNeuro.org,
including running examples of simulations in this package, and the
Rubicon model of goal-driven, motivated cognition.
*/
package axon
