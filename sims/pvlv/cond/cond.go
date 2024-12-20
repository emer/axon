// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

// Condition defines parameters for running a specific type of conditioning expt
type Condition struct {

	// identifier for this type of configuration
	Name string

	// description of this configuration
	Desc string

	// mix of trial types per block to run -- must be listed in AllBlocks
	Block string

	// use a permuted list to ensure an exact number of trials have US -- else random draw each time
	FixedProb bool

	// number of full blocks of different trial types to run (like Epochs)
	NBlocks int

	// number of sequences (behavioral trials) per block, with the different types of Sequences allocated across these sequences.  More different sequence types and greater stochasticity (lower probability) of US presentation requires more sequences.
	NSequences int

	// permute list of generated trials in random order after generation -- otherwise presented in order specified in the Block type
	Permute bool
}
