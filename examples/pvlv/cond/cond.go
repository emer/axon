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

	// number of behavioral trials per block -- blocks, with the different types of Trials specified in Block allocated across these Trials.  More different trial types and greater stochasticity (lower probability) of US presentation requires more trials.
	NTrials int

	// permute list of generated trials in random order after generation -- otherwise presented in order specified in the Block type
	Permute bool
}
