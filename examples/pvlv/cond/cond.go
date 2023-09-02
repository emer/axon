// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cond

// Condition defines parameters for running a specific type of conditioning expt
type Condition struct {

	// identifier for this type of configuration
	Name string `desc:"identifier for this type of configuration"`

	// description of this configuration
	Desc string `desc:"description of this configuration"`

	// mix of trial types per block to run -- must be listed in AllBlocks
	Block string `desc:"mix of trial types per block to run -- must be listed in AllBlocks"`

	// use a permuted list to ensure an exact number of trials have US -- else random draw each time
	FixedProb bool `desc:"use a permuted list to ensure an exact number of trials have US -- else random draw each time"`

	// number of full blocks of different trial types to run (like Epochs)
	NBlocks int `desc:"number of full blocks of different trial types to run (like Epochs)"`

	// number of behavioral trials per block -- blocks, with the different types of Trials specified in Block allocated across these Trials.  More different trial types and greater stochasticity (lower probability) of US presentation requires more trials.
	NTrials int `desc:"number of behavioral trials per block -- blocks, with the different types of Trials specified in Block allocated across these Trials.  More different trial types and greater stochasticity (lower probability) of US presentation requires more trials."`

	// permute list of generated trials in random order after generation -- otherwise presented in order specified in the Block type
	Permute bool `desc:"permute list of generated trials in random order after generation -- otherwise presented in order specified in the Block type"`
}
