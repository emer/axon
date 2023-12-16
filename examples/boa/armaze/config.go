// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import "goki.dev/etable/v2/minmax"

// Params are misc environment parameters
type Params struct {

	// effort for turning
	TurnEffort minmax.F32 `nest:"+" def:"{'Min':0.5, 'Max':0.5}"`

	// effort for consuming US
	ConsumeEffort minmax.F32 `nest:"+" def:"{'Min':0.5, 'Max':0.5}"`

	// always turn left -- zoolander style -- reduces degrees of freedom in evaluating behavior
	AlwaysLeft bool `def:"true"`

	// permute the order of CSs prior to applying them to arms -- having this off makes it easier to visually determine match between Drive and arm approach, and shouldn't make any difference to behavior (model doesn't know about this ordering).
	PermuteCSs bool `def:"false"`

	// after running down an Arm, a new random starting location is selected (otherwise same arm as last run)
	RandomStart bool `def:"true"`

	// if true, allow movement between arms just by going Left or Right -- otherwise once past the start, no switching is allowed
	OpenArms bool `def:"true"`

	// strength of inactive inputs (e.g., Drives in Approach paradigm)
	Inactive minmax.F32 `nest:"+" def:"{'Min':0, 'Max':0}" view:"inline"`

	// number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets
	NYReps int `def:"4"`
}

// Config has environment configuration
type Config struct {

	// experimental paradigm that governs the configuration and updating of environment state over time and the appropriate evaluation criteria.
	Paradigm Paradigms

	// for debugging, print out key steps including a trace of the action generation logic
	Debug bool

	// number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding positive US outcome -- this does not include the first curiosity drive
	NDrives int

	// number of negative US outcomes -- these are added after NDrives positive USs to total US list
	NNegUSs int

	// total number of USs = NDrives + NNegUSs
	NUSs int `inactive:"+"`

	// number of different arms
	NArms int

	// maximum arm length (distance)
	MaxArmLength int

	// number of different CSs -- typically at least a unique CS per US -- relationship is determined in the US params
	NCSs int

	// parameters associated with each US.  The first NDrives are positive USs, and beyond that are negative USs
	USs []*USParams

	// state of each arm: dist, effort, US, CS
	Arms []*Arm

	// misc params
	Params Params `view:"add-fields"`
}

func (cfg *Config) Defaults() {
	if cfg.NDrives == 0 {
		cfg.NDrives = 4
	}
	cfg.Update()
	if cfg.NCSs == 0 {
		cfg.NCSs = cfg.NUSs
	}
}

func (cfg *Config) Update() {
	cfg.NUSs = cfg.NDrives + cfg.NNegUSs
}
