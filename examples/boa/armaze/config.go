// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import "github.com/emer/etable/minmax"

// Params are misc environment parameters
type Params struct {

	// [def: {'Min':0.5, 'Max':0.5}] effort for turning
	TurnEffort minmax.F32 `def:"{'Min':0.5, 'Max':0.5}" desc:"effort for turning"`

	// [def: {'Min':0.5, 'Max':0.5}] effort for consuming US
	ConsumeEffort minmax.F32 `def:"{'Min':0.5, 'Max':0.5}" desc:"effort for consuming US"`

	// [def: true] always turn left -- zoolander style -- reduces degrees of freedom in evaluating behavior
	AlwaysLeft bool `def:"true" desc:"always turn left -- zoolander style -- reduces degrees of freedom in evaluating behavior"`

	// [def: true] after running down an Arm, a new random starting location is selected (otherwise same arm as last run)
	RandomStart bool `def:"true" desc:"after running down an Arm, a new random starting location is selected (otherwise same arm as last run)"`

	// [def: true] if true, allow movement between arms just by going Left or Right -- otherwise once past the start, no switching is allowed
	OpenArms bool `def:"true" desc:"if true, allow movement between arms just by going Left or Right -- otherwise once past the start, no switching is allowed"`

	// [def: {'Min':0, 'Max':0}] [view: inline] strength of inactive inputs (e.g., Drives in Approach paradigm)
	Inactive minmax.F32 `def:"{'Min':0, 'Max':0}" view:"inline" desc:"strength of inactive inputs (e.g., Drives in Approach paradigm)"`

	// [def: 4] number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets
	NYReps int `def:"4" desc:"number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets"`
}

// Config has environment configuration
type Config struct {

	// experimental paradigm that governs the configuration and updating of environment state over time and the appropriate evaluation criteria.
	Paradigm Paradigms `desc:"experimental paradigm that governs the configuration and updating of environment state over time and the appropriate evaluation criteria."`

	// for debugging, print out key steps including a trace of the action generation logic
	Debug bool `desc:"for debugging, print out key steps including a trace of the action generation logic"`

	// number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding positive US outcome -- this does not include the first curiosity drive
	NDrives int `desc:"number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding positive US outcome -- this does not include the first curiosity drive"`

	// number of negative US outcomes -- these are added after NDrives positive USs to total US list
	NNegUSs int `desc:"number of negative US outcomes -- these are added after NDrives positive USs to total US list"`

	// total number of USs = NDrives + NNegUSs
	NUSs int `inactive:"+" desc:"total number of USs = NDrives + NNegUSs"`

	// number of different arms
	NArms int `desc:"number of different arms"`

	// number of different CSs -- typically at least a unique CS per US -- relationship is determined in the US params
	NCSs int `desc:"number of different CSs -- typically at least a unique CS per US -- relationship is determined in the US params"`
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
