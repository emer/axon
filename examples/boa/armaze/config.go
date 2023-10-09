// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import "github.com/emer/etable/minmax"

// Params are misc environment parameters
type Params struct {

	// [def: {'Min':0.5, 'Max':0.5}] effort for turning
	TurnEffort minmax.F32 `nest:"+" def:"{'Min':0.5, 'Max':0.5}" desc:"effort for turning"`

	// [def: {'Min':0.5, 'Max':0.5}] effort for consuming US
	ConsumeEffort minmax.F32 `nest:"+" def:"{'Min':0.5, 'Max':0.5}" desc:"effort for consuming US"`

	// [def: true] always turn left -- zoolander style -- reduces degrees of freedom in evaluating behavior
	AlwaysLeft bool `def:"true" desc:"always turn left -- zoolander style -- reduces degrees of freedom in evaluating behavior"`

	// [def: false] permute the order of USs prior to applying them to arms -- having this off makes it easier to visually determine match between Drive and arm approach, and shouldn't make any difference to behavior (model doesn't know about this ordering).
	PermuteUSs bool `def:"false" desc:"permute the order of USs prior to applying them to arms -- having this off makes it easier to visually determine match between Drive and arm approach, and shouldn't make any difference to behavior (model doesn't know about this ordering)."`

	// [def: true] evenly distribute CSs among the USs.  Otherwise the configuration must explicitly assign probabilities within each US for each CS.
	EvenCSs bool `def:"true" desc:"evenly distribute CSs among the USs.  Otherwise the configuration must explicitly assign probabilities within each US for each CS."`

	// [def: true] after running down an Arm, a new random starting location is selected (otherwise same arm as last run)
	RandomStart bool `def:"true" desc:"after running down an Arm, a new random starting location is selected (otherwise same arm as last run)"`

	// [def: true] if true, allow movement between arms just by going Left or Right -- otherwise once past the start, no switching is allowed
	OpenArms bool `def:"true" desc:"if true, allow movement between arms just by going Left or Right -- otherwise once past the start, no switching is allowed"`

	// [def: {'Min':0, 'Max':0}] [view: inline] strength of inactive inputs (e.g., Drives in Approach paradigm)
	Inactive minmax.F32 `nest:"+" def:"{'Min':0, 'Max':0}" view:"inline" desc:"strength of inactive inputs (e.g., Drives in Approach paradigm)"`

	// [def: 4] number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets
	NYReps int `def:"4" desc:"number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets"`

	// [def: false] if true, each CS awards multiple USs (def 2)
	MultiUS bool `def:"false" desc:"if true, each CS awards multiple USs (def 2)"`
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

	// parameters associated with each US.  The first NDrives are positive USs, and beyond that are negative USs
	USs []*USParams `desc:"parameters associated with each US.  The first NDrives are positive USs, and beyond that are negative USs"`

	// state of each arm: dist, effort, US, CS
	Arms []*Arm `desc:"state of each arm: dist, effort, US, CS"`

	// [view: add-fields] misc params
	Params Params `view:"add-fields" desc:"misc params"`
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
