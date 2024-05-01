// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import "cogentcore.org/core/math32/minmax"

// Params are misc environment parameters
type Params struct {

	// effort for turning
	TurnEffort minmax.F32 `nest:"+" default:"{'Min':0.5, 'Max':0.5}"`

	// effort for consuming US
	ConsumeEffort minmax.F32 `nest:"+" default:"{'Min':0.5, 'Max':0.5}"`

	// an arbitrary scaling factor for costs relative to values,
	// used in computing the expected utility ExUtil for an arm.
	// These utilities are only useful for relative comparisons,
	// that go into computing the UtilRank, which should be used for evaluating
	// overall choices.
	CostFactor float32 `default:"0.2"`

	// threshold for considering a drive to be active; used in evaluating whether
	// an Arm choice is considered to be a good option.
	ActiveDriveThr float32 `default:"0.5"`

	// always turn left -- zoolander style -- reduces degrees of freedom in evaluating behavior
	AlwaysLeft bool `default:"true"`

	// after running down an Arm, a new random starting location is selected (otherwise same arm as last run)
	RandomStart bool `default:"true"`

	// if true, allow movement between arms just by going Left or Right.
	// Otherwise once past the start, no switching is allowed
	OpenArms bool `default:"true"`

	// strength of inactive inputs (e.g., Drives in Approach paradigm)
	Inactive minmax.F32 `nest:"+" default:"{'Min':0, 'Max':0}" view:"inline"`

	// number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets
	NYReps int `default:"4"`
}

// Config has environment configuration
type Config struct {

	// experimental paradigm that governs the configuration of environment based on params,
	// e.g., how the Range values are assigned to different arms.
	Paradigm Paradigms

	// for debugging, print out key steps including a trace of the action generation logic
	Debug bool

	// number of different drive-like body states (hunger, thirst, etc),
	// that are satisfied by a corresponding positive US outcome.
	// This is in addition to the first curiosity drive, which is always present.
	NDrives int

	// number of negative US outcomes -- these are added after NDrives positive USs to total US list
	NNegUSs int

	// total number of USs = NDrives + NNegUSs
	NUSs int `edit:"-"`

	// number of different arms, each of which has its own distinctive CS.
	// This is determined by the Paradigm (e.g., 2*NUSs for the Group cases).
	NArms int `edit:"-"`

	// range of arm length sallocated across arms, per Paradigm
	LengthRange minmax.Int `nest:"+"`

	// range of effort values allocated across arms, per Paradigm
	EffortRange minmax.F32 `nest:"+"`

	// range of US magnitudes allocated across arms, per Paradigm
	USMagRange minmax.F32 `nest:"+"`

	// range of US probabilities allocated across arms, per Paradigm
	USProbRange minmax.F32 `nest:"+"`

	// parameters for each arm option: dist, effort, US
	Arms []*Arm

	// misc params
	Params Params `view:"add-fields"`
}

func (cfg *Config) Defaults() {
	if cfg.NDrives == 0 {
		cfg.NDrives = 4
		cfg.LengthRange.Set(4, 4)
		cfg.EffortRange.Set(1, 1)
		cfg.USMagRange.Set(1, 1)
		cfg.USProbRange.Set(1, 1)
	}
	cfg.Update()
}

func (cfg *Config) Update() {
	cfg.NUSs = cfg.NDrives + cfg.NNegUSs
}
