// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import "github.com/emer/etable/minmax"

// Arm represents the properties of a given arm of the N-maze.
// Arms have characteristic distance and effort factors for getting
// down the arm, and typically have a distinctive CS visible at the start
// and a US at the end, which has US-specific parameters for
// actually delivering reward or punishment.
type Arm struct {

	// length of arm: distance from CS start to US end for this arm
	Length int `desc:"length of arm: distance from CS start to US end for this arm"`

	// range of different effort levels per step (uniformly randomly sampled per step) for going down this arm
	Effort minmax.F32 `desc:"range of different effort levels per step (uniformly randomly sampled per step) for going down this arm"`

	// todo: later
	// indexes of US[s] present at the end of this arm -- nil if none
	// USs []int `desc:"indexes of US[s] present at the end of this arm -- nil if none"`

	// index of US present at the end of this arm -- -1 if none
	US int `desc:"index of US present at the end of this arm -- -1 if none"`

	// index of CS visible at the start of this arm, -1 if none
	CS int `desc:"index of CS visible at the start of this arm, -1 if none"`

	// current expected value = US.Prob * US.Mag * Drives[US] -- computed at start of new approach
	ExValue float32 `inactive:"+" desc:"current expected value = US.Prob * US.Mag * Drives[US] -- computed at start of new approach"`

	// current expected PVpos value = normalized ExValue -- computed at start of new approach
	ExPVpos float32 `inactive:"+" desc:"current expected PVpos value = normalized ExValue -- computed at start of new approach"`

	// current expected PVneg value = normalized time and effort costs
	ExPVneg float32 `inactive:"+" desc:"current expected PVneg value = normalized time and effort costs"`

	// current expected utility = effort discounted version of ExPVpos -- computed at start of new approach
	ExUtil float32 `inactive:"+" desc:"current expected utility = effort discounted version of ExPVpos -- computed at start of new approach"`
}

func (arm *Arm) Defaults() {
	arm.Length = 4
	arm.Effort.Set(1, 1)
	arm.Empty()
}

// Empty sets all state to -1
func (arm *Arm) Empty() {
	arm.US = -1
	arm.CS = -1
	arm.ExValue = 0
	arm.ExUtil = 0
}

// USParams has parameters for different USs
type USParams struct {

	// if true is a negative valence US -- these are after the first NDrives USs
	Negative bool `desc:"if true is a negative valence US -- these are after the first NDrives USs"`

	// range of different magnitudes (uniformly sampled)
	Mag minmax.F32 `desc:"range of different magnitudes (uniformly sampled)"`

	// probability of delivering the US
	Prob float32 `desc:"probability of delivering the US"`

	// probabilities of each CS being active for this US
	// CSProbs []float32 `desc:"probabilities of each CS being active for this US"`
}
