// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import "github.com/emer/etable/v2/minmax"

// Arm represents the properties of a given arm of the N-maze,
// representing a different choice option with different cost / benefit
// tradeoffs, in terms of distance and effort factors for getting
// down the arm, and US present at the end, which is delivered with
// a given probability and magnitude range.
// Each arm has its own distinctive CS visible at the start, which is the
// only cue used for the agent to decide whether to choose this arm or not.
type Arm struct {
	// CS == index of Arm
	CS int

	// length of arm: distance from CS start to US end for this arm
	Length int

	// index of US present at the end of this arm.
	// Indexes [0:NDrives] are positive USs, and beyond that are negative USs.
	US int

	// range of different effort levels per step (uniformly randomly sampled per step) for going down this arm
	Effort minmax.F32

	// range of different US magnitudes (uniformly sampled)
	USMag minmax.F32

	// probability of delivering the US
	USProb float32

	// nominal expected value = US.Prob * US.Mag
	ExValue float32 `edit:"-"`

	// nominal expected cost = effort + normalized length
	ExCost float32 `edit:"-"`

	// nominal expected utility = ExValue - CostFactor * ExCost.
	// This is only meaningful relative to other options, not in any absolute terms.
	ExUtil float32 `edit:"-"`

	// UtilGroup is the group id for computing the BestOption utility for this arm:
	// = US for positive, and NDrives for all negative USs
	UtilGroup int

	// BestOption is true if this arm represents the best option in terms of ExUtil
	// relative to other options _for the same US_.
	// All negative USs are considered as one group for ranking.
	BestOption bool `edit:"-"`
}

func (arm *Arm) Defaults() {
	arm.Length = 4
	arm.Effort.Set(1, 1)
	arm.USMag.Set(1, 1)
	arm.USProb = 1
	arm.Empty()
}

// Empty sets all state to -1
func (arm *Arm) Empty() {
	arm.US = -1
	arm.CS = -1
	arm.ExValue = 0
	arm.ExUtil = 0
}
