// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import "github.com/goki/ki/kit"

// Paradigms is a list of experimental paradigms that
// govern the configuration and updating of environment
// state over time and the appropriate evaluation criteria.
type Paradigms int

const (
	// Approach is a basic case where one Drive (chosen at random each trial) is fully active and others are at InactiveDrives levels -- goal is to approach the CS associated with the Drive-satisfying US, and avoid negative any negative USs.  USs are always placed in same Arms (NArms must be >= NUSs -- any additional Arms are filled at random with additional US copies)
	Approach Paradigms = iota

	ParadigmsN
)

//go:generate stringer -type=Paradigms

var KiT_Paradigms = kit.Enums.AddEnum(ParadigmsN, kit.NotBitFlag, nil)

///////////////////////////////////////////////
// Approach

// ConfigApproach does initial config for Approach paradigm
func (ev *Maze) ConfigApproach() {
	if ev.Config.NArms < ev.Config.NUSs {
		ev.Config.NArms = ev.Config.NUSs
	}
	if ev.Config.NCSs < ev.Config.NUSs {
		ev.Config.NCSs = ev.Config.NUSs
	}
}

// StartApproach does new start state setting for Approach
func (ev *Maze) StartApproach() {
	ev.TrgDrive = ev.Rand.Intn(ev.NDrives, -1)
	var armOpts []int
	for i, arm := range ev.Arms {
		if arm.US == ev.TrgDrive {
			armOpts = append(armOpts, i)
		}
	}
	// todo: select arm from opts, with error if none
}

func (ev *Maze) StepApproach() {

}
