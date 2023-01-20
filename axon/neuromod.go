// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/goki/gosl/slbool"

//gosl: start neuromod

// NeuroModVals neuromodulatory values -- they are global to the layer and
// affect learning rate and other neural activity parameters of neurons.
type NeuroModVals struct {
	Rew      float32     `inactive:"+" desc:"reward value -- this is set here in the Context struct, and the RL Rew layer grabs it from there -- must also set HasRew flag when rew is set -- otherwise is ignored."`
	HasRew   slbool.Bool `inactive:"+" desc:"must be set to true when a reward is present -- otherwise Rew is ignored"`
	RewPred  float32     `inactive:"+" desc:"reward prediction -- computed by a special reward prediction layer"`
	PrevPred float32     `inactive:"+" desc:"previous time step reward prediction -- e.g., for TDPredLayer"`
	DA       float32     `inactive:"+" desc:"dopamine -- represents reward prediction error, signaled as phasic increases or decreases in activity relative to a tonic baseline.  Released by the VTA -- ventral tegmental area, or SNc -- substantia nigra pars compacta."`
	ACh      float32     `inactive:"+" desc:"acetylcholine -- activated by salient events, particularly at the onset of a reward / punishment outcome (US), or onset of a conditioned stimulus (CS).  Driven by BLA -> PPtg that detects changes in BLA activity"`
	NE       float32     `inactive:"+" desc:"norepinepherine -- not yet in use"`
	Ser      float32     `inactive:"+" desc:"serotonin -- not yet in use"`

	pad float32
}

func (nm *NeuroModVals) Reset() {
	nm.Rew = 0
	nm.HasRew.SetBool(false)
	nm.RewPred = 0
	nm.DA = 0
	nm.ACh = 0
	nm.NE = 0
	nm.Ser = 0
}

// SetRew is a convenience function for setting the external reward
func (nm *NeuroModVals) SetRew(rew float32, hasRew bool) {
	nm.HasRew.SetBool(hasRew)
	if hasRew {
		nm.Rew = rew
	} else {
		nm.Rew = 0
	}
}

//gosl: end neuromod
