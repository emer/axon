// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start neuromod

// NeuroModVals neuromodulatory values -- they are global to the layer and
// affect learning rate and other neural activity parameters of neurons.
type NeuroModVals struct {
	DA  float32 `inactive:"+" desc:"dopamine -- represents reward prediction error, signaled as phasic increases or decreases in activity relative to a tonic baseline.  Released by the VTA -- ventral tegmental area, or SNc -- substantia nigra pars compacta."`
	ACh float32 `inactive:"+" desc:"acetylcholine -- activated by salient events, particularly at the onset of a reward / punishment outcome (US), or onset of a conditioned stimulus (CS).  Driven by BLA -> PPtg that detects changes in BLA activity"`
	NE  float32 `inactive:"+" desc:"norepinepherine -- not yet in use"`
	Ser float32 `inactive:"+" desc:"serotonin -- not yet in use"`
}

func (nm *NeuroModVals) Reset() {
	nm.DA = 0
	nm.ACh = 0
	nm.NE = 0
	nm.Ser = 0
}

//gosl: end neuromod
