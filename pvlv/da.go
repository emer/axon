// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import "github.com/goki/ki/kit"

// Dopamine receptor type, for D1R and D2R dopamine receptors
type DARs int

const (
	// D1R: primarily expresses Dopamine D1 Receptors, which are excitatory from DA bursts
	D1R DARs = iota

	// D2R: primarily expresses Dopamine D2 Receptors, which are inhibitory from DA dips
	D2R

	DARsN
)

var KiT_DARs = kit.Enums.AddEnum(DARsN, kit.NotBitFlag, nil)

// DaModParams specifies parameters shared by all layers that receive dopaminergic modulatory input.
type DaModParams struct {
	On        bool    `desc:"whether to use dopamine modulation"`
	DAR       DARs    `desc:"dopamine receptor type, D1 or D2"`
	BurstGain float32 `desc:"multiplicative gain factor applied to positive dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign!"`
	DipGain   float32 `desc:"multiplicative gain factor applied to negative dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign! should be small for acq, but roughly equal to burst for ext"`
}

func (dp *DaModParams) Defaults() {
	dp.On = true
	dp.BurstGain = 1
	dp.DipGain = 1
}

// Gain returns effective DA gain factor given raw da +/- burst / dip value
func (dp *DaModParams) Gain(da float32) float32 {
	if !dp.On {
		return 0
	}
	if da > 0 {
		da *= dp.BurstGain
	} else {
		da *= dp.DipGain
	}
	if dp.DAR == D2R {
		return -da
	}
	return da
}
