// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/emer/etable/minmax"

//gosl: start layervals

// ActAvgVals are running-average activation levels used for Ge scaling and adaptive inhibition
type ActAvgVals struct {
	ActMAvg   float32 `inactive:"+" desc:"running-average minus-phase activity integrated at Dt.LongAvgTau -- used for adapting inhibition relative to target level"`
	ActPAvg   float32 `inactive:"+" desc:"running-average plus-phase activity integrated at Dt.LongAvgTau"`
	AvgMaxGeM float32 `inactive:"+" desc:"running-average max of minus-phase Ge value across the layer integrated at Dt.LongAvgTau -- for monitoring and adjusting Prjn scaling factors: Prjn PrjnScale"`
	AvgMaxGiM float32 `inactive:"+" desc:"running-average max of minus-phase Gi value across the layer integrated at Dt.LongAvgTau -- for monitoring and adjusting Prjn scaling factors: Prjn PrjnScale"`
	GiMult    float32 `inactive:"+" desc:"multiplier on inhibition -- adapted to maintain target activity level"`

	pad, pad1, pad2 float32

	CaSpkPM minmax.AvgMax32 `inactive:"+" desc:"avg and maximum CaSpkP value in layer in the minus phase -- for monitoring network activity levels"`
	CaSpkP  minmax.AvgMax32 `inactive:"+" desc:"avg and maximum CaSpkP value in layer, updated in plus phase and used for normalizing CaSpkP values in RLRate sigmoid derivative computation"`
	CaSpkD  minmax.AvgMax32 `inactive:"+" desc:"avg and maximum CaSpkD value in layer, updated in plus phase and used for normalizing CaSpkD values in RLRate sigmoid derivative computation"`
}

// CorSimStats holds correlation similarity (centered cosine aka normalized dot product)
// statistics at the layer level
type CorSimStats struct {
	Cor float32 `inactive:"+" desc:"correlation (centered cosine aka normalized dot product) activation difference between ActP and ActM on this alpha-cycle for this layer -- computed by CorSimFmActs called by PlusPhase"`
	Avg float32 `inactive:"+" desc:"running average of correlation similarity between ActP and ActM -- computed with CorSim.Tau time constant in PlusPhase"`
	Var float32 `inactive:"+" desc:"running variance of correlation similarity between ActP and ActM -- computed with CorSim.Tau time constant in PlusPhase"`

	pad float32
}

func (cd *CorSimStats) Init() {
	cd.Cor = 0
	cd.Avg = 0
}

// LayerVals holds extra layer state that is updated per layer
type LayerVals struct {
	ActAvg   ActAvgVals   `view:"inline" desc:"running-average activation levels used for Ge scaling and adaptive inhibition"`
	CorSim   CorSimStats  `desc:"correlation (centered cosine aka normalized dot product) similarity between ActM, ActP states"`
	NeuroMod NeuroModVals `desc:"neuromodulatory values: global to the layer, copied from Context"`
}

//gosl: end layervals
