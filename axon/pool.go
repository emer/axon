// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/axon/fsfffb"
	"github.com/emer/etable/minmax"
)

// Pool contains computed values for FS-FFFB inhibition,
// and various other state values for layers
// and pools (unit groups) that can be subject to inhibition
type Pool struct {
	StIdx, EdIdx int             `inactive:"+" desc:"starting and ending (exlusive) indexes for the list of neurons in this pool"`
	Inhib        fsfffb.Inhib    `inactive:"+" desc:"fast-slow FFFB inhibition values"`
	ActM         minmax.AvgMax32 `inactive:"+" desc:"minus phase average and max Act activation values, for ActAvg updt"`
	ActP         minmax.AvgMax32 `inactive:"+" desc:"plus phase average and max Act activation values, for ActAvg updt"`
	GeM          minmax.AvgMax32 `inactive:"+" desc:"stats for GeM minus phase averaged Ge values"`
	GiM          minmax.AvgMax32 `inactive:"+" desc:"stats for GiM minus phase averaged Gi values"`
	AvgDif       minmax.AvgMax32 `inactive:"+" desc:"absolute value of AvgDif differences from actual neuron ActPct relative to TrgAvg"`
}

func (pl *Pool) Init() {
	pl.Inhib.Init()
}

// NNeurons returns the number of neurons in the pool: EdIdx - StIdx
func (pl *Pool) NNeurons() int {
	return pl.EdIdx - pl.StIdx
}

// ActAvgVals are running-average activation levels used for Ge scaling and adaptive inhibition
type ActAvgVals struct {
	ActMAvg   float32         `inactive:"+" desc:"running-average minus-phase activity integrated at Dt.LongAvgTau -- used for adapting inhibition relative to target level"`
	ActPAvg   float32         `inactive:"+" desc:"running-average plus-phase activity integrated at Dt.LongAvgTau"`
	AvgMaxGeM float32         `inactive:"+" desc:"running-average max of minus-phase Ge value across the layer integrated at Dt.LongAvgTau -- for monitoring and adjusting Prjn scaling factors: Prjn PrjnScale"`
	AvgMaxGiM float32         `inactive:"+" desc:"running-average max of minus-phase Gi value across the layer integrated at Dt.LongAvgTau -- for monitoring and adjusting Prjn scaling factors: Prjn PrjnScale"`
	GiMult    float32         `inactive:"+" desc:"multiplier on inhibition -- adapted to maintain target activity level"`
	CaSpkPM   minmax.AvgMax32 `inactive:"+" desc:"maximum CaSpkP value in layer in the minus phase -- for monitoring network activity levels"`
	CaSpkP    minmax.AvgMax32 `inactive:"+" desc:"maximum CaSpkP value in layer, updated in plus phase and used for normalizing CaSpkP values in RLrate sigmoid derivative computation"`
}

// CorSimStats holds correlation similarity (centered cosine aka normalized dot product)
// statistics at the layer level
type CorSimStats struct {
	Cor float32 `inactive:"+" desc:"correlation (centered cosine aka normalized dot product) activation difference between ActP and ActM on this alpha-cycle for this layer -- computed by CorSimFmActs called by PlusPhase"`
	Avg float32 `inactive:"+" desc:"running average of correlation similarity between ActP and ActM -- computed with CorSim.Tau time constant in PlusPhase"`
	Var float32 `inactive:"+" desc:"running variance of correlation similarity between ActP and ActM -- computed with CorSim.Tau time constant in PlusPhase"`
}

func (cd *CorSimStats) Init() {
	cd.Cor = 0
	cd.Avg = 0
}
