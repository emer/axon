// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/Astera-org/axon/fsfffb"
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

/* todo: fixme below -- dumping this here so layer is clean

// TopoGi computes topographic Gi inhibition
// todo: this does not work for 2D layers, and in general needs more testing
func (ly *Layer) TopoGi(ctime *Time) {
	if !ly.Inhib.Topo.On {
		return
	}
	pyn := ly.Shp.Dim(0)
	pxn := ly.Shp.Dim(1)
	wd := ly.Inhib.Topo.Width
	wrap := ly.Inhib.Topo.Wrap

	ssq := ly.Inhib.Topo.Sigma * float32(wd)
	ssq *= ssq
	ff0 := ly.Inhib.Topo.FF0

	l4d := ly.Is4D()

	var clip bool
	for py := 0; py < pyn; py++ {
		for px := 0; px < pxn; px++ {
			var tge, tact, twt float32
			for iy := -wd; iy <= wd; iy++ {
				ty := py + iy
				if ty, clip = edge.Edge(ty, pyn, wrap); clip {
					continue
				}
				for ix := -wd; ix <= wd; ix++ {
					tx := px + ix
					if tx, clip = edge.Edge(tx, pxn, wrap); clip {
						continue
					}
					ds := float32(iy*iy + ix*ix)
					df := mat32.Sqrt(ds)
					di := int(mat32.Round(df))
					if di > wd {
						continue
					}
					wt := mat32.FastExp(-0.5 * ds / ssq)
					twt += wt
					ti := ty*pxn + tx
					if l4d {
						pl := &ly.Pools[ti+1]
						tge += wt * pl.OldInhib.Ge.Avg
						tact += wt * pl.OldInhib.Act.Avg
					} else {
						nrn := &ly.Neurons[ti]
						tge += wt * nrn.Ge
						tact += wt * nrn.Act
					}
				}
			}

			gi := ly.Inhib.Topo.GiFmGeAct(tge, tact, ff0*twt)
			pi := py*pxn + px
			if l4d {
				pl := &ly.Pools[pi+1]
				pl.OldInhib.Gi += gi
				// } else {
				// 	nrn := &ly.Neurons[pi]
				// 	nrn.GiSelf = gi
			}
		}
	}
}
*/
