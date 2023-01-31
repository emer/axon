// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/axon/fsfffb"
	"github.com/emer/etable/minmax"
	"github.com/goki/gosl/slbool"
)

//gosl: start pool

// AvgMaxPhases contains the average and maximum values over a Pool of neurons,
// at different time scales within a standard ThetaCycle of updating.
// It is much more efficient on the GPU to just grab everything in one pass at
// the cycle level, and then take snapshots from there.
// All of the cycle level values are updated at the *start* of the cycle
// based on values from the prior cycle -- thus are 1 cycle behind in general.
type AvgMaxPhases struct {
	Cycle minmax.AvgMax32 `inactive:"+" view:"inline" desc:"updated every cycle -- this is the source of all subsequent time scales"`
	Minus minmax.AvgMax32 `inactive:"+" view:"inline" desc:"at the end of the minus phase"`
	Plus  minmax.AvgMax32 `inactive:"+" view:"inline" desc:"at the end of the plus phase"`
}

// CycleToMinus grabs current Cycle values into the Minus phase values
func (am *AvgMaxPhases) CycleToMinus() {
	am.Minus = am.Cycle
}

// CycleToPlus grabs current Cycle values into the Plus phase values
func (am *AvgMaxPhases) CycleToPlus() {
	am.Plus = am.Cycle
}

// PoolAvgMax contains the average and maximum values over a Pool of neurons
// for different variables of interest, at Cycle, Minus and Plus phase timescales.
// All of the cycle level values are updated at the *start* of the cycle
// based on values from the prior cycle -- thus are 1 cycle behind in general.
type PoolAvgMax struct {
	CaSpkP AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum CaSpkP (continuously updated at roughly 40 msec integration window timescale, ends up capturing potentiation, plus-phase signal) -- this is the primary variable to use for tracking overall pool activity"`
	CaSpkD AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum CaSpkD longer-term depression / DAPK1 signal in layer"`
	SpkMax AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum SpkMax value (based on CaSpkP) -- reflects peak activity at any point across the cycle"`
	Act    AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum Act firing rate value"`
	Ge     AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum Ge excitatory conductance value"`
	Gi     AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum Gi inhibitory conductance value"`
}

// CycleToMinus grabs current Cycle values into the Minus phase values
func (am *PoolAvgMax) CycleToMinus() {
	am.CaSpkP.CycleToMinus()
	am.CaSpkD.CycleToMinus()
	am.SpkMax.CycleToMinus()
	am.Act.CycleToMinus()
	am.Ge.CycleToMinus()
	am.Gi.CycleToMinus()
}

// CycleToPlus grabs current Cycle values into the Plus phase values
func (am *PoolAvgMax) CycleToPlus() {
	am.CaSpkP.CycleToPlus()
	am.CaSpkD.CycleToPlus()
	am.SpkMax.CycleToPlus()
	am.Act.CycleToPlus()
	am.Ge.CycleToPlus()
	am.Gi.CycleToPlus()
}

// Init does Init on Cycle level -- for update start
func (am *PoolAvgMax) Init() {
	am.CaSpkP.Cycle.Init()
	am.CaSpkD.Cycle.Init()
	am.SpkMax.Cycle.Init()
	am.Act.Cycle.Init()
	am.Ge.Cycle.Init()
	am.Gi.Cycle.Init()
}

// CalcAvg does CalcAvg on Cycle level
func (am *PoolAvgMax) CalcAvg() {
	am.CaSpkP.Cycle.CalcAvg()
	am.CaSpkD.Cycle.CalcAvg()
	am.SpkMax.Cycle.CalcAvg()
	am.Act.Cycle.CalcAvg()
	am.Ge.Cycle.CalcAvg()
	am.Gi.Cycle.CalcAvg()
}

// UpdateVals for neuron values
func (am *PoolAvgMax) UpdateVals(nrn *Neuron, ni int32) {
	am.CaSpkP.Cycle.UpdateVal(nrn.CaSpkP, ni)
	am.CaSpkD.Cycle.UpdateVal(nrn.CaSpkD, ni)
	am.SpkMax.Cycle.UpdateVal(nrn.SpkMax, ni)
	am.Act.Cycle.UpdateVal(nrn.Act, ni)
	am.Ge.Cycle.UpdateVal(nrn.Ge, ni)
	am.Gi.Cycle.UpdateVal(nrn.Gi, ni)
}

// Pool contains computed values for FS-FFFB inhibition,
// and various other state values for layers
// and pools (unit groups) that can be subject to inhibition
type Pool struct {
	StIdx, EdIdx uint32      `inactive:"+" desc:"starting and ending (exlusive) layer-wise indexes for the list of neurons in this pool"`
	LayIdx       uint32      `view:"-" desc:"layer index in global layer list"`
	PoolIdx      uint32      `view:"-" desc:"pool index in global pool list: [Layer][Pool]"`
	IsLayPool    slbool.Bool `inactive:"+" desc:"is this a layer-wide pool?  if not, it represents a sub-pool of units within a 4D layer"`
	Gated        slbool.Bool `inactive:"+" desc:"for special types where relevant (e.g., MatrixLayer, VThalLayer), indicates if the pool was gated"`

	pad, pad1 uint32

	Inhib  fsfffb.Inhib    `inactive:"+" desc:"fast-slow FFFB inhibition values"`
	AvgMax PoolAvgMax      `desc:"average and max values for relevant variables in this pool, at different time scales"`
	AvgDif minmax.AvgMax32 `inactive:"+" view:"inline" desc:"absolute value of AvgDif differences from actual neuron ActPct relative to TrgAvg"`
}

// Init is callled during InitActs
func (pl *Pool) Init() {
	pl.Inhib.Init()
	pl.AvgMax.Init()
	pl.Gated.SetBool(false)
}

// NNeurons returns the number of neurons in the pool: EdIdx - StIdx
func (pl *Pool) NNeurons() int {
	return int(pl.EdIdx - pl.StIdx)
}

//gosl: end pool

/* todo: fixme below -- dumping this here so layer is clean

// TopoGi computes topographic Gi inhibition
// todo: this does not work for 2D layers, and in general needs more testing
func (ly *Layer) TopoGi(ctx *Context) {
	if !ly.Params.Inhib.Topo.On {
		return
	}
	pyn := ly.Shp.Dim(0)
	pxn := ly.Shp.Dim(1)
	wd := ly.Params.Inhib.Topo.Width
	wrap := ly.Params.Inhib.Topo.Wrap

	ssq := ly.Params.Inhib.Topo.Sigma * float32(wd)
	ssq *= ssq
	ff0 := ly.Params.Inhib.Topo.FF0

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

			gi := ly.Params.Inhib.Topo.GiFmGeAct(tge, tact, ff0*twt)
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
