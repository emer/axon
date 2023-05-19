// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/axon/fsfffb"
	"github.com/goki/gosl/slbool"
	"github.com/goki/mat32"
)

//gosl: hlsl pool
// #include "avgmaxi.hlsl"
//gosl: end pool

//gosl: start pool

// AvgMaxPhases contains the average and maximum values over a Pool of neurons,
// at different time scales within a standard ThetaCycle of updating.
// It is much more efficient on the GPU to just grab everything in one pass at
// the cycle level, and then take snapshots from there.
// All of the cycle level values are updated at the *start* of the cycle
// based on values from the prior cycle -- thus are 1 cycle behind in general.
type AvgMaxPhases struct {
	Cycle AvgMaxI32 `view:"inline" desc:"updated every cycle -- this is the source of all subsequent time scales"`
	Minus AvgMaxI32 `view:"inline" desc:"at the end of the minus phase"`
	Plus  AvgMaxI32 `view:"inline" desc:"at the end of the plus phase"`
	Prev  AvgMaxI32 `view:"inline" desc:"at the end of the previous plus phase"`
}

// CycleToMinus grabs current Cycle values into the Minus phase values
func (am *AvgMaxPhases) CycleToMinus() {
	am.Minus = am.Cycle
	am.Prev = am.Plus
}

// CycleToPlus grabs current Cycle values into the Plus phase values
func (am *AvgMaxPhases) CycleToPlus() {
	am.Plus = am.Cycle
}

// Calc does Calc on Cycle, which is then ready for aggregation again
func (am *AvgMaxPhases) Calc(refIdx int32) {
	am.Cycle.Calc(refIdx)
}

// Zero does a full reset on everything -- for InitActs
func (am *AvgMaxPhases) Zero() {
	am.Cycle.Zero()
	am.Minus.Zero()
	am.Plus.Zero()
	am.Prev.Zero()
}

// PoolAvgMax contains the average and maximum values over a Pool of neurons
// for different variables of interest, at Cycle, Minus and Plus phase timescales.
// All of the cycle level values are updated at the *start* of the cycle
// based on values from the prior cycle -- thus are 1 cycle behind in general.
type PoolAvgMax struct {
	CaSpkP   AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum CaSpkP (continuously updated at roughly 40 msec integration window timescale, ends up capturing potentiation, plus-phase signal) -- this is the primary variable to use for tracking overall pool activity"`
	CaSpkD   AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum CaSpkD longer-term depression / DAPK1 signal in layer"`
	SpkMax   AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum SpkMax value (based on CaSpkP) -- reflects peak activity at any point across the cycle"`
	Act      AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum Act firing rate value"`
	GeInt    AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum GeInt integrated running-average excitatory conductance value"`
	GeIntMax AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum GeIntMax integrated running-average excitatory conductance value"`
	GiInt    AvgMaxPhases `inactive:"+" view:"inline" desc:"avg and maximum GiInt integrated running-average inhibitory conductance value"`
}

// SetN sets the N for aggregation
func (am *PoolAvgMax) SetN(n int32) {
	am.CaSpkP.Cycle.N = n
	am.CaSpkD.Cycle.N = n
	am.SpkMax.Cycle.N = n
	am.Act.Cycle.N = n
	am.GeInt.Cycle.N = n
	am.GeIntMax.Cycle.N = n
	am.GiInt.Cycle.N = n
}

// CycleToMinus grabs current Cycle values into the Minus phase values
func (am *PoolAvgMax) CycleToMinus() {
	am.CaSpkP.CycleToMinus()
	am.CaSpkD.CycleToMinus()
	am.SpkMax.CycleToMinus()
	am.Act.CycleToMinus()
	am.GeInt.CycleToMinus()
	am.GeIntMax.CycleToMinus()
	am.GiInt.CycleToMinus()
}

// CycleToPlus grabs current Cycle values into the Plus phase values
func (am *PoolAvgMax) CycleToPlus() {
	am.CaSpkP.CycleToPlus()
	am.CaSpkD.CycleToPlus()
	am.SpkMax.CycleToPlus()
	am.Act.CycleToPlus()
	am.GeInt.CycleToPlus()
	am.GeIntMax.CycleToPlus()
	am.GiInt.CycleToPlus()
}

// Init does Init on Cycle vals-- for update start.
// always left init'd so generally unnecessary
func (am *PoolAvgMax) Init() {
	am.CaSpkP.Cycle.Init()
	am.CaSpkD.Cycle.Init()
	am.SpkMax.Cycle.Init()
	am.Act.Cycle.Init()
	am.GeInt.Cycle.Init()
	am.GeIntMax.Cycle.Init()
	am.GiInt.Cycle.Init()
}

// Zero does full reset on everything -- for InitActs
func (am *PoolAvgMax) Zero() {
	am.CaSpkP.Zero()
	am.CaSpkD.Zero()
	am.SpkMax.Zero()
	am.Act.Zero()
	am.GeInt.Zero()
	am.GeIntMax.Zero()
	am.GiInt.Zero()
}

// Calc does Calc on Cycle level, and re-inits
func (am *PoolAvgMax) Calc(refIdx int32) {
	am.CaSpkP.Calc(refIdx)
	am.CaSpkD.Calc(refIdx)
	am.SpkMax.Calc(refIdx)
	am.Act.Calc(refIdx)
	am.GeInt.Calc(refIdx)
	am.GeIntMax.Calc(refIdx)
	am.GiInt.Calc(refIdx)
}

// UpdateVals for neuron values
func (am *PoolAvgMax) UpdateVals(ctx *Context, ni, di uint32) {
	am.CaSpkP.Cycle.UpdateVal(NeurVar(ctx, ni, di, CaSpkP))
	am.CaSpkD.Cycle.UpdateVal(NeurVar(ctx, ni, di, CaSpkD))
	am.SpkMax.Cycle.UpdateVal(NeurVar(ctx, ni, di, SpkMax))
	am.Act.Cycle.UpdateVal(mat32.Abs(NeurVar(ctx, ni, di, Act))) // can be neg
	am.GeInt.Cycle.UpdateVal(NeurVar(ctx, ni, di, GeInt))
	am.GeIntMax.Cycle.UpdateVal(NeurVar(ctx, ni, di, GeIntMax))
	am.GiInt.Cycle.UpdateVal(NeurVar(ctx, ni, di, GiInt))
}

// note: the following is actually being used despite appearing to be
// commented out!  it is auto-uncommented when copied to hlsl
// MUST update whenever above UpdateVals code is updated.

//gosl: end pool

//gosl: hlsl pool
/*
// // AtomicUpdatePoolAvgMax provides an atomic update using atomic ints
// // implemented by InterlockedAdd HLSL intrinsic.
// // This is a #define because it doesn't work on arg values --
// // must be directly operating on a RWStorageBuffer entity.
#define AtomicUpdatePoolAvgMax(am, nrn) \
	AtomicUpdateAvgMaxI32(am.CaSpkP.Cycle, NeurVar(ctx, ni, di, CaSpkP); \
	AtomicUpdateAvgMaxI32(am.CaSpkD.Cycle, NeurVar(ctx, ni, di, CaSpkD); \
	AtomicUpdateAvgMaxI32(am.SpkMax.Cycle, NeurVar(ctx, ni, di, SpkMax); \
	AtomicUpdateAvgMaxI32(am.Act.Cycle, NeurVar(ctx, ni, di, Act); \
	AtomicUpdateAvgMaxI32(am.GeInt.Cycle, NeurVar(ctx, ni, di, GeInt); \
	AtomicUpdateAvgMaxI32(am.GeIntMax.Cycle, NeurVar(ctx, ni, di, GeIntMax); \
	AtomicUpdateAvgMaxI32(am.GiInt.Cycle, NeurVar(ctx, ni, di, GiInt)
*/
//gosl: end pool

//gosl: start pool

// Pool contains computed values for FS-FFFB inhibition,
// and various other state values for layers
// and pools (unit groups) that can be subject to inhibition
type Pool struct {
	StIdx, EdIdx uint32      `inactive:"+" desc:"starting and ending (exlusive) layer-wise indexes for the list of neurons in this pool"`
	LayIdx       uint32      `view:"-" desc:"layer index in global layer list"`
	DataIdx      uint32      `view:"-" desc:"data parallel index (innermost index per layer)"`
	PoolIdx      uint32      `view:"-" desc:"pool index in global pool list: [Layer][Pool][Data]"`
	IsLayPool    slbool.Bool `inactive:"+" desc:"is this a layer-wide pool?  if not, it represents a sub-pool of units within a 4D layer"`
	Gated        slbool.Bool `inactive:"+" desc:"for special types where relevant (e.g., MatrixLayer, BGThalLayer), indicates if the pool was gated"`

	pad uint32

	Inhib  fsfffb.Inhib `inactive:"+" desc:"fast-slow FFFB inhibition values"`
	AvgMax PoolAvgMax   `desc:"average and max values for relevant variables in this pool, at different time scales"`
	AvgDif AvgMaxI32    `inactive:"+" view:"inline" desc:"absolute value of AvgDif differences from actual neuron ActPct relative to TrgAvg"`
}

// Init is callled during InitActs
func (pl *Pool) Init() {
	pl.Inhib.Init()
	pl.AvgMax.Zero()
	pl.AvgMax.SetN(int32(pl.NNeurons()))
	pl.AvgDif.N = int32(pl.NNeurons())
	pl.AvgDif.Init()
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
						tge += wt * pl.OldInhib.GeInt.Avg
						tact += wt * pl.OldInhib.Act.Avg
					} else {
						nrn := &ly.Neurons[ti]
						tge += wt * NeurVar(ctx, ni, di, Ge)
						tact += wt * NeurVar(ctx, ni, di, Act)
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
				// 	SetNeurVar(ctx, ni, di, GiSelf, gi)
			}
		}
	}
}
*/
