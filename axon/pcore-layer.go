// Code generated by "goal build"; DO NOT EDIT.
//line pcore-layer.goal:1
// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"
	"strings"

	"cogentcore.org/core/base/num"
	"cogentcore.org/core/goal/gosl/slbool"
)

//gosl:start

// MatrixParams has parameters for BG Striatum Matrix MSN layers
// These are the main Go / NoGo gating units in BG.
// DA, ACh learning rate modulation is pre-computed on the recv neuron
// RLRate variable via NeuroMod.  Also uses Pool.Gated for InvertNoGate,
// updated in PlusPhase prior to DWt call.
// Must set Learn.NeuroMod.DAMod = D1Mod or D2Mod via SetBuildConfig("DAMod").
type MatrixParams struct {

	// threshold on layer Avg SpkMax for Matrix Go and VThal layers to count as having gated
	GateThr float32 `default:"0.05"`

	// is this a ventral striatum (VS) matrix layer?  if true, the gating status of this layer is recorded in the Global state, and used for updating effort and other factors.
	IsVS slbool.Bool

	// index of other matrix (Go if we are NoGo and vice-versa).    Set during Build from BuildConfig OtherMatrixName
	OtherMatrixIndex int32 `edit:"-"`

	// index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay1Name if present -- -1 if not used
	ThalLay1Index int32 `edit:"-"`

	// index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay2Name if present -- -1 if not used
	ThalLay2Index int32 `edit:"-"`

	// index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay3Name if present -- -1 if not used
	ThalLay3Index int32 `edit:"-"`

	// index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay4Name if present -- -1 if not used
	ThalLay4Index int32 `edit:"-"`

	// index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay5Name if present -- -1 if not used
	ThalLay5Index int32 `edit:"-"`

	// index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay6Name if present -- -1 if not used
	ThalLay6Index int32 `edit:"-"`

	pad, pad1, pad2 int32
}

func (mp *MatrixParams) Defaults() {
	mp.GateThr = 0.05
}

func (mp *MatrixParams) Update() {
}

// GPLayerTypes is a GPLayer axon-specific layer type enum.
type GPLayerTypes int32 //enums:enum

// The GPLayer types
const (
	// GPePr is the set of prototypical GPe neurons, mediating classical NoGo
	GPePr GPLayerTypes = iota

	// GPeAk is arkypallidal layer of GPe neurons, receiving inhibition from GPePr
	// and projecting inhibition to Mtx
	GPeAk

	// GPi is the inner globus pallidus, functionally equivalent to SNr,
	// receiving from MtxGo and GPePr, and sending inhibition to VThal
	GPi
)

// GPLayer represents a globus pallidus layer, including:
// GPePr, GPeAk (arkypallidal), and GPi (see GPType for type).
// Typically just a single unit per Pool representing a given stripe.
type GPParams struct {

	// type of GP Layer -- must set during config using SetBuildConfig of GPType.
	GPType GPLayerTypes

	pad, pad1, pad2 uint32
}

func (gp *GPParams) Defaults() {
}

func (gp *GPParams) Update() {
}

//gosl:end

// MatrixGated is called after std PlusPhase, on CPU, has Pool info
// downloaded from GPU, to set Gated flag based on SpkMax activity
func (ly *Layer) MatrixGated(ctx *Context) {
	if ly.Params.Learn.NeuroMod.DAMod != D1Mod {
		lpi := ly.Params.PoolIndex(0)
		oly := ly.Network.Layers[int(ly.Params.Matrix.OtherMatrixIndex)]
		opi := oly.Params.PoolIndex(0)
		// note: NoGo layers don't track gating at the sub-pool level!
		for di := uint32(0); di < ctx.NData; di++ {
			PoolsInt.Set(PoolsInt.Value(int(opi), int(PoolGated), int(di)), int(lpi), int(PoolGated), int(di))
		}
		return
	}
	// todo: Context requires data parallel state!

	for di := uint32(0); di < ctx.NData; di++ {
		mtxGated, poolIndex := ly.GatedFromSpkMax(di, ly.Params.Matrix.GateThr)

		thalGated := false
		if ly.Params.Matrix.ThalLay1Index >= 0 {
			tly := ly.Network.Layers[int(ly.Params.Matrix.ThalLay1Index)]
			gt, _ := tly.GatedFromSpkMax(di, ly.Params.Matrix.GateThr)
			thalGated = thalGated || gt
		}
		if ly.Params.Matrix.ThalLay2Index >= 0 {
			tly := ly.Network.Layers[int(ly.Params.Matrix.ThalLay2Index)]
			gt, _ := tly.GatedFromSpkMax(di, ly.Params.Matrix.GateThr)
			thalGated = thalGated || gt
		}
		if ly.Params.Matrix.ThalLay3Index >= 0 {
			tly := ly.Network.Layers[int(ly.Params.Matrix.ThalLay3Index)]
			gt, _ := tly.GatedFromSpkMax(di, ly.Params.Matrix.GateThr)
			thalGated = thalGated || gt
		}
		if ly.Params.Matrix.ThalLay4Index >= 0 {
			tly := ly.Network.Layers[int(ly.Params.Matrix.ThalLay4Index)]
			gt, _ := tly.GatedFromSpkMax(di, ly.Params.Matrix.GateThr)
			thalGated = thalGated || gt
		}
		if ly.Params.Matrix.ThalLay5Index >= 0 {
			tly := ly.Network.Layers[int(ly.Params.Matrix.ThalLay5Index)]
			gt, _ := tly.GatedFromSpkMax(di, ly.Params.Matrix.GateThr)
			thalGated = thalGated || gt
		}
		if ly.Params.Matrix.ThalLay6Index >= 0 {
			tly := ly.Network.Layers[int(ly.Params.Matrix.ThalLay6Index)]
			gt, _ := tly.GatedFromSpkMax(di, ly.Params.Matrix.GateThr)
			thalGated = thalGated || gt
		}

		mtxGated = mtxGated && thalGated

		// note: in principle with multi-pool GP, could try to establish
		// a correspondence between thal and matrix pools, such that
		// a failure to gate at the thal level for a given pool would veto
		// just the one corresponding pool.  However, we're not really sure
		// that this will make sense and not doing yet..

		if !mtxGated { // nobody did if thal didn't
			for spi := uint32(0); spi < ly.NPools; spi++ {
				pi := ly.Params.PoolIndex(spi)
				PoolsInt.Set(0, int(pi), int(PoolGated), int(di))
			}
		}
		if ctx.PlusPhase.IsTrue() && ly.Params.Matrix.IsVS.IsTrue() {
			GlobalScalars.Set(num.FromBool[float32](mtxGated), int(GvVSMatrixJustGated), int(di))
			if mtxGated {
				GlobalVectors.Set(1, int(GvVSMatrixPoolGated), int(poolIndex), int(di))
			}
		}
	}
}

// GatedFromSpkMax updates the Gated state in Pools of given layer,
// based on Avg SpkMax being above given threshold.
// returns true if any gated, and the pool index if 4D layer (0 = first).
func (ly *Layer) GatedFromSpkMax(di uint32, thr float32) (bool, int) {
	anyGated := false
	poolIndex := -1
	lpi := ly.Params.PoolIndex(0)
	if ly.Is4D() {
		for spi := uint32(1); spi < ly.NPools; spi++ {
			pi := ly.Params.PoolIndex(spi)
			spkavg := PoolAvgMax(AMSpkMax, AMCycle, Avg, pi, di)
			gthr := spkavg > thr
			if gthr {
				anyGated = true
				if poolIndex < 0 {
					poolIndex = int(spi) - 1
				}
				PoolsInt.Set(1, int(pi), int(PoolGated), int(di))
			} else {
				PoolsInt.Set(0, int(pi), int(PoolGated), int(di))
			}
		}
	} else {
		spkavg := PoolAvgMax(AMSpkMax, AMCycle, Avg, lpi, di)
		if spkavg > thr {
			anyGated = true
		}
	}
	if anyGated {
		PoolsInt.Set(1, int(lpi), int(PoolGated), int(di))
	} else {
		PoolsInt.Set(0, int(lpi), int(PoolGated), int(di))
	}
	return anyGated, poolIndex
}

// AnyGated returns true if the layer-level pool Gated flag is true,
// which indicates if any of the layers gated.
func (ly *Layer) AnyGated(di uint32) bool {
	lpi := ly.Params.PoolIndex(0)
	return PoolsInt.Value(int(lpi), int(PoolGated), int(di)) > 0
}

func (ly *Layer) MatrixDefaults() {
	ly.Params.Acts.Decay.Act = 1
	ly.Params.Acts.Decay.Glong = 1 // prevent carryover of NMDA
	ly.Params.Acts.Kir.Gbar = 10
	ly.Params.Acts.GabaB.Gbar = 0 // Kir replaces GabaB
	// ly.Params.Acts.NMDA.Gbar = 0    // Matrix needs nmda, default is fine
	ly.Params.Inhib.Layer.FB = 0 // pure FF
	ly.Params.Inhib.Layer.Gi = 0.5
	ly.Params.Inhib.Pool.On.SetBool(true) // needs both pool and layer if has pools
	ly.Params.Inhib.Pool.FB = 0           // pure FF
	ly.Params.Inhib.Pool.Gi = 0.5
	ly.Params.Inhib.ActAvg.Nominal = 0.25   // pooled should be lower
	ly.Params.Learn.RLRate.On.SetBool(true) // key: sig deriv used outside of rew trials
	ly.Params.Learn.RLRate.Diff.SetBool(false)
	ly.Params.Learn.TrgAvgAct.RescaleOn.SetBool(false) // major effect

	// ly.Params.Learn.NeuroMod.DAMod needs to be set via BuildConfig
	ly.Params.Learn.NeuroMod.DALRateSign.SetBool(true) // critical
	ly.Params.Learn.NeuroMod.DALRateMod = 1
	ly.Params.Learn.NeuroMod.AChLRateMod = 0
	ly.Params.Learn.NeuroMod.DAModGain = 0
	ly.Params.Learn.NeuroMod.BurstGain = 0.1
	ly.Params.Learn.RLRate.SigmoidMin = 0.001

	if ly.Class == "VSMatrixLayer" {
		ly.Params.Inhib.Layer.On.SetBool(true)
		ly.Params.Matrix.IsVS.SetBool(true)
		ly.Params.Acts.Dend.ModBase = 0
		ly.Params.Acts.Dend.ModGain = 2 // for VS case -- otherwise irrelevant
		ly.Params.Learn.NeuroMod.AChDisInhib = 5
		ly.Params.Learn.NeuroMod.BurstGain = 1
	} else {
		ly.Params.Inhib.Layer.On.SetBool(false)
		ly.Params.Matrix.IsVS.SetBool(false)
		ly.Params.Acts.Dend.ModBase = 1
		ly.Params.Acts.Dend.ModGain = 0
		ly.Params.Learn.NeuroMod.AChDisInhib = 0
	}

	// important: user needs to adjust wt scale of some PFC inputs vs others:
	// drivers vs. modulators

	for _, pj := range ly.RecvPaths {
		pj.Params.SWts.Init.SPct = 0
		if pj.Send.Type == GPLayer { // GPeAkToMtx
			pj.Params.SetFixedWts()
			pj.Params.PathScale.Abs = 3
			pj.Params.SWts.Init.Mean = 0.75
			pj.Params.SWts.Init.Var = 0.0
			if ly.Class == "DSMatrixLayer" {
				if strings.Contains(ly.Name, "No") {
					pj.Params.PathScale.Abs = 6
				}
			}
		}
	}
}

func (ly *Layer) MatrixPostBuild() {
	ly.Params.Matrix.ThalLay1Index = ly.BuildConfigFindLayer("ThalLay1Name", false) // optional
	ly.Params.Matrix.ThalLay2Index = ly.BuildConfigFindLayer("ThalLay2Name", false) // optional
	ly.Params.Matrix.ThalLay3Index = ly.BuildConfigFindLayer("ThalLay3Name", false) // optional
	ly.Params.Matrix.ThalLay4Index = ly.BuildConfigFindLayer("ThalLay4Name", false) // optional
	ly.Params.Matrix.ThalLay5Index = ly.BuildConfigFindLayer("ThalLay5Name", false) // optional
	ly.Params.Matrix.ThalLay6Index = ly.BuildConfigFindLayer("ThalLay6Name", false) // optional

	ly.Params.Matrix.OtherMatrixIndex = ly.BuildConfigFindLayer("OtherMatrixName", true)

	dm, err := ly.BuildConfigByName("DAMod")
	if err == nil {
		err = ly.Params.Learn.NeuroMod.DAMod.SetString(dm)
		if err != nil {
			log.Println(err)
		}
	}
}

////////////////////////////////////////////////////////////////////
//  GP

func (ly *Layer) GPDefaults() {
	// GP is tonically self-active and has no FFFB inhibition
	// Defaults are for GPePr, Ak has special values below
	ly.Params.Acts.Init.GeBase = 0.4
	ly.Params.Acts.Init.GeVar = 0.2
	ly.Params.Acts.Init.GiVar = 0.1
	ly.Params.Acts.Decay.Act = 0
	ly.Params.Acts.Decay.Glong = 1
	ly.Params.Acts.NMDA.Gbar = 0 // carryover of NMDA was causing issues!
	ly.Params.Acts.GabaB.Gbar = 0
	ly.Params.Inhib.ActAvg.Nominal = 1 // very active!
	ly.Params.Inhib.Layer.On.SetBool(false)
	ly.Params.Inhib.Pool.On.SetBool(false)

	if ly.Params.GP.GPType == GPeAk {
		ly.Params.Acts.Init.GeBase = 0.2 // definitely lower in bio data, necessary
		ly.Params.Acts.Init.GeVar = 0.1
	}

	for _, pj := range ly.RecvPaths {
		pj.Params.SetFixedWts()
		pj.Params.SWts.Init.Mean = 0.75 // 0.75 -- very similar -- maybe a bit more reliable with 0.8 / 0
		pj.Params.SWts.Init.Var = 0.25  // 0.25
		switch ly.Params.GP.GPType {
		case GPePr:
			switch pj.Send.Type {
			case MatrixLayer:
				pj.Params.PathScale.Abs = 1 // MtxNoToGPePr -- primary NoGo pathway
			case GPLayer:
				pj.Params.PathScale.Abs = 4 // 4 best for DS; GPePrToGPePr -- must be very strong
			case STNLayer:
				pj.Params.PathScale.Abs = 0.5 // STNToGPePr
			}
		case GPeAk:
			switch pj.Send.Type {
			case MatrixLayer:
				pj.Params.PathScale.Abs = 0.5 // MtxGoToGPeAk
			case GPLayer:
				pj.Params.PathScale.Abs = 1 // GPePrToGPeAk
			case STNLayer:
				pj.Params.PathScale.Abs = 0.1 // STNToGPAk
			}
		}
	}

	if ly.Params.GP.GPType == GPi {
		ly.GPiDefaults()
	}
}

func (ly *Layer) GPiDefaults() {
	ly.Params.Acts.Init.GeBase = 0.3
	ly.Params.Acts.Init.GeVar = 0.1
	ly.Params.Acts.Init.GiVar = 0.1
	// note: GPLayer took care of STN input paths

	for _, pj := range ly.RecvPaths {
		pj.Params.SetFixedWts()
		pj.Params.SWts.Init.Mean = 0.75  // 0.75  see above
		pj.Params.SWts.Init.Var = 0.25   // 0.25
		if pj.Send.Type == MatrixLayer { // MtxGoToGPi
			if pj.Send.Class == "VSMatrixLayer" {
				pj.Params.PathScale.Abs = 0.2
			} else {
				pj.Params.PathScale.Abs = 1
			}
		} else if pj.Send.Type == GPLayer { // GPePrToGPi
			pj.Params.PathScale.Abs = 1
		} else if pj.Send.Type == STNLayer { // STNToGPi
			pj.Params.PathScale.Abs = 0.2
		}
	}
}

func (ly *Layer) GPPostBuild() {
	gpnm, err := ly.BuildConfigByName("GPType")
	if err == nil {
		err = ly.Params.GP.GPType.SetString(gpnm)
		if err != nil {
			log.Println(err)
		}
	}
}

////////////////////////////////////////////////////////////////////
//  STN

func (ly *Layer) STNDefaults() {
	// STN is tonically self-active and has no FFFB inhibition
	ly.Params.Acts.Init.GeBase = 0.1 // was 0.3
	ly.Params.Acts.Init.GeVar = 0.1
	ly.Params.Acts.Init.GiVar = 0.1
	ly.Params.Acts.SKCa.Gbar = 2
	ly.Params.Acts.SKCa.CaRDecayTau = 80 // 80 > 150 for longer theta windows
	ly.Params.Acts.Kir.Gbar = 10         // 10 > 5 -- key for pause
	ly.Params.Acts.Decay.Act = 0
	ly.Params.Acts.Decay.Glong = 0
	ly.Params.Acts.Decay.LearnCa = 1 // key for non-spaced trials, to refresh immediately
	ly.Params.Acts.Dend.SSGi = 0
	ly.Params.Acts.NMDA.Gbar = 0 // fine with 0
	ly.Params.Acts.GabaB.Gbar = 0
	ly.Params.Inhib.Layer.On.SetBool(true)
	ly.Params.Inhib.Layer.Gi = 0.5
	ly.Params.Inhib.Layer.FB = 0
	ly.Params.Inhib.Pool.On.SetBool(false)
	ly.Params.Inhib.Pool.Gi = 0.5
	ly.Params.Inhib.Pool.FB = 0
	ly.Params.Inhib.ActAvg.Nominal = 0.15
	ly.Params.Learn.NeuroMod.AChDisInhib = 0 // was 2,

	// if ly.Cls == "VSTNLayer" {
	// 	ly.Params.Inhib.Layer.On.SetBool(false)
	// } else {
	// 	ly.Params.Inhib.Layer.On.SetBool(true)
	// }

	for _, pj := range ly.RecvPaths {
		pj.Params.SetFixedWts()
		pj.Params.SWts.Init.Mean = 0.75
		pj.Params.SWts.Init.Var = 0.25
		if pj.Send.Type == GPLayer { // GPePrToSTN
			pj.Params.PathScale.Abs = 0.5
		} else {
			pj.Params.PathScale.Abs = 2.0 // pfc inputs
		}
	}
}

////////////////////////////////////////////////////////////////////
//  BGThal

func (ly *Layer) BGThalDefaults() {
	// note: not tonically active
	// ly.Params.Acts.NMDA.Gbar = 0 // needs NMDA
	ly.Params.Acts.Decay.Act = 1
	ly.Params.Acts.Decay.Glong = 0.6
	ly.Params.Acts.Dend.SSGi = 0
	ly.Params.Inhib.ActAvg.Nominal = 0.1
	ly.Params.Inhib.Layer.On.SetBool(true)
	ly.Params.Inhib.Layer.Gi = 0.6
	ly.Params.Inhib.Pool.On.SetBool(false)
	ly.Params.Inhib.Pool.Gi = 0.6

	ly.Params.Learn.NeuroMod.AChDisInhib = 1

	for _, pj := range ly.RecvPaths {
		pj.Params.SetFixedWts()
		pj.Params.SWts.Init.Mean = 0.75
		pj.Params.SWts.Init.Var = 0.0
		if strings.HasSuffix(pj.Send.Name, "GPi") { // GPiToBGThal
			pj.Params.PathScale.Abs = 5 // can now be much stronger with PTMaint mod and maint dynamics
			pj.AddClass("GPiToBGThal")
		}
	}
}

////////////////////////////////////////////////////////////////////
//  VSGated

func (ly *LayerParams) VSGatedDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.5
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 1
	ly.Inhib.Pool.On.SetBool(false)
	ly.Inhib.Pool.Gi = 1
	ly.Acts.Decay.Act = 1
	ly.Acts.Decay.Glong = 1
}