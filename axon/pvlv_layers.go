// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"
	"strings"
)

//gosl: start pvlv_layers

// PVLVParams has parameters for readout of values as inputs to PVLV equations.
type PVLVParams struct {
	Thr  float32 `desc:"threshold on value prior to multiplying by Gain"`
	Gain float32 `desc:"multiplier applied after Thr threshold"`

	pad, pad1 float32
}

func (pp *PVLVParams) Defaults() {
	pp.Thr = 0.2
	pp.Gain = 4
}

func (pp *PVLVParams) Update() {

}

func (pp *PVLVParams) Val(val float32) float32 {
	vs := val - pp.Thr
	if vs < 0 {
		return 0
	}
	return pp.Gain * vs
}

// VSPatchParams parameters for VSPatch learning
type VSPatchParams struct {
	NoDALRate float32 `def:"0.1" desc:"learning rate when no positive dopamine is present (i.e., when not learning to predict a positive valence PV / US outcome.  if too high, extinguishes too quickly.  if too low, doesn't discriminate US vs. non-US trials as well."`
	NoDAThr   float32 `def:"0.01" desc:"threshold on DA level to engage the NoDALRate -- use a small positive number just in case"`

	pad, pad1 float32
}

func (pp *VSPatchParams) Defaults() {
	pp.NoDALRate = 0.1
	pp.NoDAThr = 0.01
}

func (pp *VSPatchParams) Update() {

}

// DALRate returns the learning rate modulation factor modlr based on dopamine level
func (pp *VSPatchParams) DALRate(da, modlr float32) float32 {
	if da <= pp.NoDAThr {
		if modlr < -pp.NoDALRate { // big dip: use it
			return modlr
		}
		return -pp.NoDALRate
	}
	return modlr
}

//gosl: end pvlv_layers

func (ly *Layer) BLADefaults() {
	lp := ly.Params
	lp.Act.Decay.Act = 0
	lp.Act.Decay.Glong = 0
	lp.Act.Dend.SSGi = 0
	lp.Inhib.Layer.On.SetBool(true)
	lp.Inhib.Layer.Gi = 1.8
	lp.Inhib.Pool.On.SetBool(true)
	lp.Inhib.Pool.Gi = 0.9
	lp.Inhib.ActAvg.Nominal = 0.025
	lp.Learn.RLRate.SigmoidMin = 1.0
	lp.Learn.TrgAvgAct.On.SetBool(false)

	// lp.Learn.NeuroMod.DAMod needs to be set via BuildConfig
	// because it depends on the configured D1 vs. D2 status
	isAcq := strings.Contains(ly.Nm, "Acq")

	if isAcq {
		lp.Learn.NeuroMod.DALRateMod = 0.5
		lp.Learn.NeuroMod.BurstGain = 0.2
		lp.Learn.NeuroMod.DipGain = 0
		lp.Learn.RLRate.Diff.SetBool(true)
		lp.Learn.RLRate.DiffThr = 0.01
	} else {
		lp.Learn.NeuroMod.DALRateSign.SetBool(true) // yes for Extinction
		lp.Learn.NeuroMod.BurstGain = 1
		lp.Learn.NeuroMod.DipGain = 1
		lp.Learn.RLRate.Diff.SetBool(false)
	}
	lp.Learn.NeuroMod.AChLRateMod = 1
	lp.Learn.NeuroMod.AChDisInhib = 0 // needs to be always active

	for _, pj := range ly.RcvPrjns {
		slay := pj.Send
		if slay.LayerType() == BLALayer { // inhibition from Ext
			pj.Params.SetFixedWts()
			pj.Params.PrjnScale.Abs = 2
		}
	}
}

// PVLVPostBuild is used for BLA, VSPatch, and PVLayer types to set NeuroMod params
func (ly *Layer) PVLVPostBuild() {
	dm, err := ly.BuildConfigByName("DAMod")
	if err == nil {
		err = ly.Params.Learn.NeuroMod.DAMod.FromString(dm)
		if err != nil {
			log.Println(err)
		}
	}
	vl, err := ly.BuildConfigByName("Valence")
	if err == nil {
		err = ly.Params.Learn.NeuroMod.Valence.FromString(vl)
		if err != nil {
			log.Println(err)
		}
	}
}

func (ly *Layer) CeMDefaults() {
	lp := ly.Params
	lp.Act.Decay.Act = 1
	lp.Act.Decay.Glong = 1
	lp.Act.Dend.SSGi = 0
	lp.Inhib.Layer.On.SetBool(true)
	lp.Inhib.Layer.Gi = 0.5
	lp.Inhib.Pool.On.SetBool(true)
	lp.Inhib.Pool.Gi = 0.3
	lp.Inhib.ActAvg.Nominal = 0.15
	lp.Learn.RLRate.SigmoidMin = 1.0
	lp.Learn.TrgAvgAct.On.SetBool(false)

	for _, pj := range ly.RcvPrjns {
		pj.Params.SetFixedWts()
		pj.Params.PrjnScale.Abs = 1
		// slay := pj.Send.(AxonLayer).AsAxon()
		// if ly.Params.NeuroMod.Valence == Positive {
		// 	if slay.Params.NeuroMod.DAMod == D2Mod {
		// 	}
		// } else {
		//
		// }
	}
}

func (ly *Layer) PPTgDefaults() {
	lp := ly.Params
	lp.Inhib.ActAvg.Nominal = 0.1
	lp.Inhib.Layer.On.SetBool(true)
	lp.Inhib.Layer.Gi = 1 // todo: explore
	lp.Inhib.Pool.On.SetBool(true)
	lp.Inhib.Pool.Gi = 0.5   // todo: could be lower!
	lp.Inhib.Pool.FFPrv = 10 // key for temporal derivative
	lp.Act.Decay.Act = 1
	lp.Act.Decay.Glong = 1
	lp.Learn.TrgAvgAct.On.SetBool(false)
	lp.PVLV.Thr = 0.2
	lp.PVLV.Gain = 2

	for _, pj := range ly.RcvPrjns {
		pj.Params.SetFixedWts()
		pj.Params.PrjnScale.Abs = 1
	}
}

func (ly *LayerParams) VSPatchDefaults() {
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Layer.On.SetBool(false)
	ly.Inhib.Layer.Gi = 0.5
	ly.Inhib.Layer.FB = 0
	ly.Inhib.Pool.FB = 0
	ly.Inhib.Pool.Gi = 0.5
	ly.Inhib.ActAvg.Nominal = 0.2
	ly.Learn.RLRate.Diff.SetBool(false)
	ly.Learn.RLRate.SigmoidMin = 1
	ly.Learn.TrgAvgAct.On.SetBool(false)

	// ms.Learn.NeuroMod.DAMod needs to be set via BuildConfig
	ly.Learn.NeuroMod.DALRateSign.SetBool(true)
	ly.Learn.NeuroMod.AChLRateMod = 0.8 // ACh now active for extinction, so this is ok
	ly.Learn.NeuroMod.AChDisInhib = 0   // 5 for matrix -- not sure about this?
	ly.Learn.NeuroMod.BurstGain = 1
	ly.Learn.NeuroMod.DipGain = 1 // extinction -- works fine at 1
	ly.PVLV.Thr = 0.3
	ly.PVLV.Gain = 6
}

func (ly *LayerParams) DrivesDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.03
	ly.Inhib.Layer.On.SetBool(false)
	ly.Inhib.Layer.Gi = 0.1
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 0.5
	ly.Act.PopCode.On.SetBool(true)
	ly.Act.PopCode.MinAct = 0.2 // low activity for low drive -- also has special 0 case = nothing
	ly.Act.PopCode.MinSigma = 0.08
	ly.Act.PopCode.MaxSigma = 0.12
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Learn.TrgAvgAct.On.SetBool(false)
}

func (ly *LayerParams) EffortDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.2
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 0.5
	ly.Inhib.Pool.On.SetBool(false)
	ly.Act.PopCode.On.SetBool(true) // use only popcode
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Learn.TrgAvgAct.On.SetBool(false)
}

func (ly *LayerParams) USDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.2
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 0.5
	ly.Inhib.Pool.On.SetBool(false)
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Learn.TrgAvgAct.On.SetBool(false)
}

func (ly *LayerParams) PVDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.2
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 0.5
	ly.Inhib.Pool.On.SetBool(false)
	ly.Act.PopCode.On.SetBool(true)
	// note: may want to modulate rate code as well:
	// ly.Act.PopCode.MinAct = 0.2
	// ly.Act.PopCode.MinSigma = 0.08
	// ly.Act.PopCode.MaxSigma = 0.12
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Learn.TrgAvgAct.On.SetBool(false)
}
