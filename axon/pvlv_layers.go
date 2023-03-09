// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"
)

//gosl: start pvlv_layers

// BLAParams has parameters for basolateral amygdala.  Most of BLA
// learning is handled by NeuroMod settings for DA and ACh modulation.
type BLAParams struct {
	NegLRate float32 `desc:"negative DWt learning rate multiplier -- weights go down much more slowly than up -- extinction is separate learning in extinction layer"`

	pad, pad1, pad2 float32
}

func (bp *BLAParams) Defaults() {
	bp.NegLRate = 0.1
}

func (bp *BLAParams) Update() {

}

// PPTgParams has parameters for PPTg = pedunculopontine tegmental nucleus layer.
type PPTgParams struct {
	Thr  float32 `desc:"threshold on PPTg activation prior to multiplying by Gain"`
	Gain float32 `desc:"extra multiplier on raw PPTg Max CaSpkP activity for driving effects of PPTg on ACh and DA neuromodulation"`

	pad, pad1 float32
}

func (pp *PPTgParams) Defaults() {
	pp.Thr = 0.2
	pp.Gain = 2
}

func (pp *PPTgParams) Update() {

}

func (pp *PPTgParams) PPTgVal(val float32) float32 {
	if val < pp.Thr {
		return 0
	}
	return pp.Gain * val
}

//gosl: end pvlv_layers

func (ly *LayerParams) BLADefaults() {
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.Dend.SSGi = 0
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 1.8
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 1.0
	ly.Inhib.ActAvg.Nominal = 0.025

	// ly.Learn.NeuroMod.DAMod needs to be set via BuildConfig
	// because it depends on the configured D1 vs. D2 status
	ly.Learn.NeuroMod.DALRateMod = 1
	ly.Learn.NeuroMod.AChLRateMod = 1
	ly.Learn.NeuroMod.AChDisInhib = 0 // needs to be always active
}

// PVLVPostBuild is used for BLA, VSPatch, and PVLayer types to sett NeuroMod params
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

func (ly *LayerParams) VSPatchDefaults() {
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 0.5
	ly.Inhib.Layer.FB = 0
	ly.Inhib.Pool.FB = 0
	ly.Inhib.Pool.Gi = 0.5
	ly.Inhib.ActAvg.Nominal = 0.25
	ly.Learn.RLRate.Diff.SetBool(false)
	ly.Learn.RLRate.SigmoidMin = 1

	// ms.Learn.NeuroMod.DAMod needs to be set via BuildConfig
	ly.Learn.NeuroMod.DALRateMod = 1
	ly.Learn.NeuroMod.AChLRateMod = 1
	ly.Learn.NeuroMod.AChDisInhib = 0 // 5 for matrix -- not sure about this?
}

func (ly *LayerParams) PVDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.1
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 0.1
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 0.1
	ly.Act.PopCode.On.SetBool(true)
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
}

func (ly *LayerParams) DrivesDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.1
	ly.Inhib.Layer.On.SetBool(false)
	ly.Inhib.Layer.Gi = 0.1
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 0.9
	ly.Act.PopCode.On.SetBool(true)
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
}

func (ly *LayerParams) PPTgDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.1
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 1 // todo: explore
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 0.5   // todo: could be lower!
	ly.Inhib.Pool.FFPrv = 10 // key for temporal derivative
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
}

func (ly *LayerParams) USDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.1
	ly.Inhib.Layer.On.SetBool(false)
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 0.5
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
}
