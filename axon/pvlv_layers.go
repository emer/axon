// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"
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

//gosl: end pvlv_layers

func (ly *LayerParams) BLADefaults() {
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.Dend.SSGi = 0
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 1.8
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 0.9
	ly.Inhib.ActAvg.Nominal = 0.025
	ly.Learn.RLRate.SigmoidMin = 1.0
	ly.Learn.TrgAvgAct.On.SetBool(false)

	// ly.Learn.NeuroMod.DAMod needs to be set via BuildConfig
	// because it depends on the configured D1 vs. D2 status
	if ly.Learn.NeuroMod.DAMod == D2Mod {
		ly.Learn.NeuroMod.DALRateSign.SetBool(true) // set this for Extinction type
		ly.Learn.NeuroMod.BurstGain = 1
		ly.Learn.NeuroMod.DipGain = 1
	} else {
		ly.Learn.NeuroMod.DALRateMod = 0.5
		ly.Learn.NeuroMod.BurstGain = 0.2
		ly.Learn.NeuroMod.DipGain = 0
	}
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

func (ly *LayerParams) PVDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.1
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 1
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 1
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
	ly.Learn.TrgAvgAct.On.SetBool(false)
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
	ly.Learn.TrgAvgAct.On.SetBool(false)
	ly.PVLV.Thr = 0.2
	ly.PVLV.Gain = 2
}

func (ly *LayerParams) USDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.1
	ly.Inhib.Layer.On.SetBool(false)
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 0.5
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Learn.TrgAvgAct.On.SetBool(false)
}
