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

func (ly *Layer) BLAPostBuild() {
	dm, err := ly.BuildConfigByName("DAMod")
	if err == nil {
		err = ly.Params.Learn.NeuroMod.DAMod.FromString(dm)
		if err != nil {
			log.Println(err)
		}
	}
}

func (ly *Layer) VSPatchDefaults() {
	// ly.Params.Act.Decay.Act = 0
	// ly.Params.Act.Decay.Glong = 0
	// ly.Params.Inhib.Pool.On.SetBool(false)
	// ly.Params.Inhib.Layer.On.SetBool(true)
	// ly.Params.Inhib.Layer.Gi = 0.5
	// ly.Params.Inhib.Layer.FB = 0
	// ly.Params.Inhib.Pool.FB = 0
	// ly.Params.Inhib.Pool.Gi = 0.5
	ly.Params.Inhib.ActAvg.Nominal = 0.25
	ly.Params.Learn.RLRate.Diff.SetBool(false)

	// ly.Params.Learn.NeuroMod.DAMod needs to be set via BuildConfig
	ly.Params.Learn.NeuroMod.DALRateMod = 1
	ly.Params.Learn.NeuroMod.AChLRateMod = 1
	ly.Params.Learn.NeuroMod.AChDisInhib = 0 // 5 for matrix -- not sure about this?

	// important: user needs to adjust wt scale of some PFC inputs vs others:
	// drivers vs. modulators

	// for _, pji := range ly.RcvPrjns {
	// 	pj := pji.(AxonPrjn).AsAxon()
	// 	pj.Params.SWt.Init.SPct = 0
	// 	if pj.Send.(AxonLayer).LayerType() == GPLayer { // From GPe TA or In
	// 		pj.Params.PrjnScale.Abs = 1
	// 		pj.Params.Learn.Learn.SetBool(false)
	// 		pj.Params.SWt.Adapt.SigGain = 1
	// 		pj.Params.SWt.Init.Mean = 0.75
	// 		pj.Params.SWt.Init.Var = 0.0
	// 		pj.Params.SWt.Init.Sym.SetBool(false)
	// 		if strings.HasSuffix(pj.Send.Name(), "GPeIn") { // GPeInToMtx
	// 			pj.Params.PrjnScale.Abs = 0.5 // counterbalance for GPeTA to reduce oscillations
	// 		} else if strings.HasSuffix(pj.Send.Name(), "GPeTA") { // GPeTAToMtx
	// 			if strings.HasSuffix(ly.Nm, "MtxGo") {
	// 				pj.Params.PrjnScale.Abs = 2 // was .8
	// 			} else {
	// 				pj.Params.PrjnScale.Abs = 1 // was .3 GPeTAToMtxNo must be weaker to prevent oscillations, even with GPeIn offset
	// 			}
	// 		}
	// 	}
	// }
}
