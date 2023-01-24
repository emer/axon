// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "log"

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
	ly.Learn.NeuroMod.DALRateMod.SetBool(true)
	ly.Learn.NeuroMod.AChLRateMod.SetBool(true)
	ly.Learn.NeuroMod.DALRatePct = 1
	ly.Learn.NeuroMod.AChLRatePct = 1
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
