// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start pvlv_layers

//gosl: end pvlv_layers

func (ly *LayerParams) BLALayerDefaults() {
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.Dend.SSGi = 0
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 1.8
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 1.0
	ly.Inhib.ActAvg.Nominal = 0.025
}
