// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consat

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers: lower gain, slower, soft clamp",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Acts.Dt.GeTau = 5
				ly.Acts.Dt.GiTau = 7
				ly.Acts.Gbar.I = 100
				ly.Acts.Gbar.L = 20
				ly.Acts.Decay.Act = 0.0   // 0.2 def
				ly.Acts.Decay.Glong = 0.0 // 0.6 def
				ly.Acts.Noise.On.SetBool(false)
				ly.Acts.Noise.GeHz = 100
				ly.Acts.Noise.Ge = 0.002 // 0.001 min
				ly.Acts.Noise.GiHz = 200
				ly.Acts.Noise.Gi = 0.002 // 0.001 min
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "no learning",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false)
				pt.SWts.Init.Mean = 0.5
				pt.SWts.Init.Var = 0.25
				pt.Com.Delay = 2
			}},
	},
}
