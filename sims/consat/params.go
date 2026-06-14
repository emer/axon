// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package consat

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Acts.Noise.On.SetBool(true)
				ly.Acts.Noise.GeHz = 100
				ly.Acts.Noise.Ge = 0.01 // 0.001 min
				ly.Acts.Noise.GiHz = 200
				ly.Acts.Noise.Gi = 0.01 // 0.001 min
				ly.Acts.Decay.Act = 1
				ly.Acts.Decay.Glong = 1
				ly.Acts.Decay.LearnCa = 1
				ly.Acts.Decay.GBuffs.SetBool(true)
			}},
		{Sel: "#Input", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.0
				ly.Acts.Init.GeBase = 0.3
				ly.Acts.Init.GeVar = 0.05
				ly.Acts.Noise.Ge = 0.02 // 0.001 min
				ly.Acts.Noise.Gi = 0.02 // 0.001 min
			}},
		{Sel: "#Cities", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.9 // 0.9 min
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 0.5
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Acts.Init.GeBase = 0
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
				pt.SWts.Init.Mean = 0.8
				pt.SWts.Init.Var = 0.25
				pt.Com.Delay = 2
			}},
		{Sel: ".InhibPath", Doc: "inhib gain",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5
			}},
		{Sel: ".LateralPath", Doc: "lateral gain",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4
			}},
	},
}
