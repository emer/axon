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
				// ly.Inhib.Layer.On.SetBool(true)
				// ly.Inhib.Layer.Gi = 1.0
				// ly.Inhib.ActAvg.Nominal = 0.1
				// ly.Acts.Noise.On.SetBool(false)
				// ly.Acts.Noise.GeHz = 100
				// ly.Acts.Noise.Ge = 0.01 // 0.001 min
				// ly.Acts.Noise.GiHz = 200
				// ly.Acts.Noise.Gi = 0.01 // 0.001 min
			}},
		{Sel: "#Input", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.0
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "no learning",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.01
				pt.SWts.Adapt.LRate = 0.1
				pt.SWts.Init.SPct = 0.5
				pt.Learn.DWt.SubMean = 0
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.3 // 0.3 > 0.2 > 0.1 > 0.5
			}},
	},
}
