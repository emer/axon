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
				ly.Inhib.ActAvg.AdaptRate = 0.1 // 0.1 default > 0.05?
				ly.Inhib.ActAvg.AdaptMax = 0.05 // 0.05 > 0.01
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
				ly.Acts.Clamp.Ge = 1.5 // 1.5 for fsffffb
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.Layer.Gi = 1.0
			}},
		{Sel: "#Hidden1", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.ActAvg.Nominal = 0.03
				ly.Inhib.ActAvg.Offset = 0.02
				ly.Inhib.Layer.Gi = 1.0
			}},
		{Sel: "#Hidden2", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.ActAvg.Nominal = 0.03
				ly.Inhib.ActAvg.Offset = 0.02
				ly.Inhib.Layer.Gi = 1.0
			}},
		{Sel: "#Output", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 1.1
				ly.Inhib.Pool.FB = 4
				ly.Acts.Clamp.Ge = 1.4
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "std",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.1
				pt.SWts.Adapt.LRate = 0.1
				pt.SWts.Init.SPct = 1
				pt.SWts.Adapt.LRate = 0.0001 // 0.005 == .1 == .01
				pt.Learn.DWt.SubMean = 1
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2 // 0.3 > 0.2 > 0.1 > 0.5
			}},
	},
}
