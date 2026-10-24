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
		{Sel: ".Input", Doc: "note: class for Input and Recon",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 1.5          // 1.5 for fsffffb
				ly.Inhib.ActAvg.Nominal = 0.005 // 0.01 actual
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.9
				ly.Inhib.Pool.Gi = 0.95
				// ly.Inhib.Layer.Gi = 1
			}},
		{Sel: "#Full", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02 // .025 actual
				// ly.Acts.Clamp.Add.SetBool(true) // allows off act
				// ly.Acts.Clamp.Ge = 0.5
			}},
		{Sel: "#Encode", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.01 // 0.02 actual
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.Pool.Gi = 0.95
				// ly.Inhib.Layer.Gi = 1
			}},
		{Sel: "#Hidden1", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03 // 0.5 actual
				ly.Inhib.ActAvg.Offset = 0.02
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.Pool.Gi = 1.05
			}},
		{Sel: "#Hidden2", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.08 //
				ly.Inhib.ActAvg.Offset = 0.0
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				// ly.Inhib.Pool.On.SetBool(true)
				// ly.Inhib.Pool.Gi = 1.0
				ly.Inhib.Layer.Gi = 1.05
			}},
		// {Sel: "#Hidden2", Doc: "",
		// 	Set: func(ly *axon.LayerParams) {
		// 		ly.Inhib.ActAvg.Nominal = 0.03
		// 		ly.Inhib.ActAvg.Offset = 0.02
		// 		ly.Inhib.ActAvg.AdaptGi.SetBool(true)
		// 		ly.Inhib.Layer.Gi = 1.0
		// 	}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "std",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.05 // 0.05 >= 0.1 > 0.02
				pt.SWts.Adapt.LRate = 0.1
				pt.SWts.Init.SPct = 1
				pt.SWts.Adapt.LRate = 0.0001 // 0.005 == .1 == .01
				pt.Learn.DWt.SubMean = 1     // 1 > 0
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.2 > 0.3
			}},
	},
}
