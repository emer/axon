// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "#Input", Doc: "input fixed act",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Decay.Act = 1
				ly.Acts.Decay.Glong = 1
				ly.Inhib.ActAvg.Nominal = 0.05
			}},
		{Sel: "#Rew", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.2
				ly.Inhib.ActAvg.Nominal = 1
			}},
	},
	"RW": {
		{Sel: ".RWPredLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.2
				ly.Inhib.ActAvg.Nominal = 1
				ly.Acts.Dt.GeTau = 40
			}},
	},
	"TD": {
		{Sel: ".TDPredLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.2
				ly.Inhib.ActAvg.Nominal = 1
				ly.Acts.Dt.GeTau = 40
			}},
		{Sel: ".TDIntegLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.2
				ly.Inhib.ActAvg.Nominal = 1
				ly.TDInteg.Discount = 0.9
				ly.TDInteg.PredGain = 1.0
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {},
	"RW": {
		{Sel: ".RWPath", Doc: "RW pred",
			Set: func(pt *axon.PathParams) {
				pt.SWts.Init.Mean = 0
				pt.SWts.Init.Var = 0
				pt.SWts.Init.Sym.SetBool(false)
				pt.Learn.LRate.Base = 0.1
				pt.RLPred.OppSignLRate = 1.0
				pt.RLPred.DaTol = 0.0
			}},
	},
	"TD": {
		{Sel: "#InputToRewPred", Doc: "input to rewpred",
			Set: func(pt *axon.PathParams) {
				pt.SWts.Init.Mean = 0
				pt.SWts.Init.Var = 0
				pt.SWts.Init.Sym.SetBool(false)
				pt.Learn.LRate.Base = 0.1
				pt.RLPred.OppSignLRate = 1.0
			}},
	},
}
