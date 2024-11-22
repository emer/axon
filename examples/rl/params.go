// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "#Input", Doc: "input fixed act",
			Params: params.Params{
				ly.Acts.Decay.Act =       "1",
				ly.Acts.Decay.Glong =     "1",
				ly.Inhib.ActAvg.Nominal = "0.05",
			}},
		{Sel: "#Rew", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi =       "0.2",
				ly.Inhib.ActAvg.Nominal = "1",
			}},
	},
	"RW": {
		{Sel: ".RWPredLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi =       "0.2",
				ly.Inhib.ActAvg.Nominal = "1",
				ly.Acts.Dt.GeTau =        "40",
			}},
		{Sel: ".RWPath", Doc: "RW pred",
			Params: params.Params{
				pt.SWts.Init.Mean =      "0",
				pt.SWts.Init.Var =       "0",
				pt.SWts.Init.Sym =       "false",
				pt.Learn.LRate.Base =    "0.1",
				pt.RLPred.OppSignLRate = "1.0",
				pt.RLPred.DaTol =        "0.0",
			}},
	},
	"TD": {
		{Sel: ".TDPredLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi =       "0.2",
				ly.Inhib.ActAvg.Nominal = "1",
				ly.Acts.Dt.GeTau =        "40",
			}},
		{Sel: ".TDIntegLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi =       "0.2",
				ly.Inhib.ActAvg.Nominal = "1",
				ly.TDInteg.Discount =     "0.9",
				ly.TDInteg.PredGain =     "1.0",
			}},
		{Sel: "#InputToRewPred", Doc: "input to rewpred",
			Params: params.Params{
				pt.SWts.Init.Mean =      "0",
				pt.SWts.Init.Var =       "0",
				pt.SWts.Init.Sym =       "false",
				pt.Learn.LRate.Base =    "0.1",
				pt.RLPred.OppSignLRate = "1.0",
			}},
	},
}
