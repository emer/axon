// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Layer", Doc: "clamp gain makes big diff on overall excitation, gating propensity",
			Params: params.Params{
				ly.Acts.Clamp.Ge = "1.0", // 1.5 is def, was 0.6 (too low)
			}},
		{Sel: "#State", Doc: "",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.2",
			}},
		{Sel: ".VSPatchLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Pool.On =              "false",
				ly.Inhib.Layer.On =             "true",
				ly.Inhib.Layer.Gi =             "0.5", // 0.5 needed for differentiated reps
				ly.Learn.NeuroMod.DipGain =     "1",   // boa requires balanced..
				ly.Learn.TrgAvgAct.GiBaseInit = "0",   // 0.5 default; 0 better
				ly.Learn.RLRate.SigmoidMin =    "1",   // 0.05 def; 1 causes positive DA bias
				ly.Learn.NeuroMod.AChLRateMod = "0",
				ly.Learn.NeuroMod.DAModGain =   "0", // this is actual perf mod
			}},
		{Sel: ".VSPatchPath", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs =        "2",
				pt.Learn.Trace.LearnThr = "0",
				pt.Learn.LRate.Base =     "0.02", // 0.02 necc to fit closely; no bene for 0.01
				pt.SWts.Init.Mean =       "0.5",
				pt.SWts.Init.Var =        "0.25",
			}},
	},
}
