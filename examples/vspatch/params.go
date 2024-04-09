// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/netparams"
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: "Layer", Desc: "clamp gain makes big diff on overall excitation, gating propensity",
			Params: params.Params{
				"Layer.Acts.Clamp.Ge": "1.0", // 1.5 is def, was 0.6 (too low)
			}},
		{Sel: "#State", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.2",
			}},
		{Sel: ".VSPatchLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Pool.On":              "false",
				"Layer.Inhib.Layer.On":             "true",
				"Layer.Inhib.Layer.Gi":             "0.5", // 0.5 needed for differentiated reps
				"Layer.Learn.NeuroMod.DipGain":     "1",   // boa requires balanced..
				"Layer.Learn.TrgAvgAct.GiBaseInit": "0",   // 0.5 default; 0 better
				"Layer.Learn.RLRate.SigmoidMin":    "1",   // 0.05 def; 1 causes positive DA bias
				"Layer.Learn.NeuroMod.AChLRateMod": "0",
				"Layer.Learn.NeuroMod.DAModGain":   "0", // this is actual perf mod
				"Layer.VSPatch.MaxLRateFactor":     "1", // even 2 causes too much distortion -- remove
			}},
		{Sel: ".VSPatchPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":        "2",
				"Prjn.Learn.Trace.LearnThr": "0",
				"Prjn.Learn.LRate.Base":     "0.02", // 0.02 necc to fit closely
				"Prjn.SWts.Init.Mean":       "0.5",
				"Prjn.SWts.Init.Var":        "0.25",
			}},
	},
}
