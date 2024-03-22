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
		{Sel: ".VSPatchLayer", Desc: "",
			Params: params.Params{
				"Layer.Learn.NeuroMod.DipGain":     "1",      // boa requires balanced..
				"Layer.VSPatch.Gain":               "5",      // 3 def
				"Layer.VSPatch.ThrInit":            "0.2",    // thr .2
				"Layer.VSPatch.ThrLRate":           "0.0001", // 0.0001 good for flexible cycle test
				"Layer.VSPatch.ThrNonRew":          "10",
				"Layer.Learn.TrgAvgAct.GiBaseInit": "0.5",
				"Layer.Learn.RLRate.SigmoidMin":    "0.05", // 0.05 def
				"Layer.Learn.NeuroMod.AChLRateMod": "0",
			}},
		{Sel: ".VSPatchPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":        "6",
				"Prjn.Learn.Trace.LearnThr": "0",
				"Prjn.Learn.LRate.Base":     "0.05", // 0.05 def
			}},
	},
}
