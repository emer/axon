// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build notyet

package main

import (
	"github.com/emer/emergent/v2/netparams"
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the active set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: "Layer", Desc: "generic params for all layers",
			Params: params.Params{
				"Layer.Acts.Clamp.Ge": "1.5",
			}},
		{Sel: ".PFCLayer", Desc: "pfc layers: slower trgavgact",
			Params: params.Params{
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002", // also now set by default
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":             "2.4",
				"Layer.Inhib.Pool.Gi":              "2.4",
				"Layer.Acts.Dend.ModGain":          "1.5", // 2 min -- reduces maint early
				"Layer.Learn.NeuroMod.AChDisInhib": "0.0", // not much effect here..
			}},
		{Sel: ".PTPredLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal":       "0.1",
				"Layer.CT.GeGain":                  "0.05", // 0.05 key for stronger activity
				"Layer.CT.DecayTau":                "50",
				"Layer.Learn.NeuroMod.AChDisInhib": "0", // 0.2, 0.5 not much diff
			}},
		{Sel: ".CS", Desc: "need to adjust Nominal for number of CSs -- now down automatically",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.1", // 0.1 for 4, divide by N/4 from there
			}},
		{Sel: ".MatrixLayer", Desc: "all mtx",
			Params: params.Params{
				"Layer.Inhib.Layer.On":    "false", // todo: explore -- could be bad for gating
				"Layer.Inhib.Pool.Gi":     "0.3",   // go lower, get more inhib from elsewhere?
				"Layer.Inhib.Pool.FB":     "1",
				"Layer.Acts.Dend.ModGain": "1", // todo: 2 is default
			}},
		////////////////////////////////////////////
		// Cortical Paths
		{Sel: ".PFCPath", Desc: "pfc path params -- more robust to long-term training",
			Params: params.Params{
				"Path.Learn.Trace.SubMean": "1",    // 1 > 0 for long-term stability
				"Path.Learn.LRate.Base":    "0.01", // 0.04 def; 0.02 more stable; 0.01 even more
			}},
		{Sel: ".PTtoPred", Desc: "stronger drive on pt pred",
			Params: params.Params{
				"Path.PathScale.Abs": "6",
			}},
		{Sel: ".PTSelfMaint", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs":    "4",
				"Path.Learn.LRate.Base": "0.0001", // this is not a problem
			}},
		{Sel: ".ToPTp", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs": "4",
			}},
		////////////////////////////////////////////
		// Rubicon Paths
		{Sel: ".MatrixPath", Desc: "",
			Params: params.Params{
				"Path.Matrix.NoGateLRate":   "1", // this is KEY for robustness when failing initially!
				"Path.Learn.Trace.LearnThr": "0.0",
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs": "4", // 4 def
			}},
		{Sel: ".SuperToPT", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs": "0.5", // 0.5 def
			}},
		{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
			Params: params.Params{
				"Path.PathScale.Abs": "5", // with new mod, this can be stronger
			}},
	},
}
