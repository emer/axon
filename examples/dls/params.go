// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build notyet

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the active set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers",
			Params: params.Params{
				ly.Acts.Clamp.Ge = "1.5",
			}},
		{Sel: ".PFCLayer", Doc: "pfc layers: slower trgavgact",
			Params: params.Params{
				ly.Learn.TrgAvgAct.SynScaleRate = "0.0002", // also now set by default
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Params: params.Params{
				ly.Inhib.Layer.Gi =             "2.4",
				ly.Inhib.Pool.Gi =              "2.4",
				ly.Acts.Dend.ModGain =          "1.5", // 2 min -- reduces maint early
				ly.Learn.NeuroMod.AChDisInhib = "0.0", // not much effect here..
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal =       "0.1",
				ly.CT.GeGain =                  "0.05", // 0.05 key for stronger activity
				ly.CT.DecayTau =                "50",
				ly.Learn.NeuroMod.AChDisInhib = "0", // 0.2, 0.5 not much diff
			}},
		{Sel: ".CS", Doc: "need to adjust Nominal for number of CSs -- now down automatically",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.1", // 0.1 for 4, divide by N/4 from there
			}},
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Params: params.Params{
				ly.Inhib.Layer.On =    "false", // todo: explore -- could be bad for gating
				ly.Inhib.Pool.Gi =     "0.3",   // go lower, get more inhib from elsewhere?
				ly.Inhib.Pool.FB =     "1",
				ly.Acts.Dend.ModGain = "1", // todo: 2 is default
			}},
		////////////////////////////////////////////
		// Cortical Paths
		{Sel: ".PFCPath", Doc: "pfc path params -- more robust to long-term training",
			Params: params.Params{
				pt.Learn.Trace.SubMean = "1",    // 1 > 0 for long-term stability
				pt.Learn.LRate.Base =    "0.01", // 0.04 def; 0.02 more stable; 0.01 even more
			}},
		{Sel: ".PTtoPred", Doc: "stronger drive on pt pred",
			Params: params.Params{
				pt.PathScale.Abs = "6",
			}},
		{Sel: ".PTSelfMaint", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs =    "4",
				pt.Learn.LRate.Base = "0.0001", // this is not a problem
			}},
		{Sel: ".ToPTp", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "4",
			}},
		////////////////////////////////////////////
		// Rubicon Paths
		{Sel: ".MatrixPath", Doc: "",
			Params: params.Params{
				pt.Matrix.NoGateLRate =   "1", // this is KEY for robustness when failing initially!
				pt.Learn.Trace.LearnThr = "0.0",
			}},
		{Sel: ".SuperToThal", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "4", // 4 def
			}},
		{Sel: ".SuperToPT", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "0.5", // 0.5 def
			}},
		{Sel: ".GPiToBGThal", Doc: "inhibition from GPi to MD",
			Params: params.Params{
				pt.PathScale.Abs = "5", // with new mod, this can be stronger
			}},
	},
}
