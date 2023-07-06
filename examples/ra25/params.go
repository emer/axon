// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets sets the minimal non-default params
// Base is always applied, and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {Desc: "these are the best params", Sheets: params.Sheets{
		"NetSize": &params.Sheet{
			{Sel: "Layer", Desc: "all layers",
				Params: params.Params{
					"Layer.X": "8", // 10 orig, 8 is similar, faster
					"Layer.Y": "8",
				}},
		},
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "1.05", // 1.05 > 1.1 for short-term; 1.1 better long-run stability
					"Layer.Inhib.Layer.FB":       "0.5",  // 0.5 > 0.2 > 0.1 > 1.0 -- usu 1.0
					"Layer.Inhib.ActAvg.Nominal": "0.06", // 0.6 > 0.5
					"Layer.Acts.NMDA.MgC":        "1.2",  // 1.2 > 1.4 here, still..
				},
				Hypers: params.Hypers{
					"Layer.Inhib.Layer.Gi":       {"StdDev": "0.1", "Min": "0.5"},
					"Layer.Inhib.ActAvg.Nominal": {"StdDev": "0.01", "Min": "0.01"},
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "0.9",  // 0.9 > 1.0
					"Layer.Acts.Clamp.Ge":        "1.5",  // 1.5 > 1.0
					"Layer.Inhib.ActAvg.Nominal": "0.15", // .24 nominal, lower to give higher excitation
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "0.65", // 0.65
					"Layer.Inhib.ActAvg.Nominal":    "0.24",
					"Layer.Acts.Spikes.Tr":          "1",    // 1 is new minimum.. > 3
					"Layer.Acts.Clamp.Ge":           "0.8",  // 0.8 > 0.6
					"Layer.Learn.RLRate.SigmoidMin": "0.05", // sigmoid derivative actually useful here!
				}},
			{Sel: "Prjn", Desc: "basic prjn params",
				Params: params.Params{
					"Prjn.Learn.LRate.Base":    "0.1", // 0.1 learns fast but dies early, .02 is stable long term
					"Prjn.SWts.Adapt.LRate":    "0.1", // .1 >= .2,
					"Prjn.SWts.Init.SPct":      "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.Trace.SubMean": "0",   // 1 > 0 for long run stability
				}},
			{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
	}},
}
