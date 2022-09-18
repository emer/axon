// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets sets the minimal non-default params
// Base is always applied, and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
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
					"Layer.Inhib.Layer.Gi":    "1.0",  // 1.2 > 1.1
					"Layer.Inhib.ActAvg.Init": "0.05", // 0.4 for 1.2, 0.3 for 1.1
					"Layer.Act.NMDA.MgC":      "1.2",  // 1.2 > 1.4 for SynSpkTheta
					"Layer.Act.NMDA.Voff":     "0",    // 0 > 5 for SynSpkTheta
				},
				Hypers: params.Hypers{
					"Layer.Inhib.Layer.Gi":    {"StdDev": "0.1", "Min": "0.5"},
					"Layer.Inhib.ActAvg.Init": {"StdDev": "0.01", "Min": "0.01"},
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 > 1.0
					"Layer.Act.Clamp.Ge":      "1.0",  // 1.0 > 0.6 >= 0.7 == 0.5
					"Layer.Inhib.ActAvg.Init": "0.15", // .24 nominal, lower to give higher excitation
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "0.9", // 0.9 >= 0.8 > 1.0 > 0.7
					"Layer.Inhib.ActAvg.Init":       "0.24",
					"Layer.Act.Spike.Tr":            "1",    // 1 is new minimum.. > 3
					"Layer.Act.Clamp.Ge":            "0.6",  // .6 > .5 v94
					"Layer.Learn.RLrate.SigmoidMin": "0.05", // sigmoid derivative actually useful here!
				}},
			{Sel: "Prjn", Desc: "basic prjn params",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.1", // 0.1 learns fast but dies early, .02 is stable long term
					"Prjn.SWt.Adapt.Lrate":  "0.1", // .1 >= .2,
					"Prjn.SWt.Init.SPct":    "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
	}},
}
