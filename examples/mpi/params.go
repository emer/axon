// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets sets the minimal non-default params
// Base is always applied, and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Layer", Doc: "all defaults",
			Params: params.Params{
				ly.Inhib.Layer.Gi =       "1.05", // 1.05 > 1.1 for short-term; 1.1 better long-run stability
				ly.Inhib.Layer.FB =       "0.5",  // 0.5 > 0.2 > 0.1 > 1.0 -- usu 1.0
				ly.Inhib.ActAvg.Nominal = "0.06", // 0.6 > 0.5
				ly.Acts.NMDA.MgC =        "1.2",  // 1.2 > 1.4 here, still..
			},
			Hypers: params.Hypers{
				ly.Inhib.Layer.Gi =       {"StdDev = "0.1", "Min = "0.5"},
				ly.Inhib.ActAvg.Nominal = {"StdDev = "0.01", "Min = "0.01"},
			}},
		{Sel: "#Input", Doc: "critical now to specify the activity level",
			Params: params.Params{
				ly.Inhib.Layer.Gi =       "0.9",  // 0.9 > 1.0
				ly.Acts.Clamp.Ge =        "1.5",  // 1.5 > 1.0
				ly.Inhib.ActAvg.Nominal = "0.15", // .24 nominal, lower to give higher excitation
			}},
		{Sel: "#Output", Doc: "output definitely needs lower inhib -- true for smaller layers in general",
			Params: params.Params{
				ly.Inhib.Layer.Gi =          "0.65", // 0.65
				ly.Inhib.ActAvg.Nominal =    "0.24",
				ly.Acts.Spikes.Tr =          "1",    // 1 is new minimum.. > 3
				ly.Acts.Clamp.Ge =           "0.8",  // 0.8 > 0.6
				ly.Learn.RLRate.SigmoidMin = "0.05", // sigmoid derivative actually useful here!
			}},
		{Sel: "Path", Doc: "basic path params",
			Params: params.Params{
				pt.Learn.LRate.Base =    "0.1", // 0.1 learns fast but dies early, .02 is stable long term
				pt.SWts.Adapt.LRate =    "0.1", // .1 >= .2,
				pt.SWts.Init.SPct =      "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
				pt.Learn.Trace.SubMean = "0",   // 1 > 0 for long run stability
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				pt.PathScale.Rel = "0.3", // 0.3 > 0.2 > 0.1 > 0.5
			}},
	},
}
