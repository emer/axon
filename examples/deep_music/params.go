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
		{Sel: "Layer", Desc: "generic layer params",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal":         "0.1",   // 0.05 needed to get hidden2 high to .1, 0.1 keeps it too low!
				"Layer.Inhib.Layer.Gi":               "0.9",   // 0.9 > 0.95 > 1.0 > 1.1  SSGi = 2
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.005", // 0.005 best
				"Layer.Learn.TrgAvgAct.SubMean":      "1",     // 1 > 0
				"Layer.Acts.Dend.SSGi":               "2",
				"Layer.Acts.Gbar.L":                  "0.2", // std
				"Layer.Acts.Decay.Act":               "0.0", // 0 == 0.2
				"Layer.Acts.Decay.Glong":             "0.0",
				"Layer.Acts.NMDA.MgC":                "1.4", // 1.4, 5 > 1.2, 0 ?
				"Layer.Acts.NMDA.Voff":               "0",
				"Layer.Acts.NMDA.Gbar":               "0.006",
				"Layer.Acts.GabaB.Gbar":              "0.015", // 0.015 > 0.012 lower
				"Layer.Acts.Mahp.Gbar":               "0.04",  // 0.04 == 0.05+ > 0.02 -- reduces hidden activity
				"Layer.Acts.Sahp.Gbar":               "0.1",   // 0.1 == 0.02 no real diff
				"Layer.Acts.Sahp.Off":                "0.8",   //
				"Layer.Acts.Sahp.Slope":              "0.02",  //
				"Layer.Acts.Sahp.CaTau":              "5",     // 5 > 10
			}},
		{Sel: ".SuperLayer", Desc: "super layer params",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.1",
				"Layer.Bursts.ThrRel":        "0.1", // 0.1 > 0.2 > 0
				"Layer.Bursts.ThrAbs":        "0.1",
			}},
		{Sel: ".InLay", Desc: "input layers need more inhibition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.025", // 0.025 for full song
				// "Layer.Inhib.ActAvg.Nominal": "0.05", // 0.08 for 18 notes -- 30 rows
			}},
		{Sel: ".CTLayer", Desc: "CT NMDA gbar factor is key",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.12", // CT in general more active
				"Layer.Inhib.Layer.Gi":       "2.2",  // 2.2 >= 2.4 > 2.8
				"Layer.CT.GeGain":            "1.0",  // 1.0 >= 1.5 > 2.0 (very bad) > 0.5
				"Layer.Acts.Dend.SSGi":       "0",    // 0 > higher -- kills nmda maint!
				"Layer.Acts.Decay.Act":       "0.0",
				"Layer.Acts.Decay.Glong":     "0.0",
				"Layer.Acts.MaintNMDA.Gbar":  "0.007", // 0.007 > 0.008 -- same w/ reg better than not
				"Layer.Acts.MaintNMDA.Tau":   "300",   // 300 > 200
				"Layer.Acts.NMDA.Gbar":       "0.007", // 0.007?
				"Layer.Acts.NMDA.Tau":        "300",   // 300 > 200
				"Layer.Acts.GabaB.Gbar":      "0.015", // 0.015 def
				"Layer.Acts.Noise.On":        "false", // todo?
				"Layer.Acts.Noise.Ge":        "0.005",
				"Layer.Acts.Noise.Gi":        "0.005",
			}},
		{Sel: ".PulvinarLayer", Desc: "Pulv = Pulvinar",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":          "1.0", // 1.0 > 1.1 >> 1.2
				"Layer.Pulv.DriveScale":         "0.1", // 0.1 > 0.15 > 0.2; .05 doesn't work at all
				"Layer.Pulv.FullDriveAct":       "0.6", // 0.6 def
				"Layer.Acts.Decay.Act":          "0.0",
				"Layer.Acts.Decay.Glong":        "0.0", // clear long
				"Layer.Learn.RLRate.SigmoidMin": "1.0", // 1 > .05
			}},

		// Projections below
		{Sel: "Path", Desc: "std",
			Params: params.Params{
				"Path.Learn.LRate.Base":    "0.002",  // full song and 30n: 0.002 > 0.005, 0.001 in the end
				"Path.Learn.Trace.SubMean": "0",      // 0 > 1 -- doesn't work at all with 1
				"Path.SWts.Adapt.LRate":    "0.0001", // 0.01 == 0.0001 but 0.001 not as good..
				"Path.SWts.Init.SPct":      "1.0",    // 1 works fine here -- .5 also ok
				"Path.Com.PFail":           "0.0",
				"Path.Learn.Trace.Tau":     "1", // 1 > 2 v0.0.9
			}},
		{Sel: ".BackPath", Desc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				"Path.PathScale.Rel": "0.1", // 0.1 > 0.2
			}},
		{Sel: ".CTCtxtPath", Desc: "all CT context paths",
			Params: params.Params{
				"Path.Learn.LRate.Base":    "0.001", // 0.001 >> 0.002 for full
				"Path.Learn.Trace.Tau":     "2",     // 1 > 2 > 4 v0.0.9
				"Path.Learn.Trace.SubMean": "0",     // 0 > 1 -- 1 is especially bad
				"Path.Com.PFail":           "0.0",   // .2, .3 too high -- very slow learning
			}},
		{Sel: ".CTFromSuper", Desc: "1to1 > full",
			Params: params.Params{
				"Path.Learn.Learn":    "true", // learning > fixed 1to1
				"Path.SWts.Init.Mean": "0.5",  // if fixed, 0.8 > 0.5, var = 0
				"Path.SWts.Init.Var":  "0.25",
			}},
		{Sel: ".FromPulv", Desc: "defaults to .Back but generally weaker is better",
			Params: params.Params{
				"Path.PathScale.Rel": "0.1", // 0.1 == 0.15 > 0.05
			}},
		{Sel: ".CTSelfCtxt", Desc: "",
			Params: params.Params{
				"Path.PathScale.Rel": "0.5",  // 0.5 > 0.2 > 0.8
				"Path.Com.PFail":     "0.0",  // never useful for random gen
				"Path.SWts.Init.Sym": "true", // true > false
			}},
		{Sel: ".CTSelfMaint", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs": "0.2", // 0.2 > lower, higher
				"Path.Com.GType":     "MaintG",
				"Path.SWts.Init.Sym": "true", // no effect?  not sure why
			}},
		{Sel: "#HiddenCTToInputP", Desc: "differential contributions",
			Params: params.Params{
				"Path.PathScale.Rel": "1.0", // .5 is almost as good as 1, .1 is a bit worse
			}},
	},
	"Hid2": {
		{Sel: "#Hidden2CT", Desc: "CT NMDA gbar factor is key",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.12", // 2 even more active -- maybe try higher inhib
				"Layer.Acts.GabaB.Gbar":      "0.3",
				"Layer.Acts.NMDA.Gbar":       "0.3", // higher layer has more nmda..
				"Layer.Acts.NMDA.Tau":        "300", // 300 > 200
				"Layer.Acts.Sahp.CaTau":      "10",  // todo
			}},
		// {Sel: "#HiddenP", Desc: "distributed hidden-layer pulvinar",
		// 	Params: params.Params{
		// 		"Layer.Inhib.Layer.Gi":  "0.9",  // 0.9 > 0.8 > 1
		// 		"Layer.Pulv.DriveScale": "0.05", // 0.05 > .1
		// 		"Layer.Acts.NMDA.Gbar":   "0.1",
		// 	}},
		{Sel: "#Hidden2CTToHiddenCT", Desc: "ct top-down",
			Params: params.Params{
				"Path.PathScale.Rel": "0.1", // 0.1 > 0.2
			}},
		{Sel: "#HiddenToHidden2", Desc: "jack up fwd pathway",
			Params: params.Params{
				"Path.PathScale.Abs": "2.0", // this mostly serves to get Hidden2 active -- but why is it so low?
			}},
		{Sel: "#Hidden2CTToInputP", Desc: "differential contributions",
			Params: params.Params{
				"Path.PathScale.Abs": "1.0", // 1 is best..
			}},
	},
	"30Notes": {
		{Sel: ".InLay", Desc: "input layers need more inhibition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.05", // 0.08 for 18 notes -- 30 rows
			}},
	},
	"FullSong": {
		{Sel: ".InLay", Desc: "input layers need more inhibition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.025", // 0.025 for full song
			}},
	},
}
