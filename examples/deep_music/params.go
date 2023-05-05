// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "generic layer params",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal":         "0.1",   // 0.05 needed to get hidden2 high to .1, 0.1 keeps it too low!
					"Layer.Inhib.Layer.Gi":               "0.9",   // 0.9 > 0.95 > 1.0 > 1.1  SSGi = 2
					"Layer.Learn.TrgAvgAct.SynScaleRate": "0.005", // 0.005 best
					"Layer.Learn.TrgAvgAct.SubMean":      "1",     // 1 > 0
					"Layer.Act.Dend.SSGi":                "2",
					"Layer.Act.Gbar.L":                   "0.2", // std
					"Layer.Act.Decay.Act":                "0.0", // 0 == 0.2
					"Layer.Act.Decay.Glong":              "0.0",
					"Layer.Act.NMDA.MgC":                 "1.4", // 1.4, 5 > 1.2, 0 ?
					"Layer.Act.NMDA.Voff":                "0",
					"Layer.Act.NMDA.Gbar":                "0.006",
					"Layer.Act.GABAB.Gbar":               "0.015", // 0.015 > 0.012 lower
					"Layer.Act.Mahp.Gbar":                "0.04",  // 0.04 == 0.05+ > 0.02 -- reduces hidden activity
					"Layer.Act.Sahp.Gbar":                "0.1",   // 0.1 == 0.02 no real diff
					"Layer.Act.Sahp.Off":                 "0.8",   //
					"Layer.Act.Sahp.Slope":               "0.02",  //
					"Layer.Act.Sahp.CaTau":               "5",     // 5 > 10
				}},
			{Sel: ".SuperLayer", Desc: "super layer params",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1",
					"Layer.Burst.ThrRel":         "0.1", // 0.1 > 0.2 > 0
					"Layer.Burst.ThrAbs":         "0.1",
				}},
			{Sel: ".InLay", Desc: "input layers need more inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.025", // 0.025 for full song
					// "Layer.Inhib.ActAvg.Nominal": "0.05", // 0.08 for 18 notes -- 30 rows
				}},
			{Sel: ".CTLayer", Desc: "CT NMDA gbar factor is key",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.12", // CT in general more active
					"Layer.Inhib.Layer.Gi":       "2.8",  // 2.2 >= 2.4 > 2.8
					"Layer.CT.GeGain":            "1.0",  // 1.0 >= 1.5 > 2.0 (very bad)
					"Layer.CT.DecayTau":          "50",   // 50 > 30 -- 30 ok but takes a bit to get going
					"Layer.Act.Dend.SSGi":        "0",    // 0 > higher -- kills nmda maint!
					"Layer.Act.Decay.Act":        "0.0",
					"Layer.Act.Decay.Glong":      "0.0",
					"Layer.Act.MaintNMDA.Gbar":   "0.007", // 0.007 > 0.008 -- same w/ reg better than not
					"Layer.Act.MaintNMDA.Tau":    "300",   // 300 > 200
					"Layer.Act.NMDA.Gbar":        "0.007", // 0.007?
					"Layer.Act.NMDA.Tau":         "300",   // 300 > 200
					"Layer.Act.GABAB.Gbar":       "0.015", // 0.015 def
					"Layer.Act.Noise.On":         "false", // todo?
					"Layer.Act.Noise.Ge":         "0.005",
					"Layer.Act.Noise.Gi":         "0.005",
				}},
			{Sel: ".PulvinarLayer", Desc: "Pulv = Pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "1.0", // 1.0 > 1.1 >> 1.2
					"Layer.Pulv.DriveScale":         "0.1", // 0.1 shows up -- was 0.02
					"Layer.Pulv.FullDriveAct":       "0.6", // 0.6 def
					"Layer.Act.Decay.Act":           "0.0",
					"Layer.Act.Decay.Glong":         "0.0", // clear long
					"Layer.Learn.RLRate.SigmoidMin": "1.0", // 1 > .05
				}},

			// Projections below
			{Sel: "Prjn", Desc: "std",
				Params: params.Params{
					"Prjn.Learn.LRate.Base":    "0.002",  // full song: 0.002 > 0.005, 0.001 in the end; 30 notes: .02
					"Prjn.Learn.Trace.SubMean": "0",      // 0 > 1 -- doesn't work at all with 1
					"Prjn.SWt.Adapt.LRate":     "0.0001", // 0.01 == 0.0001 but 0.001 not as good..
					"Prjn.SWt.Adapt.DreamVar":  "0.0",    // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":       "1.0",    // 1 works fine here -- .5 also ok
					"Prjn.Com.PFail":           "0.0",
					"Prjn.Learn.Trace.Tau":     "2", // 2 > 1 (small bene) > 4 (worse at end on full)
				}},
			{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // 0.1 > 0.2
				}},
			{Sel: ".CTCtxtPrjn", Desc: "all CT context prjns",
				Params: params.Params{
					"Prjn.Learn.LRate.Base":    "0.001", // 0.001 >> 0.002 for full
					"Prjn.Learn.Trace.Tau":     "4",     // 4 > 2?
					"Prjn.Learn.Trace.SubMean": "0",     // 0 > 1 -- 1 is especially bad
					"Prjn.Com.PFail":           "0.0",   // .2, .3 too high -- very slow learning
				}},
			{Sel: ".CTFmSuper", Desc: "1to1 > full",
				Params: params.Params{
					"Prjn.Learn.Learn":   "true", // learning > fixed 1to1
					"Prjn.SWt.Init.Mean": "0.5",  // if fixed, 0.8 > 0.5, var = 0
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: ".FmPulv", Desc: "defaults to .Back but generally weaker is better",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // 0.1 == 0.15 > 0.05
				}},
			{Sel: ".CTSelfCtxt", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5",  // 0.5 > 0.2 > 0.8
					"Prjn.Com.PFail":     "0.0",  // never useful for random gen
					"Prjn.SWt.Init.Sym":  "true", // true > false
				}},
			{Sel: ".CTSelfMaint", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.5", // 0.2 > lower, higher
					"Prjn.Com.GType":     "MaintG",
					"Prjn.SWt.Init.Sym":  "true", // no effect?  not sure why
				}},
			{Sel: "#HiddenCTToInputP", Desc: "differential contributions",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1.0", // .5 is almost as good as 1, .1 is a bit worse
				}},
		},
	}},
	{Name: "Hid2", Desc: "Hid2 config", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#Hidden2CT", Desc: "CT NMDA gbar factor is key",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.12", // 2 even more active -- maybe try higher inhib
					"Layer.Act.GABAB.Gbar":       "0.3",
					"Layer.Act.NMDA.Gbar":        "0.3", // higher layer has more nmda..
					"Layer.Act.NMDA.Tau":         "300", // 300 > 200
					"Layer.Act.Sahp.CaTau":       "10",  // todo
				}},
			// {Sel: "#HiddenP", Desc: "distributed hidden-layer pulvinar",
			// 	Params: params.Params{
			// 		"Layer.Inhib.Layer.Gi":  "0.9",  // 0.9 > 0.8 > 1
			// 		"Layer.Pulv.DriveScale": "0.05", // 0.05 > .1
			// 		"Layer.Act.NMDA.Gbar":   "0.1",
			// 	}},
			{Sel: "#Hidden2CTToHiddenCT", Desc: "ct top-down",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // 0.1 > 0.2
				}},
			{Sel: "#HiddenToHidden2", Desc: "jack up fwd pathway",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0", // this mostly serves to get Hidden2 active -- but why is it so low?
				}},
			{Sel: "#Hidden2CTToInputP", Desc: "differential contributions",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0", // 1 is best..
				}},
		},
	}},
	{Name: "30Notes", Desc: "for the small 30 note test case", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".InLay", Desc: "input layers need more inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.05", // 0.08 for 18 notes -- 30 rows
				}},
		},
	}},
	{Name: "FullSong", Desc: "for the full song", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".InLay", Desc: "input layers need more inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.025", // 0.025 for full song
				}},
		},
	}},
}
