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
					"Layer.Inhib.Inhib.AvgTau": "30",
					"Layer.Inhib.ActAvg.Init":  "0.15",
					"Layer.Inhib.Layer.Gi":     "1.0", // 1.0 > 1.1  trace
					"Layer.Act.Gbar.L":         "0.2", // std
					"Layer.Act.Decay.Act":      "0.2", // 0 == 0.2
					"Layer.Act.Decay.Glong":    "0.6",
					"Layer.Act.Dt.LongAvgTau":  "20",  // 20 > higher for objrec, lvis
					"Layer.Act.Dend.GbarExp":   "0.2", // 0.2 > 0.5 > 0.1 > 0
					"Layer.Act.Dend.GbarR":     "3",   // 3 / 0.2 > 6 / 0.5
					"Layer.Act.Dt.VmDendTau":   "5",   // old: 8 > 5 >> 2.81 -- big diff
					"Layer.Act.AK.Gbar":        "1.0",
					"Layer.Act.NMDA.MgC":       "1.4", // 1.4, 5 > 1.2, 0 ?
					"Layer.Act.NMDA.Voff":      "5",
					"Layer.Act.Mahp.Gbar":      "0.04", // 0.04 == 0.05+ > 0.02 -- just reduces hidden activity levels
					"Layer.Act.Sahp.Gbar":      "0.1",  // 0.1 == 0.02 no real diff
					"Layer.Act.Sahp.Off":       "0.8",  //
					"Layer.Act.Sahp.Slope":     "0.02", //
					"Layer.Act.Sahp.CaTau":     "5",    // 5 > 10
				}},
			{Sel: ".Hidden", Desc: "fix avg act",
				Params: params.Params{}},
			{Sel: ".InLay", Desc: "input layers need more inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.05", // 0.08 for 12 notes, 0.05 for 18
				}},
			{Sel: ".CT", Desc: "CT gain factor is key",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.6", // 1.5 > 1.3 == 1.2
					"Layer.Inhib.Pool.Gi":  "1.2", // 1.2 > 1.3 > 1.1
					// "Layer.Inhib.Pool.On":   "true",
					"Layer.CT.GeGain":       "0.5", // 0.5 > 1 ok with stronger maint
					"Layer.CT.DecayTau":     "50",  // 50 > 30 -- 30 ok but takes a bit to get going
					"Layer.Act.KNa.On":      "true",
					"Layer.Act.Decay.Act":   "0.0",
					"Layer.Act.Decay.Glong": "0.0",
					"Layer.Act.GABAB.Gbar":  "0.4",
					"Layer.Act.NMDA.Gbar":   "0.35",  // 0.35 > 0.4 -- 0.3, 200 works but a bit worse
					"Layer.Act.NMDA.Tau":    "300",   // 300 > 200
					"Layer.Act.Noise.On":    "false", // todo?
					"Layer.Act.Noise.Ge":    "0.005",
					"Layer.Act.Noise.Gi":    "0.005",
				}},
			{Sel: "TRCLayer", Desc: "standard weight is .3 here for larger distributed reps. no learn",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "1.0", // 1.0 > 0.9 > 1.1
					"Layer.TRC.DriveScale":          "0.1", // 0.1 > 0.05 > 0.15
					"Layer.TRC.FullDriveAct":        "0.6", // 0.6 def
					"Layer.Act.Spike.Tr":            "3",   // 1 is best for ra25..
					"Layer.Act.Decay.Act":           "0.0",
					"Layer.Act.Decay.Glong":         "0.0", // clear long
					"Layer.Act.GABAB.Gbar":          "0.2", // .2 > old: 0.005
					"Layer.Act.NMDA.Gbar":           "0.1", // now .15 best, .4, .6 sig worse
					"Layer.Learn.RLrate.SigmoidMin": "1",   // 1 > .05
				}},
			{Sel: "Prjn", Desc: "std",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":   "0.002", // 0.002 > 0.001 > 0.005 higher
					"Prjn.SWt.Adapt.Lrate":    "0.01",  // 0.01 == 0.0001 but 0.001 not as good..
					"Prjn.SWt.Adapt.DreamVar": "0.0",   // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":      "1.0",   // 1 works fine here -- .5 also ok
					"Prjn.Com.PFail":          "0.0",
					"Prjn.Learn.Trace.Tau":    "4", // 4 == 2 > 1
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.2 > 0.3
				}},
			{Sel: ".CTCtxt", Desc: "all CT context prjns",
				Params: params.Params{
					"Prjn.Trace":     "false", // not as good with Trace here..
					"Prjn.Com.PFail": "0.0",   // .2, .3 too high -- very slow learning
				}},
			{Sel: ".CTFmSuper", Desc: "initial weight = 0.5 much better than 0.8",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.5",
					// "Prjn.Learn.Lrate.Base":   "0.03", // .04 for rlr too!
				}},
			{Sel: "#InputPToHiddenCT", Desc: "critical to make this small so deep context dominates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // 0.1 > .05 lba
				}},
			{Sel: ".CTToCT", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5", // 0.5 > 0.2 > 0.8
					"Prjn.Com.PFail":     "0.0",
					"Prjn.SWt.Init.Sym":  "true", // true > false
				}},
			{Sel: ".CTLateral", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // 0.1  >= 0.05 > 0.2
					"Prjn.Com.PFail":     "0.0",
					"Prjn.SWt.Init.Sym":  "true", // no effect?  not sure why
				}},
			// {Sel: "#HiddenCTToInputP", Desc: "special",
			// 	Params: params.Params{
			// 		"Prjn.Learn.Lrate.Base": "0.01", // .03 std
			// 	}},
		},
	}},
}
