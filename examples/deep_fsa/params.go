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
					"Layer.Act.Decay.Act":      "0.0", // 0 == 0.2
					"Layer.Act.Decay.Glong":    "0.0",
					"Layer.Act.Dt.LongAvgTau":  "20",  // 20 > higher for objrec, lvis
					"Layer.Act.Dend.GbarExp":   "0.2", // 0.2 > 0.5 > 0.1 > 0
					"Layer.Act.Dend.GbarR":     "3",   // 3 / 0.2 > 6 / 0.5
					"Layer.Act.Dt.VmDendTau":   "5",   // old: 8 > 5 >> 2.81 -- big diff
					"Layer.Act.AK.Gbar":        "0.1",
					"Layer.Act.NMDA.MgC":       "1.4", // 1.4, 5 > 1.2, 0 ?
					"Layer.Act.NMDA.Voff":      "5",
					"Layer.Act.Sahp.Gbar":      "0.1",  //
					"Layer.Act.Sahp.Off":       "0.8",  //
					"Layer.Act.Sahp.Slope":     "0.02", //
					"Layer.Act.Sahp.CaTau":     "10",   //
				}},
			{Sel: ".Hidden", Desc: "fix avg act",
				Params: params.Params{}},
			{Sel: ".InLay", Desc: "input layers need more inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
				}},
			{Sel: ".CT", Desc: "CT NMDA gbar factor is key",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "1.4", // 1.2 > 1.3 > 1.1
					"Layer.CT.GeGain":               "0.8", // 0.8 > 0.5 > 1
					"Layer.CT.DecayTau":             "50",  // 50 > 30 -- 30 ok but takes a bit to get going
					"Layer.Act.Decay.Act":           "0.0",
					"Layer.Act.Decay.Glong":         "0.0",
					"Layer.Act.Dt.VmDendTau":        "5",
					"Layer.Act.Dt.GeTau":            "5",
					"Layer.Act.GABAB.Gbar":          "0.25",  //
					"Layer.Act.NMDA.Gbar":           "0.25",  // 0.25+ > .2, .15 only at start -- others catch up
					"Layer.Act.NMDA.Tau":            "200",   // 200 slightly better than 300 early, same later; 100 fails
					"Layer.Act.Noise.On":            "false", // todo?
					"Layer.Act.Noise.Ge":            "0.005",
					"Layer.Act.Noise.Gi":            "0.005",
					"Layer.Learn.RLrate.On":         "true", // beneficial for trace
					"Layer.Learn.RLrate.SigmoidMin": "0.05", // 0.05 > .1 > .02
					"Layer.Learn.RLrate.Diff":       "true",
					"Layer.Learn.RLrate.ActDiffThr": "0.02", // 0.02 def - todo
					"Layer.Learn.RLrate.ActThr":     "0.1",  // 0.1 def
					"Layer.Learn.RLrate.Min":        "0.001",
				}},
			{Sel: "TRCLayer", Desc: "pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "1.0", // 1.0 > 0.9 > 1.1
					"Layer.TRC.DriveScale":          "0.1", // 0.1 music
					"Layer.TRC.FullDriveAct":        "0.6", // 0.6 def
					"Layer.Act.Spike.Tr":            "3",   // 1 is best for ra25..
					"Layer.Act.Decay.Act":           "0.0",
					"Layer.Act.Decay.Glong":         "0.0", // clear long
					"Layer.Act.Decay.AHP":           "0.0", // clear ahp
					"Layer.Act.GABAB.Gbar":          "0.2", // .2 > old: 0.005
					"Layer.Act.NMDA.Gbar":           "0.1", // .1 music
					"Layer.Learn.RLrate.SigmoidMin": "1",   // auto = 1: not useful in output layer
				}},
			{Sel: "Prjn", Desc: "std",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":   "0.03", // .03 > others -- same as CtCtxt
					"Prjn.SWt.Adapt.Lrate":    "0.01", // 0.01 or 0.0001 music
					"Prjn.SWt.Adapt.DreamVar": "0.0",  // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":      "1.0",  // 1 works fine here -- .5 also ok
					"Prjn.Com.PFail":          "0.0",
					"Prjn.Learn.Trace.Tau":    "2", // 2 > 1 -- more-or-less a ceiling effect..
					"Prjn.Learn.Trace.NoExp2": "true",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.2 > 0.3
				}},
			{Sel: ".CTCtxt", Desc: "all CT context prjns",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":   "0.01", // .03 > .02 > .01 -- .03 std
					"Prjn.Trace":              "true",
					"Prjn.Learn.Trace.Tau":    "1",
					"Prjn.Learn.Trace.NoExp2": "true",
					"Prjn.Com.PFail":          "0.0", // .2, .3 too high -- very slow learning
				}},
			{Sel: ".CTFmSuper", Desc: "full > 1to1",
				Params: params.Params{
					"Prjn.Learn.Learn":   "true",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: "#InputPToHiddenCT", Desc: "critical to make this small so deep context dominates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // 0.1 > .05 lba
				}},
			{Sel: ".CTSelfCtxt", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5",  // 0.5 > 0.2 > 0.8
					"Prjn.SWt.Init.Sym":  "true", // true > false
				}},
			{Sel: ".CTSelfMaint", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",  // 0.1  >= 0.05 > 0.2
					"Prjn.SWt.Init.Sym":  "true", // no effect?  not sure why
				}},
		},
	}},
}
