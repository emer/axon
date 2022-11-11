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
					"Layer.Inhib.ActAvg.Init": "0.1", // 0.05 needed to get hidden2 high to .1, 0.1 keeps it too low!
					"Layer.Inhib.Layer.Gi":    "1.0", // 1.0 > 1.1  trace
					"Layer.Act.Gbar.L":        "0.2", // std
					"Layer.Act.Decay.Act":     "0.0", // 0 == 0.2
					"Layer.Act.Decay.Glong":   "0.0",
					"Layer.Act.NMDA.MgC":      "1.4", // 1.4, 5 > 1.2, 0 ?
					"Layer.Act.NMDA.Voff":     "5",
					"Layer.Act.Mahp.Gbar":     "0.04", // 0.04 == 0.05+ > 0.02 -- reduces hidden activity
					"Layer.Act.Sahp.Gbar":     "0.1",  // 0.1 == 0.02 no real diff
					"Layer.Act.Sahp.Off":      "0.8",  //
					"Layer.Act.Sahp.Slope":    "0.02", //
					"Layer.Act.Sahp.CaTau":    "5",    // 5 > 10
				}},
			{Sel: "SuperLayer", Desc: "super layer params",
				Params: params.Params{
					"Layer.Burst.ThrRel": "0.1", // no diffs here -- music makes a diff
					"Layer.Burst.ThrAbs": "0.1",
				}},
			{Sel: ".Hidden", Desc: "fix avg act",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.1",
				}},
			{Sel: ".DepthIn", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.2", // was .13 -- Ge very high b/c of topo prjn
					"Layer.Inhib.Layer.Gi":    "0.9", //
				}},
			{Sel: ".HeadDirIn", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.13", // 0.13 > 0.2 -- 0.13 is accurate but Ge is high..
					"Layer.Inhib.Layer.Gi":    "0.9",  //
				}},
			{Sel: ".CT", Desc: "CT NMDA gbar factor is key",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12", // CT in general more active
					"Layer.Inhib.Layer.Gi":    "2.0",  // 2.0 is fine -- was 1.4
					"Layer.CT.GeGain":         "1",    // 1
					"Layer.CT.DecayTau":       "0",    // 50 > 30 -- 30 ok but takes a bit to get going
					"Layer.Act.Decay.Act":     "0.0",
					"Layer.Act.Decay.Glong":   "0.0",
					"Layer.Act.GABAB.Gbar":    "0.2",  // standard gaba
					"Layer.Act.NMDA.Gbar":     "0.15", // .15 for copy mode
					"Layer.Act.NMDA.Tau":      "100",  // 100 best here
				}},
			{Sel: "#DepthHid", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.2",  // 1.2 tiny bit > 1.4
					"Layer.Inhib.ActAvg.Init": "0.07", // 0.07 actual
				}},
			{Sel: "#DepthHidCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "2.8",  // 2.8 is reasonable; was 2.0
					"Layer.Inhib.ActAvg.Init": "0.07", // 0.07 reasonable -- actual is closer to .15 but this produces stronger drive on Pulvinar which produces *slightly* better performance.
				}},
			{Sel: "#DepthHid2CT", Desc: "CT NMDA gbar factor is key",
				Params: params.Params{
					"Layer.CT.GeGain":         "0.8",  // 0.8, 50 small benefit
					"Layer.CT.DecayTau":       "50",   // 50 > 0
					"Layer.Inhib.ActAvg.Init": "0.12", // 2 even more active -- maybe try higher inhib
					"Layer.Inhib.Layer.Gi":    "1.4",  // todo
					"Layer.Act.GABAB.Gbar":    "0.3",
					"Layer.Act.NMDA.Gbar":     "0.3", // higher layer has more nmda..
					"Layer.Act.NMDA.Tau":      "300", // 300 > 200
					"Layer.Act.Sahp.CaTau":    "10",  // todo
				}},
			{Sel: "PulvLayer", Desc: "Pulv = Pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "0.8", // was 0.9
					"Layer.Pulv.DriveScale":         "0.1", // 0.1 -- does not work with 0.05 -- no plus
					"Layer.Pulv.FullDriveAct":       "0.6", // 0.6 def
					"Layer.Act.Decay.Act":           "0.0",
					"Layer.Act.Decay.Glong":         "0.0", // clear long
					"Layer.Act.Decay.AHP":           "0.0", // clear long
					"Layer.Act.GABAB.Gbar":          "0.2",
					"Layer.Act.NMDA.Gbar":           "0.1", // .1 was important
					"Layer.Learn.RLrate.SigmoidMin": "1",   // 1 > .05
				}},
			{Sel: "#DepthHidP", Desc: "distributed hidden-layer pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":  "0.9",  // 0.9 > 0.8 > 1
					"Layer.Pulv.DriveScale": "0.05", // 0.05 > .1
					"Layer.Act.NMDA.Gbar":   "0.1",
				}},
			{Sel: "#Action", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.25", // 0.25 is accurate -- good MaxGe levels
					"Layer.Inhib.Layer.Gi":    "0.9",  //
				}},

			// Projections below
			{Sel: "Prjn", Desc: "std",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":   "0.005", // 0.005 > 0.002 > 0.01
					"Prjn.SWt.Adapt.Lrate":    "0.01",  // 0.01 == 0.0001 but 0.001 not as good..
					"Prjn.SWt.Adapt.DreamVar": "0.0",   // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":      "1.0",   // 1 works fine here -- .5 also ok
					"Prjn.Com.PFail":          "0.0",
					"Prjn.Learn.Trace.Tau":    "2", // 4 == 2 > 1
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: "#HeadDirHidCTToDepthHidCT", Desc: "ct top-down",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // not much diff here
				}},
			{Sel: ".CTCtxt", Desc: "all CT context prjns",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.002", // has almost no effect in 1to1
					"Prjn.Learn.Trace.Tau":  "2",     // late in learning 2 does best
					"Prjn.Com.PFail":        "0.0",   // .2, .3 too high -- very slow learning
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
					"Prjn.PrjnScale.Rel": "0.1", // 0.1  >= 0.05 > 0.2
					"Prjn.Com.PFail":     "0.0",
					"Prjn.SWt.Init.Sym":  "true", // no effect?  not sure why
				}},
			{Sel: "#ActionToDepthHidCT", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.5 is not better
				}},
			{Sel: "#ActionToDepthHid", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "2.0", // 2.0 > 3.0 > 1.0
				}},
			{Sel: "#DepthHid2CTToDepthP", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // 0.1 == 0.15 > 0.05
				}},
		},
	}},
}
