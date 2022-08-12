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
					"Layer.Inhib.Layer.Gi":          "1.0",  // 1.2 > 1.1
					"Layer.Inhib.ActAvg.Init":       "0.04", // 0.03 > 0.04 but can replicate with Act.NMDA.Gbar
					"Layer.Act.NMDA.MgC":            "1.4",  // 1.4 > 1.2 for trace
					"Layer.Act.NMDA.Gbar":           "0.15", // 0.3 > 0.25 > .15 default -- key!
					"Layer.Learn.LrnNMDA.Gbar":      "0.15", // .15 default
					"Layer.Act.NMDA.Voff":           "5",    // 5 > 0 for trace
					"Layer.Act.GABAB.Gbar":          "0.2",  // 0.2 def > higher
					"Layer.Act.AK.Gbar":             "1",    // 1 def
					"Layer.Act.VGCC.Gbar":           "0.02", // .02 def
					"Layer.Learn.NeurCa.Trace":      "true",
					"Layer.Learn.NeurCa.TrGeG":      "1",
					"Layer.Learn.NeurCa.TrPlusG":    "1",    // 1.2 corrects negative trend but doesn't improve learning
					"Layer.Learn.NeurCa.CaMax":      "200",  // 200 def
					"Layer.Learn.NeurCa.CaThr":      "0.05", // 0.05 def -- todo: test more
					"Layer.Learn.NeurCa.MTau":       "5",    // 5 > 10 > 2 for runs after first
					"Layer.Learn.NeurCa.PTau":       "40",   // 40 > 30
					"Layer.Learn.NeurCa.DTau":       "40",   // 40 > 30
					"Layer.Learn.NeurCa.SynTau":     "30",   // 30 > 20, 40
					"Layer.Learn.TrgAvgAct.On":      "true", // not much diff
					"Layer.Learn.RLrate.On":         "true", // beneficial for NMDA = .3
					"Layer.Learn.RLrate.ActDiffThr": "0.02", // 0.02 def
					"Layer.Learn.RLrate.ActThr":     "0.1",  // 0.1 def
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 > 1.0
					"Layer.Act.Clamp.Ge":      "1.0",  // 1.0 > 0.6 >= 0.7 == 0.5
					"Layer.Inhib.ActAvg.Init": "0.15", // .24 nominal, lower to give higher excitation
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9", // 0.9 >= 0.8 > 1.0 > 0.7
					"Layer.Inhib.ActAvg.Init": "0.24",
					"Layer.Act.Spike.Tr":      "1", // 1 is new minimum.. > 3
					"Layer.Act.Clamp.Ge":      "1", // .6 > .5 v94
					// "Layer.Learn.NeurCa.Trace": "false", // auto excluded
				}},
			{Sel: "#Hidden1", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.0",  // 1.1 > 1.2 -- otherwise 1.2 too inactive
					"Layer.Inhib.ActAvg.Init": "0.04", // 0.02 > higher -- fixed by nmda gbar higher
				}},
			{Sel: "Prjn", Desc: "basic prjn params",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":       "0.02",  // 0.1 is default
					"Prjn.SWt.Adapt.Lrate":        "0.1",   // .1 >= .2,
					"Prjn.SWt.Init.SPct":          "0.5",   // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.XCal.On":          "false", // no diff
					"Prjn.Learn.XCal.LrnThr":      "0",
					"Prjn.Learn.XCal.SubMean":     "0",    // no real diff -- amazing..
					"Prjn.Learn.XCal.PThrMin":     "0.01", // 0.01 here; 0.05 best for bigger nets
					"Prjn.Learn.Trace.On":         "true",
					"Prjn.Learn.Trace.Tau":        "1",     // no longer: 5-10 >> 1 -- longer tau, lower lrate needed
					"Prjn.Learn.KinaseCa.NeurCa":  "false", // NeurCa is significantly worse!
					"Prjn.Learn.KinaseCa.UpdtThr": "0",
					"Prjn.Learn.KinaseCa.MTau":    "5", // 5 ==? 2 > 10
					"Prjn.Learn.KinaseCa.PTau":    "40",
					"Prjn.Learn.KinaseCa.DTau":    "40",
				}},
			{Sel: "#Hidden2ToOutput", Desc: "key to use activation-based learning for output layers",
				Params: params.Params{
					// "Prjn.Learn.Trace.On":   "false",
					"Prjn.Learn.Lrate.Base": "0.1", // 0.1 is default
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
	}},
	{Name: "NoTrace", Desc: "non-trace values", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Learn.NeurCa.Trace":  "false",
					"Layer.Learn.NeurCa.SpikeG": "8", // note: makes a diff that can't be fixed by lrate..
					"Layer.Act.Decay.Glong":     "0",
				}},
			{Sel: "Prjn", Desc: "basic prjn params",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":       ".1", // 0.1 is default
					"Prjn.Learn.Trace.On":         "false",
					"Prjn.Learn.KinaseCa.UpdtThr": "0",
				}},
		},
	}},
}
