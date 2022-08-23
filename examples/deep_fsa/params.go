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
					"Layer.Inhib.Inhib.AvgTau":      "30",
					"Layer.Inhib.ActAvg.Init":       "0.15",
					"Layer.Inhib.Layer.Gi":          "1.0", // 1.0 > 1.1  trace
					"Layer.Act.Gbar.L":              "0.2", // std
					"Layer.Act.Decay.Act":           "0.2", // 0 == 0.2
					"Layer.Act.Decay.Glong":         "0.6",
					"Layer.Act.Dt.LongAvgTau":       "20",  // 20 > higher for objrec, lvis
					"Layer.Act.Dend.GbarExp":        "0.2", // 0.2 > 0.5 > 0.1 > 0
					"Layer.Act.Dend.GbarR":          "3",   // 3 / 0.2 > 6 / 0.5
					"Layer.Act.Dt.VmDendTau":        "5",   // old: 8 > 5 >> 2.81 -- big diff
					"Layer.Act.AK.Gbar":             "1.0",
					"Layer.Act.NMDA.MgC":            "1.4", // 1.4, 5 > 1.2, 0 ?
					"Layer.Act.NMDA.Voff":           "5",
					"Layer.Act.VGCC.Gbar":           "0.02",
					"Layer.Act.VGCC.Ca":             "20",   // 20 / 10tau similar to spk
					"Layer.Learn.CaLrn.Norm":        "80",   // 80 works
					"Layer.Learn.CaLrn.SpkVGCC":     "true", // sig better..
					"Layer.Learn.CaLrn.SpkVgccCa":   "35",   // 20? or 35?
					"Layer.Learn.CaLrn.VgccTau":     "10",   // 10 > 5 ?
					"Layer.Learn.CaLrn.Dt.MTau":     "2",    // 2 > 1 ?
					"Layer.Learn.CaSpk.SpikeG":      "8",    // 8 > 12 with CaSpk trace learning
					"Layer.Learn.CaSpk.SynTau":      "30",   // 30 > 20, 40
					"Layer.Learn.CaSpk.Dt.MTau":     "5",    // 5 > 10?
					"Layer.Learn.LrnNMDA.MgC":       "1.4",  // copy act
					"Layer.Learn.LrnNMDA.Voff":      "5",
					"Layer.Learn.LrnNMDA.Tau":       "100",  // 100 def
					"Layer.Learn.TrgAvgAct.On":      "true", // critical!
					"Layer.Learn.TrgAvgAct.SubMean": "1",    // 1 == 0
					"Layer.Learn.RLrate.On":         "true", // beneficial for trace
					"Layer.Learn.RLrate.MidRange":   "0.4",  // 0.4 > 0.35 > 0.45
					"Layer.Learn.RLrate.NonMid":     "1",    // not useful in this model!
					"Layer.Learn.RLrate.DiffMod":    "true",
					"Layer.Learn.RLrate.ActDiffThr": "0.02", // 0.02 def - todo
					"Layer.Learn.RLrate.ActThr":     "0.1",  // 0.1 def
				}},
			{Sel: ".Hidden", Desc: "fix avg act",
				Params: params.Params{}},
			{Sel: ".InLay", Desc: "input layers need more inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
				}},
			{Sel: ".CT", Desc: "CT gain factor is key",
				Params: params.Params{
					"Layer.CtxtGeGain":      "0.25", // 0.25 > 0.3 (most blow up) > 0.2 (was .2)
					"Layer.Act.KNa.On":      "true",
					"Layer.Act.Decay.Act":   "0.0",
					"Layer.Act.Decay.Glong": "0.0",
				}},
			{Sel: "TRCLayer", Desc: "standard weight is .3 here for larger distributed reps. no learn",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":   "1.0",  // 1.0 > 0.9 > 1.1
					"Layer.TRC.DriveScale":   "0.05", // 0.05 > .1 > .15 for trace w/ gi1.0 -- repl10
					"Layer.TRC.FullDriveAct": "0.6",  // 0.6 def
					"Layer.Act.Spike.Tr":     "3",    // 1 is best for ra25..
					"Layer.Act.Decay.Act":    "0.5",
					"Layer.Act.Decay.Glong":  "1",    // clear long
					"Layer.Act.GABAB.Gbar":   "0.2",  // .2 > old: 0.005
					"Layer.Act.NMDA.Gbar":    "0.15", // now .15 best, .4, .6 sig worse
				}},
			{Sel: "Prjn", Desc: "std",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":       "0.02", // .02 > .03 > .01 -- .03 std
					"Prjn.SWt.Adapt.Lrate":        "0.1",  // 0.01 seems to work fine, but .1 maybe more reliable
					"Prjn.SWt.Adapt.DreamVar":     "0.0",  // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":          "1.0",  // 1 works fine here -- .5 also ok
					"Prjn.Com.PFail":              "0.0",
					"Prjn.Learn.Trace.Tau":        "2",    // 2 > 1?
					"Prjn.Learn.KinaseCa.SpikeG":  "12",   // 12 def -- produces reasonable ~1ish max vals
					"Prjn.Learn.KinaseCa.UpdtThr": "0.01", // 0.01 def
					"Prjn.Learn.KinaseCa.Dt.MTau": "5",    // 5 ==? 2 > 10
					"Prjn.Learn.KinaseCa.Dt.PTau": "40",
					"Prjn.Learn.KinaseCa.Dt.DTau": "40",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.2 > 0.3
				}},
			{Sel: ".CTCtxt", Desc: "all CT context prjns",
				Params: params.Params{
					"Prjn.Trace": "false",
				}},
			{Sel: ".CTFmSuper", Desc: "initial weight = 0.5 much better than 0.8",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.5",
					// "Prjn.Learn.Lrate.Base":   "0.03", // .04 for rlr too!
					// "Prjn.Learn.XCal.PThrMin": "0.02", //
				}},
			{Sel: "#InputPToHiddenCT", Desc: "critical to make this small so deep context dominates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // 0.1 > .05 lba
				}},
			{Sel: "#HiddenCTToHiddenCT", Desc: "testing",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1", // 1 > other
				}},
			// {Sel: "#HiddenCTToInputP", Desc: "special",
			// 	Params: params.Params{
			// 		"Prjn.Learn.Lrate.Base": "0.01", // .03 std
			// 	}},
		},
	}},
}
