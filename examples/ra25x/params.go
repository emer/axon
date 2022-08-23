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
					"Layer.Inhib.Layer.Gi":          "1.0",  // 1.0 > 1.1 > 1.2 -- diff from orig
					"Layer.Inhib.ActAvg.Init":       "0.05", // 0.05 more sensible, same perf
					"Layer.Act.NMDA.Gbar":           "0.15", // now .15 best
					"Layer.Act.NMDA.MgC":            "1.2",  // 1.4 == 1.2 for trace
					"Layer.Act.NMDA.Voff":           "0",    // 5 == 0 for trace
					"Layer.Act.NMDA.Tau":            "100",  // 100 def -- 50 is sig worse
					"Layer.Act.GABAB.Gbar":          "0.2",  // 0.2 def > higher
					"Layer.Act.AK.Gbar":             "0.1",  // 0.05 to 0.1 likely good per urakubo, but 1.0 needed to prevent vgcc blowup
					"Layer.Act.VGCC.Gbar":           "0.02", // 0.12 per urakubo / etc models, but produces too much high-burst plateau -- even 0.05 with AK = .1 blows up
					"Layer.Act.VGCC.Ca":             "20",   // 20 / 10tau similar to spk
					"Layer.Learn.CaLrn.Norm":        "80",   // 80 works
					"Layer.Learn.CaLrn.SpkVGCC":     "true", // sig better..
					"Layer.Learn.CaLrn.SpkVgccCa":   "35",   // 70 / 5 or 35 / 10 both work
					"Layer.Learn.CaLrn.VgccTau":     "10",   // 10 > 5 ?
					"Layer.Learn.CaLrn.Dt.MTau":     "2",    // 2 > 1 ?
					"Layer.Learn.CaSpk.SpikeG":      "8",    // 8 produces reasonable 0-1 norm CaSpk levels?
					"Layer.Learn.CaSpk.SynTau":      "30",   // 30 > 20, 40
					"Layer.Learn.CaSpk.Dt.MTau":     "5",    // 5 > 10?
					"Layer.Learn.LrnNMDA.MgC":       "1.2",  // 1.2 for unified Act params, else 1.4
					"Layer.Learn.LrnNMDA.Voff":      "0",    // 0 for unified Act params, else 5
					"Layer.Learn.LrnNMDA.Tau":       "100",  // 100 def
					"Layer.Learn.TrgAvgAct.On":      "true", // critical!
					"Layer.Learn.TrgAvgAct.SubMean": "0",
					"Layer.Learn.RLrate.On":         "true", // beneficial for trace
					"Layer.Learn.RLrate.MidRange":   "0.4",
					"Layer.Learn.RLrate.NonMid":     "0.05",
					"Layer.Learn.RLrate.DiffMod":    "true",
					"Layer.Learn.RLrate.ActDiffThr": "0.02", // 0.02 def - todo
					"Layer.Learn.RLrate.ActThr":     "0.1",  // 0.1 def
					"Layer.Learn.RLrate.Min":        "0.001",
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 > 1.0
					"Layer.Act.Clamp.Ge":      "1.0",  // 1.0 > 0.6 >= 0.7 == 0.5
					"Layer.Inhib.ActAvg.Init": "0.15", // .24 nominal, lower to give higher excitation
					"Layer.Act.VGCC.Ca":       "1",    // otherwise dominates display
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9", // 0.9 >= 0.8 > 1.0 > 0.7
					"Layer.Inhib.ActAvg.Init": "0.24",
					"Layer.Act.Spike.Tr":      "1",   // 1 is new minimum.. > 3
					"Layer.Act.Clamp.Ge":      "0.6", // .6 > .5 v94
					"Layer.Act.VGCC.Ca":       "1",   // otherwise dominates display
					// "Layer.Learn.NeurCa.RCa": "false", // auto excluded
				}},
			{Sel: "Prjn", Desc: "basic prjn params",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":       "0.1", // 0.1 is default, 0.05 for TrSpk = .5
					"Prjn.SWt.Adapt.Lrate":        "0.2", // .1 >= .2,
					"Prjn.SWt.Adapt.SoftBound":    "false",
					"Prjn.SWt.Adapt.SigGain":      "1",
					"Prjn.SWt.Adapt.MaxWt":        "false",
					"Prjn.SWt.Adapt.SubMean":      "0",
					"Prjn.SWt.Init.SPct":          "0.5",   // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.Trace.Tau":        "1",     // no longer: 5-10 >> 1 -- longer tau, lower lrate needed
					"Prjn.Learn.Trace.SpkErr":     "false", // comparison case: needs lrate = * .5
					"Prjn.Learn.KinaseCa.SpikeG":  "12",    // 12 def -- produces reasonable ~1ish max vals
					"Prjn.Learn.KinaseCa.UpdtThr": "0.01",  // 0.01 def
					"Prjn.Learn.KinaseCa.Dt.MTau": "5",     // 5 ==? 2 > 10
					"Prjn.Learn.KinaseCa.Dt.PTau": "40",
					"Prjn.Learn.KinaseCa.Dt.DTau": "40",
				}},
			{Sel: "#Hidden2ToOutput", Desc: "key to use activation-based learning for output layers",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.1", // 0.1 is default
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
	}},
}
