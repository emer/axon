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
					"Layer.Inhib.ActAvg.Init":       "0.05", // 0.05 more sensible, same perf
					"Layer.Act.NMDA.MgC":            "1.2",  // 1.4 == 1.2 for trace
					"Layer.Act.NMDA.Voff":           "0",    // 5 == 0 for trace
					"Layer.Act.NMDA.Gbar":           "0.15", // now .15 best
					"Layer.Learn.LrnNMDA.Gbar":      "0.15", // .15 default
					"Layer.Act.GABAB.Gbar":          "0.2",  // 0.2 def > higher
					"Layer.Act.AK.Gbar":             "0.1",  // 0.05 to 0.1 likely good per urakubo
					"Layer.Act.VGCC.Gbar":           "0.02", // 0.12 per urakubo / etc models, but produces too much high-burst plateau
					"Layer.Act.VGCC.Ca":             "800",
					"Layer.Learn.NeurCa.RCa":        "true",
					"Layer.Learn.NeurCa.SpkVGCC":    "false",
					"Layer.Learn.NeurCa.SpkVGCCa":   "1000",
					"Layer.Learn.NeurCa.CaMax":      "250",  // 200 def -- 250 best with vgcc, tau 5 gain 100
					"Layer.Learn.NeurCa.CaThr":      "0.0",  // 0 > 0.05 def
					"Layer.Learn.NeurCa.MTau":       "5",    // has an lrate-like effect: 1=slower than 5
					"Layer.Learn.NeurCa.PTau":       "40",   // 40 > 30
					"Layer.Learn.NeurCa.DTau":       "40",   // 40 > 30
					"Layer.Learn.NeurCa.SynTau":     "30",   // 30 > 20, 40
					"Layer.Learn.TrgAvgAct.On":      "true", // critical!
					"Layer.Learn.RLrate.On":         "true", // beneficial for NMDA = .3
					"Layer.Learn.RLrate.ActDiffThr": "0.02", // 0.02 def
					"Layer.Learn.RLrate.ActThr":     "0.1",  // 0.1 def
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
					"Prjn.Learn.Lrate.Base":       "0.1",   // 0.1 is default, 0.05 for TrSpk = .5
					"Prjn.SWt.Adapt.Lrate":        "0.1",   // .1 >= .2,
					"Prjn.SWt.Init.SPct":          "0.5",   // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.XCal.On":          "false", // slightly worse, makes no sense really in trace context
					"Prjn.Learn.XCal.LrnThr":      "0",
					"Prjn.Learn.XCal.SubMean":     "0",    // actually bad now
					"Prjn.Learn.XCal.PThrMin":     "0.01", // 0.01 here; 0.05 best for bigger nets
					"Prjn.Learn.Trace.On":         "true",
					"Prjn.Learn.Trace.Tau":        "1",     // no longer: 5-10 >> 1 -- longer tau, lower lrate needed
					"Prjn.Learn.KinaseCa.NeurCa":  "false", // NeurCa is significantly worse!
					"Prjn.Learn.KinaseCa.UpdtThr": "0.0",   // todo: test with new tag
					"Prjn.Learn.KinaseCa.MTau":    "5",     // 5 ==? 2 > 10
					"Prjn.Learn.KinaseCa.PTau":    "40",
					"Prjn.Learn.KinaseCa.DTau":    "40",
				}},
			{Sel: "#Hidden2ToOutput", Desc: "key to use activation-based learning for output layers",
				Params: params.Params{
					// "Prjn.Learn.Trace.On":   "false", // auto
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