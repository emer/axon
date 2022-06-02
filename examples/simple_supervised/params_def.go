// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
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
					"Layer.Inhib.Layer.Gi":        "1.2",  // 1.2 > 1.1
					"Layer.Inhib.ActAvg.Init":     "0.04", // 0.4 for 1.2, 0.3 for 1.1
					"Layer.Inhib.Layer.Bg":        "0.0",  // 0.0 > 0.3 -- 0.3 much worse
					"Layer.Act.Decay.Glong":       "0.6",  // 0.6
					"Layer.Act.Dend.GbarExp":      "0.2",  // 0.5 > 0.2 old def but not in larger or fsa
					"Layer.Act.Dend.GbarR":        "3",    // 6 > 3 old def
					"Layer.Act.Dt.VmDendTau":      "5",    // 5 > 2.81 here but small effect
					"Layer.Act.Dt.VmSteps":        "2",    // 2 > 3 -- somehow works better
					"Layer.Act.Dt.GeTau":          "5",
					"Layer.Act.Dend.SeiDeplete":   "false", // noisy!  try on larger models
					"Layer.Act.Dend.SnmdaDeplete": "false",
					"Layer.Act.GABAB.Gbar":        "0.2", // 0.2 > 0.15

					// Voff = 5, MgC = 1.4, CaMax = 90, VGCCCa = 20 is a reasonable "high voltage" config
					// Voff = 0, MgC = 1, CaMax = 100, VGCCCa = 20 is a good "default" config
					"Layer.Act.NMDA.Gbar":       "0.15", // 0.15 for !SnmdaDeplete, 1.4 for SnmdaDeplete, 7 for ITau = 100, Tau = 30, !SnmdaDeplete, still doesn't learn..
					"Layer.Act.NMDA.ITau":       "1",    // 1 = get rid of I -- 100, 100 1.5, 1.2 kinda works
					"Layer.Act.NMDA.Tau":        "100",  // 30 not good
					"Layer.Act.NMDA.MgC":        "1.2",  // 1.2 > 1.4 for SynSpkTheta
					"Layer.Act.NMDA.Voff":       "0",    // 0 > 5 for SynSpkTheta
					"Layer.Act.VGCC.Gbar":       "0.1",
					"Layer.Act.AK.Gbar":         "1",    // 1 >= 0 > 2
					"Layer.Learn.RLrate.On":     "true", // beneficial still
					"Layer.Learn.NeurCa.SpikeG": "8",
					"Layer.Learn.NeurCa.SynTau": "30",
					"Layer.Learn.NeurCa.MTau":   "10",
					"Layer.Learn.NeurCa.PTau":   "40",
					"Layer.Learn.NeurCa.DTau":   "40",
					"Layer.Learn.NeurCa.CaMax":  "100",
					"Layer.Learn.NeurCa.CaThr":  "0.05",
					"Layer.Learn.NeurCa.Decay":  "true", // synnmda needs false
					"Layer.Learn.LrnNMDA.ITau":  "1",    // urakubo = 100, does not work here..
					"Layer.Learn.LrnNMDA.Tau":   "50",   // urakubo = 30 > 20 but no major effect on PCA
				},
				Hypers: params.Hypers{
					"Layer.Inhib.Layer.Gi":    {"StdDev": "0.1", "Min": "0.5"},
					"Layer.Inhib.ActAvg.Init": {"StdDev": "0.01", "Min": "0.01"},
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 > 1.0
					"Layer.Act.Clamp.Ge":      "1.0",  // 1.0 > 0.6 >= 0.7 == 0.5
					"Layer.Inhib.ActAvg.Init": "0.15", // .24 nominal, lower to give higher excitation
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 >= 0.8 > 1.0 > 0.7 even with adapt -- not beneficial to start low
					"Layer.Inhib.ActAvg.Init": "0.24", // this has to be exact for adapt
					"Layer.Act.Spike.Tr":      "1",    // 1 is new minimum..
					"Layer.Act.Clamp.Ge":      "0.6",  // .6 > .5 v94
					// "Layer.Learn.NeurCa.CaMax": "120",
				}},
			{Sel: "Prjn", Desc: "basic prjn params",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":        "0.1",  // 0.1 for SynSpkTheta even though dwt equated
					"Prjn.SWt.Adapt.Lrate":         "0.08", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":           "0.5",  // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.KinaseCa.SpikeG":   "12",   // keep at 12 standard, adjust other things
					"Prjn.Learn.KinaseCa.NMDAG":    "2",    // 2 > 1
					"Prjn.Learn.KinaseCa.Rule":     "SynSpkTheta",
					"Prjn.Learn.KinaseCa.MTau":     "5", // 5 > 10 test more
					"Prjn.Learn.KinaseCa.PTau":     "40",
					"Prjn.Learn.KinaseCa.DTau":     "40",
					"Prjn.Learn.KinaseCa.UpdtThr":  "0.01", // 0.01 here; 0.05 best for bigger nets
					"Prjn.Learn.KinaseCa.Decay":    "true",
					"Prjn.Learn.KinaseDWt.TWindow": "10",
					"Prjn.Learn.KinaseDWt.DMaxPct": "0.5",
					"Prjn.Learn.KinaseDWt.DScale":  "1",
					"Prjn.Learn.XCal.On":           "true",
					"Prjn.Learn.XCal.PThrMin":      "0.01", // 0.01 here; 0.05 best for bigger nets
					"Prjn.Learn.XCal.LrnThr":       "0.01", // 0.01 here; 0.05 best for bigger nets
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
	}},
}
