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
					// resting = -65 vs. 70 -- not working -- debug later
					// "Layer.Act.Spike.Thr": ".55", // also bump up
					// "Layer.Act.Spike.VmR": ".35",
					// "Layer.Act.Init.Vm":   ".35",
					// "Layer.Act.Erev.L":    ".35",
					// "Layer.Act.Erev.I":    ".15",
					// "Layer.Act.Erev.K":    ".15",

					"Layer.Inhib.Layer.Gi":        "1.2",  // 1.2 > 1.1
					"Layer.Inhib.ActAvg.Init":     "0.04", // 0.4 for 1.2, 0.3 for 1.1
					"Layer.Inhib.Layer.Bg":        "0.3",  // 0.3 > 0.0
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
					"Layer.Act.NMDA.MgC":        "1.4",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					"Layer.Act.NMDA.Voff":       "5",    // 5 > 0 but need to reduce gbar -- too much
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
					"Prjn.Learn.KinaseCa.SpikeG":   "10",   // keep at 12 standard, adjust other things
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
	{Name: "SynSpkCa", Desc: "SynSpkCa params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Learn.NeurCa.SynTau": "30", // 30 best in larger models now
				}},
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":      "0.1", // 0.1 for SynSpkCa even though dwt equated
					"Prjn.SWt.Adapt.Lrate":       "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":         "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.Kinase.SpikeG":   "8",   // 42 nominal for spkca, but 8 is reliable
					"Prjn.Learn.Kinase.Rule":     "SynSpkCa",
					"Prjn.Learn.Kinase.OptInteg": "true",
					"Prjn.Learn.Kinase.MTau":     "5", // 5 > 10 = 2 - todo test more
					"Prjn.Learn.Kinase.PTau":     "40",
					"Prjn.Learn.Kinase.DTau":     "40",
					"Prjn.Learn.Kinase.DScale":   "1",
					"Prjn.Learn.XCal.On":         "true",
					"Prjn.Learn.XCal.PThrMin":    "0",    // 0.05 best for objrec, higher worse
					"Prjn.Learn.XCal.LrnThr":     "0.01", // 0.05 best in lvis, bad for objrec
				}},
		},
	}},
	{Name: "SynNMDACa", Desc: "SynNMDACa learning settings", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Act.Decay.Glong":  "0.6", // 0.6
					"Layer.Act.Dend.GbarExp": "0.5", // 0.2 > 0.1 > 0
					"Layer.Act.Dend.GbarR":   "6",   // 3 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
					"Layer.Act.Dt.VmDendTau": "5",   // 5 > 2.81 here but small effect
					// Voff = 5, MgC = 1.4, CaMax = 90, VGCCCa = 20 is a reasonable "high voltage" config
					// Voff = 5, MgC = 1.4 is significantly better for PCA Top5
					// Voff = 0, MgC = 1, CaMax = 100, VGCCCa = 20 is a good "default" config
					"Layer.Act.NMDA.Gbar":      "0.15", // 0.15 for !SnmdaDeplete, 1.4 for SnmdaDeplete, 7 for ITau = 100, Tau = 30, !SnmdaDeplete, still doesn't learn..
					"Layer.Act.NMDA.ITau":      "1",    // 1 = get rid of I -- 100, 100 1.5, 1.2 kinda works
					"Layer.Act.NMDA.Tau":       "100",  // 100 > 80 > 70 -- 30 def not good
					"Layer.Act.NMDA.MgC":       "1.4",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					"Layer.Act.NMDA.Voff":      "5",    // 5 > 0 but need to reduce gbar -- too much
					"Layer.Act.Dend.VGCCCa":    "20",   // 20 seems reasonable, but not obviously better than 0
					"Layer.Act.Dend.CaMax":     "100",
					"Layer.Act.Dend.CaThr":     "0.2",
					"Layer.Learn.LrnNMDA.ITau": "1",  // urakubo = 100, does not work here..
					"Layer.Learn.LrnNMDA.Tau":  "30", // urakubo = 30 > 20 but no major effect on PCA
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
					// "Layer.Act.NMDA.Gbar":     "0.3",  // higher not better
				}},
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":      "0.1", // 0.2 std; kinase: 0.08 - 0.1 with RCa normalized
					"Prjn.SWt.Adapt.Lrate":       "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":         "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.Kinase.Rule":     "SynNMDACa",
					"Prjn.Learn.Kinase.OptInteg": "false",
					"Prjn.Learn.Kinase.MTau":     "5", // 5 > 10 > 1 maybe
					"Prjn.Learn.Kinase.PTau":     "40",
					"Prjn.Learn.Kinase.DTau":     "40",
					"Prjn.Learn.Kinase.DScale":   "0.93",  // 0.93 > 1
					"Prjn.Learn.XCal.On":         "false", // no diff
					"Prjn.Learn.XCal.PThrMin":    "0.05",  // can handle this -- todo: try bigger, test more
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
	}},
	{Name: "NeurSpkCa", Desc: "these are the original best NeurSpkCa params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":        "1.2",  // 1.2 > 1.1
					"Layer.Inhib.ActAvg.Init":     "0.04", // 0.4 for 1.2, 0.3 for 1.1
					"Layer.Inhib.Layer.Bg":        "0.3",  // 0.3 > 0.0
					"Layer.Act.Decay.Glong":       "0.6",  // 0.6
					"Layer.Act.Dend.GbarExp":      "0.5",  // 0.2 > 0.1 > 0
					"Layer.Act.Dend.GbarR":        "6",    // 3 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
					"Layer.Act.Dt.VmDendTau":      "5",    // 5 > 2.81 here but small effect
					"Layer.Act.Dt.VmSteps":        "2",    // 2 > 3 -- somehow works better
					"Layer.Act.Dt.GeTau":          "5",
					"Layer.Act.Dend.SeiDeplete":   "false", // noisy!  try on larger models
					"Layer.Act.Dend.SnmdaDeplete": "false",
					"Layer.Act.GABAB.Gbar":        "0.2",  // 0.2 > 0.15
					"Layer.Act.NMDA.Gbar":         "0.15", // 0.15
					"Layer.Act.NMDA.ITau":         "1",    // 1 = get rid of I -- 100, 100 1.5, 1.2 kinda works
					"Layer.Act.NMDA.Tau":          "100",  // 30 not good
					"Layer.Act.NMDA.MgC":          "1.4",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					"Layer.Act.NMDA.Voff":         "5",    // 5 > 0 but need to reduce gbar -- too much
					"Layer.Learn.NeurCa.MTau":     "10",
					"Layer.Learn.NeurCa.PTau":     "40",
					"Layer.Learn.NeurCa.DTau":     "40",
					"Layer.Learn.LrnNMDA.ITau":    "1",   // urak 100
					"Layer.Learn.LrnNMDA.Tau":     "100", // urak 30
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
					// "Layer.Act.NMDA.Gbar":     "0.3",  // higher not better
				}},
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":      "0.2", // 0.2 std; kinase: 0.08 - 0.1 with RCa normalized
					"Prjn.SWt.Adapt.Lrate":       "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":         "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.Kinase.Rule":     "NeurSpkCa",
					"Prjn.Learn.Kinase.OptInteg": "false",
					"Prjn.Learn.Kinase.MTau":     "10",
					"Prjn.Learn.Kinase.PTau":     "40",
					"Prjn.Learn.Kinase.DTau":     "40",
					"Prjn.Learn.Kinase.DScale":   "1",
					"Prjn.Learn.XCal.On":         "true",
					"Prjn.Learn.XCal.PThrMin":    "0.05", // 0.05 max for objrec
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
	}},
}
