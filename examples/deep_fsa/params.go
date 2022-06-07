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
					"Layer.Inhib.Layer.Gi":     "1.1", // 1.1 > 1.2 > 1.0
					"Layer.Act.Gbar.L":         "0.2", // std
					"Layer.Act.Decay.Act":      "0.2", // lvis best = .2, .6 good here too
					"Layer.Act.Decay.Glong":    "0.6",
					"Layer.Act.Dt.LongAvgTau":  "20",  // 20 > higher for objrec, lvis
					"Layer.Act.Dend.GbarExp":   "0.2", // 0.2 > 0.5 > 0.1 > 0
					"Layer.Act.Dend.GbarR":     "3",   // 3 / 0.2 > 6 / 0.5
					"Layer.Act.Dt.VmDendTau":   "8",   // 8 > 5 >> 2.81 -- big diff
					// "Layer.Act.NMDA.MgC":        "1.0",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					// "Layer.Act.NMDA.Voff":       "0",    // 5 > 0 but need to reduce gbar -- too much
					"Layer.Act.VGCC.Gbar":       "0.02",
					"Layer.Act.AK.Gbar":         "1.0",
					"Layer.Act.NMDA.MgC":        "1.2",  // 1.2 > 1.4 for SynSpkTheta
					"Layer.Act.NMDA.Voff":       "0",    // 0 > 5 for SynSpkTheta
					"Layer.Learn.RLrate.On":     "true", // beneficial still
					"Layer.Learn.NeurCa.SpikeG": "8",
					"Layer.Learn.NeurCa.SynTau": "30", // 40 best in larger models
					"Layer.Learn.NeurCa.MTau":   "10",
					"Layer.Learn.NeurCa.PTau":   "40",
					"Layer.Learn.NeurCa.DTau":   "40",
					"Layer.Learn.NeurCa.CaMax":  "200",
					"Layer.Learn.NeurCa.CaThr":  "0.05",
					"Layer.Learn.NeurCa.Decay":  "false",
					"Layer.Learn.LrnNMDA.ITau":  "1",  // urakubo = 100, does not work here..
					"Layer.Learn.LrnNMDA.Tau":   "50", // urakubo = 30 > 20 but no major effect on PCA
				}},
			{Sel: ".Hidden", Desc: "fix avg act",
				Params: params.Params{}},
			{Sel: ".InLay", Desc: "input layers need more inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
				}},
			{Sel: ".CT", Desc: "CT gain factor is key",
				Params: params.Params{
					"Layer.CtxtGeGain":      "0.2", // 0.2 > 0.3 > 0.1
					"Layer.Inhib.Layer.Gi":  "1.1", // 1.1 > 1.0
					"Layer.Act.KNa.On":      "true",
					"Layer.Act.Decay.Act":   "0.0",
					"Layer.Act.Decay.Glong": "0.0",
				}},
			{Sel: "TRCLayer", Desc: "standard weight is .3 here for larger distributed reps. no learn",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":   "1.1",  // 1.1 > 1.2 with new GeSyn
					"Layer.TRC.DriveScale":   "0.15", // .15 >= .1
					"Layer.TRC.FullDriveAct": "0.6",  // 0.6 def
					"Layer.Act.Spike.Tr":     "3",    // 1 is best for ra25..
					"Layer.Act.Decay.Act":    "0.5",
					"Layer.Act.Decay.Glong":  "1",   // clear long
					"Layer.Act.GABAB.Gbar":   "0.2", // .2 > old: 0.005
					"Layer.Act.NMDA.Gbar":    "0.6", // .6 > .4 > .2 std -- strange but real!
				}},
			{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":        "0.03", // .03 std
					"Prjn.SWt.Adapt.Lrate":         "0.1",  // 0.01 seems to work fine, but .1 maybe more reliable
					"Prjn.SWt.Adapt.DreamVar":      "0.0",  // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":           "1.0",  // 1 works fine here -- .5 also ok
					"Prjn.Com.PFail":               "0.0",
					"Prjn.Learn.KinaseCa.SpikeG":   "12", // 12 good
					"Prjn.Learn.KinaseCa.NMDAG":    "1",
					"Prjn.Learn.KinaseCa.Rule":     "SynSpkTheta", // NeurSpkTheta, SynSpkTheta good, *Cont bad
					"Prjn.Learn.KinaseCa.MTau":     "5",           // 5 > 10 test more
					"Prjn.Learn.KinaseCa.PTau":     "40",
					"Prjn.Learn.KinaseCa.DTau":     "40",
					"Prjn.Learn.KinaseCa.UpdtThr":  "0.01", //
					"Prjn.Learn.KinaseCa.Decay":    "true",
					"Prjn.Learn.KinaseDWt.TWindow": "10",
					"Prjn.Learn.KinaseDWt.DMaxPct": "0.5",
					"Prjn.Learn.KinaseDWt.DScale":  "1",
					"Prjn.Learn.XCal.On":           "true",
					"Prjn.Learn.XCal.PThrMin":      "0.01", // 0.01 > 0.05
					"Prjn.Learn.XCal.LrnThr":       "0.01",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.2 > 0.3
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
			// {Sel: "#HiddenCTToInputP", Desc: "no swt to output layers",
			// 	Params: params.Params{
			// 		"Prjn.Com.PFail":          "0.0",
			// 		"Prjn.Com.PFailWtMax":     "0.0",
			// 		"Prjn.SWt.Adapt.DreamVar": "0.0",   // nope
			// 		"Prjn.SWt.Adapt.On":       "false", // off > on
			// 		"Prjn.SWt.Init.SPct":      "0",     // when off, 0
			// 	}},
		},
	}},
}
