// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the active set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "minimal base params needed for this model", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "generic params for all layers",
				Params: params.Params{
					"Layer.Act.Clamp.Ge": "1.5",
				}},
			{Sel: ".MatrixLayer", Desc: "all mtx",
				Params: params.Params{
					"Layer.Inhib.Layer.On":   "false",
					"Layer.Inhib.Pool.Gi":    "0.3",
					"Layer.Inhib.Pool.FB":    "1",
					"Layer.Act.Dend.ModGain": "1", // todo: try with lower drive
				}},
			{Sel: "#BLAPosAcqD1", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2.4", // 2.2 not enough to knock out novelty
				}},
			{Sel: ".PTMaintLayer", Desc: "time integration params",
				Params: params.Params{
					"Layer.Act.Dend.ModGain": "2", // 2 min -- reduces maint early
				}},
			{Sel: "#SC", Desc: "",
				Params: params.Params{
					"Layer.Act.Decay.Act": "1",
				}},
			////////////////////////////////////////////
			// Cortical Prjns
			{Sel: "#BLAPosAcqD1ToOFCus", Desc: "stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // 4 is too strong?
				}},
			{Sel: "#OFCusToOFCval", Desc: "stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3",
				}},
			{Sel: "#ACCcostToACCutil", Desc: "stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3",
				}},
			{Sel: "#DistToACCcost", Desc: "stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: "#EffortToACCcost", Desc: "stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			////////////////////////////////////////////
			// PVLV Prjns
			{Sel: ".ToSC", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2",
				}},
			{Sel: ".DrivesToMtx", Desc: "", Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			}},
			{Sel: ".BLAAcqToGo", Desc: "must dominate",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1",
					"Prjn.PrjnScale.Abs": "3",
				}},
			{Sel: ".PFCToVSMtx", Desc: "contextual, should be weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // 0.1 def
				}},
			{Sel: ".VSPatchPrjn", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs":    "1",     // 3 orig
					"Prjn.Learn.LRate.Base": "0.001", // 0.05 def
				}},
			{Sel: "#CSToBLAPosAcqD1", Desc: "",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.1", // was 0.5 -- too fast!?
				}},
			{Sel: ".SuperToThal", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "4", // lower default to prevent maint early
				}},
			{Sel: "#ACCcostToACCcostMD", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3", // supertothal for us stronger
				}},
			{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3", // 4 prevents some gating, 2 leaks with supertothal 4
				}},
			{Sel: ".BLAFromNovel", Desc: "if too strong, isn't out-competed",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // 3 is too strong.
				}},
			{Sel: "#UrgencyToVsMtxGo", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0", // todo: not working -- no ach by time this happens!  need to drive ach too.
				}},
			{Sel: ".BLAExtPrjn", Desc: "ext learns relatively fast",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.01",
				}},
		}},
	},
}

// ParamSetsDefs is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSetsDefs = params.Sets{
	{Name: "Base", Desc: "minimal base params needed for this model", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					// "Layer.Act.Decay.Act":   "0.0", // don't do this generically..
					// "Layer.Act.Decay.Glong": "0.0",
					"Layer.Act.Clamp.Ge": "1.5",
				}},
			{Sel: ".CTLayer", Desc: "corticothalamic context -- using FSA-based params -- intermediate",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.12",
					"Layer.CT.GeGain":            "1.0",
					"Layer.CT.DecayTau":          "50",
					"Layer.Inhib.Layer.Gi":       "2.2",
					"Layer.Inhib.Pool.Gi":        "2.2",
					"Layer.Act.GABAB.Gbar":       "0.25",
					"Layer.Act.NMDA.Gbar":        "0.25",
					"Layer.Act.NMDA.Tau":         "200",
					"Layer.Act.Decay.Act":        "0.0",
					"Layer.Act.Decay.Glong":      "0.0",
					"Layer.Act.Sahp.Gbar":        "1.0",
				}},
			{Sel: ".PTPredLayer", Desc: "PTPred prediction layer -- more dynamic acts",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.12",
					"Layer.CT.GeGain":            "0.01",
					"Layer.CT.DecayTau":          "50",
					"Layer.Inhib.Layer.Gi":       "0.8", // 0.8 overridden by OFCPTPred below
					"Layer.Inhib.Pool.Gi":        "0.8",
					"Layer.Act.GABAB.Gbar":       "0.2", // regular
					"Layer.Act.NMDA.Gbar":        "0.15",
					"Layer.Act.NMDA.Tau":         "100",
					"Layer.Act.Decay.Act":        "0.2",
					"Layer.Act.Decay.Glong":      "0.6",
					"Layer.Act.Sahp.Gbar":        "0.1",
					"Layer.Act.KNa.Slow.Max":     "0.2", // maybe too random if higher?
				}},
			{Sel: ".PTMaintLayer", Desc: "time integration params",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":             "1.8", // was 1.0
					"Layer.Inhib.Pool.Gi":              "1.8", // was 1.8
					"Layer.Act.GABAB.Gbar":             "0.3",
					"Layer.Act.NMDA.Gbar":              "0.3", // 0.3 enough..
					"Layer.Act.NMDA.Tau":               "300",
					"Layer.Act.Decay.Act":              "0.0",
					"Layer.Act.Decay.Glong":            "0.0",
					"Layer.Act.Sahp.Gbar":              "0.01", // not much pressure -- long maint
					"Layer.Act.Dend.ModGain":           "10",   // 10?
					"Layer.Learn.NeuroMod.AChDisInhib": "1.0",
				}},
			{Sel: ".PTNotMaintLayer", Desc: "",
				Params: params.Params{
					"Layer.Act.Init.GeBase": "1.2",
					"Layer.Act.Decay.Act":   "1.0",
					"Layer.Act.Decay.Glong": "1.0",
					"Layer.Inhib.Layer.Gi":  "0.5",
				}},
			{Sel: ".BGThalLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.6",
					"Layer.Inhib.Pool.Gi":  "0.6", // 0.6 > 0.5 -- 0.8 too high
				}},
			{Sel: ".DrivesLayer", Desc: "expect act",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.03",
					"Layer.Inhib.Layer.On":       "false",
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Pool.Gi":        "0.5",
				}},
			{Sel: ".USLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.2",
					"Layer.Inhib.Layer.On":       "true",
					"Layer.Inhib.Pool.On":        "false",
					"Layer.Inhib.Layer.Gi":       "0.5",
				}},
			{Sel: ".PVLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.2",
					"Layer.Inhib.Layer.On":       "true",
					"Layer.Inhib.Pool.On":        "false",
					"Layer.Inhib.Layer.Gi":       "0.5",
				}},
			{Sel: ".VSPatchLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal":       "0.2",
					"Layer.Inhib.Pool.Gi":              "0.3",
					"Layer.Learn.NeuroMod.AChLRateMod": "0.8",
					"Layer.Learn.NeuroMod.BurstGain":   "3",   // extra key for compensating for NoDALRate
					"Layer.Learn.NeuroMod.DipGain":     "1",   // controls extinction -- works fine at 1
					"Layer.PVLV.Thr":                   "0.3", // .2 -- .25 doesn't get strong enough to overcome DA, .15 is too low
					"Layer.PVLV.Gain":                  "20",  // 20 needed for faster learning
					"Layer.VSPatch.NoDALRate":          "0.2", // 0.05 is just not quite sufficient
				}},
			{Sel: "#CS", Desc: "expect act",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1", // 1 / css
				}},
			{Sel: ".Dist", Desc: "expect act",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1", // 1 / maxdist -- low to jack up Ge
				}},
			{Sel: "#DistP", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.6",
				}},
			{Sel: ".OFC", Desc: "",
				Params: params.Params{
					"Layer.Act.Decay.Act":        "0.0", // do this only where needed
					"Layer.Act.Decay.Glong":      "0.0",
					"Layer.Act.Decay.OnRew":      "true", // everything clears
					"Layer.Inhib.ActAvg.Nominal": "0.025",
					"Layer.Inhib.Layer.Gi":       "1.1", // todo: 1.4?
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Pool.Gi":        "1.0",
					"Layer.Act.Dend.SSGi":        "0",
				}},
			{Sel: "#OFCCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2.8", // 2.4 not strong enough to prevent diffuse activity
					"Layer.Inhib.Pool.Gi":  "1.2", // was 1.4
				}},
			{Sel: "#OFC", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.2", // todo: 1.4?
					"Layer.Inhib.Pool.Gi":  "1.0", // was 1.1
				}},
			{Sel: "#OFCPT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8", // was 1.3
					"Layer.Inhib.Pool.Gi":  "2.0", // was 0.6
				}},
			{Sel: "#OFCPTp", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.8",
					"Layer.Inhib.Pool.Gi":  "0.8",
				}},
			{Sel: "#OFCMD", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Pool.Gi": "0.6",
				}},
			{Sel: ".ACC", Desc: "",
				Params: params.Params{
					"Layer.Act.Decay.Act":        "0.0", // do this only where needed
					"Layer.Act.Decay.Glong":      "0.0",
					"Layer.Act.Decay.OnRew":      "true", // everything clears
					"Layer.Inhib.ActAvg.Nominal": "0.1",
					"Layer.Inhib.Layer.On":       "true",
					"Layer.Inhib.Layer.Gi":       "0.9",
					"Layer.Inhib.Pool.On":        "false",
					"Layer.Act.Dend.SSGi":        "0",
				}},
			{Sel: "#ACCCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.4",
				}},
			{Sel: "#ACCPT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2.5",
				}},
			{Sel: "#ACCPTp", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.8",
				}},
			{Sel: "#ACCMD", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.6", // 0.6 > 0.5 -- 0.8 too high
				}},
			{Sel: ".ALM", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9",
				}},
			{Sel: "#ALMCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2.4",
				}},
			{Sel: "#M1", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9",
					// "Layer.Act.Decay.Act":   "1.0",
					// "Layer.Act.Decay.Glong": "1.0",
				}},
			{Sel: "#M1P", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.5",
				}},
			{Sel: "#VL", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":  "0.8",  // was 0.8
					"Layer.Pulv.DriveScale": "0.12", // 0.12 > 0.1
				}},
			{Sel: ".BLALayer", Desc: "",
				Params: params.Params{
					"Layer.Act.Decay.Act":             "0.0",
					"Layer.Act.Decay.Glong":           "0.0",
					"Layer.Inhib.ActAvg.Nominal":      "0.025",
					"Layer.Inhib.Layer.Gi":            "1.8", // needs to be strong to prevent random off-US act
					"Layer.Inhib.Pool.Gi":             "0.9",
					"Layer.Learn.NeuroMod.DALRateMod": "0.5",
					"Layer.Learn.NeuroMod.BurstGain":  "0.2",
					"Layer.Learn.NeuroMod.DipGain":    "0", // ignore small negative DA
				}},
			{Sel: "#BLAPosExtD2", Desc: "",
				Params: params.Params{
					"Layer.Learn.NeuroMod.BurstGain":   "1",
					"Layer.Learn.NeuroMod.DipGain":     "1",
					"Layer.Learn.NeuroMod.AChLRateMod": "1",
					"Layer.Learn.RLRate.Diff":          "false",
				}},
			{Sel: "#CeMPos", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.15",
					"Layer.Act.Dend.SSGi":        "0",
					"Layer.Inhib.Layer.Gi":       "0.5",
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Pool.Gi":        "0.3",
				}},
			{Sel: ".LDTLayer", Desc: "",
				Params: params.Params{
					"Layer.Act.Decay.Act":        "1.0", // critical to not decay -- otherwise FFPrv reset!
					"Layer.Act.Decay.Glong":      "1.0",
					"Layer.Inhib.ActAvg.Nominal": "0.1",
					"Layer.Inhib.Layer.Gi":       "1.0",
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Pool.Gi":        "0.5",
					"Layer.Inhib.Pool.FFPrv":     "10", // key td param
					"Layer.PVLV.Thr":             "0.2",
					"Layer.PVLV.Gain":            "2", // key for impact on CS bursting
					"Layer.LDT.RewThr":           "0.2",
				}},
			{Sel: "#VpSTNp", Desc: "Pausing STN",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal":       "0.15",
					"Layer.Inhib.Layer.On":             "true", // this is critical, else too active
					"Layer.Inhib.Layer.Gi":             "0.6",
					"Layer.Act.SKCa.Gbar":              "3",
					"Layer.Learn.NeuroMod.AChDisInhib": "2",
				}},
			{Sel: "#VpSTNs", Desc: "Sustained STN",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal":       "0.15",
					"Layer.Inhib.Layer.On":             "true",
					"Layer.Inhib.Layer.Gi":             "0.2",
					"Layer.Learn.NeuroMod.AChDisInhib": "2",
				}},
			{Sel: ".GPLayer", Desc: "all gp",
				Params: params.Params{
					"Layer.Act.Init.GeBase":      "0.3",
					"Layer.Act.Init.GeVar":       "0.1",
					"Layer.Act.Init.GiVar":       "0.1",
					"Layer.Act.Dt.GeTau":         "6", // 5 def; 6 looks better..
					"Layer.Act.Dt.GiTau":         "7", // 7 def; slower = more gating (dramatic)
					"Layer.Inhib.ActAvg.Nominal": "1",
				}},
			{Sel: "#VpGPi", Desc: "",
				Params: params.Params{
					"Layer.Act.Init.GeBase": "0.6", // todo: test 0.5
					"Layer.Act.Dt.GeTau":    "7",   // 7 > 6 > 5
					"Layer.Act.Dt.GiTau":    "7",   //
				}},
			{Sel: ".MatrixLayer", Desc: "all mtx",
				Params: params.Params{
					"Layer.Matrix.GateThr":             "0.05", // 0.05 > 0.08 maybe
					"Layer.Matrix.IsVS":                "true",
					"Layer.Learn.NeuroMod.DALRateSign": "true",
					"Layer.Learn.NeuroMod.BurstGain":   "1",
					"Layer.Learn.NeuroMod.DipGain":     "1",
					"Layer.Learn.NeuroMod.AChDisInhib": "5", // key to be 5
					"Layer.Act.Dend.ModGain":           "2",
					"Layer.Inhib.ActAvg.Nominal":       ".03",
					"Layer.Inhib.Layer.On":             "false",
					"Layer.Inhib.Layer.Gi":             "0.0", // was .8
					"Layer.Inhib.Pool.On":              "true",
					"Layer.Inhib.Pool.Gi":              "0.5", // 0.7 > 0.6 more sparse
				}},
			////////////////////////////////////////////////////////////////
			// cortical prjns
			{Sel: "Prjn", Desc: "all prjns",
				Params: params.Params{
					"Prjn.Learn.Trace.Tau":  "1", // was 4..
					"Prjn.Learn.LRate.Base": "0.04",
				}},
			{Sel: ".BackPrjn", Desc: "back is weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".SuperToPT", Desc: "one-to-one from super -- just use fixed nonlearning prjn so can control behavior easily",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1",    // keep this constant -- only self vs. this -- thal is modulatory
					"Prjn.PrjnScale.Abs": "0.01", // monitor maint early and other maint stats with PTMaintLayer ModGain = 0 to set this so super alone is not able to drive it.
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".PTSelfMaint", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel":    "1",      // use abs to manipulate
					"Prjn.PrjnScale.Abs":    "2",      // 2 > 1
					"Prjn.Learn.LRate.Base": "0.0001", // slower > faster
					"Prjn.SWt.Init.Mean":    "0.5",
					"Prjn.SWt.Init.Var":     "0.5", // high variance so not just spreading out over time
				}},
			{Sel: ".SuperToThal", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1.0",
					"Prjn.PrjnScale.Abs": "2.0", // if this is too strong, it gates to the wrong CS
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".ThalToSuper", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".ThalToPT", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1.0",
					"Prjn.Com.GType":     "ModulatoryG", // this marks as modulatory with extra ModGain factor
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".CTtoThal", Desc: "",
				Params: params.Params{
					"Prjn.SWt.Init.Var":  "0.25",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".CTCtxtPrjn", Desc: "all CT context prjns",
				Params: params.Params{
					"Prjn.Learn.LRate.Base":    "0.01", // trace: .01 > .005 > .02; .03 > .02 > .01 -- .03 std
					"Prjn.Learn.Trace.Tau":     "1",    // 2 > 1
					"Prjn.Learn.Trace.SubMean": "0",    // 0 > 1 -- 1 is especially bad
				}},
			{Sel: ".ToPTPred", Desc: "needs more activation",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: "#OFCCTToOFCPTp", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.5",
				}},
			{Sel: "#OFCToOFCMD", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "4.0", // 6? if this is too strong, it gates to the wrong CS
				}},
			/*
				{Sel: "#OFCToALM", Desc: "",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "0.2",
					}},
				{Sel: "#ACCToALM", Desc: "",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "0.2",
					}},
				{Sel: "#ALMToALMd", Desc: "selects action based on alm -- nominally weaker?",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "1",
					}},
			*/
			// {Sel: "#ACCToACCPT", Desc: "",
			// 	Params: params.Params{
			// 		"Prjn.PrjnScale.Abs": "1.5",
			// 	}},
			{Sel: "#ACCPTToACCMD", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
				}},
			{Sel: "#ACCPTToACCPTp", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2",
				}},
			{Sel: "#DistToACC", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0",
				}},
			{Sel: "#EffortToACC", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: ".ToM1", Desc: "",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.1",
				}},
			{Sel: ".ToVL", Desc: "",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.1",
				}},
			{Sel: "#DistToM1", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: "#VLToM1", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0", // not 2.0..
				}},
			//////////////////////////////////////////////
			// PVLV
			{Sel: ".BLAAcqPrjn", Desc: "",
				Params: params.Params{
					"Prjn.Learn.LRate.Base":  "0.02",
					"Prjn.BLA.NegDeltaLRate": "0.01", // todo: explore
					"Prjn.BLA.NonUSLRate":    "0.0",  // def 0.01
				}},
			{Sel: ".BLAExtPrjn", Desc: "",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.01",
					"Prjn.PrjnScale.Abs":    "1",
				}},
			{Sel: ".USToBLA", Desc: "starts strong, learns slow",
				Params: params.Params{
					"Prjn.SWt.Init.SPct":    "0",
					"Prjn.SWt.Init.Mean":    "0.5",
					"Prjn.SWt.Init.Var":     "0.25",
					"Prjn.Learn.LRate.Base": "0.001",
					"Prjn.PrjnScale.Rel":    "0.5",
				}},
			{Sel: "#USposToBLAPosAcqD1", Desc: "stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3.0", // if weaker, other BLA pools get active
				}},
			{Sel: "#CSToBLAPosAcqD1", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs":    "1.0",
					"Prjn.Learn.LRate.Base": "0.1", // was 0.5 -- too fast!?
				}},
			{Sel: "#BLAPosAcqD1ToOFC", Desc: "strong",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "4", // 4 orig
				}},
			{Sel: ".BLAToCeM_Excite", Desc: "",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".BLAToCeM_Inhib", Desc: "",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".BLAExtToAcq", Desc: "fixed inhibitory",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0", // 1 is ok but some spontaneous activity -- 2 is more reliable
				}},
			{Sel: ".VSPatchPrjn", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs":    "3",   // 3 orig
					"Prjn.Learn.LRate.Base": "0.1", // 0.2 > 0.02..?
				}},
			// {Sel: "#OFCPTpToVSPatch", Desc: "",
			// 	Params: params.Params{
			// 		"Prjn.PrjnScale.Abs": "3", // let ofc is more dominant, has pools
			// 	}},
			{Sel: "#ALMCTToVSPatch", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // todo: explore
				}},

			// BG prjns
			{Sel: ".MatrixPrjn", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs":    "1.0", // stronger
					"Prjn.SWt.Init.SPct":    "0",
					"Prjn.SWt.Init.Mean":    "0.5",
					"Prjn.SWt.Init.Var":     "0.25",
					"Prjn.Matrix.CurTrlDA":  "true",
					"Prjn.Matrix.UseHasRew": "true", // hack to use US-only timing
					"Prjn.Matrix.AChDecay":  "0",    // not used if UseHasRew is on
					"Prjn.Learn.Learn":      "true",
					"Prjn.Learn.LRate.Base": "0.1",
				}},
			{Sel: ".BgFixed", Desc: "fixed, non-learning params",
				Params: params.Params{
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.Learn.Learn":   "false",
				}},
			{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // 2 > 1 -- inhib stronger, prevent unwanted gating
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.SWt.Adapt.On":  "false",
					"Prjn.Learn.Learn":   "false",
				}},
			{Sel: "#USposToVpMtxGo", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "5", // 5 orig
					"Prjn.PrjnScale.Rel": ".2",
				}},
			{Sel: ".BLAToBG", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "8",
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: ".DrivesToMtx", Desc: "this is modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.PrjnScale.Rel": "1",
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.Com.GType":     "ModulatoryG",
				}},
			{Sel: ".DrivesToOFC", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // 2 > 1
					"Prjn.PrjnScale.Rel": ".5",
					// "Prjn.Learn.Learn":   "false",
					// "Prjn.SWt.Init.Mean": "0.8",
					// "Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".DrivesToVSPatch", Desc: "this is modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.PrjnScale.Rel": "1",
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.Com.GType":     "ModulatoryG",
				}},
			{Sel: ".FmSTNp", Desc: "increase to prevent repeated gating",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.2", // 1.2 > 1.0 > 1.5 (too high)
				}},
			{Sel: ".GPeTAToMtx", Desc: "nonspecific gating activity surround inhibition -- wta",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // 2 nominal
				}},
			{Sel: ".GPeInToMtx", Desc: "provides weak counterbalance for GPeTA -> Mtx to reduce oscillations",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.5",
				}},
		}},
	},
	{Name: "NoTrainMtx", Desc: "turn off training in Mtx", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".MatrixPrjn", Desc: "learning in mtx",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
				}},
		}},
	},
}
