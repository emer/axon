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
			{Sel: ".CS", Desc: "need to adjust Nominal for number of CSs -- now down automatically",
				Params: params.Params{
					// "Layer.Inhib.Layer.Gi":  "1.0", // 1.0 for CSP, 0.9 for CS -- should be higher for CSPerDrive > 1
					"Layer.Inhib.ActAvg.Nominal": "0.1", // 0.1 for 4, divide by N/4 from there
				}},
			{Sel: ".BLAFromNovel", Desc: "must be strong enough to compete with CS at start -- now done automatically",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3", // 2 is good for .CS nominal .1, but 3 needed for .03
				}},
			{Sel: ".MatrixLayer", Desc: "all mtx",
				Params: params.Params{
					"Layer.Inhib.Layer.On":   "false", // todo: explore -- could be bad for gating
					"Layer.Inhib.Pool.Gi":    "0.3",   // go lower, get more inhib from elsewhere?
					"Layer.Inhib.Pool.FB":    "1",
					"Layer.Act.Dend.ModGain": "1", // todo: try with lower drive
				}},
			{Sel: "#BLAPosAcqD1", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2.4", // 2.2 not enough to knock out novelty
				}},
			{Sel: ".PTMaintLayer", Desc: "time integration params",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":             "2.4",
					"Layer.Inhib.Pool.Gi":              "2.4",
					"Layer.Act.Dend.ModGain":           "1.5", // 2 min -- reduces maint early
					"Layer.Learn.NeuroMod.AChDisInhib": "0",   // todo: explore!  might be bad..
				}},
			// {Sel: ".PTPredLayer", Desc: "",
			// 	Params: params.Params{
			// 		"Layer.Learn.NeuroMod.AChDisInhib": "0", // todo: explore!  might be bad..
			// 	}},
			{Sel: ".VSPatchLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Pool.Gi":          "0.5", // todo: go lower, get more inhib from elsewhere?
					"Layer.Inhib.Pool.FB":          "0",
					"Layer.Learn.NeuroMod.DipGain": "0.01", // rate of extinction -- reduce to slow
					"Layer.PVLV.Thr":               "0.4",
					"Layer.PVLV.Gain":              "20",
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
					"Prjn.PrjnScale.Abs": "3", // fairly sensitive to this param..
				}},
			{Sel: "#OFCvalToACCutil", Desc: "not good to make this stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: ".PTSelfMaint", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "4",
				}},
			{Sel: ".ToPTp", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "4",
				}},
			////////////////////////////////////////////
			// PVLV Prjns
			{Sel: ".MatrixPrjn", Desc: "",
				Params: params.Params{
					"Prjn.Matrix.NoGateLRate": "1", // this is KEY for robustness when failing initially!
				}},
			{Sel: ".ToSC", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2",
				}},
			{Sel: ".DrivesToMtx", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: ".BLAExtPrjn", Desc: "ext learns relatively fast",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.02",
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
					"Prjn.PrjnScale.Abs":    "2",    // 3 orig
					"Prjn.Learn.LRate.Base": "0.01", // 0.05 def
				}},
			{Sel: ".DrivesToVSPatch", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1", // 3 orig
				}},
			{Sel: "#OFCusPTpToVsPatch", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "5", // 3 orig
				}},
			{Sel: "#CSToBLAPosAcqD1", Desc: "",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.1", // was 0.5 -- too fast!?
				}},
			{Sel: ".SuperToThal", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "4", // 4 def
				}},
			{Sel: ".SuperToPT", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.5", // 4 def
				}},
			{Sel: "#ACCcostToACCcostMD", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3", // supertothal for us stronger
				}},
			{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "5", // with new mod, this can be stronger
				}},
			{Sel: "#UrgencyToVsMtxGo", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0", // todo: not working -- no ach by time this happens!  need to drive ach too.
				}},
		}},
	},
}
