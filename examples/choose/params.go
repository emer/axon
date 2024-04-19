// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/netparams"
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the active set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: "Layer", Desc: "generic params for all layers",
			Params: params.Params{
				"Layer.Acts.Clamp.Ge": "1.5",
			}},
		{Sel: ".PFCLayer", Desc: "pfc layers: slower trgavgact",
			Params: params.Params{
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002", // also now set by default
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				// "Layer.Inhib.Layer.Gi":             "2.4",
				// "Layer.Inhib.Pool.Gi":              "2.4",
				"Layer.Acts.Dend.ModGain":          "1.5", // 2 min -- reduces maint early
				"Layer.Learn.NeuroMod.AChDisInhib": "0.0", // not much effect here..
			}},
		{Sel: ".BLALayer", Desc: "",
			Params: params.Params{
				"Layer.Learn.NeuroMod.DAModGain": "0.5",
			}},
		{Sel: ".PTPredLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal":       "0.1",
				"Layer.CT.GeGain":                  "0.05", // 0.05 key for stronger activity
				"Layer.CT.DecayTau":                "50",
				"Layer.Learn.NeuroMod.AChDisInhib": "0", // 0.2, 0.5 not much diff
			}},
		{Sel: ".CS", Desc: "need to adjust Nominal for number of CSs -- now down automatically",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.1", // 0.1 for 4, divide by N/4 from there
			}},
		// {Sel: "#OFCpos", Desc: "",
		// 	Params: params.Params{
		// 		"Layer.Inhib.Pool.Gi": "1",
		// 	}},
		// {Sel: "#OFCposPT", Desc: "",
		// 	Params: params.Params{
		// 		"Layer.Inhib.Pool.Gi":        "0.5",
		// 	}},
		{Sel: "#OFCposPTp", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.1", // 0.1 -- affects how strongly BLA is driven -- key param
				"Layer.Inhib.Pool.Gi":        "1.4",
			}},
		{Sel: "#ILposPTp", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.2",
			}},
		{Sel: "#ILnegPTp", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.2",
			}},
		{Sel: "#OFCneg", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.1",
				// "Layer.Inhib.Layer.Gi":       "0.5", // weaker in general so needs to be lower
			}},
		// {Sel: "#OFCnegPT", Desc: "",
		// 	Params: params.Params{
		// 		"Layer.Inhib.ActAvg.Nominal": "0.2",
		// 		"Layer.Inhib.Pool.Gi":        "3.0",
		// 	}},
		// {Sel: "#OFCnegPTp", Desc: "",
		// 	Params: params.Params{
		// 		"Layer.Inhib.Pool.Gi": "1.4",
		// 	}},
		// {Sel: "#ILpos", Desc: "",
		// 	Params: params.Params{
		// 		"Layer.Inhib.Pool.Gi": "1",
		// 	}},
		{Sel: ".VSMatrixLayer", Desc: "vs mtx",
			Params: params.Params{
				"Layer.Inhib.Layer.On":           "false", // todo: explore -- could be bad for gating
				"Layer.Inhib.Pool.Gi":            "0.5",   // go lower, get more inhib from elsewhere?
				"Layer.Inhib.Pool.FB":            "0",
				"Layer.Acts.Dend.ModGain":        "1", // todo: 2 is default
				"Layer.Acts.Kir.Gbar":            "2",
				"Layer.Learn.NeuroMod.BurstGain": "1",
				"Layer.Learn.NeuroMod.DAModGain": "0",     // no bias is better!
				"Layer.Learn.RLRate.SigmoidMin":  "0.001", // 0.01 better than .05
			}},
		{Sel: "#BLAposAcqD1", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "2.4", // 2.2 not enough to knock out novelty
				"Layer.Inhib.Pool.Gi":  "1",
			}},
		{Sel: "#BLAnegAcqD2", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.2", // weaker
			}},
		{Sel: ".VSPatchLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Pool.Gi":              "0.5", // 0.5 ok?
				"Layer.Inhib.Pool.FB":              "0",   // only fb
				"Layer.Learn.NeuroMod.DipGain":     "1",   // if < 1, overshoots, more -DA
				"Layer.Learn.NeuroMod.BurstGain":   "1",
				"Layer.Learn.RLRate.SigmoidMin":    "0.01", // 0.01 > 0.05 def
				"Layer.Learn.TrgAvgAct.GiBaseInit": "0",    // 0.2 gets too diffuse
			}},
		{Sel: ".LDTLayer", Desc: "",
			Params: params.Params{
				"Layer.LDT.MaintInhib": "2.0", // 0.95 is too weak -- depends on activity..
			}},
		{Sel: "#SC", Desc: "",
			Params: params.Params{
				"Layer.Acts.KNa.Slow.Max": "0.8", // .8 reliable decreases -- could go higher
			}},
		////////////////////////////////////////////
		// Cortical Prjns
		{Sel: ".PFCPrjn", Desc: "pfc prjn params -- more robust to long-term training",
			Params: params.Params{
				"Prjn.Learn.Trace.SubMean": "1",    // 1 > 0 for long-term stability
				"Prjn.Learn.LRate.Base":    "0.01", // 0.04 def; 0.02 more stable; 0.01 even more
			}},
		{Sel: ".PTtoPred", Desc: "stronger drive on pt pred",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			}},
		{Sel: "#BLAposAcqD1ToOFCpos", Desc: "stronger",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.5", // stronger = bad later
			}},
		{Sel: "#OFCposToILpos", Desc: "stronger",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3",
			}},
		{Sel: ".USToBLAExtInhib", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2",
			}},
		{Sel: "#ILposToPLutil", Desc: "not good to make this stronger",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // todo: try 3?
			}},
		{Sel: ".MToACC", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3",
			}},
		// {Sel: ".PTSelfMaint", Desc: "",
		// 	Params: params.Params{
		// 		"Prjn.PrjnScale.Abs":    "4",
		// 		"Prjn.Learn.LRate.Base": "0.0001", // this is not a problem
		// 	}},
		////////////////////////////////////////////
		// Rubicon Prjns
		{Sel: ".VSMatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":        "1.5", // 3 orig
				"Prjn.Learn.Trace.LearnThr": "0.1",
				"Prjn.Learn.LRate.Base":     "0.01", // 0.05 def
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
				"Prjn.Learn.LRate.Base": "0.005",
			}},
		{Sel: ".BLAAcqToGo", Desc: "must dominate",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "1",
				"Prjn.PrjnScale.Abs": "4",
			}},
		{Sel: ".BLAExtToAcq", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5", // note: key param -- 0.5 > 1
			}},
		{Sel: ".PFCToVSMtx", Desc: "contextual, should be weaker",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.1", // 0.1 def
			}},
		{Sel: ".VSPatchPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":        "3", // 4 > 3 > 2 -- key for rapid learning
				"Prjn.Learn.Trace.LearnThr": "0",
				"Prjn.Learn.LRate.Base":     "0.02", // 0.02 needed in test
			}},
		{Sel: "#CSToBLAposAcqD1", Desc: "",
			Params: params.Params{
				"Prjn.Learn.LRate.Base": "0.1", // was 0.5 -- too fast!?
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4", // 4 def
			}},
		{Sel: ".SuperToPT", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5", // 0.5 def
			}},
		{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "5", // with new mod, this can be stronger
			}},
		{Sel: ".BLAFromNovel", Desc: "Note: this setting is overwritten in boa.go ApplyParams",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.1", // weak rel to not dilute rest of bla prjns
				"Prjn.PrjnScale.Abs": "3",   // 2 is good for .CS nominal .1, but 3 needed for .03
			}},
	},
}