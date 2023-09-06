// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/netparams"
	"github.com/emer/emergent/params"
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
				"Layer.Inhib.Layer.Gi":             "2.4",
				"Layer.Inhib.Pool.Gi":              "2.4",
				"Layer.Acts.Dend.ModGain":          "1.5", // 2 min -- reduces maint early
				"Layer.Learn.NeuroMod.AChDisInhib": "0",   // todo: explore!  might be bad..
			}},
		{Sel: ".PTPredLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.1",
				"Layer.CT.GeGain":            "0.05", // 0.05 key for stronger activity
				"Layer.CT.DecayTau":          "50",
				// 		"Layer.Learn.NeuroMod.AChDisInhib": "0", // todo: explore!  might be bad..
			}},
		{Sel: ".CS", Desc: "need to adjust Nominal for number of CSs -- now down automatically",
			Params: params.Params{
				// "Layer.Inhib.Layer.Gi":  "1.0", // 1.0 for CSP, 0.9 for CS -- should be higher for CSPerDrive > 1
				"Layer.Inhib.ActAvg.Nominal": "0.1", // 0.1 for 4, divide by N/4 from there
			}},
		{Sel: "#OFCposUS", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Pool.Gi": "1",
			}},
		{Sel: "#OFCposUSPT", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.2",
				"Layer.Inhib.Pool.Gi":        "3.0",
			}},
		{Sel: "#OFCposUSPTp", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Pool.Gi": "1.4",
			}},
		{Sel: "#OFCposVal", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Pool.Gi": "1",
			}},
		{Sel: ".MatrixLayer", Desc: "all mtx",
			Params: params.Params{
				"Layer.Inhib.Layer.On":    "false", // todo: explore -- could be bad for gating
				"Layer.Inhib.Pool.Gi":     "0.3",   // go lower, get more inhib from elsewhere?
				"Layer.Inhib.Pool.FB":     "1",
				"Layer.Acts.Dend.ModGain": "1", // todo: try with lower drive
			}},
		{Sel: "#BLAPosAcqD1", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "2.4", // 2.2 not enough to knock out novelty
				"Layer.Inhib.Pool.Gi":  "1",
			}},
		{Sel: ".VSPatchLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Pool.Gi":              "0.5",   // 0.5 ok?
				"Layer.Inhib.Pool.FB":              "0",     // only fb
				"Layer.Learn.NeuroMod.DipGain":     "1",     // if < 1, overshoots, more -DA
				"Layer.Learn.RLRate.SigmoidMin":    "0.01",  // 0.05 def
				"Layer.VSPatch.Gain":               "3",     // 3 smoother than higher
				"Layer.VSPatch.ThrInit":            "0.2",   // could be a bit lower
				"Layer.VSPatch.ThrLRate":           "0.002", // 0.002 good
				"Layer.VSPatch.ThrNonRew":          "10",    // 10 to prevent creeping up NR
				"Layer.Learn.TrgAvgAct.GiBaseInit": "0.5",
			}},
		{Sel: ".LDTLayer", Desc: "",
			Params: params.Params{
				"Layer.LDT.MaintInhib": "2.0", // 0.95 is too weak -- depends on activity..
			}},
		{Sel: "#SC", Desc: "",
			Params: params.Params{
				"Layer.Acts.KNa.Slow.Max": "0.05", // .1 still enough to shut off -- was .2
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
				"Prjn.PrjnScale.Abs": "6",
			}},
		{Sel: "#BLAPosAcqD1ToOFCposUS", Desc: "stronger",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.5", // stronger = bad later
			}},
		{Sel: "#OFCposUSToOFCposVal", Desc: "stronger",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3",
			}},
		{Sel: "#ACCnegValToACCutil", Desc: "stronger",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3", // fairly sensitive to this param..
			}},
		{Sel: "#OFCposValToACCutil", Desc: "not good to make this stronger",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			}},
		{Sel: ".PTSelfMaint", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":    "4",
				"Prjn.Learn.LRate.Base": "0.0001", // this is not a problem
			}},
		{Sel: ".ToPTp", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4",
			}},
		////////////////////////////////////////////
		// PVLV Prjns
		{Sel: ".BLAFromNovel", Desc: "must be strong enough to compete with CS at start -- now done automatically",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3", // 2 is good for .CS nominal .1, but 3 needed for .03
			}},
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
				"Prjn.Learn.LRate.Base": "0.005",
			}},
		{Sel: ".BLAAcqToGo", Desc: "must dominate",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "1",
				"Prjn.PrjnScale.Abs": "3",
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
				"Prjn.PrjnScale.Abs":    "2",    // 3 orig
				"Prjn.Learn.LRate.Base": "0.05", // 0.05 def
			}},
		{Sel: ".DrivesToVSPatch", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 3 orig
			}},
		{Sel: "#OFCposUSPTpToVsPatch", Desc: "",
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
				"Prjn.PrjnScale.Abs": "0.5", // 0.5 def
			}},
		{Sel: "#ACCnegValToACCnegValMD", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3", // supertothal for us stronger
			}},
		{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "5", // with new mod, this can be stronger
			}},
		{Sel: "#UrgencyToVsMtxGo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4", // 4 good -- 1,2 too weak
			}},
	},
}
