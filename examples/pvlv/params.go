// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/netparams"
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: ".InputLayer", Desc: "",
			Params: params.Params{
				"Layer.Acts.Decay.Act":   "1.0",
				"Layer.Acts.Decay.Glong": "1.0",
			}},
		{Sel: "#CS", Desc: "expect act",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.05", // 1 / css
			}},
		{Sel: "#ContextIn", Desc: "expect act",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.025", // 1 / css
			}},
		{Sel: ".VSPatchLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Pool.Gi":              "0.5", // 0.5 needed for differentiation
				"Layer.Inhib.Layer.Gi":             "0.5",
				"Layer.Learn.NeuroMod.DipGain":     "1",    // boa requires balanced..
				"Layer.Learn.TrgAvgAct.GiBaseInit": "0",    // 0.5 def; 0 faster
				"Layer.Learn.RLRate.SigmoidMin":    "0.05", // 0.05 def
				"Layer.Learn.NeuroMod.AChLRateMod": "0",
			}},
		{Sel: ".VTALayer", Desc: "",
			Params: params.Params{
				"Layer.VTA.CeMGain": "0.5",  // 0.75 def -- controls size of CS burst
				"Layer.VTA.LHbGain": "1.25", // 1.25 def -- controls size of PV DA
				"Layer.VTA.AChThr":  "0.5",  // prevents non-CS-onset CS DA
			}},
		{Sel: ".MatrixLayer", Desc: "all mtx",
			Params: params.Params{
				"Layer.Acts.Kir.Gbar":     "2",    // 10 > 5  > 2 -- key for pause
				"Layer.Learn.RLRate.On":   "true", // only used for non-rew trials -- key
				"Layer.Learn.RLRate.Diff": "false",
			}},
		{Sel: "#BLAPosExtD2", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.8",
				"Layer.Inhib.Pool.Gi":  "1.0",
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Acts.Dend.ModGain": "1.5",
				// "Layer.Inhib.Layer.Gi":    "3.0",
				// "Layer.Inhib.Pool.Gi":     "3.6",
			}},
		{Sel: "#OFCposUSPT", Desc: "",
			Params: params.Params{
				"Layer.Acts.SMaint.Gbar": "0.4",
			}},
		{Sel: ".PTPredLayer", Desc: "",
			Params: params.Params{
				"Layer.CT.GeGain": "0.05", // stronger ptp
			}},
		{Sel: ".LDTLayer", Desc: "",
			Params: params.Params{
				"Layer.LDT.MaintInhib": "0.8", // 0.8 def; AChThr = 0.5 typically
			}},
		{Sel: "#OFCposUSPTp", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Pool.Gi":        "1",
				"Layer.Inhib.ActAvg.Nominal": "0.1", // 0.1 -- affects how strongly BLA is driven -- key param
			}},
		// {Sel: "#OFCposUSPT", Desc: "",
		// 	Params: params.Params{
		// 		"Layer.Acts.SMaint.Gbar": "0.4", // 0.2 def fine
		// 	}},
		{Sel: "#SC", Desc: "",
			Params: params.Params{
				"Layer.Acts.KNa.Slow.Max": "0.5", // .5 needed to shut off
			}},
		//////////////////////////////////////////////////
		// required custom params for this project
		{Sel: "#ContextInToBLAPosExtD2", Desc: "specific to this project",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":    "4",
				"Prjn.Learn.LRate.Base": "0.1",
			}},
		//////////////////////////////////////////////////
		// current experimental settings
		{Sel: ".VSMatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.Learn.Trace.LearnThr": "0.1", // prevents learning below this thr: preserves low act
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4.0", // 4 > 2 for gating sooner
			}},
		{Sel: ".BLAExtPrjn", Desc: "",
			Params: params.Params{
				"Prjn.Learn.LRate.Base":  "0.005", // 0.02 allows .5 CS for B50
				"Prjn.BLA.NegDeltaLRate": "1",
			}},
		{Sel: ".BLAExtToAcq", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			}},
		{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3", // 3 best -- 4 prevents some gating, 2 can sometimes leak
			}},
		{Sel: ".PTpToBLAExt", Desc: "modulatory, drives extinction learning based on maintained goal rep",
			Params: params.Params{
				// "Prjn.Learn.LRate.Base": "0.0",
				"Prjn.PrjnScale.Abs": "0.5", // todo: expt
			}},
		{Sel: "#BLAPosAcqD1ToOFCposUS", Desc: "strong, high variance",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // key param for OFC focusing on current cs -- expt
			}},
		{Sel: ".BLAAcqToGo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4", // 4 def
			}},
		{Sel: ".BLAExtToAcq", Desc: "fixed inhibitory",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.0", // key param for efficacy of inhibition
			}},
		{Sel: ".VSPatchPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":        "3",
				"Prjn.Learn.Trace.LearnThr": "0",
				"Prjn.Learn.LRate.Base":     "0.02", // 0.02 needed for vspatch test
			}},
		// {Sel: ".ToPTp", Desc: "",
		// 	Params: params.Params{
		// 		"Prjn.PrjnScale.Abs": "2",
		// 	}},
	},
}
