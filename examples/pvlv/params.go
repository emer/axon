// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/netparams"
	"github.com/emer/emergent/params"
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
				"Layer.Learn.NeuroMod.DipGain":     "1",    // boa requires balanced..
				"Layer.Learn.RLRate.SigmoidMin":    "0.01", // 0.05 def
				"Layer.VSPatch.Gain":               "3",
				"Layer.VSPatch.ThrInit":            "0.12",
				"Layer.VSPatch.ThrLRate":           "0.002", // .001",
				"Layer.VSPatch.ThrNonRew":          "10",
				"Layer.Learn.TrgAvgAct.GiBaseInit": "0.5",
			}},
		{Sel: "#BLAPosExtD2", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.8",
				"Layer.Inhib.Pool.Gi":  "1.0",
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Acts.Dend.ModGain": "1.5",
			}},
		{Sel: ".PTPredLayer", Desc: "",
			Params: params.Params{
				"Layer.CT.GeGain": "0.1", // stronger ptp
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
		{Sel: ".MatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.Matrix.NoGateLRate": "0.0", // 1 default, 0 needed b/c no actual contingency in pavlovian
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
		{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // 4 prevents some gating, 2 leaks with supertothal 4
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
		{Sel: ".BLAExtToAcq", Desc: "fixed inhibitory",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.0", // key param for efficacy of inhibition
			}},
		{Sel: ".VSPatchPrjn", Desc: "",
			Params: params.Params{ // todo: expt with these more..
				"Prjn.PrjnScale.Abs":        "2",
				"Prjn.Learn.Trace.LearnThr": "0",
				"Prjn.Learn.LRate.Base":     "0.05", // 0.2 to speed up
			}},
		// {Sel: "#OFCposUSPTToOFCposUSPT", Desc: "",
		// 	Params: params.Params{
		// 		"Prjn.PrjnScale.Abs": "5", // 4 needed to sustain
		// 	}},
		{Sel: ".ToPTp", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2",
			}},
	},
}
