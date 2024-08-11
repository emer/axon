// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
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
		{Sel: "#BLAposExtD2", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":           "1.8", // 1.8 puts just under water
				"Layer.Inhib.Pool.Gi":            "1.0",
				"Layer.Learn.NeuroMod.DAModGain": "0", // critical to be 0 -- otherwise prevents CS onset activity!
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Acts.Dend.ModGain": "1.5",
				// "Layer.Inhib.Layer.Gi":    "3.0",
				// "Layer.Inhib.Pool.Gi":     "3.6",
			}},
		{Sel: "#OFCposPT", Desc: "",
			Params: params.Params{
				"Layer.Acts.SMaint.Gbar": "0.4",
			}},
		{Sel: ".PTPredLayer", Desc: "",
			Params: params.Params{
				"Layer.CT.GeGain": "0.1", // 0.05 orig; stronger ptp
			}},
		{Sel: ".LDTLayer", Desc: "",
			Params: params.Params{
				"Layer.LDT.MaintInhib": "0.8", // 0.8 def; AChThr = 0.5 typically
			}},
		{Sel: "#OFCposPTp", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Pool.Gi":        "1",
				"Layer.Inhib.ActAvg.Nominal": "0.025", // 0.1 -- affects how strongly BLA is driven -- key param
			}},
		// {Sel: "#OFCposPT", Desc: "",
		// 	Params: params.Params{
		// 		"Layer.Acts.SMaint.Gbar": "0.4", // 0.2 def fine
		// 	}},
		{Sel: "#SC", Desc: "",
			Params: params.Params{
				"Layer.Acts.KNa.Slow.Max": "0.5", // .5 needed to shut off
			}},
		{Sel: "#CostP", Desc: "",
			Params: params.Params{
				"Layer.Pulv.DriveScale": "0.2", // 0.1 def
			}},
		//////////////////////////////////////////////////
		// current experimental settings
		{Sel: ".VSMatrixPath", Desc: "",
			Params: params.Params{
				"Path.Learn.Trace.LearnThr": "0.1", // prevents learning below this thr: preserves low act
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs": "4.0", // 4 > 2 for gating sooner
			}},
		{Sel: ".BLAExtPath", Desc: "",
			Params: params.Params{
				"Path.Learn.LRate.Base":  "0.05", // 0.05 is fine -- maybe a bit fast
				"Path.BLA.NegDeltaLRate": "1",
				"Path.PathScale.Abs":     "4",
			}},
		// {Sel: ".BLANovelInhib", Desc: "",
		// 	Params: params.Params{
		// 		"Path.PathScale.Abs": "0.2",
		// 	}},
		{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
			Params: params.Params{
				"Path.PathScale.Abs": "3", // 3 best -- 4 prevents some gating, 2 can sometimes leak
			}},
		{Sel: ".PTpToBLAExt", Desc: "modulatory so only active with -da, drives extinction learning based on maintained goal rep",
			Params: params.Params{
				"Path.PathScale.Abs": "1", // todo: expt
			}},
		{Sel: "#BLAposAcqD1ToOFCpos", Desc: "strong, high variance",
			Params: params.Params{
				"Path.PathScale.Abs": "2", // key param for OFC focusing on current cs -- expt
			}},
		{Sel: ".CSToBLApos", Desc: "",
			Params: params.Params{
				"Path.Learn.LRate.Base": "0.05",
			}},
		{Sel: ".BLAAcqToGo", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs": "4", // 4 def
			}},
		{Sel: ".BLAExtToAcq", Desc: "fixed inhibitory",
			Params: params.Params{
				"Path.PathScale.Abs": "1.0", // key param for efficacy of inhibition
			}},
		{Sel: ".VSPatchPath", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs":        "3",
				"Path.Learn.Trace.LearnThr": "0",
				"Path.Learn.LRate.Base":     "0.02", // 0.02 for vspatch -- essential to drive long enough extinction
			}},
		// {Sel: ".ToPTp", Desc: "",
		// 	Params: params.Params{
		// 		"Path.PathScale.Abs": "2",
		// 	}},
	},
}
