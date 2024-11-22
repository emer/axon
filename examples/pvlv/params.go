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
		{Sel: ".InputLayer", Doc: "",
			Params: params.Params{
				ly.Acts.Decay.Act =   "1.0",
				ly.Acts.Decay.Glong = "1.0",
			}},
		{Sel: "#CS", Doc: "expect act",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.05", // 1 / css
			}},
		{Sel: "#ContextIn", Doc: "expect act",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.025", // 1 / css
			}},
		{Sel: ".VSPatchLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Pool.Gi =              "0.5", // 0.5 needed for differentiation
				ly.Inhib.Layer.Gi =             "0.5",
				ly.Learn.NeuroMod.DipGain =     "1",    // boa requires balanced..
				ly.Learn.TrgAvgAct.GiBaseInit = "0",    // 0.5 def; 0 faster
				ly.Learn.RLRate.SigmoidMin =    "0.05", // 0.05 def
				ly.Learn.NeuroMod.AChLRateMod = "0",
			}},
		{Sel: ".VTALayer", Doc: "",
			Params: params.Params{
				ly.VTA.CeMGain = "0.5",  // 0.75 def -- controls size of CS burst
				ly.VTA.LHbGain = "1.25", // 1.25 def -- controls size of PV DA
				ly.VTA.AChThr =  "0.5",  // prevents non-CS-onset CS DA
			}},
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Params: params.Params{
				ly.Acts.Kir.Gbar =     "2",    // 10 > 5  > 2 -- key for pause
				ly.Learn.RLRate.On =   "true", // only used for non-rew trials -- key
				ly.Learn.RLRate.Diff = "false",
			}},
		{Sel: "#BLAposExtD2", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi =           "1.8", // 1.8 puts just under water
				ly.Inhib.Pool.Gi =            "1.0",
				ly.Learn.NeuroMod.DAModGain = "0", // critical to be 0 -- otherwise prevents CS onset activity!
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Params: params.Params{
				ly.Acts.Dend.ModGain = "1.5",
				// ly.Inhib.Layer.Gi =    "3.0",
				// ly.Inhib.Pool.Gi =     "3.6",
			}},
		{Sel: "#OFCposPT", Doc: "",
			Params: params.Params{
				ly.Acts.SMaint.Gbar = "0.4",
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Params: params.Params{
				ly.CT.GeGain = "0.1", // 0.05 orig; stronger ptp
			}},
		{Sel: ".LDTLayer", Doc: "",
			Params: params.Params{
				ly.LDT.MaintInhib = "0.8", // 0.8 def; AChThr = 0.5 typically
			}},
		{Sel: "#OFCposPTp", Doc: "",
			Params: params.Params{
				ly.Inhib.Pool.Gi =        "1",
				ly.Inhib.ActAvg.Nominal = "0.025", // 0.1 -- affects how strongly BLA is driven -- key param
			}},
		// {Sel: "#OFCposPT", Doc: "",
		// 	Params: params.Params{
		// 		ly.Acts.SMaint.Gbar = "0.4", // 0.2 def fine
		// 	}},
		{Sel: "#SC", Doc: "",
			Params: params.Params{
				ly.Acts.KNa.Slow.Max = "0.5", // .5 needed to shut off
			}},
		{Sel: "#CostP", Doc: "",
			Params: params.Params{
				ly.Pulv.DriveScale = "0.2", // 0.1 def
			}},
		//////////////////////////////////////////////////
		// current experimental settings
		{Sel: ".VSMatrixPath", Doc: "",
			Params: params.Params{
				pt.Learn.Trace.LearnThr = "0.1", // prevents learning below this thr: preserves low act
			}},
		{Sel: ".SuperToThal", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "4.0", // 4 > 2 for gating sooner
			}},
		{Sel: ".BLAExtPath", Doc: "",
			Params: params.Params{
				pt.Learn.LRate.Base =  "0.05", // 0.05 is fine -- maybe a bit fast
				pt.BLA.NegDeltaLRate = "1",
				pt.PathScale.Abs =     "4",
			}},
		// {Sel: ".BLANovelInhib", Doc: "",
		// 	Params: params.Params{
		// 		pt.PathScale.Abs = "0.2",
		// 	}},
		{Sel: ".GPiToBGThal", Doc: "inhibition from GPi to MD",
			Params: params.Params{
				pt.PathScale.Abs = "3", // 3 best -- 4 prevents some gating, 2 can sometimes leak
			}},
		{Sel: ".PTpToBLAExt", Doc: "modulatory so only active with -da, drives extinction learning based on maintained goal rep",
			Params: params.Params{
				pt.PathScale.Abs = "1", // todo: expt
			}},
		{Sel: "#BLAposAcqD1ToOFCpos", Doc: "strong, high variance",
			Params: params.Params{
				pt.PathScale.Abs = "2", // key param for OFC focusing on current cs -- expt
			}},
		{Sel: ".CSToBLApos", Doc: "",
			Params: params.Params{
				pt.Learn.LRate.Base = "0.05",
			}},
		{Sel: ".BLAAcqToGo", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "4", // 4 def
			}},
		{Sel: ".BLAExtToAcq", Doc: "fixed inhibitory",
			Params: params.Params{
				pt.PathScale.Abs = "1.0", // key param for efficacy of inhibition
			}},
		{Sel: ".VSPatchPath", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs =        "3",
				pt.Learn.Trace.LearnThr = "0",
				pt.Learn.LRate.Base =     "0.02", // 0.02 for vspatch -- essential to drive long enough extinction
			}},
		// {Sel: ".ToPTp", Doc: "",
		// 	Params: params.Params{
		// 		pt.PathScale.Abs = "2",
		// 	}},
	},
}
