// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: ".InputLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Decay.Act = 1.0
				ly.Acts.Decay.Glong = 1.0
			}},
		{Sel: "#CS", Doc: "expect act",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05 // 1 / css
			}},
		{Sel: "#ContextIn", Doc: "expect act",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.025 // 1 / css
			}},
		{Sel: ".VSPatchLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Pool.Gi = 0.5 // 0.5 needed for differentiation
				ly.Inhib.Layer.Gi = 0.5
				ly.Learn.NeuroMod.DipGain = 1     // boa requires balanced..
				ly.Learn.TrgAvgAct.GiBaseInit = 0 // 0.5 def; 0 faster
				ly.Learn.RLRate.SigmoidMin = 0.05 // 0.05 def
				ly.Learn.NeuroMod.AChLRateMod = 0
			}},
		{Sel: ".VTALayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.VTA.CeMGain = 0.5  // 0.75 def -- controls size of CS burst
				ly.VTA.LHbGain = 1.25 // 1.25 def -- controls size of PV DA
				ly.VTA.AChThr = 0.5   // prevents non-CS-onset CS DA
			}},
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Kir.Gk = 2               // 10 > 5  > 2 -- key for pause
				ly.Learn.RLRate.On.SetBool(true) // only used for non-rew trials -- key
				ly.Learn.RLRate.Diff.SetBool(false)
			}},
		{Sel: "#BLAposExtD2", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.8 // 1.8 puts just under water
				ly.Inhib.Pool.Gi = 1.0
				ly.Learn.NeuroMod.DAModGain = 0 // critical to be 0 -- otherwise prevents CS onset activity!
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Dend.ModGain = 1.5
				// ly.Inhib.Layer.Gi =    3.0
				// ly.Inhib.Pool.Gi =     3.6
			}},
		{Sel: "#OFCposPT", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.SMaint.Ge = 0.4
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.CT.GeGain = 0.1 // 0.05 orig; stronger ptp
			}},
		{Sel: ".LDTLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.LDT.MaintInhib = 0.8 // 0.8 def; AChThr = 0.5 typically
			}},
		{Sel: "#OFCposPTp", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Pool.Gi = 1
				ly.Inhib.ActAvg.Nominal = 0.025 // 0.1 -- affects how strongly BLA is driven -- key param
			}},
		// {Sel: "#OFCposPT", Doc: "",
		// 	Set: func(ly *axon.LayerParams) {
		// 		ly.Acts.SMaint.Gbar = 0.4 // 0.2 def fine
		// 	}},
		{Sel: "#SC", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.KNa.Slow.Gk = 0.5 // .5 needed to shut off
			}},
		{Sel: "#CostP", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Pulv.DriveScale = 0.2 // 0.1 def
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: ".VSMatrixPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.DWt.LearnThr = 0.1 // prevents learning below this thr: preserves low act
			}},
		{Sel: ".SuperToThal", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4.0 // 4 > 2 for gating sooner
			}},
		{Sel: ".BLAExtPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.05 // 0.05 is fine -- maybe a bit fast
				pt.BLA.NegDeltaLRate = 1
				pt.PathScale.Abs = 4
			}},
		// {Sel: ".BLANovelInhib", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Abs = 0.2
		// 	}},
		{Sel: ".GPiToBGThal", Doc: "inhibition from GPi to MD",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3 // 3 best -- 4 prevents some gating, 2 can sometimes leak
			}},
		{Sel: ".PTpToBLAExt", Doc: "modulatory so only active with -da, drives extinction learning based on maintained goal rep",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // todo: expt
			}},
		{Sel: "#BLAposAcqD1ToOFCpos", Doc: "strong, high variance",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // key param for OFC focusing on current cs -- expt
			}},
		{Sel: ".CSToBLApos", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.05
			}},
		{Sel: ".BLAAcqToGo", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4 // 4 def
			}},
		{Sel: ".BLAExtToAcq", Doc: "fixed inhibitory",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.0 // key param for efficacy of inhibition
			}},
		{Sel: ".VSPatchPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3
				pt.Learn.DWt.LearnThr = 0
				pt.Learn.LRate.Base = 0.02 // 0.02 for vspatch -- essential to drive long enough extinction
			}},
		// {Sel: ".ToPTp", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Abs = 2
		// 	}},
	},
}
