// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the active set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers",
			Params: params.Params{
				ly.Acts.Clamp.Ge = "1.5",
			}},
		{Sel: ".PFCLayer", Doc: "pfc layers: slower trgavgact",
			Params: params.Params{
				ly.Learn.TrgAvgAct.SynScaleRate = "0.0002", // also now set by default
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Params: params.Params{
				// ly.Inhib.Layer.Gi =             "2.4",
				// ly.Inhib.Pool.Gi =              "2.4",
				ly.Acts.Dend.ModGain =          "1.5", // 1.5; was 2 min -- reduces maint early
				ly.Learn.NeuroMod.AChDisInhib = "0.0", // not much effect here..
			}},
		{Sel: ".VSTNLayer", Doc: "all VSTN",
			Params: params.Params{
				ly.Acts.Init.GeBase =      "0.1",
				ly.Acts.Kir.Gbar =         "10", // 10 > 5  > 2 -- key for pause
				ly.Acts.SKCa.Gbar =        "2",  // 2 > 5 >> 1 (for Kir = 10)
				ly.Acts.SKCa.CaRDecayTau = "80", // 80 > 150
				// ly.Inhib.Layer.On =             "true", // really no inhib neurons here.  all VGPePr
				ly.Learn.NeuroMod.AChDisInhib = "0",
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.1",
				ly.CT.GeGain =            "0.05", // 0.05 key for stronger activity
				// ly.CT.DecayTau =                "50",
				ly.Learn.NeuroMod.AChDisInhib = "0", // 0.2, 0.5 not much diff
			}},
		{Sel: ".CS", Doc: "need to adjust Nominal for number of CSs -- now down automatically",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.1", // 0.1 for 4, divide by N/4 from there
			}},
		// {Sel: "#OFCpos", Doc: "",
		// 	Params: params.Params{
		// 		ly.Inhib.Pool.Gi = "1",
		// 	}},
		// {Sel: "#OFCposPT", Doc: "",
		// 	Params: params.Params{
		// 		ly.Inhib.Pool.Gi =        "0.5",
		// 	}},
		{Sel: "#OFCposPTp", Doc: "",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.1", // 0.1 -- affects how strongly BLA is driven -- key param
				ly.Inhib.Pool.Gi =        "1.4", // 1.4 orig
			}},
		{Sel: "#ILposPTp", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi = "1.2",
			}},
		{Sel: "#ILnegPTp", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi = "1.2",
			}},
		{Sel: "#OFCneg", Doc: "",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.1",
				// ly.Inhib.Layer.Gi =       "0.5", // weaker in general so needs to be lower
			}},
		// {Sel: "#OFCnegPT", Doc: "",
		// 	Params: params.Params{
		// 		ly.Inhib.ActAvg.Nominal = "0.2",
		// 		ly.Inhib.Pool.Gi =        "3.0",
		// 	}},
		// {Sel: "#OFCnegPTp", Doc: "",
		// 	Params: params.Params{
		// 		ly.Inhib.Pool.Gi = "1.4",
		// 	}},
		// {Sel: "#ILpos", Doc: "",
		// 	Params: params.Params{
		// 		ly.Inhib.Pool.Gi = "1",
		// 	}},
		{Sel: ".VSMatrixLayer", Doc: "vs mtx",
			Params: params.Params{
				ly.Inhib.Layer.On =           "false", // todo: explore -- could be bad for gating
				ly.Inhib.Pool.Gi =            "0.5",   // go lower, get more inhib from elsewhere?
				ly.Inhib.Pool.FB =            "0",
				ly.Acts.Dend.ModGain =        "1", // todo: 2 is default
				ly.Acts.Kir.Gbar =            "2",
				ly.Learn.NeuroMod.BurstGain = "1",
				ly.Learn.NeuroMod.DAModGain = "0",     // no bias is better!
				ly.Learn.RLRate.SigmoidMin =  "0.001", // 0.01 better than .05
			}},
		{Sel: "#BLAposAcqD1", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi = "2", // 2 fine with BLANovelInhib path
				ly.Inhib.Pool.Gi =  "1",
			}},
		{Sel: "#BLAposExtD2", Doc: "",
			Params: params.Params{
				ly.CT.GeGain = "0.5",
			}},
		{Sel: "#BLAnegAcqD2", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi = "1.2", // weaker
			}},
		{Sel: ".VSPatchLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Pool.Gi =              "0.5", // 0.5 ok?
				ly.Inhib.Pool.FB =              "0",   // only fb
				ly.Learn.NeuroMod.DipGain =     "1",   // if < 1, overshoots, more -DA
				ly.Learn.NeuroMod.BurstGain =   "1",
				ly.Learn.RLRate.SigmoidMin =    "0.01", // 0.01 > 0.05 def
				ly.Learn.TrgAvgAct.GiBaseInit = "0",    // 0.2 gets too diffuse
			}},
		{Sel: ".LDTLayer", Doc: "",
			Params: params.Params{
				ly.LDT.MaintInhib = "2.0", // 0.95 is too weak -- depends on activity..
			}},
		{Sel: "#SC", Doc: "",
			Params: params.Params{
				ly.Acts.KNa.Slow.Max = "0.8", // .8 reliable decreases -- could go higher
			}},
		////////////////////////////////////////////
		// Cortical Paths
		{Sel: ".PFCPath", Doc: "pfc path params -- more robust to long-term training",
			Params: params.Params{
				pt.Learn.Trace.SubMean = "1",    // 1 > 0 for long-term stability
				pt.Learn.LRate.Base =    "0.01", // 0.04 def; 0.02 more stable; 0.01 even more
			}},
		{Sel: ".PTtoPred", Doc: "stronger drive on pt pred",
			Params: params.Params{
				pt.PathScale.Abs = "1",
			}},
		{Sel: "#BLAposAcqD1ToOFCpos", Doc: "stronger",
			Params: params.Params{
				pt.PathScale.Abs = "1.5", // 1.5; stronger = bad later
			}},
		{Sel: "#OFCposToILpos", Doc: "stronger",
			Params: params.Params{
				pt.PathScale.Abs = "3",
			}},
		{Sel: ".USToBLAExtInhib", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "2",
			}},
		{Sel: "#ILposToPLutil", Doc: "not good to make this stronger",
			Params: params.Params{
				pt.PathScale.Abs = "1", // todo: try 3?
			}},
		{Sel: ".MToACC", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "3",
			}},
		// {Sel: ".PTSelfMaint", Doc: "",
		// 	Params: params.Params{
		// 		pt.PathScale.Abs =    "4",
		// 		pt.Learn.LRate.Base = "0.0001", // this is not a problem
		// 	}},
		////////////////////////////////////////////
		// Rubicon Paths
		{Sel: ".VSMatrixPath", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs =        "1.5", // 3 orig
				pt.Learn.Trace.LearnThr = "0.1",
				pt.Learn.LRate.Base =     "0.02", // 0.05 def
			}},
		{Sel: ".ToSC", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "2",
			}},
		{Sel: ".DrivesToMtx", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1",
			}},
		{Sel: ".BLAExtPath", Doc: "ext learns relatively fast",
			Params: params.Params{
				pt.Learn.LRate.Base = "0.05", // 0.05 > 0.02 = 0.01
			}},
		{Sel: ".BLAAcqToGo", Doc: "must dominate",
			Params: params.Params{
				pt.PathScale.Rel = "1",
				pt.PathScale.Abs = "4", // 4 > 3 > 2 for urgency early
			}},
		{Sel: ".BLAExtToAcq", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1.0", // 0.5 is min effective
			}},
		{Sel: ".CSToBLApos", Doc: "",
			Params: params.Params{
				pt.Learn.LRate.Base = "0.01", // 0.01 > 0.02 much better long term
			}},
		{Sel: ".PFCToVSMtx", Doc: "contextual, should be weaker",
			Params: params.Params{
				pt.PathScale.Rel = "0.1", // 0.1 def
				pt.PathScale.Abs = "1",   // 1.5def
			}},
		{Sel: "#OFCposToVMtxGo", Doc: "specific best go signal",
			Params: params.Params{
				pt.PathScale.Abs = "3",
			}},
		{Sel: "#ILposToVMtxGo", Doc: "specific best go signal",
			Params: params.Params{
				pt.PathScale.Abs = "3",
			}},
		{Sel: "#ACCcostToVMtxGo", Doc: "costs..",
			Params: params.Params{
				pt.PathScale.Abs = "3",
			}},
		{Sel: ".VSPatchPath", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs =        "4", // 4 > 3 > 2 -- key for rapid learning
				pt.Learn.Trace.LearnThr = "0",
				pt.Learn.LRate.Base =     "0.02", // 0.02  > 0.01
			}},
		{Sel: ".CSToBLANovelInhib", Doc: "learning rate here is critical to bootstrap & then fade",
			Params: params.Params{
				pt.Learn.LRate.Base = "0.01", // 0.01 slightly worse for Gate CS, but shows cost effects..
				// 0.02 too fast and Gate CS suffers significantly. 0.005 best for Gate CS, but inhibits costs
			}},
		{Sel: ".SuperToThal", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "4", // 4 = 3, 2 worse
			}},
		{Sel: ".SuperToPT", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "0.5", // 0.5 def
			}},
		{Sel: ".GPiToBGThal", Doc: "inhibition from GPi to MD",
			Params: params.Params{
				pt.PathScale.Abs = "5", // with new mod, this can be stronger
			}},
		{Sel: ".BLAFromNovel", Doc: "Note: this setting is overwritten in boa.go ApplyParams",
			Params: params.Params{
				pt.PathScale.Rel = "0.1", // weak rel to not dilute rest of bla paths
				pt.PathScale.Abs = "3",   // 2 is good for .CS nominal .1, but 3 needed for .03
			}},
	},
}
