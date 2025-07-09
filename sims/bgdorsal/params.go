// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package bgdorsal

import (
	"github.com/emer/axon/v2/axon"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "clamp gain makes big diff on overall excitation, gating propensity",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 1.0 // 1.5 is def, was 0.6 (too low)
				ly.Acts.Noise.On.SetBool(true)
				ly.Acts.Noise.Ge = 0.0001                    // 0.0001 > others; could just be noise ;)
				ly.Acts.Noise.Gi = 0.0001                    // 0.0001 perhaps better than others
				ly.Learn.RLRate.SigmoidLinear.SetBool(false) // false >> true; orig = true
			}},
		{Sel: ".PFCLayer", Doc: "pfc",
			Set: func(ly *axon.LayerParams) {
				ly.Learn.NeuroMod.DAMod = axon.NoDAMod       // NoDAMod > D1Mod
				ly.Learn.NeuroMod.DAModGain = 0.005          // 0.005 > higher
				ly.Learn.NeuroMod.DipGain = 0                // 0 > higher
				ly.Learn.RLRate.SigmoidLinear.SetBool(false) // false >> true; orig = true

				ly.Acts.Decay.Glong = 0.3                 // def 0
				ly.Learn.CaLearn.ETraceAct.SetBool(false) // false > true: act not beneficial
				ly.Learn.CaLearn.ETraceTau = 4            // 4 > 3?
				ly.Learn.CaLearn.ETraceScale = 0          // 0 > 0.02 > higher: not useful overall

				ly.Acts.KNa.On.SetBool(true)
				ly.Acts.KNa.Med.Max = 0.2 // 0.2 > 0.05
				ly.Acts.KNa.Slow.Max = 0.2
				ly.Acts.Mahp.Gk = 0.02  // 0.02 def; 0.05 might compensate for lack of KNa?
				ly.Acts.Sahp.Gk = 0.05  // 0.05 def
				ly.Acts.Sahp.CaTau = 10 // 10 (def) > 5?

				// ly.Acts.NMDA.Tau = 100                       // 100 def >> 200
				// ly.Learn.LearnNMDA.Tau = 100                 // 100 def >> 200
			}},
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Pool.Gi = 0.5                     // 0.5 > others
				ly.Learn.NeuroMod.BurstGain = 0.1          // 0.1 == 0.2 > 0.05 > 0.5 -- key lrate modulator
				ly.Learn.NeuroMod.DAModGain = 0.2          // 0.2 >= 0.5 (orig) > 0
				ly.Learn.RLRate.On.SetBool(true)           // note: applied for tr update trials
				ly.Learn.TrgAvgAct.RescaleOn.SetBool(true) // true > false
			}},
		{Sel: ".DSTNLayer", Doc: "all STN",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.1
				ly.Acts.Kir.Gk = 10             // 10 > 5  > 2 -- key for pause
				ly.Acts.SKCa.Gk = 2             // 2 > 5 >> 1 (for Kir = 10)
				ly.Acts.SKCa.CaRDecayTau = 150  // 150 > 180 > 200 > 130 >> 80 def -- key param!
				ly.Inhib.Layer.On.SetBool(true) // actually needs this
				ly.Inhib.Layer.Gi = 0.5
				ly.Learn.NeuroMod.AChDisInhib = 0
			}},
		{Sel: "#M1VM", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Learn.NeuroMod.AChDisInhib = 0
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 2.4           // 2.4 def > 1.4
				ly.Inhib.ActAvg.Nominal = 0.3     // 0.3 def -- key but wrong!
				ly.Acts.Decay.OnRew.SetBool(true) // true def -- seems better?
				ly.Acts.Dend.ModGain = 1.0        // 1.5 def
				ly.Acts.Kir.Gk = 0                // no real diff here over range 0-10
				ly.Acts.MaintNMDA.Ge = 0.007      // 0.007 default
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.8 // 0.8 def
				ly.CT.GeGain = 0.05     // 0.05 def
				ly.CT.DecayTau = 100    // was 100 -- 50 in orig -- OFCposPT ??
			}},
		{Sel: ".CTLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.4 // 0.8 def
				ly.CT.GeGain = 5        // 2 def
				ly.CT.DecayTau = 100    // was 100 -- 50 in orig -- OFCposPT ??
			}},
		{Sel: "#MotorBS", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Pool.On.SetBool(false)
				ly.Inhib.Layer.Gi = 0.2 // 0.2 def
				ly.Acts.Clamp.Ge = 2    // 2 > 1.5, >> 1 -- absolutely critical given GPi inhib
				// ly.Learn.RLRate.Diff.SetBool(false) // true > false
				// ly.Learn.RLRate.SigmoidLinear.SetBool(false) // false >> true; orig = true
				// ly.Learn.RLRate.SigmoidMin = 0.05 // 0.05 def > 0.1 > 0.2 > 0.02
			}},
		// {Sel: "#M1", Doc: "",
		// 	Set: func(ly *axon.LayerParams) {
		// 		ly.Learn.NeuroMod.DAMod = axon.D1Mod // not good here.
		// 		ly.Learn.NeuroMod.DAModGain = 0.03   // up to 0.04 good
		// 		ly.Learn.NeuroMod.DipGain = 0.1      // 0.1 > 0 > 0.2
		// 	}},
		{Sel: "#DGPeAk", Doc: "arkypallidal",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.2 // 0.2 > 0.3, 0.1
				ly.Acts.Init.GeVar = 0.1  // 0.1 == 0.2 > 0.05
			}},
	},
	"NoiseOff": {
		{Sel: "Layer", Doc: "turn off noise",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Noise.On.SetBool(false)
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.04         // 0.04 > 0.03
				pt.Learn.DWt.SynTraceTau = 1       // 1 > 2
				pt.Learn.DWt.CaPScale = 1.05       // 1.05 > 1 > 1.1
				pt.Learn.DWt.SynCa20.SetBool(true) // 20 > 10
				pt.SWts.Adapt.HiMeanDecay = 0.0008 // 0.0008 for 4x6, 0.005 for 3x10 -- not clear if real..
				pt.Learn.DWt.SubMean = 0           // 0 >> 1 -- fails at 1
				pt.Learn.DWt.LearnThr = 0          // 0 > .1
			}},
		// {Sel: ".PFCPath", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.Learn.DWt.CaPScale = 1
		// 		pt.Learn.SynCaBin.Envelope = kinase.Env30
		// 	}},
		{Sel: ".CTCtxtPath", Doc: "all CT context paths",
			Set: func(pt *axon.PathParams) {
				pt.Learn.DWt.LearnThr = 0
			}},
		{Sel: ".CTtoPred", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 1 def
			}},
		{Sel: ".PTtoPred", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // was 6
			}},
		{Sel: ".DSMatrixPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.8          // 1.8 > others
				pt.Learn.LRate.Base = 0.02      // rlr sig: .02 > .015 .025
				pt.Learn.DWt.LearnThr = 0.1     // 0.1  > 0.2
				pt.Matrix.Credit = 0.6          // key param, 0.6 > 0.5, 0.4, 0.7, 1 with pf modulation
				pt.Matrix.BasePF = 0.005        // 0.005 > 0.01, 0.002 etc
				pt.Matrix.Delta = 1             // should always be 1 except for testing; adjust lrate to compensate
				pt.SWts.Adapt.On.SetBool(false) // false > true here
			}},
		{Sel: ".SuperToPT", Doc: "one-to-one from super",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5
			}},
		{Sel: ".PTSelfMaint", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1 def
			}},
		{Sel: ".SuperToThal", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3.0 // 3
			}},
		// {Sel: ".FmState", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.5 // abs, rel < 1 worse
		// 	}},
		{Sel: ".ToM1", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.5     // now 1.5 > 2 > 1 ..
				pt.Learn.LRate.Base = 0.04 // 0.04 > 0.02
			}},
		{Sel: ".ToMotor", Doc: "all excitatory paths to MotorBS; see #DGPiToMotorBS too",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.02 // 0.02 > 0.04 > 0.01 -- still key
				// note: MotorBS is a target, key for learning; SWts not used.
				// pt.Learn.SynCaBin.Envelope = kinase.Env10
				// pt.Learn.DWt.CaPScale = 1 // tbd in Env
			}},
		{Sel: ".VLM1", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.02 // 0.02 > 0.04 > 0.01 -- still key
				// note: VL is a target layer; SWts not used.
				// pt.Learn.SynCaBin.Envelope = kinase.Env10
				// pt.Learn.DWt.CaPScale = 1 // tbd in Env
			}},
		{Sel: "#DGPiToM1VM", Doc: "final inhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2
				// learn = false by default
			}},
		{Sel: "#DGPiToMotorBS", Doc: "final inhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3       // 3 > 2.5, 3.5
				pt.Learn.LRate.Base = 0.04 // 0.04 > 0.02 > 0.0005 with STN 150
				// pt.Learn.SynCaBin.Envelope = kinase.Env10
				// pt.Learn.DWt.CaPScale = 1 // tbd in Env
			}},
		{Sel: "#DGPiToPF", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.4     // 0.4 >= 0.5, 0.3, 0.2 >> higher
				pt.Learn.LRate.Base = 0.04 // 0.4 prev default
			}},
		{Sel: "#StateToM1", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1 > 1.5, 2, 0.5 etc
			}},
		{Sel: "#MotorBSToPF", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1       // 1 > 1.1 > 0.9 >> 0.5
				pt.Learn.LRate.Base = 0.04 // 0.04 > 0.02
				// fixed is not better:
				// pt.Learn.Learn.SetBool(false)
				// pt.SWts.Init.SPct = 0
				// pt.SWts.Init.Mean = 0.8
				// pt.SWts.Init.Var = 0.0
			}},
		{Sel: ".M1ToMotorBS", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2 > 1.5, 2.5
			}},
		{Sel: "#M1PTToMotorBS", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2
				pt.PathScale.Rel = 1 // 1
				// note: lr = 0.04 in orig
			}},
		{Sel: "#M1PTToVL", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1   // 1
				pt.PathScale.Rel = 0.1 // 0.1 > 0.2, .05, 0
			}},
		{Sel: "#M1PTToM1PT", Doc: "self path",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.0001 // 0.0001 > .04 but not a major diff
			}},
		// {Sel: "#M1PTpToMotorBS", Doc: "not used",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Abs = 2
		// 		pt.PathScale.Rel = 1
		// 	}},
		{Sel: "#M1ToMotorBS", Doc: "weaker; note: this is a proxy for cerebellum etc inputs",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.5 // 1.5 > 1, 2, 2.5
			}},
		{Sel: "#DMtxNoToDMtxGo", Doc: "weakish no->go inhibition is beneficial",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1        // 0.1 > 0.05
				pt.Learn.Learn.SetBool(false) // no-learn better than learn
			}},
		{Sel: "#DGPeAkToDMtxNo", Doc: "go disinhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 6
			}},
		// {Sel: ".StateToDMtx", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Abs = 1.5 // 1.8 def
		// 	}},
		// {Sel: ".CLToDMtx", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.Learn.Learn =   false
		// 		pt.PathScale.Rel = 0.001
		// 	}},
	},
}

/////////

/*

// ParamSetsDefs contains the full set of parameters, many of which are at default values
// and have informed the default values in the first place.
var ParamSetsDefs = params.Sets{
	"Defaults": {
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Params: params.Params{
				ly.Inhib.Layer.On =             "true",
				ly.Inhib.Pool.On =              "true",
				ly.Inhib.Pool.FB =              "0",
				ly.Inhib.Pool.Gi =              "0.5",  // 0.5 > others
				ly.Matrix.GateThr =             "0.05", // .05 default
				ly.Acts.Kir.Gbar =              "10",   // 10 > 5 > 20
				ly.Acts.GabaB.Gk =            "0",
				ly.Acts.NMDA.Ge =             "0.006", // 0.006 default, necessary (0 very bad)
				ly.Acts.Dend.ModBase =          "1",
				ly.Acts.Dend.ModGain =          "0",   // has no effect
				ly.Learn.NeuroMod.AChLRateMod = "0",   // dorsal should not use
				ly.Learn.NeuroMod.BurstGain =   "0.1", // 0.1 == 0.2 > 0.05 > 0.5 -- key lrate modulator
				ly.Learn.NeuroMod.AChDisInhib = "0",
			},
		{Sel: ".DSTNLayer", Doc: "all STN",
			Params: params.Params{
				ly.Acts.Init.GeBase =           "0.1",
				ly.Acts.Kir.Gbar =              "10",   // 10 > 5  > 2 -- key for pause
				ly.Acts.SKCa.Gk =             "2",    // 2 > 5 >> 1 (for Kir = 10)
				ly.Inhib.Layer.On =             "true", // actually needs this
				ly.Inhib.Layer.Gi =             "0.5",
				ly.Inhib.Layer.FB =             "0",
				ly.Learn.NeuroMod.AChDisInhib = "0",
			},
		{Sel: "#PF", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.On = "false",
				ly.Inhib.Pool.On =  "false",
			}},
		{Sel: "#DGPePr", Doc: "prototypical",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.4", // 0.4 > 0.3, 0.5
				ly.Acts.Init.GeVar =  "0.2",
			},
		{Sel: "#DGPeAk", Doc: "arkypallidal",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.2", // 0.2 > 0.3, 0.1
				ly.Acts.Init.GeVar =  "0.1", // 0.1 == 0.2 > 0.05
			},
		{Sel: "#DGPi", Doc: "",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.3", // 0.3 > 0.2, 0.1
				ly.Acts.Init.GeVar =  "0.1",
			},
		{Sel: "#DGPeAkToDMtxGo", Doc: "go disinhibition",
			Params: params.Params{
				pt.PathScale.Abs = "3",
			},
		{Sel: "#DMtxGoToDGPeAk", Doc: "go inhibition",
			Params: params.Params{
				pt.PathScale.Abs = ".5", // stronger = more binary
			},
		{Sel: "#DMtxNoToDGPePr", Doc: "proto = primary classical NoGo pathway",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 fully inhibits Pr
			},
		{Sel: ".ToDSTN", Doc: "excitatory inputs",
			Params: params.Params{
				pt.PathScale.Abs = "1",
			},
		{Sel: "#DMtxGoToDGPi", Doc: "go influence on gating -- slightly weaker than integrated GPePr",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 > 1.1, .9 and lower (not huge diffs)
			},
		{Sel: "#DGPePrToDSTN", Doc: "enough to kick off the ping-pong dynamics for STN.",
			Params: params.Params{
				pt.PathScale.Abs = "0.5",
			},
		{Sel: "#DSTNToDGPePr", Doc: "stronger STN -> DGPePr to kick it high at start",
			Params: params.Params{
				pt.PathScale.Abs = "0.5",
			},
		{Sel: "#DSTNToDGPeAk", Doc: "this is weak biologically -- could try 0",
			Params: params.Params{
				pt.PathScale.Abs = "0.1",
			},
		{Sel: "#DGPePrToDGPePr", Doc: "self-inhib -- only source of self reg",
			Params: params.Params{
				pt.PathScale.Abs = "4",
			},
		{Sel: "#DGPePrToDGPeAk", Doc: "just enough to knock down in baseline state",
			Params: params.Params{
				pt.PathScale.Abs = "1",
			},
		{Sel: "#DGPePrToDGPi", Doc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 >> 2
			},
		{Sel: "#DSTNToDGPi", Doc: "strong initial phasic activation",
			Params: params.Params{
				pt.PathScale.Abs = ".2",
			},
		{Sel: ".PFToDMtx", Doc: "",
			Params: params.Params{
				pt.Learn.Learn =   "false",
				pt.Com.GType =     "ModulatoryG",
				pt.PathScale.Abs = "1",
			}},
	},
}

*/
