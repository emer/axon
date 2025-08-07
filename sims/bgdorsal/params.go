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
				ly.Acts.Clamp.Ge = 1.0         // 1.5 is def, was 0.6 (too low)
				ly.Acts.Noise.On.SetBool(true) // true >= false (minor)
				ly.Acts.Noise.Ge = 0.0001      // 0.0001 > others; could just be noise ;)
				ly.Acts.Noise.Gi = 0.0001      // 0.0001 perhaps better than others
			}},
		{Sel: ".PFCLayer", Doc: "pfc",
			Set: func(ly *axon.LayerParams) {
				// ly.Learn.NeuroMod.DAMod = axon.NoDAMod       // NoDAMod > D1Mod
				// ly.Learn.NeuroMod.DAModGain = 0.005          // 0.005 > higher
				// ly.Learn.NeuroMod.DipGain = 0                // 0 > higher
				ly.Learn.RLRate.SigmoidLinear.SetBool(false) // false >> true; orig = true

				ly.Acts.Decay.Glong = 0                   // 0 ==? 0.1; > higher
				ly.Learn.CaLearn.ETraceAct.SetBool(false) // false > true: act not beneficial
				ly.Learn.CaLearn.ETraceTau = 4            // 4 > 3?
				ly.Learn.CaLearn.ETraceScale = 0.02       // 0 == 0.02 >= 0.05 > 0.1 -- todo..

				ly.Acts.KNa.On.SetBool(true)
				ly.Acts.KNa.Med.Max = 0.2 // 0.2 > 0.1 > 0.05
				ly.Acts.KNa.Slow.Max = 0.2
				ly.Acts.Mahp.Gk = 0.05  // 0.05
				ly.Acts.Sahp.Gk = 0.05  // 0.05
				ly.Acts.Sahp.CaTau = 10 // 10 (def) > 5?

				// ly.Acts.NMDA.Tau = 100                       // 100 def >> 200
				// ly.Learn.LearnNMDA.Tau = 100                 // 100 def >> 200
			}},
		{Sel: ".DSMatrixLayer", Doc: "all matrix",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Pool.Gi = 0.5                     // 0.5 > others
				ly.Learn.NeuroMod.BurstGain = 0.1          // 0.1 > 0.2 still v53
				ly.Learn.NeuroMod.DAModGain = 0            // 0 > higher?
				ly.DSMatrix.PatchBurstGain = 1.0           // 1 > others
				ly.DSMatrix.PatchDAModGain = 0.02          // .02 > .01 > .05 > 0; 0 not that bad
				ly.DSMatrix.PatchD1Range.Set(0.1, 0.3)     // 0.3 > 0.35, .4
				ly.DSMatrix.PatchD2Range.Set(0.05, 0.25)   // 0.05, 0.25 > 0.1, 0.3
				ly.Learn.RLRate.On.SetBool(true)           // note: applied for tr update trials
				ly.Learn.RLRate.SigmoidMin = 0.001         // 0.001 >= 0.01 -- minor
				ly.Learn.TrgAvgAct.RescaleOn.SetBool(true) // true > false
			}},
		{Sel: ".DSPatchLayer", Doc: "all matrix",
			Set: func(ly *axon.LayerParams) {
				ly.Learn.NeuroMod.AChLRateMod = 1 // 1 is now default
			}},
		{Sel: ".DSTNLayer", Doc: "all STN",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.1
				ly.Acts.Kir.Gk = 10             // 10 >= 8 > 12 > 5  > 2 -- key for pause
				ly.Acts.SKCa.Gk = 2             // 2 > 1.8 >> 2.5 >> 3 >> 1 (for Kir = 10)
				ly.Acts.SKCa.CaRDecayTau = 150  // 150 >= 140 >= 160 > 180 > 200 > 130 >> 80 def -- key param!
				ly.Inhib.Layer.On.SetBool(true) // actually needs this
				ly.Inhib.Layer.Gi = 0.5         // 0.5 > 0.4 >> 0.6
				ly.Learn.NeuroMod.AChDisInhib = 0
			}},
		{Sel: "#M1VM", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Learn.NeuroMod.AChDisInhib = 0
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 2.4           // 2.4 >= 2.2, 2.6
				ly.Inhib.ActAvg.Nominal = 0.3     // 0.3 def -- key but wrong!
				ly.Acts.Decay.OnRew.SetBool(true) // true def -- seems better?
				ly.Acts.Dend.ModGain = 1.0        // 1.5 def
				ly.Acts.Kir.Gk = 0                // no real diff here over range 0-10
				ly.Acts.MaintNMDA.Ge = 0.007      // 0.007 >= 0.006 > 0.005 > 0.004 > 0.008
				ly.Acts.MaintNMDA.Tau = 200       // 200 > 250, 180
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.8 // 0.8 > 0.9, 0.7
				ly.CT.GeGain = 0.05     // 0.05 >= 0.07 > 0.03
				ly.CT.DecayTau = 100    // 100 >= 120, 80
			}},
		{Sel: ".CTLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.4 // 1.4 > 1.2 >= 1.6
				ly.CT.GeGain = 5        // 5 > 3, 8
				ly.CT.DecayTau = 100    // 100 > 120 >> 80
			}},
		{Sel: "#MotorBS", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Pool.On.SetBool(false)
				ly.Inhib.Layer.Gi = 0.2 // 0.2 > 0.3 > 0.1
				ly.Acts.Clamp.Ge = 2.0  // 2 > 2.5 > 2.2 > 1.5, >> 1 -- absolutely critical given GPi inhib
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
				ly.Acts.Init.GeBase = 0.2 // 0.2 >= 0.15 >> 0.25 0.3
				ly.Acts.Init.GeVar = 0.1  // 0.1 > 0.15 > 0.2 > 0.05
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
		{Sel: ".DSPatchPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.04 // 0.04 std best
			}},
		{Sel: ".DSMatrixPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.8      // 1.8 > others
				pt.Learn.LRate.Base = 0.02  // rlr sig: .02 > .015 .025
				pt.Learn.DWt.LearnThr = 0.1 // 0.1  > 0.2
				pt.DSMatrix.PatchDA = 0.5   // 0.5 > 0.8 >> 0.2
				pt.DSMatrix.Credit = 0.6    // key param, 0.6 > 0.5, 0.4, 0.7, 1 with pf modulation
				pt.DSMatrix.Delta = 1       // verified essential v0.2.40
				// Delta should always be 1 except for testing; adjust lrate to compensate
				pt.DSMatrix.OffTrace = 0.1      // 0.1 > 0.2, 0.5 > 0.05 > 0
				pt.DSMatrix.D2Scale = 1         // 1 >= .9, 1.1
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
				pt.PathScale.Abs = 2 // 2 > 1.6, 2.4
				// learn = false by default
			}},
		{Sel: "#DGPiToMotorBS", Doc: "final inhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3       // 3 >= 3.5 > 2.5
				pt.Learn.LRate.Base = 0.04 // 0.04 > 0.02 > 0.0005 with STN 150
				// pt.Learn.SynCaBin.Envelope = kinase.Env10
				// pt.Learn.DWt.CaPScale = 1 // tbd in Env
			}},
		{Sel: "#DGPiToPF", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.7     // 0.7 >= 0.6 >= 0.5 > lower
				pt.Learn.LRate.Base = 0.04 // 0.4 prev default
			}},
		{Sel: "#StateToM1", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1.2 >= 1 > 0.8
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
		// {Sel: ".PFToDMatrix", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		// std random sig better
		// 		// pt.SWts.Init.Mean = 0.5
		// 		// pt.SWts.Init.Var = 0.0
		// 	}},
		{Sel: ".M1ToMotorBS", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2 > 1.5, 2.5
			}},
		{Sel: "#M1ToMotorBS", Doc: "weaker; note: this is a proxy for cerebellum etc inputs",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.5 // 1.5 > 1.2, 1.8, 1, 2, 2.5 sensitive
			}},
		{Sel: "#M1PTToMotorBS", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2 > 1.6, 2.4 sensitive
				pt.PathScale.Rel = 1 // 1
				// note: lr = 0.04 in orig
			}},
		{Sel: "#M1PTToVL", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1   // 1 > 0.8, 1.2
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
		{Sel: "#DMatrixNoToDMatrixGo", Doc: "weakish no->go inhibition is beneficial",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1        // 0.1 > 0.08, 0.12 > 0.05 not too sensitive
				pt.Learn.Learn.SetBool(false) // no-learn better than learn
			}},
		{Sel: "#DMatrixGoToDGPeAk", Doc: "go inhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.6 // 0.6 > 0.5 > 0.4 > 0.7
			}},
		{Sel: "#DGPeAkToDMatrixNo", Doc: "go disinhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4 // 4 > 5 > 6 > 3 >> 2
			}},
		{Sel: "#DGPePrToDGPePr", Doc: "self-inhib -- only source of self reg",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4.5 // 4.5 >= 4 >= 4.8 >> 3.2
			}},
		{Sel: "#DGPePrToDSTN", Doc: "enough to kick off the ping-pong dynamics for STN.",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.4 // 0.4 >= 0.5 > 0.6
			}},
		// {Sel: ".StateToDMatrix", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Abs = 1.5 // 1.8 def
		// 	}},
		// {Sel: ".CLToDMatrix", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.Learn.Learn =   false
		// 		pt.PathScale.Rel = 0.001
		// 	}},
	},
}

/////////

// LayerParamsDefs has builtin default values.
var LayerParamsDefs = axon.LayerSheets{
	"Base": {
		{Sel: ".DSMatrixLayer", Doc: "all matrix",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.FB = 0
				ly.Inhib.Pool.Gi = 0.5     // 0.5 > others
				ly.Striatum.GateThr = 0.05 // .05 default
				ly.Acts.Kir.Gk = 10        // 10 > 5 > 20
				ly.Acts.GabaB.Gk = 0
				ly.Acts.NMDA.Ge = 0.006 // 0.006 default, necessary (0 very bad)
				ly.Acts.Dend.ModBase = 1
				ly.Acts.Dend.ModGain = 0          // has no effect
				ly.Learn.NeuroMod.AChLRateMod = 0 // dorsal should not use
				ly.Learn.NeuroMod.BurstGain = 0.1 // 0.1 == 0.2 > 0.05 > 0.5 -- key lrate modulator
				ly.Learn.NeuroMod.AChDisInhib = 0
			}},
		{Sel: ".DSTNLayer", Doc: "all STN",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.1
				ly.Acts.Kir.Gk = 10             // 10 > 5  > 2 -- key for pause
				ly.Acts.SKCa.Gk = 2             // 2 > 5 >> 1 (for Kir = 10)
				ly.Inhib.Layer.On.SetBool(true) // actually needs this
				ly.Inhib.Layer.Gi = 0.5
				ly.Inhib.Layer.FB = 0
				ly.Learn.NeuroMod.AChDisInhib = 0
			}},
		{Sel: "#PF", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(false)
			}},
		{Sel: "#DGPePr", Doc: "prototypical",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.4 // 0.4 > 0.3, 0.5
				ly.Acts.Init.GeVar = 0.2
			}},
		{Sel: "#DGPeAk", Doc: "arkypallidal",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.2 // 0.2 > 0.3, 0.1
				ly.Acts.Init.GeVar = 0.1  // 0.1 == 0.2 > 0.05
			}},
		{Sel: "#DGPi", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.3 // 0.3 > 0.2, 0.1
				ly.Acts.Init.GeVar = 0.1
			}},
	},
}

// PathParamsDefs are builtin default params
var PathParamsDefs = axon.PathSheets{
	"Base": {
		{Sel: "#DGPePrToDGPi", Doc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1 > 0.8, 1.2
			}},
		{Sel: "#DMatrixGoToDGPi", Doc: "go influence on gating -- slightly weaker than integrated GPePr",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1 >> 1.2, 0.8
			}},
		{Sel: "#DSTNToDGPi", Doc: "strong initial phasic activation",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = .2 // .2 >= .16 >> .24
			}},

		{Sel: "#DMatrixNoToDGPePr", Doc: "proto = primary classical NoGo pathway",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1 > 0.8, 1.2
			}},
		{Sel: "#DGPePrToDGPePr", Doc: "self-inhib -- only source of self reg",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4 // 4 >= 4.8 >> 3.2 // todo: try 4.5
			}},
		{Sel: "#DSTNToDGPePr", Doc: "stronger STN -> DGPePr to kick it high at start",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5 // 0.5 > 0.4 >> 0.6
			}},

		{Sel: "#DGPePrToDGPeAk", Doc: "just enough to knock down in baseline state",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1 > 1.2 >> 0.8
			}},
		{Sel: "#DMatrixGoToDGPeAk", Doc: "go inhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.6 // 0.6 > 0.5 > 0.4 -- todo: set 0.6 and try higher
			}},
		{Sel: "#DSTNToDGPeAk", Doc: "this is weak biologically -- but relatively sensitive..",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.1 // 0.1 > 0.12 >> 0.08
			}},

		{Sel: ".ToDSTN", Doc: "excitatory inputs",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2
			}},
		{Sel: "#DGPePrToDSTN", Doc: "enough to kick off the ping-pong dynamics for STN.",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.4 // 0.4 > 0.5 > 0.6 -- todo: try 0.4
			}},
		{Sel: "#StateToDSTN", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2 > 1.6, 2.4
			}},
		{Sel: "#S1ToDSTN", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2 >= 2.4 > 1.6
			}},

		{Sel: "#DMatrixNoToDMatrixGo", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1 > 0.8 >> 1.2
			}},
		{Sel: "#DGPeAkToDMatrixGo", Doc: "go disinhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3 // 3 > 2.4, 3.6
			}},

		{Sel: ".PFToDMatrix", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false)
				pt.Com.GType = axon.ModulatoryG
				pt.PathScale.Abs = 1
			}},
	},
}
