// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package main

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "clamp gain makes big diff on overall excitation, gating propensity",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 1.0 // 1.5 is def, was 0.6 (too low)
				ly.Acts.Noise.On.SetBool(true)
				ly.Acts.Noise.Ge = 0.0001 // 0.0001 > others; could just be noise ;)
				ly.Acts.Noise.Gi = 0.0001 // 0.0001 perhaps better than others
				ly.Learn.GateSync.On.SetBool(false)
				ly.Learn.GateSync.Offset = 80
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
				ly.Acts.Kir.Gbar = 10           // 10 > 5  > 2 -- key for pause
				ly.Acts.SKCa.Gbar = 2           // 2 > 5 >> 1 (for Kir = 10)
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
				ly.Acts.Kir.Gbar = 0              // no real diff here over range 0-10
				ly.Acts.MaintNMDA.Gbar = 0.007    // 0.007 default
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.8 // 0.8 def
				ly.CT.GeGain = 0.05     // 0.05 def
				ly.CT.DecayTau = 50     // 50 def
			}},
		{Sel: ".CTLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.4 // 0.8 def
				ly.CT.GeGain = 5        // 2 def
				ly.CT.DecayTau = 50     // 50 def
			}},
		{Sel: "#MotorBS", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Pool.On.SetBool(false)
				ly.Inhib.Layer.Gi = 0.2 // 0.2 def
				ly.Acts.Clamp.Ge = 2    // 2 > 1.5, >> 1 -- absolutely critical given GPi inhib
			}},
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
				pt.Learn.LRate.Base = 0.04  // 0.04 def
				pt.Learn.Trace.CaGain = 0.7 // 0.7 for 300 cycles
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
				pt.Learn.Trace.LearnThr = 0.1   // 0.1 > 0 > 0.2
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
		{Sel: ".ToM1", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.5 // now 1.5 > 2 > 1 ..
			}},
		{Sel: "#StateToM1", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1 > 1.5, 2, 0.5 etc
			}},
		{Sel: "#DGPiToPF", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.4 // 0.4 >= 0.5, 0.3, 0.2 >> higher
			}},
		{Sel: "#MotorBSToPF", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1 > 1.1 > 0.9 >> 0.5
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
		{Sel: "#DGPiToM1VM", Doc: "final inhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2
			}},
		{Sel: "#DGPiToMotorBS", Doc: "final inhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3 // 3 > 2.5, 3.5
			}},
		{Sel: ".M1ToMotorBS", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2 > 1.5, 2.5
			}},
		{Sel: "#M1PTToMotorBS", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 2
				pt.PathScale.Rel = 1 // 1
			}},
		{Sel: "#M1PTToVL", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1   // 1
				pt.PathScale.Rel = 0.1 // 0.1 > 0.2, .05, 0
			}},
		// {Sel: "#M1PTpToMotorBS", Doc: "",
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
				ly.Acts.GabaB.Gbar =            "0",
				ly.Acts.NMDA.Gbar =             "0.006", // 0.006 default, necessary (0 very bad)
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
				ly.Acts.SKCa.Gbar =             "2",    // 2 > 5 >> 1 (for Kir = 10)
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
