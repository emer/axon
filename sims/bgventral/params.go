// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgventral

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "clamp gain makes big diff on overall excitation, gating propensity",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 1.0 // 1.5 is def, was 0.6 (too low)
			}},
		{Sel: ".VBG", Doc: "all ModACh",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Dend.ModACh.SetBool(true)
			}},
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Pool.On.SetBool(false)
				ly.Inhib.Layer.Gi = 0.5 // 0.5 needed for differentiated reps
				ly.Inhib.ActAvg.Nominal = 0.25
				ly.Striatum.GateThr = 0.05       // todo: .01 should be new default
				ly.Striatum.IsVS.SetBool(true)   // key for resetting urgency
				ly.Learn.RLRate.On.SetBool(true) // only used for non-rew trials -- key
				ly.Learn.RLRate.Diff.SetBool(false)
				ly.Learn.RLRate.SigmoidMin = 0.01 // 0.01 better than .05
				ly.Learn.TrgAvgAct.RescaleOn.SetBool(true)
				ly.Learn.NeuroMod.BurstGain = 0.1 // 1 def -- must be smaller given rew dynamics
				ly.Learn.NeuroMod.DAModGain = 0   // strongly biases the gating
			}},
		{Sel: ".VSTNLayer", Doc: "all VSTN",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.1 // todo: re-param with more stn, increase..
				ly.Acts.SKCa.CaRDecayTau = 80
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Set: func(ly *axon.LayerParams) {
				// ly.Inhib.Layer.Gi =    3.2 // 3.2 def
				ly.Acts.Dend.ModGain = 1.5 // 1.5 def
				ly.Acts.Kir.Gk = 0         // no real diff here over range 0-10
				ly.Acts.Dend.ModACh.SetBool(true)
			}},
		{Sel: ".ACC", Doc: "manipulate noise to see about integration over time",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Noise.On.SetBool(false)
				ly.Acts.Noise.Ge = 0.1  // 0.1 is visibly impactful
				ly.Acts.Noise.Gi = 0.01 // 0.01 -- if to strong, rep becomes very weak
			}},
		{Sel: "#VGPi", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.3 // 0.3 > 0.2, 0.1
				ly.Acts.Init.GeVar = 0.1
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: ".VSMatrixPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.01          // 0.01, vs .02 default
				pt.Learn.DWt.LearnThr = 0.1         // prevents learning below this thr: preserves low act
				pt.Matrix.VSRewLearn.SetBool(false) // significantly cleaner
				pt.SWts.Adapt.On.SetBool(false)     // not much diff: false >= true
			}},
		{Sel: "#UrgencyToVMatrixGo", Doc: "strong urgency factor",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // don't dilute from others
				pt.PathScale.Abs = 1
				pt.Learn.Learn.SetBool(false)
			}},
		{Sel: ".SuperToPT", Doc: "one-to-one from super",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5
			}},
		{Sel: ".SuperToThal", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3.0 // was 4
			}},
		{Sel: ".ACCToVMatrix", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.5 // 1.5 good; 1.8 causes some breakthrough
			}},
		{Sel: "#VMatrixNoToVMatrixGo", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.05
				pt.PathScale.Abs = 1
				pt.Learn.Learn.SetBool(false)
			}},
		{Sel: "#VSTNToVGPi", Doc: "strong initial phasic activation",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = .2
			}},
	},
}

/*
// ParamSetsDefs contains the full set of parameters, many of which are at default values
// and have informed the default values in the first place.
var ParamSetsDefs = params.Sets{
	"Defaults": {
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Pool.On =              "false",
				ly.Inhib.Layer.Gi =             "0.5",
				ly.Inhib.Layer.FB =             "0",
				ly.Striatum.GateThr =             "0.05", // .05 default
				ly.Acts.Kir.Gbar =              "10",   // 10 > 5 > 20
				ly.Acts.GabaB.Gk =            "0",
				ly.Acts.NMDA.Ge =             "0.006", // 0.006 default, necessary (0 very bad)
				ly.Learn.NeuroMod.AChLRateMod = "0",     // no diff here -- always ACh
				ly.Learn.NeuroMod.BurstGain =   "0.1",   // 0.1 == 0.2 > 0.05 > 0.5; only for weird rew case here; 1 def
			},
			Hypers: params.Hypers{
				ly.Learn.NeuroMod.BurstGain = {"Tweak = "-"},
				ly.Acts.Kir.Gbar =            {"Tweak = "-"},
				ly.Acts.NMDA.Ge =           {"Tweak = "-"},
				ly.Inhib.Layer.Gi =           {"Tweak = "-"},
			}},
		{Sel: ".VSTNLayer", Doc: "all VSTN",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase =           "0.1",
				ly.Acts.Kir.Gbar =              "10",   // 10 > 5  > 2 -- key for pause
				ly.Acts.SKCa.Gk =             "2",    // 2 > 5 >> 1 (for Kir = 10)
				ly.Inhib.Layer.On =             "true", // really no inhib neurons here.  all VGPePr
				ly.Learn.NeuroMod.AChDisInhib = "0",
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
				ly.Acts.Kir.Gbar =    {"Tweak = "-"},
				ly.Acts.SKCa.Gk =   {"Tweak = "-"},
			}},
		{Sel: "#VGPePr", Doc: "prototypical",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = "0.4", // 0.4 > 0.3, 0.5
				ly.Acts.Init.GeVar =  "0.2",
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
			}},
		{Sel: "#VGPeAk", Doc: "arkypallidal",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = "0.2", // 0.2 > 0.3, 0.1
				ly.Acts.Init.GeVar =  "0.1", // 0.1 == 0.2 > 0.05
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
				ly.Acts.Init.GeVar =  {"Tweak = "-"},
			}},
		{Sel: "#VGPi", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = "0.3", // 0.3 > 0.2, 0.1
				ly.Acts.Init.GeVar =  "0.1",
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
			}},
		{Sel: ".VGPeAkToVMatrix", Doc: "go disinhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = "3", // 3 >= 2, 4
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VMatrixGoToVGPeAk", Doc: "go inhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = ".5", // stronger = more binary
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPePrToVSTN", Doc: "enough to kick off the ping-pong dynamics for VSTN.",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = "0.5",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VSTNToVGPePr", Doc: "stronger VSTN -> VGPePr to kick it high at start",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = "0.5",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VSTNToVGPeAk", Doc: "this is weak biologically -- could try 0",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = "0.1",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VMatrixNoToVGPePr", Doc: "proto = primary classical NoGo pathway",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = "1", // 1 fully inhibits Pr
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPePrToVGPePr", Doc: "self-inhib -- only source of self reg",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = "4", // 4 best for DS
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPePrToVGPeAk", Doc: "just enough to knock down in baseline state",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = "1",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VMatrixGoToVGPi", Doc: "go influence on gating",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = ".2", // .1 too weak
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPePrToVGPi", Doc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = "1", // 2 is much worse.. keep at 1
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VSTNToVGPi", Doc: "strong initial phasic activation",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = ".2",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPiToACCPosVM", Doc: "final inhibition",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = "5", // needs to be very strong -- 5
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
	},
}

*/
