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
		{Sel: "Layer", Doc: "clamp gain makes big diff on overall excitation, gating propensity",
			Params: params.Params{
				ly.Acts.Clamp.Ge = "1.0", // 1.5 is def, was 0.6 (too low)
			}},
		{Sel: ".VBG", Doc: "all ModACh",
			Params: params.Params{
				ly.Acts.Dend.ModACh = "true",
			}},
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Params: params.Params{
				ly.Inhib.Pool.On =             "false",
				ly.Inhib.Layer.Gi =            "0.5", // 0.5 needed for differentiated reps
				ly.Inhib.ActAvg.Nominal =      "0.25",
				ly.Matrix.GateThr =            "0.05", // todo: .01 should be new default
				ly.Matrix.IsVS =               "true", // key for resetting urgency
				ly.Learn.RLRate.On =           "true", // only used for non-rew trials -- key
				ly.Learn.RLRate.Diff =         "false",
				ly.Learn.RLRate.SigmoidMin =   "0.01", // 0.01 better than .05
				ly.Learn.TrgAvgAct.RescaleOn = "true",
				ly.Learn.NeuroMod.BurstGain =  "0.1", // 1 def -- must be smaller given rew dynamics
				ly.Learn.NeuroMod.DAModGain =  "0",   // strongly biases the gating
			}},
		{Sel: ".VSTNLayer", Doc: "all VSTN",
			Params: params.Params{
				ly.Acts.Init.GeBase =      "0.1", // todo: re-param with more stn, increase..
				ly.Acts.SKCa.CaRDecayTau = "80",
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Params: params.Params{
				// ly.Inhib.Layer.Gi =    "3.2", // 3.2 def
				ly.Acts.Dend.ModGain = "1.5", // 1.5 def
				ly.Acts.Kir.Gbar =     "0",   // no real diff here over range 0-10
				ly.Acts.Dend.ModACh =  "true",
			}},
		{Sel: ".ACC", Doc: "manipulate noise to see about integration over time",
			Params: params.Params{
				ly.Acts.Noise.On = "false",
				ly.Acts.Noise.Ge = "0.1",  // 0.1 is visibly impactful
				ly.Acts.Noise.Gi = "0.01", // 0.01 -- if to strong, rep becomes very weak
			}},
		////////////////////////////////////////////
		// Paths
		{Sel: ".VSMatrixPath", Doc: "",
			Params: params.Params{
				pt.Learn.LRate.Base =     "0.01",  // 0.01, vs .02 default
				pt.Learn.Trace.LearnThr = "0.1",   // prevents learning below this thr: preserves low act
				pt.Matrix.VSRewLearn =    "false", // significantly cleaner
				pt.SWts.Adapt.On =        "false", // not much diff: false >= true
			},
			Hypers: params.Hypers{
				pt.Learn.LRate.Base =     {"Tweak = "-"},
				pt.Learn.Trace.LearnThr = {"Tweak = "-"},
			}},
		{Sel: "#UrgencyToVMtxGo", Doc: "strong urgency factor",
			Params: params.Params{
				pt.PathScale.Rel = "0.1", // don't dilute from others
				pt.PathScale.Abs = "0",   // todo: is misbehaving here
				pt.Learn.Learn =   "false",
			}},
		{Sel: ".SuperToPT", Doc: "one-to-one from super",
			Params: params.Params{
				pt.PathScale.Abs = "0.5",
			}},
		{Sel: ".SuperToThal", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "3.0", // was 4
			}},
		{Sel: ".ACCToVMtx", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1.5", // 1.5 good; 1.8 causes some breakthrough
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VMtxNoToVMtxGo", Doc: "",
			Params: params.Params{
				pt.PathScale.Rel = "0.05",
				pt.PathScale.Abs = "1",
				pt.Learn.Learn =   "false",
			},
			Hypers: params.Hypers{
				pt.PathScale.Rel = {"Tweak = "log"},
			}},
		{Sel: "#VGPi", Doc: "",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.3", // 0.3 > 0.2, 0.1
				ly.Acts.Init.GeVar =  "0.1",
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
			}},
		{Sel: "#VSTNToVGPi", Doc: "strong initial phasic activation",
			Params: params.Params{
				pt.PathScale.Abs = ".2",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
	},
}

// ParamSetsDefs contains the full set of parameters, many of which are at default values
// and have informed the default values in the first place.
var ParamSetsDefs = params.Sets{
	"Defaults": {
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Params: params.Params{
				ly.Inhib.Pool.On =              "false",
				ly.Inhib.Layer.Gi =             "0.5",
				ly.Inhib.Layer.FB =             "0",
				ly.Matrix.GateThr =             "0.05", // .05 default
				ly.Acts.Kir.Gbar =              "10",   // 10 > 5 > 20
				ly.Acts.GabaB.Gbar =            "0",
				ly.Acts.NMDA.Gbar =             "0.006", // 0.006 default, necessary (0 very bad)
				ly.Learn.NeuroMod.AChLRateMod = "0",     // no diff here -- always ACh
				ly.Learn.NeuroMod.BurstGain =   "0.1",   // 0.1 == 0.2 > 0.05 > 0.5; only for weird rew case here; 1 def
			},
			Hypers: params.Hypers{
				ly.Learn.NeuroMod.BurstGain = {"Tweak = "-"},
				ly.Acts.Kir.Gbar =            {"Tweak = "-"},
				ly.Acts.NMDA.Gbar =           {"Tweak = "-"},
				ly.Inhib.Layer.Gi =           {"Tweak = "-"},
			}},
		{Sel: ".VSTNLayer", Doc: "all VSTN",
			Params: params.Params{
				ly.Acts.Init.GeBase =           "0.1",
				ly.Acts.Kir.Gbar =              "10",   // 10 > 5  > 2 -- key for pause
				ly.Acts.SKCa.Gbar =             "2",    // 2 > 5 >> 1 (for Kir = 10)
				ly.Inhib.Layer.On =             "true", // really no inhib neurons here.  all VGPePr
				ly.Learn.NeuroMod.AChDisInhib = "0",
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
				ly.Acts.Kir.Gbar =    {"Tweak = "-"},
				ly.Acts.SKCa.Gbar =   {"Tweak = "-"},
			}},
		{Sel: "#VGPePr", Doc: "prototypical",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.4", // 0.4 > 0.3, 0.5
				ly.Acts.Init.GeVar =  "0.2",
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
			}},
		{Sel: "#VGPeAk", Doc: "arkypallidal",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.2", // 0.2 > 0.3, 0.1
				ly.Acts.Init.GeVar =  "0.1", // 0.1 == 0.2 > 0.05
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
				ly.Acts.Init.GeVar =  {"Tweak = "-"},
			}},
		{Sel: "#VGPi", Doc: "",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.3", // 0.3 > 0.2, 0.1
				ly.Acts.Init.GeVar =  "0.1",
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
			}},
		{Sel: ".VGPeAkToVMtx", Doc: "go disinhibition",
			Params: params.Params{
				pt.PathScale.Abs = "3", // 3 >= 2, 4
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VMtxGoToVGPeAk", Doc: "go inhibition",
			Params: params.Params{
				pt.PathScale.Abs = ".5", // stronger = more binary
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPePrToVSTN", Doc: "enough to kick off the ping-pong dynamics for VSTN.",
			Params: params.Params{
				pt.PathScale.Abs = "0.5",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VSTNToVGPePr", Doc: "stronger VSTN -> VGPePr to kick it high at start",
			Params: params.Params{
				pt.PathScale.Abs = "0.5",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VSTNToVGPeAk", Doc: "this is weak biologically -- could try 0",
			Params: params.Params{
				pt.PathScale.Abs = "0.1",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VMtxNoToVGPePr", Doc: "proto = primary classical NoGo pathway",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 fully inhibits Pr
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPePrToVGPePr", Doc: "self-inhib -- only source of self reg",
			Params: params.Params{
				pt.PathScale.Abs = "4", // 4 best for DS
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPePrToVGPeAk", Doc: "just enough to knock down in baseline state",
			Params: params.Params{
				pt.PathScale.Abs = "1",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VMtxGoToVGPi", Doc: "go influence on gating",
			Params: params.Params{
				pt.PathScale.Abs = ".2", // .1 too weak
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPePrToVGPi", Doc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 2 is much worse.. keep at 1
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VSTNToVGPi", Doc: "strong initial phasic activation",
			Params: params.Params{
				pt.PathScale.Abs = ".2",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#VGPiToACCPosVM", Doc: "final inhibition",
			Params: params.Params{
				pt.PathScale.Abs = "5", // needs to be very strong -- 5
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
	},
}
