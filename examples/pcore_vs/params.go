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
		{Sel: "Layer", Desc: "clamp gain makes big diff on overall excitation, gating propensity",
			Params: params.Params{
				"Layer.Acts.Clamp.Ge": "1.0", // 1.5 is def, was 0.6 (too low)
			}},
		{Sel: ".VBG", Desc: "all ModACh",
			Params: params.Params{
				"Layer.Acts.Dend.ModACh": "true",
			}},
		{Sel: ".MatrixLayer", Desc: "all mtx",
			Params: params.Params{
				"Layer.Inhib.Pool.On":             "false",
				"Layer.Inhib.Layer.Gi":            "0.5", // 0.5 needed for differentiated reps
				"Layer.Inhib.ActAvg.Nominal":      "0.25",
				"Layer.Matrix.GateThr":            "0.05", // todo: .01 should be new default
				"Layer.Matrix.IsVS":               "true", // key for resetting urgency
				"Layer.Learn.RLRate.On":           "true", // only used for non-rew trials -- key
				"Layer.Learn.RLRate.Diff":         "false",
				"Layer.Learn.RLRate.SigmoidMin":   "0.01", // 0.01 better than .05
				"Layer.Learn.TrgAvgAct.RescaleOn": "true",
				"Layer.Learn.NeuroMod.BurstGain":  "0.1", // 1 def -- must be smaller given rew dynamics
				"Layer.Learn.NeuroMod.DAModGain":  "0",   // strongly biases the gating
			}},
		{Sel: ".VSTNLayer", Desc: "all VSTN",
			Params: params.Params{
				"Layer.Acts.Init.GeBase":      "0.1", // todo: re-param with more stn, increase..
				"Layer.Acts.SKCa.CaRDecayTau": "80",
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				// "Layer.Inhib.Layer.Gi":    "3.2", // 3.2 def
				"Layer.Acts.Dend.ModGain": "1.5", // 1.5 def
				"Layer.Acts.Kir.Gbar":     "0",   // no real diff here over range 0-10
				"Layer.Acts.Dend.ModACh":  "true",
			}},
		{Sel: ".ACC", Desc: "manipulate noise to see about integration over time",
			Params: params.Params{
				"Layer.Acts.Noise.On": "false",
				"Layer.Acts.Noise.Ge": "0.1",  // 0.1 is visibly impactful
				"Layer.Acts.Noise.Gi": "0.01", // 0.01 -- if to strong, rep becomes very weak
			}},
		////////////////////////////////////////////
		// Paths
		{Sel: ".VSMatrixPath", Desc: "",
			Params: params.Params{
				"Path.Learn.LRate.Base":     "0.01",  // 0.01, vs .02 default
				"Path.Learn.Trace.LearnThr": "0.1",   // prevents learning below this thr: preserves low act
				"Path.Matrix.VSRewLearn":    "false", // significantly cleaner
				"Path.SWts.Adapt.On":        "false", // not much diff: false >= true
			},
			Hypers: params.Hypers{
				"Path.Learn.LRate.Base":     {"Tweak": "-"},
				"Path.Learn.Trace.LearnThr": {"Tweak": "-"},
			}},
		{Sel: "#UrgencyToVMtxGo", Desc: "strong urgency factor",
			Params: params.Params{
				"Path.PathScale.Rel": "0.1", // don't dilute from others
				"Path.PathScale.Abs": "0",   // todo: is misbehaving here
				"Path.Learn.Learn":   "false",
			}},
		{Sel: ".SuperToPT", Desc: "one-to-one from super",
			Params: params.Params{
				"Path.PathScale.Abs": "0.5",
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs": "3.0", // was 4
			}},
		{Sel: ".ACCToVMtx", Desc: "",
			Params: params.Params{
				"Path.PathScale.Abs": "1.5", // 1.5 good; 1.8 causes some breakthrough
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VMtxNoToVMtxGo", Desc: "",
			Params: params.Params{
				"Path.PathScale.Rel": "0.05",
				"Path.PathScale.Abs": "1",
				"Path.Learn.Learn":   "false",
			},
			Hypers: params.Hypers{
				"Path.PathScale.Rel": {"Tweak": "log"},
			}},
		{Sel: "#VGPi", Desc: "",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.3", // 0.3 > 0.2, 0.1
				"Layer.Acts.Init.GeVar":  "0.1",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
			}},
		{Sel: "#VSTNToVGPi", Desc: "strong initial phasic activation",
			Params: params.Params{
				"Path.PathScale.Abs": ".2",
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
	},
}

// ParamSetsDefs contains the full set of parameters, many of which are at default values
// and have informed the default values in the first place.
var ParamSetsDefs = params.Sets{
	"Defaults": {
		{Sel: ".MatrixLayer", Desc: "all mtx",
			Params: params.Params{
				"Layer.Inhib.Pool.On":              "false",
				"Layer.Inhib.Layer.Gi":             "0.5",
				"Layer.Inhib.Layer.FB":             "0",
				"Layer.Matrix.GateThr":             "0.05", // .05 default
				"Layer.Acts.Kir.Gbar":              "10",   // 10 > 5 > 20
				"Layer.Acts.GabaB.Gbar":            "0",
				"Layer.Acts.NMDA.Gbar":             "0.006", // 0.006 default, necessary (0 very bad)
				"Layer.Learn.NeuroMod.AChLRateMod": "0",     // no diff here -- always ACh
				"Layer.Learn.NeuroMod.BurstGain":   "0.1",   // 0.1 == 0.2 > 0.05 > 0.5; only for weird rew case here; 1 def
			},
			Hypers: params.Hypers{
				"Layer.Learn.NeuroMod.BurstGain": {"Tweak": "-"},
				"Layer.Acts.Kir.Gbar":            {"Tweak": "-"},
				"Layer.Acts.NMDA.Gbar":           {"Tweak": "-"},
				"Layer.Inhib.Layer.Gi":           {"Tweak": "-"},
			}},
		{Sel: ".VSTNLayer", Desc: "all VSTN",
			Params: params.Params{
				"Layer.Acts.Init.GeBase":           "0.1",
				"Layer.Acts.Kir.Gbar":              "10",   // 10 > 5  > 2 -- key for pause
				"Layer.Acts.SKCa.Gbar":             "2",    // 2 > 5 >> 1 (for Kir = 10)
				"Layer.Inhib.Layer.On":             "true", // really no inhib neurons here.  all VGPePr
				"Layer.Learn.NeuroMod.AChDisInhib": "0",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
				"Layer.Acts.Kir.Gbar":    {"Tweak": "-"},
				"Layer.Acts.SKCa.Gbar":   {"Tweak": "-"},
			}},
		{Sel: "#VGPePr", Desc: "prototypical",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.4", // 0.4 > 0.3, 0.5
				"Layer.Acts.Init.GeVar":  "0.2",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
			}},
		{Sel: "#VGPeAk", Desc: "arkypallidal",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.2", // 0.2 > 0.3, 0.1
				"Layer.Acts.Init.GeVar":  "0.1", // 0.1 == 0.2 > 0.05
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
				"Layer.Acts.Init.GeVar":  {"Tweak": "-"},
			}},
		{Sel: "#VGPi", Desc: "",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.3", // 0.3 > 0.2, 0.1
				"Layer.Acts.Init.GeVar":  "0.1",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
			}},
		{Sel: ".VGPeAkToVMtx", Desc: "go disinhibition",
			Params: params.Params{
				"Path.PathScale.Abs": "3", // 3 >= 2, 4
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VMtxGoToVGPeAk", Desc: "go inhibition",
			Params: params.Params{
				"Path.PathScale.Abs": ".5", // stronger = more binary
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPePrToVSTN", Desc: "enough to kick off the ping-pong dynamics for VSTN.",
			Params: params.Params{
				"Path.PathScale.Abs": "0.5",
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VSTNToVGPePr", Desc: "stronger VSTN -> VGPePr to kick it high at start",
			Params: params.Params{
				"Path.PathScale.Abs": "0.5",
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VSTNToVGPeAk", Desc: "this is weak biologically -- could try 0",
			Params: params.Params{
				"Path.PathScale.Abs": "0.1",
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VMtxNoToVGPePr", Desc: "proto = primary classical NoGo pathway",
			Params: params.Params{
				"Path.PathScale.Abs": "1", // 1 fully inhibits Pr
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPePrToVGPePr", Desc: "self-inhib -- only source of self reg",
			Params: params.Params{
				"Path.PathScale.Abs": "4", // 4 best for DS
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPePrToVGPeAk", Desc: "just enough to knock down in baseline state",
			Params: params.Params{
				"Path.PathScale.Abs": "1",
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VMtxGoToVGPi", Desc: "go influence on gating",
			Params: params.Params{
				"Path.PathScale.Abs": ".2", // .1 too weak
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPePrToVGPi", Desc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Params: params.Params{
				"Path.PathScale.Abs": "1", // 2 is much worse.. keep at 1
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VSTNToVGPi", Desc: "strong initial phasic activation",
			Params: params.Params{
				"Path.PathScale.Abs": ".2",
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPiToACCPosVM", Desc: "final inhibition",
			Params: params.Params{
				"Path.PathScale.Abs": "5", // needs to be very strong -- 5
			},
			Hypers: params.Hypers{
				"Path.PathScale.Abs": {"Tweak": "-"},
			}},
	},
}
