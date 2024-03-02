// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/netparams"
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = netparams.Sets{
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
				"Layer.Inhib.Pool.On":        "false",
				"Layer.Inhib.ActAvg.Nominal": "0.25",
				"Layer.Matrix.IsVS":          "true", // key for resetting urgency
				"Layer.Learn.RLRate.On":      "true", // only used for non-rew trials -- key
				"Layer.Learn.RLRate.Diff":    "false",
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":    "3.2", // 3.2 def
				"Layer.Acts.Dend.ModGain": "1.5", // 1.5 def
				"Layer.Acts.Kir.Gbar":     "0",   // no real diff here over range 0-10
				"Layer.Acts.Dend.ModACh":  "true",
			}},
		////////////////////////////////////////////
		// Prjns
		{Sel: ".VSMatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.Learn.LRate.Base":     "0.01", // 0.01, vs .02 default
				"Prjn.Learn.Trace.LearnThr": "0.0",  // prevents learning below this thr: preserves low act
				"Prjn.Matrix.VSRewLearn":    "false",
			},
			Hypers: params.Hypers{
				"Prjn.Learn.LRate.Base":     {"Tweak": "-"},
				"Prjn.Learn.Trace.LearnThr": {"Tweak": "-"},
			}},
		{Sel: "#UrgencyToVMtxGo", Desc: "strong urgency factor",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.1", // don't dilute from others
				"Prjn.PrjnScale.Abs": "0",   // todo: is misbehaving here
				"Prjn.Learn.Learn":   "false",
			}},
		{Sel: ".SuperToPT", Desc: "one-to-one from super",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			}},
		{Sel: ".PTSelfMaint", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4",
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3.0", // was 4
			}},
		{Sel: ".ACCToVMtx", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.5", // 1.5 good; 1.8 causes some breakthrough
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VMtxNoToVMtxGo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.05",
				"Prjn.PrjnScale.Abs": "1",
				"Prjn.Learn.Learn":   "false",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Rel": {"Tweak": "log"},
			}},
	},
}

// ParamSetsDefs contains the full set of parameters, many of which are at default values
// and have informed the default values in the first place.
var ParamSetsDefs = netparams.Sets{
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
				"Layer.Learn.NeuroMod.AChLRateMod": "1",     // no diff here -- always ACh
				"Layer.Learn.NeuroMod.BurstGain":   "0.1",   // 0.1 == 0.2 > 0.05 > 0.5 -- key lrate modulator
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
				"Prjn.PrjnScale.Abs": "3", // 3 >= 2, 4
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VMtxGoToVGPeAk", Desc: "go inhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": ".5", // stronger = more binary
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPePrToVSTN", Desc: "enough to kick off the ping-pong dynamics for VSTN.",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VSTNToVGPePr", Desc: "stronger VSTN -> VGPePr to kick it high at start",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VSTNToVGPeAk", Desc: "this is weak biologically -- could try 0",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.1",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VMtxNoToVGPePr", Desc: "proto = primary classical NoGo pathway",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 fully inhibits Pr
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPePrToVGPePr", Desc: "self-inhib -- only source of self reg",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4", // 4 best for DS
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPePrToVGPeAk", Desc: "just enough to knock down in baseline state",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VMtxGoToVGPi", Desc: "go influence on gating",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": ".2", // .1 too weak
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPePrToVGPi", Desc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 2 is much worse.. keep at 1
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VSTNToVGPi", Desc: "strong initial phasic activation",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": ".2",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#VGPiToACCPosVM", Desc: "final inhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "5", // needs to be very strong -- 5
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
	},
}
