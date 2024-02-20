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
		{Sel: ".MatrixLayer", Desc: "all mtx",
			Params: params.Params{
				"Layer.Inhib.Layer.On":             "false",
				"Layer.Inhib.Pool.On":              "true",
				"Layer.Inhib.Pool.FB":              "0",
				"Layer.Inhib.Pool.Gi":              "0.5",
				"Layer.Matrix.IsVS":                "false",
				"Layer.Acts.Dend.ModBase":          "1",
				"Layer.Acts.Dend.ModGain":          "0.1", // todo: could be 0
				"Layer.Learn.NeuroMod.AChDisInhib": "0",
			}},
		{Sel: ".STNLayer", Desc: "all STN",
			Params: params.Params{
				"Layer.Acts.Init.GeBase":           "0.1",
				"Layer.Acts.Kir.Gbar":              "10",   // 10 > 5  > 2 -- key for pause
				"Layer.Acts.SKCa.Gbar":             "2",    // 2 > 5 >> 1 (for Kir = 10)
				"Layer.Inhib.Layer.On":             "true", // actually needs this
				"Layer.Inhib.Layer.Gi":             "0.5",
				"Layer.Inhib.Layer.FB":             "0",
				"Layer.Learn.NeuroMod.AChDisInhib": "0",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
				"Layer.Acts.Kir.Gbar":    {"Tweak": "-"},
				"Layer.Acts.SKCa.Gbar":   {"Tweak": "-"},
			}},
		{Sel: "#M1VM", Desc: "all mtx",
			Params: params.Params{
				"Layer.Learn.NeuroMod.AChDisInhib": "0",
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":      "2.2",   // 3.2 def
				"Layer.Acts.Dend.ModGain":   "1.0",   // 1.5 def
				"Layer.Acts.Kir.Gbar":       "0",     // no real diff here over range 0-10
				"Layer.Acts.MaintNMDA.Gbar": "0.007", // 0.007 default
			}},
		{Sel: "#MotorBS", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.On": "true",
				"Layer.Inhib.Pool.On":  "false",
				"Layer.Inhib.Layer.Gi": "0.2",
			}},
		////////////////////////////////////////////
		// Prjns
		{Sel: ".DSMatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":        "1.8",  // 1.5 good; 1.8 causes some breakthrough
				"Prjn.Learn.LRate.Base":     "0.01", // .02 default
				"Prjn.Learn.Trace.LearnThr": "0.1",  // 0.1 slightly > 0.05
				"Prjn.Matrix.NonDelta":      "0.3",  // key param, must be >= 0.2, 0.3 -- with pf modulation
			},
			Hypers: params.Hypers{
				"Prjn.Learn.LRate.Base":     {"Tweak": "-"},
				"Prjn.Learn.Trace.LearnThr": {"Tweak": "-"},
				"Prjn.PrjnScale.Abs":        {"Tweak": "-"},
			}},
		{Sel: ".SuperToPT", Desc: "one-to-one from super",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			}},
		{Sel: ".PTSelfMaint", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 4 def
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3.0", // was 4
			}},
		{Sel: ".PFToMtx", Desc: "",
			Params: params.Params{
				"Prjn.Learn.Learn":   "false",
				"Prjn.Com.GType":     "ModulatoryG",
				"Prjn.PrjnScale.Abs": "1",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#MtxNoToGPePr", Desc: "proto = primary classical NoGo pathway",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 fully inhibits Pr
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#MtxGoToGPi", Desc: "go influence on gating -- slightly weaker than integrated GPePr",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // .5 too weak
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#GPiToM1VM", Desc: "final inhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // needs to be very strong -- 5
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#GPiToMotorBS", Desc: "final inhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: ".M1ToMotorBS", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3", // needs to be very strong -- 5
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
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
		{Sel: "#GPePr", Desc: "prototypical",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.4", // 0.4 > 0.3, 0.5
				"Layer.Acts.Init.GeVar":  "0.2",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
			}},
		{Sel: "#GPeAk", Desc: "arkypallidal",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.2", // 0.2 > 0.3, 0.1
				"Layer.Acts.Init.GeVar":  "0.1", // 0.1 == 0.2 > 0.05
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
				"Layer.Acts.Init.GeVar":  {"Tweak": "-"},
			}},
		{Sel: "#GPi", Desc: "",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.3", // 0.3 > 0.2, 0.1
				"Layer.Acts.Init.GeVar":  "0.1",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
			}},
		{Sel: ".GPeAkToMtx", Desc: "go disinhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3", // 3 >= 2, 4
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: ".MtxToGPeAk", Desc: "go inhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": ".4", // stronger = more binary
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: ".ToSTN", Desc: "excitatory inputs",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#GPePrToSTN", Desc: "enough to kick off the ping-pong dynamics for STN.",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#STNToGPePr", Desc: "stronger STN -> GPePr to kick it high at start",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#STNToGPeAk", Desc: "this is weak biologically -- could try 0",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.1",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#GPePrToGPePr", Desc: "self-inhib -- only source of self reg",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "5",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#GPePrToGPeAk", Desc: "just enough to knock down in baseline state",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#GPePrToGPi", Desc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 >> 2
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#STNToGPi", Desc: "strong initial phasic activation",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": ".2",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
	},
}
