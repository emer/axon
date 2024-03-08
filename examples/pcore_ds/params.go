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
				"Layer.Acts.Noise.On": "true",
				"Layer.Acts.Noise.Ge": "0.0001", // 0.0001 > others; could just be noise ;)
				"Layer.Acts.Noise.Gi": "0.0001", // 0.0001 perhaps better than others
			},
			Hypers: params.Hypers{
				"Layer.Acts.Noise.Ge": {"Tweak": "-"},
			}},
		{Sel: ".DBG", Desc: "all bg",
			Params: params.Params{
				"Layer.Acts.Mahp.Gbar": "0.0",
				"Layer.Acts.Sahp.Gbar": "0.05", // note: Pr getting lots of Sahp
			},
			Hypers: params.Hypers{
				"Layer.Acts.Mahp.Gbar": {"Tweak": "-"},
				"Layer.Acts.Sahp.Gbar": {"Tweak": "-"},
			}},
		{Sel: ".MatrixLayer", Desc: "all mtx",
			Params: params.Params{
				"Layer.Inhib.Pool.Gi":            "0.5",   // 0.5 > others
				"Layer.Learn.NeuroMod.BurstGain": "0.1",   // 0.1 == 0.2 > 0.05 > 0.5 -- key lrate modulator
				"Layer.Learn.RLRate.On":          "true",  // note: applied for tr update trials
				"Layer.Learn.TrgAvgAct.On":       "true",  // true > false
				"Layer.Acts.Mahp.Gbar":           "0.015", // 0.01 > 0.02 > 0
				"Layer.Acts.Sahp.Gbar":           "0.03",  // todo test
			},
			Hypers: params.Hypers{
				"Layer.Learn.NeuroMod.BurstGain": {"Tweak": "-"},
				"Layer.Acts.Kir.Gbar":            {"Tweak": "-"},
				"Layer.Inhib.Pool.Gi":            {"Tweak": "-"},
				"Layer.Acts.Mahp.Gbar":           {"Tweak": "-"},
				"Layer.Acts.Sahp.Gbar":           {"Tweak": "-"},
			}},
		{Sel: ".DSTNLayer", Desc: "all STN",
			Params: params.Params{
				"Layer.Acts.Init.GeBase":           "0.1",
				"Layer.Acts.Kir.Gbar":              "10",   // 10 > 5  > 2 -- key for pause
				"Layer.Acts.SKCa.Gbar":             "2",    // 2 > 5 >> 1 (for Kir = 10)
				"Layer.Inhib.Layer.On":             "true", // actually needs this
				"Layer.Inhib.Layer.Gi":             "0.5",
				"Layer.Learn.NeuroMod.AChDisInhib": "0",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
				"Layer.Acts.Kir.Gbar":    {"Tweak": "-"},
				"Layer.Acts.SKCa.Gbar":   {"Tweak": "-"},
			}},
		{Sel: "#M1VM", Desc: "",
			Params: params.Params{
				"Layer.Learn.NeuroMod.AChDisInhib": "0",
			}},
		{Sel: ".PFCLayer", Desc: "",
			Params: params.Params{
				"Layer.Acts.Mahp.Gbar": "0.02", // 0.02 def
				"Layer.Acts.Sahp.Gbar": "0.05", // 0.05 def
			},
			Hypers: params.Hypers{
				"Layer.Acts.Mahp.Gbar": {"Tweak": "[0.01,0.015,0.005]"},
				"Layer.Acts.Sahp.Gbar": {"Tweak": "[0.04,0.03,0.02]"},
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":      "2.2",   // 2.2 def
				"Layer.Acts.Dend.ModGain":   "1.0",   // 1.5 def
				"Layer.Acts.Kir.Gbar":       "0",     // no real diff here over range 0-10
				"Layer.Acts.MaintNMDA.Gbar": "0.007", // 0.007 default
				"Layer.Acts.Mahp.Gbar":      "0",     // 0.02 def
				"Layer.Acts.Sahp.Gbar":      "0.05",  // 0.05 def
			},
			Hypers: params.Hypers{
				"Layer.Acts.Mahp.Gbar": {"Tweak": "[0.01,0.015,0.005]"},
				"Layer.Acts.Sahp.Gbar": {"Tweak": "[0.04,0.03,0.02]"},
			}},
		{Sel: ".PTPredLayer", Desc: "",
			Params: params.Params{
				"Layer.Acts.Mahp.Gbar": "0",    // 0.02 def
				"Layer.Acts.Sahp.Gbar": "0.05", // 0.05 def
			},
			Hypers: params.Hypers{
				"Layer.Acts.Mahp.Gbar": {"Tweak": "[0.01,0.015,0.005]"},
				"Layer.Acts.Sahp.Gbar": {"Tweak": "[0.04,0.03,0.02]"},
			}},
		{Sel: "#MotorBS", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.On": "true",
				"Layer.Inhib.Pool.On":  "false",
				"Layer.Inhib.Layer.Gi": "0.2", // 0.2 def
				"Layer.Acts.Clamp.Ge":  "2",   // 2 > 1.5, >> 1 -- absolutely critical given GPi inhib
			},
			Hypers: params.Hypers{
				"Layer.Acts.Clamp.Ge": {"Tweak": "-"},
			}},
		{Sel: "#DGPeAk", Desc: "arkypallidal",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.2", // 0.2 > 0.3, 0.1
				"Layer.Acts.Init.GeVar":  "0.1", // 0.1 == 0.2 > 0.05
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
				"Layer.Acts.Init.GeVar":  {"Tweak": "-"},
			}},
		////////////////////////////////////////////
		// Prjns
		// {Sel: "Prjn", Desc: "",
		// 	Params: params.Params{
		// 		"Prjn.Learn.LRate.Base": "0.02", // 0.04 > 0.02 probably
		// 	}},
		{Sel: ".DSMatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":        "1.8",   // 1.8 > others
				"Prjn.Learn.LRate.Base":     "0.02",  // rlr sig: .02 > .015 .025
				"Prjn.Learn.Trace.LearnThr": "0.1",   // 0.1 > 0 > 0.2
				"Prjn.Matrix.Credit":        "0.6",   // key param, 0.6 > 0.5, 0.4, 0.7, 1 with pf modulation
				"Prjn.Matrix.BasePF":        "0.005", // 0.005 > 0.01, 0.002 etc
				"Prjn.Matrix.Delta":         "1",     // should always be 1 except for testing; adjust lrate to compensate
				"Prjn.SWts.Adapt.On":        "false", // false > true here
			},
			Hypers: params.Hypers{
				"Prjn.Learn.LRate.Base": {"Tweak": "-"},
				"Prjn.PrjnScale.Abs":    {"Tweak": "-"},
				"Prjn.Matrix.BasePF":    {"Tweak": "-"},
			}},
		{Sel: ".SuperToPT", Desc: "one-to-one from super",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: ".PTSelfMaint", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 def
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3.0", // 3
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: ".ToM1", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.5", // now 1.5 > 2 > 1 ..
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#StateToM1", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 > 1.5, 2, 0.5 etc
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DGPiToPF", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.4", // 0.4 >= 0.5, 0.3, 0.2 >> higher
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#MotorBSToPF", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 > 1.1 > 0.9 >> 0.5
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		// {Sel: ".StateToDMtx", Desc: "",
		// 	Params: params.Params{
		// 		"Prjn.PrjnScale.Abs": "1.5", // 1.8 def
		// 	},
		// 	Hypers: params.Hypers{
		// 		"Prjn.PrjnScale.Abs": {"Tweak": "-"},
		// 	}},
		// {Sel: ".CLToDMtx", Desc: "",
		// 	Params: params.Params{
		// 		"Prjn.Learn.Learn":   "false",
		// 		"Prjn.PrjnScale.Rel": "0.001",
		// 	},
		// 	Hypers: params.Hypers{
		// 		"Prjn.PrjnScale.Rel": {"Tweak": "-"},
		// 	}},
		{Sel: "#DGPiToM1VM", Desc: "final inhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // 2
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DGPiToMotorBS", Desc: "final inhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3", // 3 > 2.5, 3.5
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: ".M1ToMotorBS", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // 2 > 1.5, 2.5
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#M1PTToMotorBS", Desc: "PT to motor is strong, key",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // 2 > 1.5, 2.5 per above
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#M1ToMotorBS", Desc: "weaker; note: this is a proxy for cerebellum etc inputs",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.5", // 1.5 > 1, 2, 2.5
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DMtxNoToDMtxGo", Desc: "weakish no->go inhibition is beneficial",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.1",   // 0.1 > 0.05
				"Prjn.Learn.Learn":   "false", // no-learn better than learn
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Rel": {"Tweak": "-"},
			}},
		{Sel: "#DGPeAkToDMtxNo", Desc: "go disinhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "6",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
	},
	"NoiseOff": {
		{Sel: "Layer", Desc: "turn off noise",
			Params: params.Params{
				"Layer.Acts.Noise.On": "false",
			}},
	},
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

// ParamSetsDefs contains the full set of parameters, many of which are at default values
// and have informed the default values in the first place.
var ParamSetsDefs = netparams.Sets{
	"Defaults": {
		{Sel: ".MatrixLayer", Desc: "all mtx",
			Params: params.Params{
				"Layer.Inhib.Layer.On":             "true",
				"Layer.Inhib.Pool.On":              "true",
				"Layer.Inhib.Pool.FB":              "0",
				"Layer.Inhib.Pool.Gi":              "0.5",  // 0.5 > others
				"Layer.Matrix.GateThr":             "0.05", // .05 default
				"Layer.Acts.Kir.Gbar":              "10",   // 10 > 5 > 20
				"Layer.Acts.GabaB.Gbar":            "0",
				"Layer.Acts.NMDA.Gbar":             "0.006", // 0.006 default, necessary (0 very bad)
				"Layer.Acts.Dend.ModBase":          "1",
				"Layer.Acts.Dend.ModGain":          "0",   // has no effect
				"Layer.Learn.NeuroMod.AChLRateMod": "0",   // dorsal should not use
				"Layer.Learn.NeuroMod.BurstGain":   "0.1", // 0.1 == 0.2 > 0.05 > 0.5 -- key lrate modulator
				"Layer.Learn.NeuroMod.AChDisInhib": "0",
			},
			Hypers: params.Hypers{
				"Layer.Learn.NeuroMod.BurstGain": {"Tweak": "-"},
				"Layer.Acts.Kir.Gbar":            {"Tweak": "-"},
				"Layer.Acts.NMDA.Gbar":           {"Tweak": "-"},
				"Layer.Inhib.Layer.Gi":           {"Tweak": "-"},
			}},
		{Sel: ".DSTNLayer", Desc: "all STN",
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
		{Sel: "#PF", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.On": "false",
				"Layer.Inhib.Pool.On":  "false",
			}},
		{Sel: "#DGPePr", Desc: "prototypical",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.4", // 0.4 > 0.3, 0.5
				"Layer.Acts.Init.GeVar":  "0.2",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
			}},
		{Sel: "#DGPeAk", Desc: "arkypallidal",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.2", // 0.2 > 0.3, 0.1
				"Layer.Acts.Init.GeVar":  "0.1", // 0.1 == 0.2 > 0.05
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
				"Layer.Acts.Init.GeVar":  {"Tweak": "-"},
			}},
		{Sel: "#DGPi", Desc: "",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.3", // 0.3 > 0.2, 0.1
				"Layer.Acts.Init.GeVar":  "0.1",
			},
			Hypers: params.Hypers{
				"Layer.Acts.Init.GeBase": {"Tweak": "-"},
			}},
		{Sel: "#DGPeAkToDMtxGo", Desc: "go disinhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "3",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DMtxGoToDGPeAk", Desc: "go inhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": ".5", // stronger = more binary
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DMtxNoToDGPePr", Desc: "proto = primary classical NoGo pathway",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 fully inhibits Pr
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: ".ToDSTN", Desc: "excitatory inputs",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DMtxGoToDGPi", Desc: "go influence on gating -- slightly weaker than integrated GPePr",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 > 1.1, .9 and lower (not huge diffs)
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DGPePrToDSTN", Desc: "enough to kick off the ping-pong dynamics for STN.",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DSTNToDGPePr", Desc: "stronger STN -> DGPePr to kick it high at start",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DSTNToDGPeAk", Desc: "this is weak biologically -- could try 0",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.1",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DGPePrToDGPePr", Desc: "self-inhib -- only source of self reg",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DGPePrToDGPeAk", Desc: "just enough to knock down in baseline state",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DGPePrToDGPi", Desc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 >> 2
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: "#DSTNToDGPi", Desc: "strong initial phasic activation",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": ".2",
			},
			Hypers: params.Hypers{
				"Prjn.PrjnScale.Abs": {"Tweak": "-"},
			}},
		{Sel: ".PFToDMtx", Desc: "",
			Params: params.Params{
				"Prjn.Learn.Learn":   "false",
				"Prjn.Com.GType":     "ModulatoryG",
				"Prjn.PrjnScale.Abs": "1",
			}},
	},
}
