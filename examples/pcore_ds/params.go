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
				ly.Acts.Noise.On = "true",
				ly.Acts.Noise.Ge = "0.0001", // 0.0001 > others; could just be noise ;)
				ly.Acts.Noise.Gi = "0.0001", // 0.0001 perhaps better than others
			},
			Hypers: params.Hypers{
				ly.Acts.Noise.Ge = {"Tweak = "-"},
			}},
		{Sel: ".MatrixLayer", Doc: "all mtx",
			Params: params.Params{
				ly.Inhib.Pool.Gi =             "0.5",  // 0.5 > others
				ly.Learn.NeuroMod.BurstGain =  "0.1",  // 0.1 == 0.2 > 0.05 > 0.5 -- key lrate modulator
				ly.Learn.NeuroMod.DAModGain =  "0.2",  // 0.2 >= 0.5 (orig) > 0
				ly.Learn.RLRate.On =           "true", // note: applied for tr update trials
				ly.Learn.TrgAvgAct.RescaleOn = "true", // true > false
			},
			Hypers: params.Hypers{
				ly.Learn.NeuroMod.BurstGain = {"Tweak": "-"},
				ly.Acts.Kir.Gbar":            {"Tweak": "-"},
				ly.Inhib.Pool.Gi":            {"Tweak": "-"},
			}},
		{Sel: ".DSTNLayer", Doc: "all STN",
			Params: params.Params{
				ly.Acts.Init.GeBase =           "0.1",
				ly.Acts.Kir.Gbar =              "10",   // 10 > 5  > 2 -- key for pause
				ly.Acts.SKCa.Gbar =             "2",    // 2 > 5 >> 1 (for Kir = 10)
				ly.Inhib.Layer.On =             "true", // actually needs this
				ly.Inhib.Layer.Gi =             "0.5",
				ly.Learn.NeuroMod.AChDisInhib = "0",
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
				ly.Acts.Kir.Gbar =    {"Tweak = "-"},
				ly.Acts.SKCa.Gbar =   {"Tweak = "-"},
			}},
		{Sel: "#M1VM", Doc: "",
			Params: params.Params{
				ly.Learn.NeuroMod.AChDisInhib = "0",
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Params: params.Params{
				ly.Inhib.Layer.Gi =       "2.4",   // 2.4 def > 1.4
				ly.Inhib.ActAvg.Nominal = "0.3",   // 0.3 def -- key but wrong!
				ly.Acts.Decay.OnRew =     "true",  // true def -- seems better?
				ly.Acts.Dend.ModGain =    "1.0",   // 1.5 def
				ly.Acts.Kir.Gbar =        "0",     // no real diff here over range 0-10
				ly.Acts.MaintNMDA.Gbar =  "0.007", // 0.007 default
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi = "0.8",  // 0.8 def
				ly.CT.GeGain =      "0.05", // 0.05 def
				ly.CT.DecayTau =    "50",   // 50 def
			}},
		{Sel: ".CTLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi = "1.4", // 0.8 def
				ly.CT.GeGain =      "5",   // 2 def
				ly.CT.DecayTau =    "50",  // 50 def
			}},
		{Sel: ".CTtoPred", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "2", // 1 def
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: ".PTtoPred", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1", // was 6
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#MotorBS", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.On = "true",
				ly.Inhib.Pool.On =  "false",
				ly.Inhib.Layer.Gi = "0.2", // 0.2 def
				ly.Acts.Clamp.Ge =  "2",   // 2 > 1.5, >> 1 -- absolutely critical given GPi inhib
			},
			Hypers: params.Hypers{
				ly.Acts.Clamp.Ge = {"Tweak = "-"},
			}},
		{Sel: "#DGPeAk", Doc: "arkypallidal",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.2", // 0.2 > 0.3, 0.1
				ly.Acts.Init.GeVar =  "0.1", // 0.1 == 0.2 > 0.05
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
				ly.Acts.Init.GeVar =  {"Tweak = "-"},
			}},
		////////////////////////////////////////////
		// Paths
		// {Sel: "Path", Doc: "",
		// 	Params: params.Params{
		// 		pt.Learn.LRate.Base = "0.02", // 0.04 > 0.02 probably
		// 	}},
		{Sel: ".DSMatrixPath", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs =        "1.8",   // 1.8 > others
				pt.Learn.LRate.Base =     "0.02",  // rlr sig: .02 > .015 .025
				pt.Learn.Trace.LearnThr = "0.1",   // 0.1 > 0 > 0.2
				pt.Matrix.Credit =        "0.6",   // key param, 0.6 > 0.5, 0.4, 0.7, 1 with pf modulation
				pt.Matrix.BasePF =        "0.005", // 0.005 > 0.01, 0.002 etc
				pt.Matrix.Delta =         "1",     // should always be 1 except for testing; adjust lrate to compensate
				pt.SWts.Adapt.On =        "false", // false > true here
			},
			Hypers: params.Hypers{
				pt.Learn.LRate.Base = {"Tweak = "-"},
				pt.PathScale.Abs =    {"Tweak = "-"},
				pt.Matrix.BasePF =    {"Tweak = "-"},
			}},
		{Sel: ".SuperToPT", Doc: "one-to-one from super",
			Params: params.Params{
				pt.PathScale.Abs = "0.5",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: ".PTSelfMaint", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 def
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: ".SuperToThal", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "3.0", // 3
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: ".ToM1", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1.5", // now 1.5 > 2 > 1 ..
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#StateToM1", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 > 1.5, 2, 0.5 etc
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DGPiToPF", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "0.4", // 0.4 >= 0.5, 0.3, 0.2 >> higher
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#MotorBSToPF", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 > 1.1 > 0.9 >> 0.5
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		// {Sel: ".StateToDMtx", Doc: "",
		// 	Params: params.Params{
		// 		pt.PathScale.Abs = "1.5", // 1.8 def
		// 	},
		// 	Hypers: params.Hypers{
		// 		pt.PathScale.Abs = {"Tweak = "-"},
		// 	}},
		// {Sel: ".CLToDMtx", Doc: "",
		// 	Params: params.Params{
		// 		pt.Learn.Learn =   "false",
		// 		pt.PathScale.Rel = "0.001",
		// 	},
		// 	Hypers: params.Hypers{
		// 		pt.PathScale.Rel = {"Tweak = "-"},
		// 	}},
		{Sel: "#DGPiToM1VM", Doc: "final inhibition",
			Params: params.Params{
				pt.PathScale.Abs = "2", // 2
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DGPiToMotorBS", Doc: "final inhibition",
			Params: params.Params{
				pt.PathScale.Abs = "3", // 3 > 2.5, 3.5
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: ".M1ToMotorBS", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "2", // 2 > 1.5, 2.5
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#M1PTToMotorBS", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "2", // 2
				pt.PathScale.Rel = "1", // 1
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#M1PTToVL", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1",   // 1
				pt.PathScale.Rel = "0.1", // 0.1 > 0.2, .05, 0
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		// {Sel: "#M1PTpToMotorBS", Doc: "",
		// 	Params: params.Params{
		// 		pt.PathScale.Abs = "2",
		// 		pt.PathScale.Rel = "1",
		// 	},
		// 	Hypers: params.Hypers{
		// 		pt.PathScale.Abs = {"Tweak = "-"},
		// 	}},
		{Sel: "#M1ToMotorBS", Doc: "weaker; note: this is a proxy for cerebellum etc inputs",
			Params: params.Params{
				pt.PathScale.Abs = "1.5", // 1.5 > 1, 2, 2.5
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DMtxNoToDMtxGo", Doc: "weakish no->go inhibition is beneficial",
			Params: params.Params{
				pt.PathScale.Rel = "0.1",   // 0.1 > 0.05
				pt.Learn.Learn =   "false", // no-learn better than learn
			},
			Hypers: params.Hypers{
				pt.PathScale.Rel = {"Tweak = "-"},
			}},
		{Sel: "#DGPeAkToDMtxNo", Doc: "go disinhibition",
			Params: params.Params{
				pt.PathScale.Abs = "6",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
	},
	"NoiseOff = {
		{Sel: "Layer", Doc: "turn off noise",
			Params: params.Params{
				ly.Acts.Noise.On = "false",
			}},
	},
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

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
			Hypers: params.Hypers{
				ly.Learn.NeuroMod.BurstGain = {"Tweak = "-"},
				ly.Acts.Kir.Gbar =            {"Tweak = "-"},
				ly.Acts.NMDA.Gbar =           {"Tweak = "-"},
				ly.Inhib.Layer.Gi =           {"Tweak = "-"},
			}},
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
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
				ly.Acts.Kir.Gbar =    {"Tweak = "-"},
				ly.Acts.SKCa.Gbar =   {"Tweak = "-"},
			}},
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
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
			}},
		{Sel: "#DGPeAk", Doc: "arkypallidal",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.2", // 0.2 > 0.3, 0.1
				ly.Acts.Init.GeVar =  "0.1", // 0.1 == 0.2 > 0.05
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
				ly.Acts.Init.GeVar =  {"Tweak = "-"},
			}},
		{Sel: "#DGPi", Doc: "",
			Params: params.Params{
				ly.Acts.Init.GeBase = "0.3", // 0.3 > 0.2, 0.1
				ly.Acts.Init.GeVar =  "0.1",
			},
			Hypers: params.Hypers{
				ly.Acts.Init.GeBase = {"Tweak = "-"},
			}},
		{Sel: "#DGPeAkToDMtxGo", Doc: "go disinhibition",
			Params: params.Params{
				pt.PathScale.Abs = "3",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DMtxGoToDGPeAk", Doc: "go inhibition",
			Params: params.Params{
				pt.PathScale.Abs = ".5", // stronger = more binary
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DMtxNoToDGPePr", Doc: "proto = primary classical NoGo pathway",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 fully inhibits Pr
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: ".ToDSTN", Doc: "excitatory inputs",
			Params: params.Params{
				pt.PathScale.Abs = "1",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DMtxGoToDGPi", Doc: "go influence on gating -- slightly weaker than integrated GPePr",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 > 1.1, .9 and lower (not huge diffs)
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DGPePrToDSTN", Doc: "enough to kick off the ping-pong dynamics for STN.",
			Params: params.Params{
				pt.PathScale.Abs = "0.5",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DSTNToDGPePr", Doc: "stronger STN -> DGPePr to kick it high at start",
			Params: params.Params{
				pt.PathScale.Abs = "0.5",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DSTNToDGPeAk", Doc: "this is weak biologically -- could try 0",
			Params: params.Params{
				pt.PathScale.Abs = "0.1",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DGPePrToDGPePr", Doc: "self-inhib -- only source of self reg",
			Params: params.Params{
				pt.PathScale.Abs = "4",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DGPePrToDGPeAk", Doc: "just enough to knock down in baseline state",
			Params: params.Params{
				pt.PathScale.Abs = "1",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DGPePrToDGPi", Doc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Params: params.Params{
				pt.PathScale.Abs = "1", // 1 >> 2
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: "#DSTNToDGPi", Doc: "strong initial phasic activation",
			Params: params.Params{
				pt.PathScale.Abs = ".2",
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: ".PFToDMtx", Doc: "",
			Params: params.Params{
				pt.Learn.Learn =   "false",
				pt.Com.GType =     "ModulatoryG",
				pt.PathScale.Abs = "1",
			}},
	},
}
