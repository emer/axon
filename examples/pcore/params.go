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
				"Layer.Inhib.Pool.On":  "false",
				"Layer.Inhib.Pool.Gi":  "0.3",
				"Layer.Inhib.Pool.FB":  "1",
				"Layer.Matrix.GateThr": "0.05", // .05 default
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":    "3.2", // 3.2 def
				"Layer.Acts.Dend.ModGain": "1.5", // 1.5 def
			}},
		////////////////////////////////////////////
		// Prjns
		{Sel: ".MatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.Matrix.NoGateLRate":   "1",    // 1 is good -- drives learning on nogate which is rewarded -- more closely tracks
				"Prjn.Learn.LRate.Base":     "0.02", // .02 default
				"Prjn.Learn.Trace.LearnThr": "0.75",
			}},
		{Sel: "#UrgencyToMtxGo", Desc: "strong urgency factor",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.1", // don't dilute from others
				"Prjn.PrjnScale.Abs": "20",
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
				"Prjn.PrjnScale.Abs": "4.0", // if this is too strong, it gates to the wrong CS
			}},
	},
}

// ParamSetsDefs contains the full set of parameters, many of which are at default values
// and have informed the default values in the first place.
var ParamSetsDefs = netparams.Sets{
	"Base": {
		{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Acts.Decay.Act":   "0.0",
				"Layer.Acts.Decay.Glong": "0.0",
				"Layer.Acts.Clamp.Ge":    "0.8", // 1.5 is def, was 0.6 (too low) -- makes big diff on gating
			}},
		{Sel: "#PFC", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.0",
			}},
		{Sel: "#ACCNeg", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.1",
				"Layer.Acts.Clamp.Ge":  "0.6",
			}},
		{Sel: "#ACCPos", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.1",
				"Layer.Acts.Clamp.Ge":  "0.6",
			}},
		// {Sel: "#PFCo", Desc: "slower FB inhib for smoother dynamics",
		// 	Params: params.Params{}},
		{Sel: "#STNp", Desc: "Pausing STN",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal":       "0.15",
				"Layer.Inhib.Layer.On":             "true",
				"Layer.Inhib.Layer.Gi":             "0.5",
				"Layer.Acts.Decay.LearnCa":         "1.0", // key
				"Layer.Acts.SKCa.Gbar":             "3",
				"Layer.Acts.SKCa.C50":              "0.5",
				"Layer.Acts.SKCa.KCaR":             "0.8",
				"Layer.Acts.SKCa.CaRDecayTau":      "150",
				"Layer.Acts.SKCa.CaInThr":          "0.01",
				"Layer.Acts.SKCa.CaInTau":          "50",
				"Layer.Learn.NeuroMod.AChDisInhib": "2",
			}},
		{Sel: "#STNs", Desc: "Sustained STN",
			Params: params.Params{
				"Layer.Acts.Init.GeBase":           "0.0",
				"Layer.Acts.Init.GeVar":            "0.0",
				"Layer.Acts.Decay.LearnCa":         "1.0", // key
				"Layer.Inhib.ActAvg.Nominal":       "0.15",
				"Layer.Inhib.Layer.On":             "true",
				"Layer.Inhib.Layer.Gi":             "0.8", // was 0.8 - todo: 0.5?  lower?
				"Layer.Acts.SKCa.Gbar":             "3",
				"Layer.Acts.SKCa.C50":              "0.4",
				"Layer.Acts.SKCa.KCaR":             "0.4",
				"Layer.Acts.SKCa.CaRDecayTau":      "200",
				"Layer.Acts.SKCa.CaInThr":          "0.01",
				"Layer.Acts.SKCa.CaInTau":          "50",
				"Layer.Learn.NeuroMod.AChDisInhib": "2", // 2 is plenty to turn off, 1 sometimes not
			}},
		{Sel: ".GPLayer", Desc: "all gp",
			Params: params.Params{
				"Layer.Acts.Init.GeBase":     "0.3",
				"Layer.Acts.Init.GeVar":      "0.1",
				"Layer.Acts.Init.GiVar":      "0.1",
				"Layer.Inhib.ActAvg.Nominal": "1",
			}},
		{Sel: "#GPi", Desc: "",
			Params: params.Params{
				"Layer.Acts.Init.GeBase": "0.5", // todo: 0.6 in params
			}},
		{Sel: ".MatrixLayer", Desc: "all mtx",
			Params: params.Params{
				"Layer.Matrix.GateThr":             "0.01", // .05 default -- doesn't work..  todo..
				"Layer.Learn.NeuroMod.AChDisInhib": "5",
				"Layer.Inhib.Layer.On":             "true",
				"Layer.Inhib.Layer.Gi":             "0.5", // 0.5 > 0.4
				"Layer.Inhib.Layer.FB":             "0.0",
			}},
		{Sel: ".CTLayer", Desc: "corticothalamic context -- using FSA-based params -- intermediate",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.12",
				"Layer.CT.GeGain":            "1.0",
				"Layer.CT.DecayTau":          "50",
				"Layer.Inhib.Layer.Gi":       "2.2",
				"Layer.Inhib.Pool.Gi":        "2.2",
				"Layer.Acts.GabaB.Gbar":      "0.25",
				"Layer.Acts.NMDA.Gbar":       "0.25",
				"Layer.Acts.NMDA.Tau":        "200",
				"Layer.Acts.Decay.Act":       "0.0",
				"Layer.Acts.Decay.Glong":     "0.0",
				"Layer.Acts.Sahp.Gbar":       "1.0",
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":    "1.8", // was 1.0
				"Layer.Inhib.Pool.Gi":     "1.8", // was 1.8
				"Layer.Acts.GabaB.Gbar":   "0.3",
				"Layer.Acts.NMDA.Gbar":    "0.3", // 0.3 enough..
				"Layer.Acts.NMDA.Tau":     "300",
				"Layer.Acts.Decay.Act":    "0.0",
				"Layer.Acts.Decay.Glong":  "0.0",
				"Layer.Acts.Sahp.Gbar":    "0.01", // not much pressure -- long maint
				"Layer.Acts.Dend.ModGain": "10",   // 10?
			}},
		{Sel: ".BGThalLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.On": "true",
				"Layer.Inhib.Layer.Gi": "0.6",
				"Layer.Inhib.Pool.Gi":  "0.6", // 0.6 > 0.5 -- 0.8 too high
			}},
		// {Sel: "#SNc", Desc: "SNc -- no clamp limits",
		// 	Params: params.Params{
		// 	}},
		{Sel: ".MatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":        "1.0", // not stronger
				"Prjn.SWts.Adapt.On":        "true",
				"Prjn.SWts.Init.Mean":       "0.5",
				"Prjn.SWts.Init.Var":        "0.25",
				"Prjn.Matrix.NoGateLRate":   "0.01", // 0.01 seems fine actually
				"Prjn.Learn.Learn":          "true",
				"Prjn.Learn.LRate.Base":     "0.05",
				"Prjn.Learn.Trace.LearnThr": "0.75",
			}},
		{Sel: ".ACCPosToGo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":  "1.0", // 1.5 = faster go
				"Prjn.SWts.Init.Mean": "0.7",
			}},
		{Sel: ".ACCPosToNo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.0",
			}},
		{Sel: ".ACCNegToNo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.0",
			}},
		{Sel: ".ACCNegToGo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.0",
			}},
		{Sel: ".BgFixed", Desc: "fixed, non-learning params",
			Params: params.Params{
				"Prjn.SWts.Init.SPct": "0",
				"Prjn.SWts.Init.Mean": "0.8",
				"Prjn.SWts.Init.Var":  "0.0",
				"Prjn.Learn.Learn":    "false",
			}},
		{Sel: "#PFCVMToPFC", Desc: "usually uniform weights",
			Params: params.Params{
				"Prjn.SWts.Init.Mean": "0.9",
				"Prjn.SWts.Init.Var":  "0.0",
				"Prjn.SWts.Init.Sym":  "false",
				"Prjn.Learn.Learn":    "false",
				"Prjn.PrjnScale.Abs":  ".5", // modulatory
			}},
		{Sel: "#PFCToMtxGo", Desc: "weaker closed loop",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":  "0.1",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.25",
			}},
		{Sel: "#PFCToMtxNo", Desc: "weaker closed loop",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":  "0.1",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.25",
			}},
		{Sel: "#UrgencyToMtxGo", Desc: "strong urgency factor",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":  "0.1", // don't dilute from others
				"Prjn.PrjnScale.Abs":  "20",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.25",
				"Prjn.Learn.Learn":    "false",
			}},
		{Sel: "#PFCToSTNp", Desc: "strong pfc to stn",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":  "2",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.25",
			}},
		{Sel: "#PFCToSTNs", Desc: "strong pfc to stn",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":  "2",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.25",
			}},
		{Sel: ".SuperToPT", Desc: "one-to-one from super -- just use fixed nonlearning prjn so can control behavior easily",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":  "1",    // keep this constant -- only self vs. this -- thal is modulatory
				"Prjn.PrjnScale.Abs":  "0.01", // monitor maint early and other maint stats with PTMaintLayer ModGain = 0 to set this so super alone is not able to drive it.
				"Prjn.Learn.Learn":    "false",
				"Prjn.SWts.Init.Mean": "0.8",
				"Prjn.SWts.Init.Var":  "0.0",
			}},
		{Sel: ".PTSelfMaint", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":    "1",      // use abs to manipulate
				"Prjn.PrjnScale.Abs":    "2",      // 2 > 1
				"Prjn.Learn.LRate.Base": "0.0001", // slower > faster
				"Prjn.SWts.Init.Mean":   "0.5",
				"Prjn.SWts.Init.Var":    "0.5", // high variance so not just spreading out over time
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":  "1.0",
				"Prjn.PrjnScale.Abs":  "2.0", // if this is too strong, it gates to the wrong CS
				"Prjn.Learn.Learn":    "false",
				"Prjn.SWts.Init.Mean": "0.8",
				"Prjn.SWts.Init.Var":  "0.0",
			}},
		{Sel: ".ThalToPT", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":  "1.0",
				"Prjn.Com.GType":      "ModulatoryG", // this marks as modulatory with extra ModGain factor
				"Prjn.Learn.Learn":    "false",
				"Prjn.SWts.Init.Mean": "0.8",
				"Prjn.SWts.Init.Var":  "0.0",
			}},
		{Sel: ".CTCtxtPrjn", Desc: "all CT context prjns",
			Params: params.Params{
				"Prjn.Learn.LRate.Base":    "0.01", // trace: .01 > .005 > .02; .03 > .02 > .01 -- .03 std
				"Prjn.Learn.Trace.Tau":     "1",    // 2 > 1
				"Prjn.Learn.Trace.SubMean": "0",    // 0 > 1 -- 1 is especially bad
			}},
	},
	"NoTrainMtx": {
		{Sel: "MatrixPrjn", Desc: "learning in mtx",
			Params: params.Params{
				"Prjn.Learn.Learn": "false",
			}},
	},
	"LearnWts": {
		{Sel: "#ACCPosToMtxGo", Desc: "trained wts",
			Params: params.Params{
				"Prjn.SWts.Init.Mean": "0.8",
			}},
		{Sel: "#ACCNegToMtxGo", Desc: "trained wts",
			Params: params.Params{
				"Prjn.SWts.Init.Mean": "0.40",
			}},
		{Sel: "#PFCToMtxGo", Desc: "trained wts",
			Params: params.Params{
				"Prjn.SWts.Init.Mean": "0.8",
			}},
		{Sel: "#ACCPosToMtxNo", Desc: "trained wts",
			Params: params.Params{
				"Prjn.SWts.Init.Mean": "0.12",
			}},
		{Sel: "#ACCNegToMtxNo", Desc: "trained wts",
			Params: params.Params{
				"Prjn.SWts.Init.Mean": "0.8",
			}},
		{Sel: "#PFCToMtxNo", Desc: "trained wts",
			Params: params.Params{
				"Prjn.SWts.Init.Mean": "0.75",
			}},
	},
	"WtScales": {
		{Sel: ".GPeTAToMtx", Desc: "nonspecific gating activity surround inhibition -- wta",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // this is key param to control WTA selectivity!
			}},
		{Sel: "#GPeTAToMtxNo", Desc: "nonspecific gating activity surround inhibition -- wta",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // must be relatively weak to prevent oscillations
			}},
		{Sel: ".GPeInToMtx", Desc: "provides weak counterbalance for GPeTA -> Mtx to reduce oscillations",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			}},
		{Sel: "#GPeOutToGPeIn", Desc: "just enough to (dis)inhibit GPeIn",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5", // 0.5 def, was 0.3 -- no dif really
			}},
		{Sel: "#GPeInToSTNp", Desc: "not very relevant -- pause knocks out anyway -- if too much higher than this, causes oscillations.",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.1", // not big diff
			}},
		// {Sel: "#GPeInToSTNs", Desc: "NOT currently used -- interferes with threshold-based Ca self-inhib dynamics",
		// 	Params: params.Params{
		// 		"Prjn.PrjnScale.Abs": "0.1",
		// 	}},
		{Sel: "#STNpToGPeIn", Desc: "stronger STN -> GPeIn to kick it high at start",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // was 0.5
			}},
		{Sel: "#STNpToGPeOut", Desc: "opposes STNpToGPeIn -- weaker",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.1",
			}},
		{Sel: "#STNpToGPeTA", Desc: "GPeTA reacts later to GPeIn disinhib, not this first STN wave",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // was .1
			}},
		{Sel: "#MtxNoToGPeIn", Desc: "primary classical NoGo pathway",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			}},
		{Sel: "#GPeInToGPeTA", Desc: "just enough to knock down in baseline state",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // was .7 then .9 -- 1 better match
			}},
		{Sel: "#MtxGoToGPeOut", Desc: "This is key driver of Go threshold, along with to GPi",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 1 gives better match
			}},
		{Sel: "#MtxGoToGPi", Desc: "go influence on gating -- slightly weaker than integrated GPeIn",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // was 0.8, 1 is fine
			}},
		{Sel: "#GPeInToGPi", Desc: "nogo influence on gating -- decreasing produces more graded function of Go",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1", // 2 is much worse.. keep at 1
			}},
		{Sel: "#STNsToGPi", Desc: "keeps GPi active until GPeIn signal has been integrated a bit -- hold-your-horses",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": ".5", // .5 > .3 for RT, spurious gating
			}},
		{Sel: "#STNpToGPi", Desc: "strong initial phasic activation",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			}},
		{Sel: "#GPiToPFCVM", Desc: "final inhibition",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // 2 default
			}},
	},
}
