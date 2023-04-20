// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "minimal base params needed for this model", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Act.Decay.Act":   "0.0",
					"Layer.Act.Decay.Glong": "0.0",
					"Layer.Act.Clamp.Ge":    "0.6",
				}},
			{Sel: "#PFC", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.0",
				}},
			{Sel: "#ACCNeg", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Act.Clamp.Ge":   "0.6",
				}},
			{Sel: "#ACCPos", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Act.Clamp.Ge":   "0.6",
				}},
			// {Sel: "#PFCo", Desc: "slower FB inhib for smoother dynamics",
			// 	Params: params.Params{}},
			{Sel: "#STNp", Desc: "Pausing STN",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal":       "0.15",
					"Layer.Inhib.Layer.On":             "true",
					"Layer.Inhib.Layer.Gi":             "0.5",
					"Layer.Act.Decay.LearnCa":          "1.0", // key
					"Layer.Act.SKCa.Gbar":              "3",
					"Layer.Act.SKCa.C50":               "0.5",
					"Layer.Act.SKCa.KCaR":              "0.8",
					"Layer.Act.SKCa.CaRDecayTau":       "150",
					"Layer.Act.SKCa.CaInThr":           "0.01",
					"Layer.Act.SKCa.CaInTau":           "50",
					"Layer.Learn.NeuroMod.AChDisInhib": "2",
				}},
			{Sel: "#STNs", Desc: "Sustained STN",
				Params: params.Params{
					"Layer.Act.Init.GeBase":            "0.0",
					"Layer.Act.Init.GeVar":             "0.0",
					"Layer.Act.Decay.LearnCa":          "1.0", // key
					"Layer.Inhib.ActAvg.Nominal":       "0.15",
					"Layer.Inhib.Layer.On":             "true",
					"Layer.Inhib.Layer.Gi":             "0.8", // was 0.8 - todo: 0.5?  lower?
					"Layer.Act.SKCa.Gbar":              "3",
					"Layer.Act.SKCa.C50":               "0.4",
					"Layer.Act.SKCa.KCaR":              "0.4",
					"Layer.Act.SKCa.CaRDecayTau":       "200",
					"Layer.Act.SKCa.CaInThr":           "0.01",
					"Layer.Act.SKCa.CaInTau":           "50",
					"Layer.Learn.NeuroMod.AChDisInhib": "2", // 2 is plenty to turn off, 1 sometimes not
				}},
			{Sel: ".GPLayer", Desc: "all gp",
				Params: params.Params{
					"Layer.Act.Init.GeBase":      "0.3",
					"Layer.Act.Init.GeVar":       "0.1",
					"Layer.Act.Init.GiVar":       "0.1",
					"Layer.Inhib.ActAvg.Nominal": "1",
				}},
			{Sel: "#GPi", Desc: "",
				Params: params.Params{
					"Layer.Act.Init.GeBase": "0.5", // todo: 0.6 in params
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
					"Layer.Act.GABAB.Gbar":       "0.25",
					"Layer.Act.NMDA.Gbar":        "0.25",
					"Layer.Act.NMDA.Tau":         "200",
					"Layer.Act.Decay.Act":        "0.0",
					"Layer.Act.Decay.Glong":      "0.0",
					"Layer.Act.Sahp.Gbar":        "1.0",
				}},
			{Sel: ".PTMaintLayer", Desc: "time integration params",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":             "1.8", // was 1.0
					"Layer.Inhib.Pool.Gi":              "1.8", // was 1.8
					"Layer.Act.GABAB.Gbar":             "0.3",
					"Layer.Act.NMDA.Gbar":              "0.3", // 0.3 enough..
					"Layer.Act.NMDA.Tau":               "300",
					"Layer.Act.Decay.Act":              "0.0",
					"Layer.Act.Decay.Glong":            "0.0",
					"Layer.Act.Sahp.Gbar":              "0.01", // not much pressure -- long maint
					"Layer.Act.Dend.ModGain":           "10",   // 10?
					"Layer.Learn.NeuroMod.AChDisInhib": "1.0",
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
					"Prjn.SWt.Adapt.On":         "true",
					"Prjn.SWt.Init.Mean":        "0.5",
					"Prjn.SWt.Init.Var":         "0.25",
					"Prjn.Matrix.NoGateLRate":   "0.01", // 0.01 seems fine actually
					"Prjn.Learn.Learn":          "true",
					"Prjn.Learn.LRate.Base":     "0.05",
					"Prjn.Learn.Trace.LearnThr": "0.75",
				}},
			{Sel: ".ACCPosToGo", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0", // 1.5 = faster go
					"Prjn.SWt.Init.Mean": "0.7",
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
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.Learn.Learn":   "false",
				}},
			{Sel: ".BgFixed", Desc: "fixed, non-learning params",
				Params: params.Params{
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.Learn.Learn":   "false",
				}},
			{Sel: "#PFCVMToPFC", Desc: "usually uniform weights",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.9",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.SWt.Init.Sym":  "false",
					"Prjn.Learn.Learn":   "false",
					"Prjn.PrjnScale.Abs": ".5", // modulatory
				}},
			{Sel: "#PFCToMtxGo", Desc: "weaker closed loop",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: "#PFCToMtxNo", Desc: "weaker closed loop",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: "#UrgencyToMtxGo", Desc: "strong urgency factor",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // don't dilute from others
					"Prjn.PrjnScale.Abs": "20",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
					"Prjn.Learn.Learn":   "false",
				}},
			{Sel: "#PFCToSTNp", Desc: "strong pfc to stn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: "#PFCToSTNs", Desc: "strong pfc to stn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: ".SuperToPT", Desc: "one-to-one from super -- just use fixed nonlearning prjn so can control behavior easily",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1",    // keep this constant -- only self vs. this -- thal is modulatory
					"Prjn.PrjnScale.Abs": "0.01", // monitor maint early and other maint stats with PTMaintLayer ModGain = 0 to set this so super alone is not able to drive it.
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".PTSelfMaint", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel":    "1",      // use abs to manipulate
					"Prjn.PrjnScale.Abs":    "2",      // 2 > 1
					"Prjn.Learn.LRate.Base": "0.0001", // slower > faster
					"Prjn.SWt.Init.Mean":    "0.5",
					"Prjn.SWt.Init.Var":     "0.5", // high variance so not just spreading out over time
				}},
			{Sel: ".SuperToThal", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1.0",
					"Prjn.PrjnScale.Abs": "2.0", // if this is too strong, it gates to the wrong CS
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".ThalToSuper", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".ThalToPT", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1.0",
					"Prjn.Com.GType":     "ModulatoryG", // this marks as modulatory with extra ModGain factor
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".CTtoThal", Desc: "",
				Params: params.Params{
					"Prjn.SWt.Init.Var":  "0.25",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".CTCtxtPrjn", Desc: "all CT context prjns",
				Params: params.Params{
					"Prjn.Learn.LRate.Base":    "0.01", // trace: .01 > .005 > .02; .03 > .02 > .01 -- .03 std
					"Prjn.Learn.Trace.Tau":     "1",    // 2 > 1
					"Prjn.Learn.Trace.SubMean": "0",    // 0 > 1 -- 1 is especially bad
				}},
		}},
	},
	{Name: "NoTrainMtx", Desc: "turn off training in Mtx", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "MatrixPrjn", Desc: "learning in mtx",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
				}},
		}},
	},
	{Name: "LearnWts", Desc: "learned weights", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#ACCPosToMtxGo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.8",
				}},
			{Sel: "#ACCNegToMtxGo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.40",
				}},
			{Sel: "#PFCToMtxGo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.8",
				}},
			{Sel: "#ACCPosToMtxNo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.12",
				}},
			{Sel: "#ACCNegToMtxNo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.8",
				}},
			{Sel: "#PFCToMtxNo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.75",
				}},
		}},
	},
	{Name: "WtScales", Desc: "these should all be hard-coded", Sheets: params.Sheets{
		"Network": &params.Sheet{
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
					"Prjn.PrjnScale.Abs": "0.3", // 0.5 def
				}},
			{Sel: "#GPeInToSTNp", Desc: "not very relevant -- pause knocks out anyway -- if too much higher than this, causes oscillations.",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.1",
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
					"Prjn.PrjnScale.Abs": "0.7", // was .9
				}},
			{Sel: "#MtxGoToGPeOut", Desc: "This is key driver of Go threshold, along with to GPi",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.5",
				}},
			{Sel: "#MtxGoToGPi", Desc: "go influence on gating -- slightly weaker than integrated GPeIn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.8", // works over wide range: 0.5 - 1 -- learning controls
				}},
			{Sel: "#GPeInToGPi", Desc: "nogo influence on gating -- decreasing produces more graded function of Go",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: "#STNsToGPi", Desc: "keeps GPi active until GPeIn signal has been integrated a bit -- hold-your-horses",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": ".3", // .3 is effective in blocking GPI
				}},
			{Sel: "#STNpToGPi", Desc: "strong initial phasic activation",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: "#GPiToPFCVM", Desc: "final inhibition",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // 2 default
				}},
		}},
	},
}
