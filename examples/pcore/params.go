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
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Act.Clamp.Ge":   "0.6",
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
			{Sel: "#PFCo", Desc: "slower FB inhib for smoother dynamics",
				Params: params.Params{}},
			{Sel: "#STNp", Desc: "Pausing STN",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Layer.Gi":    "0.6",
					"Layer.Ca.SKCa.Gbar":      "2",
					"Layer.Ca.SKCa.C50":       "0.6",
					"Layer.Ca.SKCa.ActTau":    "10",
					"Layer.Ca.SKCa.DeTau":     "50",
					"Layer.Ca.CaScale":        "4",
				}},
			{Sel: "#STNs", Desc: "Sustained STN",
				Params: params.Params{
					"Layer.Act.Init.Ge":       "0.2",
					"Layer.Act.Init.GeVar":    "0.2",
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Layer.Gi":    "0.2",
					"Layer.Ca.SKCa.Gbar":      "2",
					"Layer.Ca.SKCa.C50":       "0.6",
					"Layer.Ca.SKCa.ActTau":    "10",
					"Layer.Ca.SKCa.DeTau":     "50",
					"Layer.Ca.CaScale":        "3",
				}},
			{Sel: "GPLayer", Desc: "all gp",
				Params: params.Params{
					"Layer.Act.Init.Ge":       "0.3",
					"Layer.Act.Init.GeVar":    "0.1",
					"Layer.Act.Init.GiVar":    "0.1",
					"Layer.Inhib.ActAvg.Init": "1",
				}},
			{Sel: "#GPi", Desc: "",
				Params: params.Params{
					"Layer.Act.Init.Ge": "0.5",
				}},
			{Sel: "MatrixLayer", Desc: "all mtx",
				Params: params.Params{
					"Layer.Matrix.GPHasPools":   "false",
					"Layer.Matrix.InvertNoGate": "true",
					"Layer.Matrix.GateThr":      "0.01",
					"Layer.Inhib.Layer.On":      "true",
					"Layer.Inhib.Layer.Gi":      "0.5", // 0.5 > 0.4
					"Layer.Inhib.Layer.FB":      "0.0",
				}},
			// {Sel: "#SNc", Desc: "SNc -- no clamp limits",
			// 	Params: params.Params{
			// 	}},
			{Sel: "MatrixPrjn", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs":    "1.0", // stronger
					"Prjn.SWt.Init.SPct":    "0",
					"Prjn.SWt.Init.Mean":    "0.5",
					"Prjn.SWt.Init.Var":     "0.25",
					"Prjn.Trace.CurTrlDA":   "true",
					"Prjn.Learn.Learn":      "true",
					"Prjn.Learn.Lrate.Base": "0.1",
				}},
			{Sel: ".BgFixed", Desc: "fixed, non-learning params",
				Params: params.Params{
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.Learn.Learn":   "false",
				}},
			{Sel: "#VThalToPFCo", Desc: "usually uniform weights",
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
			{Sel: "#PFCToSTNp", Desc: "strong pfc to stn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: "#PFCToSTNs", Desc: "strong pfc to stn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.3",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: "#PFCToVThal", Desc: "strong",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
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
			{Sel: "#GPeInToSTNs", Desc: "NOT currently used -- interferes with threshold-based Ca self-inhib dynamics",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.1",
				}},
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
			{Sel: "#GPiToVThal", Desc: "final inhibition",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // 2 orig
				}},
		}},
	},
}
