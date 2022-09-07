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
					"Layer.Act.Clamp.Ge":   "0.3",
				}},
			{Sel: "#ACCNeg", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Act.Clamp.Ge":   "0.3",
				}},
			{Sel: "#ACCPos", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Act.Clamp.Ge":   "0.3",
				}},
			{Sel: "#PFCo", Desc: "slower FB inhib for smoother dynamics",
				Params: params.Params{}},
			{Sel: "#STNp", Desc: "Pausing STN",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Layer.Gi":    "0.6",
					"Layer.Ca.SKCa.Gbar":      "5",
					"Layer.Ca.SKCa.C50":       "0.6",
					"Layer.Ca.SKCa.ActTau":    "10",
					"Layer.Ca.SKCa.DeTau":     "50",
					"Layer.Ca.CaScale":        "0.4",
					"Layer.Ca.CaIncTau":       "10",
					"Layer.Ca.CaDecayTau":     "30",
				}},
			{Sel: "#STNs", Desc: "Sustained STN",
				Params: params.Params{
					"Layer.Act.Init.Ge":       "0.2",
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Layer.Gi":    "0.2",
					"Layer.Ca.SKCa.Gbar":      "5",
					"Layer.Ca.SKCa.C50":       "0.6",
					"Layer.Ca.SKCa.ActTau":    "10",
					"Layer.Ca.SKCa.DeTau":     "50",
					"Layer.Ca.CaScale":        "8",
					"Layer.Ca.CaIncTau":       "1000",
					"Layer.Ca.CaDecayTau":     "1000",
				}},
			{Sel: "GPLayer", Desc: "all gp",
				Params: params.Params{
					"Layer.Act.Init.Ge":       "0.3",
					"Layer.Act.Init.GeVar":    "0.1", // use this instead of noise
					"Layer.Act.Init.GiVar":    "0.1", // use this instead of noise
					"Layer.Inhib.ActAvg.Init": "0.2",
				}},
			// {Sel: "#SNc", Desc: "SNc -- no clamp limits",
			// 	Params: params.Params{
			// 	}},
			{Sel: "MatrixPrjn", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs":    "2.0", // stronger
					"Prjn.SWt.Init.Mean":    "0.5",
					"Prjn.SWt.Init.Var":     "0.25",
					"Prjn.Trace.CurTrlDA":   "true",
					"Prjn.Learn.Learn":      "true",
					"Prjn.Learn.Lrate.Base": "0.1",
				}},
			{Sel: "#VThalToPFCo", Desc: "usually uniform weights",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.9",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.SWt.Init.Sym":  "false",
					"Prjn.Learn.Learn":   "false",
					"Prjn.PrjnScale.Abs": ".5", // modulatory
				}},
			{Sel: "#ACCPosToMtxGo", Desc: "strong",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3.0",
				}},
			{Sel: "#ACCNegToMtxGo", Desc: "strong",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3.0",
				}},
			{Sel: "#ACCPosToMtxNo", Desc: "strong",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3.0",
				}},
			{Sel: "#ACCNegToMtxNo", Desc: "strong",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3.0",
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
					"Prjn.PrjnScale.Abs": "0.1",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: "#PFCToVThal", Desc: "strong",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: "#GPeTAToMtxGo", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.5", // 0.8 def
				}},
			{Sel: "#GPeInToMtxGo", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.3",
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
					"Prjn.SWt.Init.Mean": "0.99",
				}},
			{Sel: "#ACCNegToMtxGo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.41",
				}},
			{Sel: "#PFCToMtxGo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.99",
				}},
			{Sel: "#ACCPosToMtxNo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.12",
				}},
			{Sel: "#ACCNegToMtxNo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.90",
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
					"Prjn.PrjnScale.Abs": "0.8", // this is key param to control WTA selectivity!
				}},
			{Sel: "#GPeTAToMtxNo", Desc: "nonspecific gating activity surround inhibition -- wta",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.3", // must be relatively weak to prevent oscillations
				}},
			{Sel: ".GPeInToMtx", Desc: "provides weak counterbalance for GPeTA -> Mtx to reduce oscillations",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.3",
				}},
			{Sel: "#GPeOutToGPeIn", Desc: "just enough to (dis)inhibit GPeIn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.5",
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
					"Prjn.PrjnScale.Abs": "0.1",
				}},
			{Sel: "#MtxNoToGPeIn", Desc: "primary classical NoGo pathway",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: "#GPeInToGPeTA", Desc: "just enough to knock down in baseline state",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.9",
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

// BGateParams is the full set of BGate params
var BGateParams = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Inhib.Layer.On":     "false",
					"Layer.Inhib.ActAvg.Init":  "0.25",
					"Layer.Inhib.ActAvg.Fixed": "true",
					"Layer.Inhib.Self.On":      "true",
					"Layer.Inhib.Self.Gi":      "0.4",
					"Layer.Inhib.Self.Tau":     "3.0",
					"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
					"Layer.Act.Dt.VmTau":       "4",
					// "Layer.Act.Dt.GTau":        "5", // 5 also works but less smooth RT dist
					"Layer.Act.Gbar.L":     "0.1",
					"Layer.Act.Init.Decay": "0",
				}},
			{Sel: ".GP", Desc: "all GP are tonically active",
				Params: params.Params{
					"Layer.Act.Init.Vm":   "0.9",
					"Layer.Act.Init.Act":  "0.5",
					"Layer.Act.Erev.L":    "0.9",
					"Layer.Act.Gbar.L":    "0.2", // stronger here makes them more robust to inputs -- .2 is best
					"Layer.Inhib.Self.Gi": "0.4", // this puts eq act at .5 instead of higher -- important
				}},
			{Sel: "#STN", Desc: "STN is tonically active -- same as GP",
				Params: params.Params{
					"Layer.Act.Init.Vm":   "0.9",
					"Layer.Act.Init.Act":  "0.5",
					"Layer.Act.Erev.L":    "0.9",
					"Layer.Act.Gbar.L":    "0.2", // stronger here makes them more robust to inputs -- .2 is best
					"Layer.Inhib.Self.Gi": "0.4", // this puts eq act at .5 instead of higher -- important
				}},
			{Sel: "#GPi", Desc: "slower dynamics in GPi, to allow GPe time to sort through",
				Params: params.Params{
					// "Layer.Act.Dt.GTau": "20",  // needed
					"Layer.Act.Init.Vm": "0.8", // gives .5 ish starting
					"Layer.Act.Erev.L":  "0.8",
					"Layer.Act.Gbar.L":  "0.3", // stronger here makes them more robust to inputs
				}},
			{Sel: ".Matrix", Desc: "Matrix has more self inhib -- key for proportional response",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Layer.FB": "0",
					"Layer.Inhib.Layer.Gi": "1.5",
					"Layer.Inhib.Self.Gi":  "0.6",
				}},
			{Sel: "Prjn", Desc: "defaults",
				Params: params.Params{
					"Prjn.Learn.Learn":   "true",
					"Prjn.SWt.Init.Mean": "0.9",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".Inhib", Desc: "All inhib starts at 2 by default",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: ".MatrixPrjn", Desc: "learning case -- .5 initial weights",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
					"Prjn.SWt.Init.Mean": "0.5",
				}},
			{Sel: ".GPiPrjn", Desc: "weaker weights better ",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.5",
				}},
			{Sel: "#PFCToMtxGo", Desc: "weaker closed loop",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": ".1",
				}},
			{Sel: "#PFCToMtxNo", Desc: "weaker closed loop",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": ".1",
				}},
			{Sel: "#PFCToSTN", Desc: "strong pfc to stn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
				}},
			{Sel: "#GPeInToSTN", Desc: "if too much higher than this, causes oscillations.",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": ".5",
				}},
			{Sel: "#GPeInToGPeTA", Desc: "weaker inhib of GPeTA to allow dynamics there",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: ".FromGPeTA", Desc: "reducing below 2 reduces RT effects from conflict / NoGo",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3", // stronger selection effects, but higher threshold overall
				}},
			{Sel: "#GPeOutToGPeIn", Desc: "weaker inhibition to give graded GPeIn response",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
				}},
			{Sel: "#MtxGoToGPeIn", Desc: "wta pathway -- weaker effect",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.PrjnScale.Rel": ".3", // .4 is biggest that still works
				}},
			{Sel: "#STNToGPeOut", Desc: "weaker STN -> GPeOut",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": ".1",
				}},
			{Sel: "#STNToGPeIn", Desc: "stronger STN -> GPeIn to kick it high at start",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": ".5",
				}},
			{Sel: "#STNToGPeTA", Desc: "weaker STN -> GPeTA -- no need",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": ".1",
				}},
			{Sel: "#MtxGoToGPeOut", Desc: "This is key driver of Go threshold, along with to GPi",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": ".5",
				}},
			{Sel: "#MtxGoToGPi", Desc: "go influence on gating",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": ".2",
				}},
			{Sel: "#GPeInToGPi", Desc: "nogo influence on gating",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.8",
				}},
			{Sel: "#MtxNoToGPeIn", Desc: "This is key factor for Nogo -- only way it does anything!",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.5",
				}},
			// {Sel: "#SNc", Desc: "SNc -- no clamp limits",
			// 	Params: params.Params{
			// 	}},
			// trained weights
			{Sel: "#ACCPosToMtxGo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.99",
				}},
			{Sel: "#ACCNegToMtxGo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.41",
				}},
			{Sel: "#PFCToMtxGo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.99",
				}},
			{Sel: "#ACCPosToMtxNo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.12",
				}},
			{Sel: "#ACCNegToMtxNo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.90",
				}},
			{Sel: "#PFCToMtxNo", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.75",
				}},
			{Sel: "#MtxGoToGPi", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.55",
				}},
			{Sel: "#GPeInToGPi", Desc: "trained wts",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.55",
				}},
		},
	}},
}
