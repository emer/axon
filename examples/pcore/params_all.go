package main

import "github.com/emer/emergent/params"

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
					"Layer.Act.Gbar.L":         "0.1",
					"Layer.Act.Init.Decay":     "0",
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
