// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"NetSize": &params.Sheet{
			{Sel: ".Hidden", Desc: "all hidden layers",
				Params: params.Params{
					"Layer.X":         "10", //todo layer size correspondence between areas that are connected upstream parameter - get there when we get there
					"Layer.Y":         "10",
					"NumHiddenLayers": "2", //todo not implemented
				},
				Hypers: params.Hypers{
					//"Layer.X": {"StdDev": "0.3", "Min": "2", "Type": "Int"},
					//"Layer.Y": {"StdDev": "0.3", "Min": "2", "Type": "Int"},
				},
			},
		},
		// Some network parameters chosen by WandB: https://wandb.ai/obelisk/models/reports/Good-Weights-for-RA25--VmlldzoxODMxODI1
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					// All params with importance >=5 have hypers
					"Layer.Inhib.Layer.Gi": "1.2", // 1.2 > 1.1     importance: 10
					// TODO This param should vary with Gi it looks like
					"Layer.Inhib.ActAvg.Init": "0.04",  // 0.04 for 1.2, 0.08 for 1.1  importance: 10
					"Layer.Inhib.Layer.Bg":    "0.3",   // 0.3 > 0.0   importance: 2
					"Layer.Act.Decay.Glong":   "0.6",   // 0.6   importance: 2
					"Layer.Act.Dend.GbarExp":  "0.2",   // 0.2 > 0.1 > 0   importance: 5
					"Layer.Act.Dend.GbarR":    "3",     // 3 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels importance: 5
					"Layer.Act.Dt.VmDendTau":  "5",     // 5 > 2.81 here but small effect importance: 1
					"Layer.Act.Dt.VmSteps":    "2",     // 2 > 3 -- somehow works better importance: 1
					"Layer.Act.Dt.GeTau":      "5",     // importance: 1
					"Layer.Act.NMDA.Gbar":     "0.123", //  importance: 7 // From Wandb
					"Layer.Act.NMDA.MgC":      "1.4",
					"Layer.Act.NMDA.Voff":     "5",
					"Layer.Act.GABAB.Gbar":    "0.124", // 0.2 > 0.15  importance: 7 // From Wandb
				}, Hypers: params.Hypers{
					// These shouldn't be set without also searching for the same value in specific layers like #Input, because it'll clobber them, since it's in a separate Params sheet.
					//"Layer.Inhib.Layer.Gi":    {"StdDev": "0.15"},
					//"Layer.Inhib.ActAvg.Init": {"StdDev": "0.02", "Min": "0.01"},

					//"Layer.Act.Dend.GbarExp":  {"StdDev": "0.05"},
					//"Layer.Act.Dend.GbarR":    {"StdDev": "1"},
					"Layer.Act.NMDA.Gbar":  {"StdDev": "0.05"},
					"Layer.Act.GABAB.Gbar": {"StdDev": "0.05"},
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.827", // 0.9 > 1.0 // From Wandb
					"Layer.Act.Clamp.Ge":      "1.26",  // 1.0 > 0.6 >= 0.7 == 0.5 // From Wandb
					"Layer.Inhib.ActAvg.Init": "0.15",  // .24 nominal, lower to give higher excitation
				},
				Hypers: params.Hypers{
					"Layer.Inhib.Layer.Gi": {"StdDev": ".1", "Min": "0", "Priority": "2", "Scale": "LogLinear"},
					"Layer.Act.Clamp.Ge":   {"StdDev": ".2"},
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 >= 0.8 > 1.0 > 0.7 even with adapt -- not beneficial to start low
					"Layer.Inhib.ActAvg.Init": "0.24", // this has to be exact for adapt
					"Layer.Act.Spike.Tr":      "1",    // 1 is new minimum.
					"Layer.Act.Clamp.Ge":      "0.6",  // .6 > .5 v94
					// "Layer.Act.NMDA.Gbar":     "0.3",  // higher not better
				}},
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":    "0.2",   // 0.04 no rlr, 0.2 rlr; .3, WtSig.Gain = 1 is pretty close  //importance: 10 // From Wandb
					"Prjn.SWt.Adapt.Lrate":     "0.08",  // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint //importance: 5
					"Prjn.SWt.Init.SPct":       "0.432", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..  //importance: 7 // From Wandb
					"Prjn.Learn.KinaseCa.Rule": "SynSpkTheta",
				},
				Hypers: params.Hypers{
					"Prjn.Learn.Lrate.Base": {"StdDev": "0.1"},
					//"Prjn.SWt.Adapt.Lrate":  {"StdDev": "0.025"},
					"Prjn.SWt.Init.SPct": {"StdDev": "0.25", "Min": "0.1"},
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5 //importance: 9
				},
				Hypers: params.Hypers{
					"Prjn.PrjnScale.Rel": {"StdDev": ".2", "Min": "0.01"},
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.CmdArgs.MaxEpcs": "100",
				}},
		},
	}},
}
