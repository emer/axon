// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Inhib.Layer.On":     "true",
					"Layer.Inhib.Layer.Gi":     "1.0",
					"Layer.Inhib.Layer.SS":     "30", // 30
					"Layer.Inhib.Layer.FB":     "1",
					"Layer.Inhib.Layer.FS0":    "0.1",
					"Layer.Inhib.Layer.FSTau":  "6",
					"Layer.Inhib.Layer.SSfTau": "20",
					"Layer.Inhib.Layer.SSiTau": "50",
					"Layer.Inhib.ActAvg.Init":  "0.1",
					"Layer.Inhib.Inhib.AvgTau": "30", // 20 > 30 ?
					"Layer.Act.Dt.GeTau":       "5",
					"Layer.Act.Dt.GiTau":       "7",
					"Layer.Act.Gbar.I":         "1.0",
					"Layer.Act.Gbar.L":         "0.2",
					"Layer.Act.GABAB.Gbar":     "0.2",
					"Layer.Act.NMDA.Gbar":      "0.15",
					"Layer.Act.Decay.Act":      "0.0", // 0.2 def
					"Layer.Act.Decay.Glong":    "0.0", // 0.6 def
					"Layer.Act.Noise.On":       "false",
					"Layer.Act.Noise.GeHz":     "100",
					"Layer.Act.Noise.Ge":       "0.002", // 0.001 min
					"Layer.Act.Noise.GiHz":     "200",
					"Layer.Act.Noise.Gi":       "0.002", // 0.001 min
				}},
			{Sel: ".InhibLay", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.5",
					"Layer.Act.Spike.Thr":     "0.5",
					"Layer.Act.Spike.Tr":      "1",   // 3 def
					"Layer.Act.Spike.VmR":     "0.4", // key for firing early, plus noise
					"Layer.Act.Init.Vm":       "0.4", // key for firing early, plus noise
					"Layer.Act.Erev.L":        "0.4", // more excitable
					"Layer.Act.Gbar.L":        "0.2", // smaller, less leaky..
					"Layer.Act.KNa.On":        "false",
					"Layer.Act.GABAB.Gbar":    "0", // no gabab
					"Layer.Act.NMDA.Gbar":     "0", // no nmda
					"Layer.Act.Noise.On":      "false",
					"Layer.Act.Noise.Ge":      "0.01", // 0.001 min
					"Layer.Act.Noise.Gi":      "0.0",  //
				}},
			{Sel: "#Layer0", Desc: "Input layer",
				Params: params.Params{
					"Layer.Act.Clamp.Ge": "0.6", // no inhib so needs to be lower
					"Layer.Act.Noise.On": "true",
					"Layer.Act.Noise.Gi": "0.002", // hard to disrupt strong inputs!
				}},
			{Sel: "Prjn", Desc: "no learning",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					// "Prjn.SWt.Init.Dist": "Uniform",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
					"Prjn.Com.Delay":     "2",
				}},
			{Sel: ".Back", Desc: "feedback excitatory",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: ".Inhib", Desc: "inhibitory projections",
				Params: params.Params{
					// "Prjn.SWt.Init.Dist": "Uniform",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0",
					"Prjn.SWt.Init.Sym":  "false",
					"Prjn.Com.Delay":     "0",
					"Prjn.PrjnScale.Abs": "0",
				}},
			{Sel: ".ToInhib", Desc: "to inhibitory projections",
				Params: params.Params{
					"Prjn.Com.Delay": "1",
				}},
			{Sel: ".RndSc", Desc: "random shortcut",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.001", //
					// "Prjn.Learn.Learn":      "false",
					"Prjn.PrjnScale.Rel": "0.5",   // .5 > .8 > 1 > .4 > .3 etc
					"Prjn.SWt.Adapt.On":  "false", // seems better
					// "Prjn.SWt.Init.Var":  "0.05",
				}},
		},
	}},
	{Name: "Untrained", Desc: "simulates untrained weights -- lower variance", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".Excite", Desc: "excitatory connections",
				Params: params.Params{
					// "Prjn.SWt.Init.Dist": "Uniform",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
		},
	}},
	{Name: "Trained", Desc: "simulates trained weights -- higher variance", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".Excite", Desc: "excitatory connections",
				Params: params.Params{
					// "Prjn.SWt.Init.Dist": "Gaussian",
					"Prjn.SWt.Init.Mean": "0.4",
					"Prjn.SWt.Init.Var":  "0.8",
				}},
		},
	}},
}
