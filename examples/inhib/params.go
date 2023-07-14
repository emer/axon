// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/netparams"
	"github.com/emer/emergent/params"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Inhib.Layer.On":       "false",
				"Layer.Inhib.Layer.Gi":       "1.0",
				"Layer.Inhib.ActAvg.Nominal": "0.1",
				"Layer.Acts.Dt.GeTau":        "5",
				"Layer.Acts.Dt.GiTau":        "7",
				"Layer.Acts.Gbar.I":          "1.0",
				"Layer.Acts.Gbar.L":          "0.2",
				"Layer.Acts.Decay.Act":       "0.0", // 0.2 def
				"Layer.Acts.Decay.Glong":     "0.0", // 0.6 def
				"Layer.Acts.Noise.On":        "false",
				"Layer.Acts.Noise.GeHz":      "100",
				"Layer.Acts.Noise.Ge":        "0.002", // 0.001 min
				"Layer.Acts.Noise.GiHz":      "200",
				"Layer.Acts.Noise.Gi":        "0.002", // 0.001 min
			}},
		{Sel: ".InhibLay", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.5",
				"Layer.Acts.Spikes.Thr":      "0.5",
				"Layer.Acts.Spikes.Tr":       "1",   // 3 def
				"Layer.Acts.Spikes.VmR":      "0.4", // key for firing early, plus noise
				"Layer.Acts.Init.Vm":         "0.4", // key for firing early, plus noise
				"Layer.Acts.Erev.L":          "0.4", // more excitable
				"Layer.Acts.Gbar.L":          "0.2", // smaller, less leaky..
				"Layer.Acts.KNa.On":          "false",
				"Layer.Acts.GabaB.Gbar":      "0", // no gabab
				"Layer.Acts.NMDA.Gbar":       "0", // no nmda
				"Layer.Acts.Noise.On":        "false",
				"Layer.Acts.Noise.Ge":        "0.01", // 0.001 min
				"Layer.Acts.Noise.Gi":        "0.0",  //
			}},
		{Sel: "#Layer0", Desc: "Input layer",
			Params: params.Params{
				"Layer.Acts.Clamp.Ge": "0.6", // no inhib so needs to be lower
				"Layer.Acts.Noise.On": "true",
				"Layer.Acts.Noise.Gi": "0.002", // hard to disrupt strong inputs!
			}},
		{Sel: "Prjn", Desc: "no learning",
			Params: params.Params{
				"Prjn.Learn.Learn": "false",
				// "Prjn.SWts.Init.Dist": "Uniform",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.25",
				"Prjn.Com.Delay":      "2",
			}},
		{Sel: ".BackPrjn", Desc: "feedback excitatory",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.2",
			}},
		{Sel: ".InhibPrjn", Desc: "inhibitory projections",
			Params: params.Params{
				// "Prjn.SWts.Init.Dist": "Uniform",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0",
				"Prjn.SWts.Init.Sym":  "false",
				"Prjn.Com.Delay":      "0",
				"Prjn.PrjnScale.Abs":  "6", // key param
			}},
		{Sel: ".ToInhib", Desc: "to inhibitory projections",
			Params: params.Params{
				"Prjn.Com.Delay": "1",
			}},
		{Sel: ".RndSc", Desc: "random shortcut",
			Params: params.Params{
				"Prjn.Learn.LRate.Base": "0.001", //
				// "Prjn.Learn.Learn":      "false",
				"Prjn.PrjnScale.Rel": "0.5",   // .5 > .8 > 1 > .4 > .3 etc
				"Prjn.SWts.Adapt.On": "false", // seems better
				// "Prjn.SWts.Init.Var":  "0.05",
			}},
	},
	"FSFFFB": {
		{Sel: "Layer", Desc: "use FSFFFB computed inhibition",
			Params: params.Params{
				"Layer.Inhib.Layer.On":     "true",
				"Layer.Inhib.Layer.Gi":     "1.0",
				"Layer.Inhib.Layer.SS":     "30", // 30
				"Layer.Inhib.Layer.FB":     "1",
				"Layer.Inhib.Layer.FS0":    "0.1",
				"Layer.Inhib.Layer.FSTau":  "6",
				"Layer.Inhib.Layer.SSfTau": "20",
				"Layer.Inhib.Layer.SSiTau": "50",
			}},
		{Sel: ".InhibPrjn", Desc: "inhibitory projections",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0",
			}},
	},
	"Untrained": {
		{Sel: ".Excite", Desc: "excitatory connections",
			Params: params.Params{
				// "Prjn.SWts.Init.Dist": "Uniform",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.25",
			}},
	},
	"Trained": {
		{Sel: ".Excite", Desc: "excitatory connections",
			Params: params.Params{
				// "Prjn.SWts.Init.Dist": "Gaussian",
				"Prjn.SWts.Init.Mean": "0.4",
				"Prjn.SWts.Init.Var":  "0.8",
			}},
	},
}
