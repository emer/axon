// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				ly.Inhib.Layer.On =       "false",
				ly.Inhib.Layer.Gi =       "1.0",
				ly.Inhib.ActAvg.Nominal = "0.1",
				ly.Acts.Dt.GeTau =        "5",
				ly.Acts.Dt.GiTau =        "7",
				ly.Acts.Gbar.I =          "1.0",
				ly.Acts.Gbar.L =          "0.2",
				ly.Acts.Decay.Act =       "0.0", // 0.2 def
				ly.Acts.Decay.Glong =     "0.0", // 0.6 def
				ly.Acts.Noise.On =        "false",
				ly.Acts.Noise.GeHz =      "100",
				ly.Acts.Noise.Ge =        "0.002", // 0.001 min
				ly.Acts.Noise.GiHz =      "200",
				ly.Acts.Noise.Gi =        "0.002", // 0.001 min
			}},
		{Sel: ".InhibLay", Doc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.5",
				ly.Acts.Spikes.Thr =      "0.5",
				ly.Acts.Spikes.Tr =       "1",   // 3 def
				ly.Acts.Spikes.VmR =      "0.4", // key for firing early, plus noise
				ly.Acts.Init.Vm =         "0.4", // key for firing early, plus noise
				ly.Acts.Erev.L =          "0.4", // more excitable
				ly.Acts.Gbar.L =          "0.2", // smaller, less leaky..
				ly.Acts.KNa.On =          "false",
				ly.Acts.GabaB.Gbar =      "0", // no gabab
				ly.Acts.NMDA.Gbar =       "0", // no nmda
				ly.Acts.Noise.On =        "false",
				ly.Acts.Noise.Ge =        "0.01", // 0.001 min
				ly.Acts.Noise.Gi =        "0.0",  //
			}},
		{Sel: "#Layer0", Doc: "Input layer",
			Params: params.Params{
				ly.Acts.Clamp.Ge = "0.6", // no inhib so needs to be lower
				ly.Acts.Noise.On = "true",
				ly.Acts.Noise.Gi = "0.002", // hard to disrupt strong inputs!
			}},
		{Sel: "Path", Doc: "no learning",
			Params: params.Params{
				pt.Learn.Learn = "false",
				// pt.SWts.Init.Dist = "Uniform",
				pt.SWts.Init.Mean = "0.5",
				pt.SWts.Init.Var =  "0.25",
				pt.Com.Delay =      "2",
			}},
		{Sel: ".BackPath", Doc: "feedback excitatory",
			Params: params.Params{
				pt.PathScale.Rel = "0.2",
			}},
		{Sel: ".InhibPath", Doc: "inhibitory pathways",
			Params: params.Params{
				// pt.SWts.Init.Dist = "Uniform",
				pt.SWts.Init.Mean = "0.5",
				pt.SWts.Init.Var =  "0",
				pt.SWts.Init.Sym =  "false",
				pt.Com.Delay =      "0",
				pt.PathScale.Abs =  "6", // key param
			}},
		{Sel: ".ToInhib", Doc: "to inhibitory pathways",
			Params: params.Params{
				pt.Com.Delay = "1",
			}},
		// {Sel: ".RandSc", Doc: "random shortcut",
		// 	Params: params.Params{
		// 		pt.Learn.LRate.Base = "0.001", //
		// 		// pt.Learn.Learn =      "false",
		// 		pt.PathScale.Rel = "0.5",   // .5 > .8 > 1 > .4 > .3 etc
		// 		pt.SWts.Adapt.On = "false", // seems better
		// 		// pt.SWts.Init.Var =  "0.05",
		// 	}},
	},
	"FSFFFB": = {
		{Sel: "Layer", Doc: "use FSFFFB computed inhibition",
			Params: params.Params{
				ly.Inhib.Layer.On =     "true",
				ly.Inhib.Layer.Gi =     "1.0",
				ly.Inhib.Layer.SS =     "30", // 30
				ly.Inhib.Layer.FB =     "1",
				ly.Inhib.Layer.FS0 =    "0.1",
				ly.Inhib.Layer.FSTau =  "6",
				ly.Inhib.Layer.SSfTau = "20",
				ly.Inhib.Layer.SSiTau = "50",
			}},
		{Sel: ".InhibPath", Doc: "inhibitory pathways",
			Params: params.Params{
				pt.PathScale.Abs = "0",
			}},
	},
	"Untrained": {
		{Sel: ".Excite", Doc: "excitatory connections",
			Params: params.Params{
				// pt.SWts.Init.Dist = "Uniform",
				pt.SWts.Init.Mean = "0.5",
				pt.SWts.Init.Var =  "0.25",
			}},
	},
	"Trained": {
		{Sel: ".Excite", Doc: "excitatory connections",
			Params: params.Params{
				// pt.SWts.Init.Dist = "Gaussian",
				pt.SWts.Init.Mean = "0.4",
				pt.SWts.Init.Var =  "0.8",
			}},
	},
}
