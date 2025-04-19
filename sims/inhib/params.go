// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inhib

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers: lower gain, slower, soft clamp",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Acts.Dt.GeTau = 5
				ly.Acts.Dt.GiTau = 7
				ly.Acts.Gbar.I = 100
				ly.Acts.Gbar.L = 20
				ly.Acts.Decay.Act = 0.0   // 0.2 def
				ly.Acts.Decay.Glong = 0.0 // 0.6 def
				ly.Acts.Noise.On.SetBool(false)
				ly.Acts.Noise.GeHz = 100
				ly.Acts.Noise.Ge = 0.002 // 0.001 min
				ly.Acts.Noise.GiHz = 200
				ly.Acts.Noise.Gi = 0.002 // 0.001 min
			}},
		{Sel: ".InhibLay", Doc: "generic params for all layers: lower gain, slower, soft clamp",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.5
				ly.Acts.Spikes.Thr = 0.5
				ly.Acts.Spikes.Tr = 1    // 3 def
				ly.Acts.Spikes.VmR = -60 // key for firing early, plus noise
				ly.Acts.Init.Vm = -60    // key for firing early, plus noise
				ly.Acts.Erev.L = -60     // more excitable
				ly.Acts.Gbar.L = 20
				ly.Acts.KNa.On.SetBool(false)
				ly.Acts.GabaB.Gk = 0 // no gabab
				ly.Acts.NMDA.Ge = 0  // no nmda
				ly.Acts.Noise.On.SetBool(false)
				ly.Acts.Noise.Ge = 0.01 // 0.001 min
				ly.Acts.Noise.Gi = 0.0  //
			}},
		{Sel: "#Layer0", Doc: "Input layer",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 0.6 // no inhib so needs to be lower
				ly.Acts.Noise.On.SetBool(true)
				ly.Acts.Noise.Gi = 0.002 // hard to disrupt strong inputs!
			}},
	},
	"FSFFFB": {
		{Sel: "Layer", Doc: "use FSFFFB computed inhibition",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.Layer.SS = 30 // 30
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Layer.FS0 = 0.1
				ly.Inhib.Layer.FSTau = 6
				ly.Inhib.Layer.SSfTau = 20
				ly.Inhib.Layer.SSiTau = 50
			}},
	},
	"Untrained": {},
	"Trained":   {},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "no learning",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false)
				pt.SWts.Init.Mean = 0.5
				pt.SWts.Init.Var = 0.25
				pt.Com.Delay = 2
			}},
		{Sel: ".BackPath", Doc: "feedback excitatory",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		{Sel: ".InhibPath", Doc: "inhibitory pathways",
			Set: func(pt *axon.PathParams) {
				// pt.SWts.Init.Dist = "Uniform
				pt.SWts.Init.Mean = 0.5
				pt.SWts.Init.Var = 0
				pt.SWts.Init.Sym.SetBool(false)
				pt.Com.Delay = 0
				pt.PathScale.Abs = 6 // key param
			}},
		{Sel: ".ToInhib", Doc: "to inhibitory pathways",
			Set: func(pt *axon.PathParams) {
				pt.Com.Delay = 1
			}},
	},
	"FSFFFB": {
		{Sel: ".InhibPath", Doc: "inhibitory pathways",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0
			}},
	},
	"Untrained": {
		{Sel: ".Excite", Doc: "excitatory connections",
			Set: func(pt *axon.PathParams) {
				// pt.SWts.Init.Dist = Uniform
				pt.SWts.Init.Mean = 0.5
				pt.SWts.Init.Var = 0.25
			}},
	},
	"Trained": {
		{Sel: ".Excite", Doc: "excitatory connections",
			Set: func(pt *axon.PathParams) {
				// pt.SWts.Init.Dist = Gaussian
				pt.SWts.Init.Mean = 0.4
				pt.SWts.Init.Var = 0.8
			}},
	},
}
