// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ra25

import (
	"github.com/emer/axon/v2/axon"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "all defaults",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.05       // 1.05 > 1.1 for short-term; 1.1 better long-run stability
				ly.Inhib.Layer.FB = 0.5        // 0.5 > 0.2 > 0.1 > 1.0 -- usu 1.0
				ly.Inhib.ActAvg.Nominal = 0.06 // 0.6 > 0.5
				ly.Acts.NMDA.MgC = 1.2         // 1.2 > 1.4 here, still..
				ly.Acts.VGCC.Ge = 0
				ly.Learn.CaSpike.SpikeCaSyn = 8
				// ly.Learn.CaLearn.ETraceTau = 4
				// ly.Learn.CaLearn.ETraceScale = 0.1 // 4,0.1 best in sequential
				ly.Learn.RLRate.SigmoidLinear.SetBool(false) // false > true here
				// ly.Learn.RLRate.Diff.SetBool(false)          // false = very bad
			}},
		{Sel: "#Input", Doc: "critical now to specify the activity level",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.9        // 0.9 > 1.0
				ly.Acts.Clamp.Ge = 1.5         // 1.5 > 1.0
				ly.Inhib.ActAvg.Nominal = 0.15 // .24 nominal, lower to give higher excitation
			}},
		{Sel: "#Output", Doc: "output definitely needs lower inhib -- true for smaller layers in general",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.65 // 0.65
				ly.Inhib.ActAvg.Nominal = 0.24
				ly.Acts.Spikes.Tr = 1             // 1 is new minimum.. > 3
				ly.Acts.Clamp.Ge = 0.8            // 0.8 > 0.6
				ly.Learn.RLRate.SigmoidMin = 0.05 // sigmoid derivative actually useful here!
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "basic path params",
			Set: func(pt *axon.PathParams) {
				// pt.Com.MaxDelay = 10 // robust to this
				// pt.Com.Delay = 10
				pt.Learn.LRate.Base = 0.02 // 0.06 for std, .02 for SynCaDiff?
				pt.SWts.Adapt.LRate = 0.1  // .1 >= .2,
				pt.SWts.Init.SPct = 0.5    // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
				pt.Learn.DWt.SubMean = 0   // 1 > 0 for long run stability
				pt.Learn.DWt.CaPScale = 1  // 1
				pt.Learn.DWt.SynCa20.SetBool(false)
				pt.Learn.DWt.SynCaDiff.SetBool(true) // todo: expt
				pt.Learn.DWt.LearnThr = 0.05
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.3 // 0.3 > 0.2 > 0.1 > 0.5
			}},
	},
}
