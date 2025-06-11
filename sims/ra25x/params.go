// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ra25x

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "all defaults",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.06 // 0.06 > 0.05
				ly.Inhib.Layer.Gi = 1.1        // 1.1 > 1.05 even for SSGi=2
				ly.Inhib.Layer.SS = 30         // 30 > others
				ly.Inhib.Layer.FS0 = 0.1
				ly.Inhib.Layer.FSTau = 6
				ly.Inhib.Layer.FB = 0.5         // 0.5 > 0.2 > 0.1 > 1.0
				ly.Inhib.Layer.SSfTau = 20      // 20 > 30  > 15
				ly.Inhib.Layer.SSiTau = 50      // 50 > 40 -- try 40, 60 @ gi= 1.1?
				ly.Inhib.ActAvg.AdaptRate = 0.1 // 0.1 seems good
				ly.Inhib.ActAvg.LoTol = 0.8
				ly.Inhib.ActAvg.HiTol = 0.0
				ly.Acts.Dend.SSGi = 2.0   // 2.0 > 1.5 more reliable
				ly.Acts.Decay.Act = 0.2   // 0.2 def
				ly.Acts.Decay.Glong = 0.6 // 0.6 def
				ly.Acts.NMDA.Ge = 0.006   // 0.006 def
				ly.Acts.NMDA.MgC = 1.2    // 1.2 > 1.4 here
				ly.Acts.NMDA.Voff = 0     // 5 == 0 for trace
				ly.Acts.NMDA.Tau = 100    // 100 def -- 50 is sig worse
				ly.Acts.Mahp.Gk = 0.02    // 0.05 works..
				ly.Acts.Sahp.Gk = 0.05    //
				ly.Acts.Sahp.Off = 0.8    //
				ly.Acts.Sahp.Slope = 0.02 //
				ly.Acts.Sahp.CaTau = 5    //
				ly.Acts.GabaB.Gk = 0.015  // 0.015 > lower
				ly.Acts.KNa.On.SetBool(false)
				ly.Acts.AK.Gk = 0.1                      // 0.05 to 0.1 likely good per urakubo, but 1.0 needed to prevent vgcc blowup
				ly.Acts.VGCC.Ge = 0.02                   // 0.12 per urakubo / etc models, but produces too much high-burst plateau -- even 0.05 with AK = .1 blows up
				ly.Acts.VGCC.Ca = 25                     // 25 / 10tau default
				ly.Learn.CaLearn.Norm = 80               // 80 works
				ly.Learn.CaLearn.SpikeVGCC.SetBool(true) // sig better..
				ly.Learn.CaLearn.SpikeVgccCa = 35        // 70 / 5 or 35 / 10 both work
				ly.Learn.CaLearn.VgccTau = 10            // 10 > 5 ?
				ly.Learn.CaLearn.Dt.MTau = 2             // 2 > 1 ?
				ly.Learn.CaSpike.SpikeCaM = 8            // 8 produces reasonable 0-1 norm CaSpk levels?
				ly.Learn.CaSpike.CaSynTau = 30           // 30 > 20, 40
				ly.Learn.CaSpike.Dt.MTau = 5             // 5 > 10?
				ly.Learn.LearnNMDA.MgC = 1.4             // 1.2 for unified Act params, else 1.4
				ly.Learn.LearnNMDA.Voff = 0              // 0 for unified Act params, else 5
				ly.Learn.LearnNMDA.Ge = 0.006
				ly.Learn.LearnNMDA.Tau = 100               // 100 def
				ly.Learn.TrgAvgAct.RescaleOn.SetBool(true) // true > false even with adapt gi
				ly.Learn.TrgAvgAct.SubMean = 1             // 1 > 0 essential
				ly.Learn.TrgAvgAct.SynScaleRate = 0.0002   // 0.0002 > others; 0.005 not as good
				ly.Learn.RLRate.On.SetBool(true)           // beneficial for trace
				ly.Learn.RLRate.SigmoidMin = 0.05          // 0.05 > .1 > .02
				ly.Learn.RLRate.Diff.SetBool(true)
				ly.Learn.RLRate.DiffThr = 0.02 // 0.02 def - todo
				ly.Learn.RLRate.SpikeThr = 0.1 // 0.1 def
				ly.Learn.RLRate.Min = 0.001
			}},
		{Sel: "#Input", Doc: "critical now to specify the activity level",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.9        // 0.9 > 1.0
				ly.Acts.Clamp.Ge = 1.5         // 1.5 matches old fffb for gex (v13) > 1.0
				ly.Inhib.ActAvg.Nominal = 0.15 // .24 nominal, lower to give higher excitation
				ly.Acts.VGCC.Ca = 1            // otherwise dominates display
				ly.Acts.Decay.Act = 1          // this is subtly beneficial
				ly.Acts.Decay.Glong = 1
			}},
		{Sel: ".SuperLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.1        // 1.1 > others
				ly.Inhib.ActAvg.Nominal = 0.06 // 0.06 > 0.05
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
			}},
		{Sel: "#Output", Doc: "output definitely needs lower inhib -- true for smaller layers in general",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.24 // 0.24 > 0.3
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.Gi = 0.65          // 0.65 FB0.5 best
				ly.Inhib.Layer.SS = 30            // 30 > others
				ly.Inhib.Layer.FB = 0.5           // 0 > 1 here in output
				ly.Acts.Spikes.Tr = 1             // 1 is new minimum.. > 3
				ly.Acts.Clamp.Ge = 0.8            // 0.8 > 0.7 > 1.0 > 0.6
				ly.Acts.VGCC.Ca = 1               // otherwise dominates display
				ly.Learn.RLRate.On.SetBool(true)  // beneficial for trace
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
				pt.Learn.LRate.Base = 0.1 // .1 def
				pt.SWts.Adapt.LRate = 0.1 // .1 >= .2,
				pt.SWts.Adapt.SubMean = 1 // key for stability
				pt.SWts.Init.SPct = 0.5   // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
				pt.Learn.DWt.SubMean = 1  // 1 > 0 for long-term stability
			}},
		{Sel: "#Hidden2ToOutput", Doc: "",
			Set: func(pt *axon.PathParams) {
				// pt.Learn.LRate.Base =  0.1 // 0.1 is default
				pt.SWts.Adapt.SigGain = 6 // 1 does not work
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.3 // 0.3 > 0.2 > 0.1 > 0.5
			}},
	},
}
