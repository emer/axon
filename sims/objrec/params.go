package objrec

import (
	"github.com/emer/axon/v2/axon"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "needs some special inhibition and learning params",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Decay.Act = 0.0                  // 0 > .2 -- highly sensitive
				ly.Acts.Decay.Glong = 0.6                // 0.6 def > 0.5, .7 -- highly sensitive
				ly.Acts.NMDA.MgC = 1.4                   // 1.4, 5 > 1.2, 0
				ly.Acts.NMDA.Voff = 0                    // see above
				ly.Acts.NMDA.Gbar = 0.006                // 0.006 > 7 or higher
				ly.Acts.GabaB.Gbar = 0.015               // 0.015 > lower; higher not better
				ly.Inhib.ActAvg.AdaptRate = 0.1          // 0.1 default > 0.05?
				ly.Inhib.ActAvg.AdaptMax = 0.05          // 0.05 > 0.01
				ly.Learn.CaSpike.SpikeCaM = 12           // 12 > 8 > 15 (too high) -- 12 makes everything work!
				ly.Learn.TrgAvgAct.SynScaleRate = 0.0002 // 0.0002 > others -- 0.005 sig worse
				ly.Learn.LearnNMDA.MgC = 1.4             // 1.4, 5 > 1.2, 0
				ly.Learn.LearnNMDA.Voff = 0              // see above
				ly.Learn.LearnNMDA.Tau = 100             // 100 def
				ly.Learn.LearnNMDA.Gbar = 0.006
				ly.Learn.RLRate.SigmoidLinear.SetBool(true) // true > false later; more stable
				ly.Learn.CaLearn.Norm = 80                  // 80 works
				ly.Learn.CaLearn.SpikeVGCC.SetBool(true)    // sig better..
				ly.Learn.CaLearn.SpikeVgccCa = 35           // 70 / 5 or 35 / 10 both work
				ly.Learn.CaLearn.VgccTau = 10               // 10 > 5 ?
				ly.Learn.CaLearn.Dt.MTau = 2                // 2 > 4 even with more ncycles
				ly.Learn.CaSpike.Dt.MTau = 5                // 5 > 10 even with more ncycles
				// now automatic:
				// ly.Learn.CaLearn.Dt.PTau =        40   // 60 for 300 cyc, 40 for 200 (scales linearly)
				// ly.Learn.CaLearn.Dt.DTau =        40   // "
				// ly.Learn.CaSpk.Dt.PTau =          40   // "
				// ly.Learn.CaSpk.Dt.DTau =          40   // "
			}},
		{Sel: "#V1", Doc: "pool inhib (not used), initial activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.08 // 0.08 == 0.9 just noisier
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.9 // 0.9 def
				ly.Inhib.Pool.Gi = 0.9  // 0.9 def
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Pool.FB = 1
				ly.Acts.Clamp.Ge = 1.5 // 1.5 for fsffffb
				ly.Acts.Decay.Act = 1  // 1 = slightly beneficial
				ly.Acts.Decay.Glong = 1
			}},
		{Sel: "#V4", Doc: "pool inhib, sparse activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03 // 0.03 > .04 > 0.025
				ly.Inhib.ActAvg.Offset = 0     // 0.01 not good
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.FB = 1          // 1.1 FB1 >> 4!
				ly.Inhib.Pool.FB = 4           // 4
				ly.Inhib.Layer.SS = 30         // 30 best
				ly.Inhib.Pool.SS = 30          // 0 works here..
				ly.Inhib.Layer.Gi = 1.0        // 1.1 > 1.0 -- def 1.1, 1.0 > 1.0, 1.1!
				ly.Inhib.Pool.Gi = 0.9         // 0.9
				ly.Inhib.Pool.On.SetBool(true) // needs pool-level
			}},
		{Sel: "#IT", Doc: "initial activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04 // 0.04 -- 0.05 actual at end, but starts low
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.Gi = 1.1 // 1.1 > 1.05 1.6.15 adapt
				ly.Inhib.Layer.FB = 4   // 4
			}},
		{Sel: "#Output", Doc: "high inhib for one-hot output",
			Set: func(ly *axon.LayerParams) {
				// ly.Acts.Decay.Act =     0.0  // 0.2 with glong .6 best in lvis, slows learning here
				// ly.Acts.Decay.Glong =   0.6  // 0.6 def
				ly.Inhib.ActAvg.Nominal = 0.05        // 0.05 nominal
				ly.Inhib.ActAvg.Offset = -0.005       //
				ly.Inhib.ActAvg.AdaptGi.SetBool(true) //
				ly.Inhib.Layer.Gi = 1.2               // 1.2 FB1 > 1.1 FB4
				ly.Inhib.Layer.FB = 1                 //
				ly.Acts.Clamp.Ge = 0.8                // 0.8 > 1.0 > 0.6 1.6.4
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "",
			Set: func(pt *axon.PathParams) {
				// pt.Com.MaxDelay = 10 // not much effect
				// pt.Com.Delay = 10
				pt.Learn.LRate.Base = 0.1        // 0.1 > 0.2 much better behavior overall; just slower initial learning for trace, 0.02 for notrace
				pt.Learn.DWt.SubMean = 1         // 1 -- faster if 0 until 20 epc -- prevents sig amount of late deterioration
				pt.SWts.Adapt.LRate = 0.0001     // 0.005 == .1 == .01
				pt.SWts.Adapt.HiMeanDecay = 0    // 0 > 0.0008 (best in lvis)
				pt.SWts.Adapt.HiMeanThr = 0.5    // 0.5, 0.0008 goes the distance
				pt.SWts.Init.SPct = 1            // 1 >= lower (trace-v11)
				pt.Learn.DWt.CaPScale = 1        //
				pt.Learn.DWt.Trace.SetBool(true) // no trace starts faster but is unstable
				pt.Learn.DWt.SynCa20.SetBool(false)
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2 // .2 >= .3 > .15 > .1 > .05 @176
			}},
	},
}
