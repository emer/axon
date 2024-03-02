package main

import (
	"github.com/emer/emergent/v2/netparams"
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: "Layer", Desc: "needs some special inhibition and learning params",
			Params: params.Params{
				"Layer.Acts.Decay.Act":               "0.0",    // 0 > .2 -- highly sensitive
				"Layer.Acts.Decay.Glong":             "0.6",    // 0.6 def > 0.5, .7 -- highly sensitive
				"Layer.Acts.NMDA.MgC":                "1.4",    // 1.4, 5 > 1.2, 0
				"Layer.Acts.NMDA.Voff":               "0",      // see above
				"Layer.Acts.NMDA.Gbar":               "0.006",  // 0.006 > 7 or higher
				"Layer.Acts.GabaB.Gbar":              "0.015",  // 0.015 > lower; higher not better
				"Layer.Learn.CaSpk.SpikeG":           "12",     // 12 > 8 > 15 (too high) -- 12 makes everything work!
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002", // 0.0002 > others -- 0.005 sig worse
				"Layer.Learn.LrnNMDA.MgC":            "1.4",    // 1.4, 5 > 1.2, 0
				"Layer.Learn.LrnNMDA.Voff":           "0",      // see above
				"Layer.Learn.LrnNMDA.Tau":            "100",    // 100 def
				"Layer.Learn.LrnNMDA.Gbar":           "0.006",
				"Layer.Learn.RLRate.SigmoidLinear":   "false",
			}},
		{Sel: "#V1", Desc: "pool inhib (not used), initial activity",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.08", // 0.08 == 0.9 just noisier
				"Layer.Inhib.Pool.On":        "true",
				"Layer.Inhib.Layer.Gi":       "0.9", // 0.9 def
				"Layer.Inhib.Pool.Gi":        "0.9", // 0.9 def
				"Layer.Inhib.Layer.FB":       "1",
				"Layer.Inhib.Pool.FB":        "1",
				"Layer.Acts.Clamp.Ge":        "1.5", // 1.5 for fsffffb
				"Layer.Acts.Decay.Act":       "1",   // 1 = slightly beneficial
				"Layer.Acts.Decay.Glong":     "1",
			}},
		{Sel: "#V4", Desc: "pool inhib, sparse activity",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.03", // 0.03 > .04 > 0.025
				"Layer.Inhib.ActAvg.AdaptGi": "true",
				"Layer.Inhib.Layer.FB":       "1",    // 1.1 FB1 >> 4!
				"Layer.Inhib.Pool.FB":        "4",    // 4
				"Layer.Inhib.Layer.SS":       "30",   // 30 best
				"Layer.Inhib.Pool.SS":        "30",   // 0 works here..
				"Layer.Inhib.Layer.Gi":       "1.0",  // 1.1 > 1.0 -- def 1.1, 1.0 > 1.0, 1.1!
				"Layer.Inhib.Pool.Gi":        "0.9",  // 0.9
				"Layer.Inhib.Pool.On":        "true", // needs pool-level
			}},
		{Sel: "#IT", Desc: "initial activity",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.04", // 0.04 -- 0.05 actual at end, but starts low
				"Layer.Inhib.ActAvg.AdaptGi": "true",
				"Layer.Inhib.Layer.Gi":       "1.1", // 1.1 > 1.05 1.6.15 adapt
				"Layer.Inhib.Layer.FB":       "4",   // 4
			}},
		{Sel: "#Output", Desc: "high inhib for one-hot output",
			Params: params.Params{
				// "Layer.Acts.Decay.Act":     "0.0",  // 0.2 with glong .6 best in lvis, slows learning here
				// "Layer.Acts.Decay.Glong":   "0.6",  // 0.6 def
				"Layer.Inhib.ActAvg.Nominal": "0.05",   // 0.05 nominal
				"Layer.Inhib.ActAvg.Offset":  "-0.005", //
				"Layer.Inhib.ActAvg.AdaptGi": "true",   //
				"Layer.Inhib.Layer.Gi":       "1.2",    // 1.2 FB1 > 1.1 FB4
				"Layer.Inhib.Layer.FB":       "1",      //
				"Layer.Acts.Clamp.Ge":        "0.8",    // 0.8 > 1.0 > 0.6 1.6.4
			}},
		{Sel: "Prjn", Desc: "yes extra learning factors",
			Params: params.Params{
				"Prjn.Learn.LRate.Base":    "0.2",    // 0.4 for NeuronCa; 0.2 best, 0.1 nominal
				"Prjn.Learn.Trace.SubMean": "1",      // 1 -- faster if 0 until 20 epc -- prevents sig amount of late deterioration
				"Prjn.SWts.Adapt.LRate":    "0.0001", // 0.005 == .1 == .01
				"Prjn.SWts.Init.SPct":      "1",      // 1 >= lower (trace-v11)
			}},
		{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.2", // .2 >= .3 > .15 > .1 > .05 @176
			}},
	},
}
