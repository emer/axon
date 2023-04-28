package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "needs some special inhibition and learning params",
				Params: params.Params{
					"Layer.Inhib.Layer.FS0":              "0.1", // .1 -- highly sensitive .08, .12 both sig worse
					"Layer.Inhib.Pool.FS0":               "0.1", // .1
					"Layer.Inhib.Layer.FSTau":            "6",   // 6 best
					"Layer.Inhib.Layer.FB":               "1",   // def = 1 for most layers
					"Layer.Inhib.Pool.FB":                "4",
					"Layer.Inhib.Layer.SSfTau":           "20", // 20 > 30  > 15
					"Layer.Inhib.Layer.SSiTau":           "50", // 50 > 40 -- try 40, 60 @ gi= 1.1?
					"Layer.Inhib.ActAvg.AdaptRate":       "0.1",
					"Layer.Inhib.ActAvg.LoTol":           "0.8",
					"Layer.Inhib.ActAvg.HiTol":           "0.0",
					"Layer.Act.Dt.IntTau":                "40",  // 30 == 40 no diff
					"Layer.Act.Decay.Act":                "0.0", // 0 > .2 -- highly sensitive
					"Layer.Act.Decay.Glong":              "0.6", // 0.6 def > 0.5, .7 -- highly sensitive
					"Layer.Act.Decay.AHP":                "0.0", // 0 def
					"Layer.Act.Dend.SSGi":                "2",   // 3.0o0 best
					"Layer.Act.Dend.GbarExp":             "0.2", // 0.2 > 0.5 > 0.1 > 0
					"Layer.Act.Dend.GbarR":               "3",   // 3 > 6 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
					"Layer.Act.Dt.GeTau":                 "5",   // 5 = 4 (bit slower) > 6 > 7 @176
					"Layer.Act.Dt.LongAvgTau":            "20",  // 20 > 50 > 100
					"Layer.Act.Dt.VmDendTau":             "5",   // 5 much better in fsa!
					"Layer.Act.NMDA.MgC":                 "1.4", // 1.4, 5 > 1.2, 0
					"Layer.Act.NMDA.Gbar":                "0.004",
					"Layer.Act.GABAB.Gbar":               "0.006",
					"Layer.Act.AK.Gbar":                  "0.1",    // 1 == .1 trace-v8
					"Layer.Act.VGCC.Gbar":                "0.02",   // non nmda: 0.15 good, 0.3 blows up
					"Layer.Act.VGCC.Ca":                  "25",     // 25 / 10tau best performance
					"Layer.Act.Mahp.Gbar":                "0.02",   // .02 > .05 -- 0.01 best in lvis
					"Layer.Learn.CaLrn.Norm":             "80",     // 80 default
					"Layer.Learn.CaLrn.SpkVGCC":          "true",   // sig better?
					"Layer.Learn.CaLrn.SpkVgccCa":        "35",     // 35 > 40, 45
					"Layer.Learn.CaLrn.VgccTau":          "10",     // 10 > 5 ?
					"Layer.Learn.CaLrn.Dt.MTau":          "2",      // 2 > 1 ?
					"Layer.Learn.CaSpk.SpikeG":           "12",     // 12 > 8 > 15 (too high) -- 12 makes everything work!
					"Layer.Learn.CaSpk.SynTau":           "30",     // 30 > 20, 40
					"Layer.Learn.CaSpk.Dt.MTau":          "5",      // 5 == 10 -- no big diff
					"Layer.Learn.LrnNMDA.MgC":            "1.4",    // 1.4, 5 > 1.2, 0
					"Layer.Learn.LrnNMDA.Voff":           "0",      // see above
					"Layer.Learn.LrnNMDA.Tau":            "100",    // 100 def
					"Layer.Learn.TrgAvgAct.On":           "true",   // no diff?
					"Layer.Learn.TrgAvgAct.SubMean":      "1.0",    // 1 > 0 -- doesn't slow learning -- always 1
					"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002", // 0.0002 > others -- 0.005 sig worse
					"Layer.Learn.TrgAvgAct.ErrLRate":     "0.02",   // 0.02 >= 0.05 -- less noisy
					"Layer.Learn.RLRate.On":              "true",   // beneficial for trace
					"Layer.Learn.RLRate.SigmoidMin":      "0.05",
					"Layer.Learn.RLRate.Diff":            "true", // always key
					"Layer.Learn.RLRate.DiffThr":         "0.02", // 0.02 def - todo
					"Layer.Learn.RLRate.SpkThr":          "0.1",  // 0.1 def
					"Layer.Learn.RLRate.Min":             "0.001",
				}},
			{Sel: "#V1", Desc: "pool inhib (not used), initial activity",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.08", // 0.08 == 0.9 just noisier
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Layer.Gi":       "0.9", // 0.9 def
					"Layer.Inhib.Pool.Gi":        "0.9", // 0.9 def
					"Layer.Inhib.Layer.FB":       "1",
					"Layer.Inhib.Pool.FB":        "1",
					"Layer.Inhib.Layer.SS":       "30",
					"Layer.Inhib.Pool.SS":        "30",
					"Layer.Act.Clamp.Ge":         "1.5", // 1.5 for fsffffb
					"Layer.Act.Decay.Act":        "1",   // 1 = slightly beneficial
					"Layer.Act.Decay.Glong":      "1",
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
					// "Layer.Act.Decay.Act":     "0.0",  // 0.2 with glong .6 best in lvis, slows learning here
					// "Layer.Act.Decay.Glong":   "0.6",  // 0.6 def
					"Layer.Inhib.ActAvg.Nominal":    "0.05",   // 0.05 nominal
					"Layer.Inhib.ActAvg.Offset":     "-0.005", //
					"Layer.Inhib.ActAvg.AdaptGi":    "true",   //
					"Layer.Inhib.Layer.Gi":          "1.2",    // 1.2 FB1 > 1.1 FB4
					"Layer.Inhib.Layer.FB":          "1",      //
					"Layer.Inhib.Layer.SS":          "30",     // 0 ok here -- not better in the end
					"Layer.Act.Clamp.Ge":            "0.8",    // 0.8 > 1.0 > 0.6 1.6.4
					"Layer.Learn.CaSpk.SpikeG":      "12",     // 12 > 8 -- not a big diff
					"Layer.Learn.RLRate.On":         "true",   // beneficial for trace
					"Layer.Learn.RLRate.SigmoidMin": "0.05",   // 0.05 > 1 -- key!
					"Layer.Learn.RLRate.Diff":       "true",
					"Layer.Learn.RLRate.DiffThr":    "0.02", // 0.02 def - todo
					"Layer.Learn.RLRate.SpkThr":     "0.1",  // 0.1 def
					"Layer.Learn.RLRate.Min":        "0.001",
				}},
			{Sel: "Prjn", Desc: "yes extra learning factors",
				Params: params.Params{
					"Prjn.Learn.LRate.Base":      "0.2",    // 0.4 for NeuronCa; 0.2 best, 0.1 nominal
					"Prjn.Learn.Trace.SubMean":   "1",      // 1 -- faster if 0 until 20 epc -- prevents sig amount of late deterioration
					"Prjn.SWt.Adapt.LRate":       "0.0001", // 0.005 == .1 == .01
					"Prjn.SWt.Init.SPct":         "1",      // 1 >= lower (trace-v11)
					"Prjn.SWt.Adapt.SubMean":     "1",
					"Prjn.Com.PFail":             "0.0",
					"Prjn.Learn.KinaseCa.SpikeG": "12", // 12 def / ra25
				}},
			{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",  // .2 >= .3 > .15 > .1 > .05 @176
					"Prjn.Learn.Learn":   "true", // keep random weights to enable exploration
					// "Prjn.Learn.LRate.Base":      "0.04", // lrate = 0 allows syn scaling still
				}},
			// {Sel: ".Forward", Desc: "special forward-only params: com prob",
			// 	Params: params.Params{}},
			{Sel: "#ITToOutput", Desc: "",
				Params: params.Params{
					"Prjn.Com.PFail": "0.0",
					// "Prjn.Learn.LRate.Base": "0.1", // no effect
				}},
		},
	}},
	{Name: "NovelLearn", Desc: "learning for novel objects case -- IT, Output connections learn", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "lr = 0",
				Params: params.Params{
					"Prjn.Learn.LRate":     "0",
					"Prjn.Learn.LRateInit": "0", // make sure for sched
				}},
			{Sel: ".NovLearn", Desc: "lr = 0.04",
				Params: params.Params{
					"Prjn.Learn.LRate":     "0.04",
					"Prjn.Learn.LRateInit": "0.04", // double sure
				}},
		},
	}},
}
