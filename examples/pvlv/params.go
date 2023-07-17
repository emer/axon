// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/netparams"
	"github.com/emer/emergent/params"
)

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: ".InputLayer", Desc: "",
			Params: params.Params{
				"Layer.Acts.Decay.Act":   "1.0",
				"Layer.Acts.Decay.Glong": "1.0",
			}},
		{Sel: "#CS", Desc: "expect act",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.05", // 1 / css
			}},
		{Sel: "#ContextIn", Desc: "expect act",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.025", // 1 / css
			}},
		{Sel: ".VSPatchLayer", Desc: "",
			Params: params.Params{
				"Layer.Learn.NeuroMod.DipGain": "0.01", // controls extinction -- reduce to slow
				"Layer.VSPatch.Gain":           "30",
				"Layer.VSPatch.ThrInit":        "0.4",
				"Layer.VSPatch.ThrLRate":       "0.001",
				"Layer.VSPatch.ThrNonRew":      "10",
			}},
		{Sel: "#BLAPosExtD2", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.8",
				"Layer.Inhib.Pool.Gi":  "1.0",
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":      "3.2",
				"Layer.Inhib.Pool.Gi":       "3.2",
				"Layer.Acts.Dend.ModGain":   "1.5",
				"Layer.Acts.MaintNMDA.Gbar": "0.007",
				"Layer.Acts.MaintNMDA.Tau":  "200",
			}},
		//////////////////////////////////////////////////
		// required custom params for this project
		{Sel: "#ContextInToBLAPosExtD2", Desc: "specific to this project",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":    "4",
				"Prjn.Learn.LRate.Base": "0.1",
			}},
		//////////////////////////////////////////////////
		// current experimental settings
		{Sel: ".MatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.Matrix.NoGateLRate": "0.0", // 0.01 default, 0 needed b/c no actual contingency in pavlovian
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4.0", // 4 > 2 for gating sooner
			}},
		{Sel: ".BLAExtPrjn", Desc: "",
			Params: params.Params{
				"Prjn.Learn.LRate.Base": "0.02", // 0.02 allows .5 CS for B50
			}},
		{Sel: ".GPiToBGThal", Desc: "inhibition from GPi to MD",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // 4 prevents some gating, 2 leaks with supertothal 4
			}},
		{Sel: ".PTpToBLAExt", Desc: "modulatory, drives extinction learning based on maintained goal rep",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5", // todo: expt
			}},
		{Sel: "#BLAPosAcqD1ToOFCus", Desc: "strong, high variance",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // key param for OFC focusing on current cs -- expt
			}},
		{Sel: ".BLAExtToAcq", Desc: "fixed inhibitory",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.0", // key param for efficacy of inhibition
			}},
		{Sel: ".VSPatchPrjn", Desc: "",
			Params: params.Params{ // todo: expt with these more..
				"Prjn.PrjnScale.Abs":        "2",
				"Prjn.Learn.Trace.LearnThr": "0.3",
				"Prjn.Learn.LRate.Base":     "0.05", // 0.05 def
			}},
		{Sel: ".PTSelfMaint", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4", // 4 needed to sustain
			}},
		{Sel: ".ToPTp", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2",
			}},
	},
}

// ParamSetsDefs contains the full set of parameters, many of which are at default values
// and have informed the default values in the first place.
var ParamSetsDefs = netparams.Sets{
	"Base": {
		{Sel: ".MatrixLayer", Desc: "factory defaults",
			Params: params.Params{
				"Layer.Matrix.GateThr":             "0.05",
				"Layer.Learn.NeuroMod.AChDisInhib": "5",    // key to be 5
				"Layer.Learn.NeuroMod.DALRateSign": "true", // critical
				"Layer.Learn.NeuroMod.DALRateMod":  "1",
				"Layer.Learn.NeuroMod.AChLRateMod": "1",
				"Layer.Acts.Dend.ModGain":          "5", // key for drive modulation
				"Layer.Inhib.ActAvg.Nominal":       "0.25",
				"Layer.Inhib.Layer.On":             "true",
				"Layer.Inhib.Layer.FB":             "0", // pure FF
				"Layer.Inhib.Layer.Gi":             "0.5",
				"Layer.Inhib.Pool.On":              "true", // def false
				"Layer.Inhib.Pool.FB":              "0",    // pure FF
				"Layer.Inhib.Pool.Gi":              "0.5",
			}},
		{Sel: ".BGThalLayer", Desc: "",
			Params: params.Params{
				"Layer.Acts.Dend.SSGi":             "0",
				"Layer.Inhib.ActAvg.Nominal":       "0.1",
				"Layer.Inhib.Layer.On":             "true",
				"Layer.Inhib.Layer.Gi":             "0.6",
				"Layer.Inhib.Pool.On":              "true",
				"Layer.Inhib.Pool.Gi":              "0.6",
				"Layer.Learn.NeuroMod.AChDisInhib": "1",
			}},
		{Sel: ".VSMatrixLayer", Desc: "",
			Params: params.Params{
				"Layer.Matrix.IsVS":          "true",
				"Layer.Inhib.ActAvg.Nominal": ".03",   // def .25
				"Layer.Inhib.Layer.On":       "false", // def true
				"Layer.Inhib.Pool.On":        "true",  // def false
				"Layer.Inhib.Pool.Gi":        "0.5",
			}},
		{Sel: ".VSPatchLayer", Desc: "",
			Params: params.Params{
				"Layer.Acts.Decay.Act":             "1",
				"Layer.Acts.Decay.Glong":           "1",
				"Layer.Acts.Decay.LearnCa":         "1", // uses CaSpkD as readout
				"Layer.Inhib.ActAvg.Nominal":       "0.2",
				"Layer.Inhib.Layer.On":             "true",
				"Layer.Inhib.Layer.Gi":             "0.5",
				"Layer.Inhib.Layer.FB":             "0",
				"Layer.Inhib.Pool.On":              "true",
				"Layer.Inhib.Pool.Gi":              "0.5",
				"Layer.Inhib.Pool.FB":              "0",
				"Layer.Learn.RLRate.Diff":          "false",
				"Layer.Learn.RLRate.SigmoidMin":    "1",
				"Layer.Learn.NeuroMod.DALRateSign": "true", // essential
				"Layer.Learn.NeuroMod.AChDisInhib": "0",    // essential: has to fire when expected but not present!
				"Layer.Learn.NeuroMod.AChLRateMod": "0.8",
				"Layer.Learn.NeuroMod.BurstGain":   "1",
				"Layer.Learn.NeuroMod.DipGain":     "0.01", // controls extinction -- reduce to slow
				"Layer.PVLV.Thr":                   "0.4",  // key user param
				"Layer.PVLV.Gain":                  "20",   // key user param
			}},
		{Sel: ".GPLayer", Desc: "all gp",
			Params: params.Params{
				"Layer.Acts.Decay.Act":       "0",
				"Layer.Acts.Decay.Glong":     "0",
				"Layer.Acts.Init.GeBase":     "0.3",
				"Layer.Acts.Init.GeVar":      "0.1",
				"Layer.Acts.Init.GiVar":      "0.1",
				"Layer.Inhib.ActAvg.Nominal": "1",
				"Layer.Inhib.Layer.On":       "false",
				"Layer.Inhib.Pool.On":        "false",
			}},
		{Sel: ".STNp", Desc: "Pausing STN",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal":       "0.15",
				"Layer.Inhib.Layer.On":             "true", // this is critical, else too active
				"Layer.Inhib.Layer.Gi":             "0.5",
				"Layer.Learn.NeuroMod.AChDisInhib": "2",
				"Layer.Acts.SKCa.Gbar":             "3",
			}},
		{Sel: ".STNs", Desc: "Sustained STN",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal":       "0.15",
				"Layer.Inhib.Layer.On":             "true",
				"Layer.Inhib.Layer.Gi":             "0.5",
				"Layer.Learn.NeuroMod.AChDisInhib": "2",
				"Layer.Acts.SKCa.Gbar":             "3",
				"Layer.Acts.SKCa.C50":              "0.4",
				"Layer.Acts.SKCa.KCaR":             "0.4",
				"Layer.Acts.SKCa.CaRDecayTau":      "200",
			}},
		{Sel: ".USLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.2",
				"Layer.Inhib.Layer.On":       "true",
				"Layer.Inhib.Pool.On":        "false",
				"Layer.Inhib.Layer.Gi":       "0.5",
			}},
		{Sel: ".PVLayer", Desc: "expect act",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.2",
				"Layer.Inhib.Layer.On":       "true",
				"Layer.Inhib.Pool.On":        "false",
				"Layer.Inhib.Layer.Gi":       "0.5",
			}},
		{Sel: ".DrivesLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal":  "0.01",
				"Layer.Inhib.Layer.On":        "false",
				"Layer.Inhib.Layer.Gi":        "0.5",
				"Layer.Inhib.Pool.On":         "true",
				"Layer.Inhib.Pool.Gi":         "0.5",
				"Layer.Acts.PopCode.On":       "true",
				"Layer.Acts.PopCode.MinAct":   "0.2", // low activity for low drive -- also has special 0 case = nothing
				"Layer.Acts.PopCode.MinSigma": "0.08",
				"Layer.Acts.PopCode.MaxSigma": "0.12",
				"Layer.Acts.Decay.Act":        "1",
				"Layer.Acts.Decay.Glong":      "1",
				"Layer.Learn.TrgAvgAct.On":    "false",
			}},
		{Sel: ".LDTLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.1",
				"Layer.Inhib.Layer.Gi":       "1.0",
				"Layer.Inhib.Pool.On":        "false",
				"Layer.LDT.RewThr":           "0.2",
				"Layer.Acts.Decay.Act":       "1",
				"Layer.Acts.Decay.Glong":     "1",
				"Layer.Acts.Decay.LearnCa":   "1", // uses CaSpkD as a readout!
				"Layer.Learn.TrgAvgAct.On":   "false",
				"Layer.PVLV.Thr":             "0.2",
				"Layer.PVLV.Gain":            "2",
			}},
		{Sel: ".BLALayer", Desc: "",
			Params: params.Params{
				"Layer.Acts.Decay.Act":             "0",
				"Layer.Acts.Decay.Glong":           "0",
				"Layer.CT.GeGain":                  "0.1", // 0.1 has sig effect -- can go a bit lower if need
				"Layer.Inhib.ActAvg.Nominal":       "0.025",
				"Layer.Inhib.Layer.Gi":             "2.2",
				"Layer.Inhib.Pool.Gi":              "0.9",
				"Layer.Learn.NeuroMod.AChLRateMod": "1",
				"Layer.Learn.NeuroMod.AChDisInhib": "0", // always active
			}},
		{Sel: "#BLAPosAcqD1", Desc: "",
			Params: params.Params{
				"Layer.Learn.NeuroMod.DALRateMod": "0.5",
				"Layer.Learn.NeuroMod.BurstGain":  "0.2",
				"Layer.Learn.NeuroMod.DipGain":    "0",
			}},
		{Sel: "#BLAPosExtD2", Desc: "",
			Params: params.Params{
				"Layer.Learn.NeuroMod.DALRateMod": "0",
				"Layer.Learn.NeuroMod.BurstGain":  "1",
				"Layer.Learn.NeuroMod.DipGain":    "1",
			}},
		{Sel: ".CeMLayer", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.15",
				"Layer.Acts.Dend.SSGi":       "0",
				"Layer.Inhib.Layer.Gi":       "0.5",
				"Layer.Inhib.Pool.On":        "true",
				"Layer.Inhib.Pool.Gi":        "0.3",
			}},
		{Sel: "#BLANovelCS", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.05",
				"Layer.Inhib.Layer.Gi":       "0.8",
				"Layer.Inhib.Pool.On":        "false",
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Acts.Decay.Act":     "0",
				"Layer.Acts.Decay.Glong":   "0",
				"Layer.Acts.Decay.AHP":     "0",
				"Layer.Acts.Decay.OnRew":   "true", // everything clears
				"Layer.Acts.GabaB.Gbar":    "0.3",
				"Layer.Acts.NMDA.Gbar":     "0.3", // 0.3 enough..
				"Layer.Acts.NMDA.Tau":      "300",
				"Layer.Acts.Sahp.Gbar":     "0.01", // not much pressure -- long maint
				"Layer.Acts.Dend.ModGain":  "20",   // 10?
				"Layer.Inhib.Layer.Gi":     "1.8",
				"Layer.Inhib.Pool.Gi":      "1.8",
				"Layer.Learn.TrgAvgAct.On": "false",
			}},
		{Sel: ".PTPredLayer", Desc: "PTPred prediction layer -- more dynamic acts",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.12",
				"Layer.Inhib.Layer.Gi":       "0.8",
				"Layer.Inhib.Pool.Gi":        "0.8",
				"Layer.Acts.GabaB.Gbar":      "0.2", // regular
				"Layer.Acts.NMDA.Gbar":       "0.15",
				"Layer.Acts.NMDA.Tau":        "100",
				"Layer.Acts.Decay.Act":       "0.2",
				"Layer.Acts.Decay.Glong":     "0.6",
				"Layer.Acts.Sahp.Gbar":       "0.1",
				"Layer.Acts.KNa.Slow.Max":    "0.2", // maybe too random if higher?
				"Layer.CT.GeGain":            "0.01",
				"Layer.CT.DecayTau":          "50",
			}},
		{Sel: ".OFCus", Desc: "",
			Params: params.Params{
				"Layer.Acts.Decay.Act":       "0",
				"Layer.Acts.Decay.Glong":     "0",
				"Layer.Acts.Decay.OnRew":     "true", // everything clears
				"Layer.Inhib.ActAvg.Nominal": "0.025",
				"Layer.Inhib.Layer.Gi":       "2.2",
				"Layer.Inhib.Pool.On":        "true",
				"Layer.Inhib.Pool.Gi":        "1.2",
				"Layer.Acts.Dend.SSGi":       "0",
			}},
		{Sel: "#OFCusCT", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "2.8", // 2.4 not strong enough to prevent diffuse activity
				"Layer.Inhib.Pool.Gi":  "1.2", // was 1.4
			}},
		{Sel: "#OFCusPT", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.8", // was 1.3
				"Layer.Inhib.Pool.Gi":  "2.0", // was 0.6
			}},
		{Sel: "#OFCusPTp", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "0.8",
				"Layer.Inhib.Pool.Gi":  "0.8",
			}},
		{Sel: "#OFCusMD", Desc: "",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.1",
				"Layer.Inhib.Pool.Gi":  "0.6",
			}},
		/////////////////////////////////
		// Projections
		{Sel: ".BackPrjn", Desc: "back is weaker",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.1", // 0.1 default
			}},
		{Sel: ".MatrixPrjn", Desc: "",
			Params: params.Params{
				"Prjn.SWts.Adapt.On":        "false",
				"Prjn.SWts.Adapt.SigGain":   "6", // 1 not better
				"Prjn.SWts.Init.Sym":        "false",
				"Prjn.SWts.Init.SPct":       "0",
				"Prjn.SWts.Init.Mean":       "0.5",
				"Prjn.SWts.Init.Var":        "0.4",  // more variance
				"Prjn.Learn.LRate.Base":     "0.02", // slower fine
				"Prjn.Learn.Trace.LearnThr": "0.75",
				"Prjn.Matrix.NoGateLRate":   "0.0", // 0.01 default, 0 needed b/c no actual contingency in pavlovian
			}},
		{Sel: ".VSPatchPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":        "2",
				"Prjn.SWts.Adapt.On":        "false",
				"Prjn.SWts.Adapt.SigGain":   "1",
				"Prjn.SWts.Init.SPct":       "0",
				"Prjn.SWts.Init.Mean":       "0.1",
				"Prjn.SWts.Init.Var":        "0.05",
				"Prjn.SWts.Init.Sym":        "false",
				"Prjn.Learn.Trace.Tau":      "1",
				"Prjn.Learn.Trace.LearnThr": "0.3",
				"Prjn.Learn.LRate.Base":     "0.05", // 0.05 def
			}},
		{Sel: ".USToBLAAcq", Desc: "starts strong, learns slow",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":     "0.5",
				"Prjn.SWts.Init.SPct":    "0",
				"Prjn.SWts.Init.Mean":    "0.75",
				"Prjn.SWts.Init.Var":     "0.25",
				"Prjn.Learn.LRate.Base":  "0.001", // could be 0
				"Prjn.Learn.Trace.Tau":   "1",     // increase for second order conditioning
				"Prjn.BLA.NegDeltaLRate": "0.01",  // slow for acq -- could be 0
			}},
		{Sel: ".USToBLAExtInhib", Desc: "actual US inhibits exinction -- must be strong enough to block ACh enh Ge",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":  "2",
				"Prjn.SWts.Init.SPct": "0",
				"Prjn.SWts.Init.Mean": "0.75",
				"Prjn.SWts.Init.Var":  "0.25",
				"Prjn.SWts.Adapt.On":  "false",
				"Prjn.Learn.Learn":    "false",
			}},
		{Sel: ".BLAAcqPrjn", Desc: "",
			Params: params.Params{
				"Prjn.SWts.Adapt.On":      "false",
				"Prjn.SWts.Adapt.SigGain": "1",
				"Prjn.SWts.Init.SPct":     "0",
				"Prjn.SWts.Init.Mean":     "0.1",
				"Prjn.SWts.Init.Var":      "0.05",
				"Prjn.SWts.Init.Sym":      "false",
				"Prjn.Learn.Trace.Tau":    "1", // increase for second order conditioning
				"Prjn.Learn.LRate.Base":   "0.02",
				"Prjn.BLA.NegDeltaLRate":  "0.01", // slow -- could be 0
			}},
		{Sel: ".BLAExtPrjn", Desc: "",
			Params: params.Params{
				"Prjn.SWts.Adapt.On":      "false",
				"Prjn.SWts.Adapt.SigGain": "1",
				"Prjn.SWts.Init.SPct":     "0",
				"Prjn.SWts.Init.Mean":     "0.1",
				"Prjn.SWts.Init.Var":      "0.05",
				"Prjn.SWts.Init.Sym":      "false",
				"Prjn.Learn.Trace.Tau":    "1", // increase for second order conditioning
				"Prjn.Learn.LRate.Base":   "0.02",
				"Prjn.BLA.NegDeltaLRate":  "1.0",
			}},
		{Sel: "#USposToVsMtxGo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2",
				"Prjn.PrjnScale.Rel": ".2",
			}},
		{Sel: "#USposToBLAPosAcqD1", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":  "6.0", // if weaker, e.g., 2, other pools get active
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.4",
			}},
		{Sel: ".CSToBLAPos", Desc: "stronger by default",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.5",
			}},
		{Sel: ".BLAAcqToGo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4",
				"Prjn.PrjnScale.Rel": "1",
			}},
		{Sel: ".BLAExtToNo", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.1", // extinction is mostly within BLA
				"Prjn.PrjnScale.Rel": "1",
			}},
		{Sel: ".BLAToCeM_Excite", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			}},
		{Sel: ".BLAToCeM_Inhib", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1",
			}},
		{Sel: ".BLAFromNovel", Desc: "dilutes everyone else, so make it weaker Rel, compensate with Abs",
			Params: params.Params{
				"Prjn.Learn.Learn":    "false",
				"Prjn.PrjnScale.Rel":  "0.1",
				"Prjn.PrjnScale.Abs":  "5",
				"Prjn.SWts.Init.SPct": "0",
				"Prjn.SWts.Adapt.On":  "false",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.4",
			}},
		{Sel: "#BLAPosAcqD1ToOFCus", Desc: "strong, high variance",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":  "6", // key param for OFC focusing on current cs
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.4",
			}},
		{Sel: ".PTpToBLAExt", Desc: "modulatory, drives extinction learning based on maintained goal rep",
			Params: params.Params{
				"Prjn.Com.GType":      "ModulatoryG",
				"Prjn.PrjnScale.Abs":  "0.5",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.4",
			}},
		{Sel: ".DrivesToMtx", Desc: "this is modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act",
			Params: params.Params{
				"Prjn.Learn.Learn":    "false",
				"Prjn.PrjnScale.Abs":  "2",
				"Prjn.PrjnScale.Rel":  "1",
				"Prjn.SWts.Init.SPct": "0",
				"Prjn.SWts.Init.Mean": "0.8",
				"Prjn.SWts.Init.Var":  "0.0",
				"Prjn.Com.GType":      "ModulatoryG",
			}},
		{Sel: ".DrivesToVSPatch", Desc: "this is modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act",
			Params: params.Params{
				"Prjn.Learn.Learn":    "false",
				"Prjn.PrjnScale.Abs":  "2",
				"Prjn.PrjnScale.Rel":  "1",
				"Prjn.SWts.Init.SPct": "0",
				"Prjn.SWts.Init.Mean": "0.8",
				"Prjn.SWts.Init.Var":  "0.0",
				"Prjn.Com.GType":      "ModulatoryG",
			}},
		{Sel: ".DrivesToOFC", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.2",
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":  "1.0",
				"Prjn.PrjnScale.Abs":  "2.0", // key param for driving gating -- if too strong, premature gating
				"Prjn.Learn.Learn":    "false",
				"Prjn.SWts.Adapt.On":  "false",
				"Prjn.SWts.Init.SPct": "0",
				"Prjn.SWts.Init.Mean": "0.8",
				"Prjn.SWts.Init.Var":  "0.0",
			}},
		{Sel: ".ThalToPT", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":  "1.0",
				"Prjn.Com.GType":      "ModulatoryG", // modulatory -- control with extra ModGain factor
				"Prjn.Learn.Learn":    "false",
				"Prjn.SWts.Adapt.On":  "false",
				"Prjn.SWts.Init.SPct": "0",
				"Prjn.SWts.Init.Mean": "0.8",
				"Prjn.SWts.Init.Var":  "0.0",
			}},
		{Sel: ".SuperToPT", Desc: "one-to-one from super -- just use fixed nonlearning prjn so can control behavior easily",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":  "1",    // keep this constant -- only self vs. this -- thal is modulatory
				"Prjn.PrjnScale.Abs":  "0.01", // monitor maint early and other maint stats with PTMaintLayer ModGain = 0 to set this so super alone is not able to drive it.
				"Prjn.Learn.Learn":    "false",
				"Prjn.SWts.Adapt.On":  "false",
				"Prjn.SWts.Init.SPct": "0",
				"Prjn.SWts.Init.Mean": "0.8",
				"Prjn.SWts.Init.Var":  "0.0",
			}},
		{Sel: ".PTSelfMaint", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel":    "1",      // use abs to manipulate
				"Prjn.PrjnScale.Abs":    "2",      // 2 > 1
				"Prjn.Learn.LRate.Base": "0.0001", // slower > faster
				"Prjn.SWts.Init.Mean":   "0.5",
				"Prjn.SWts.Init.Var":    "0.5", // high variance so not just spreading out over time
			}},
		{Sel: "#OFCusCTToOFCusPTp", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.5", // 0.5 -- todo: not clear if important
			}},
		{Sel: ".FmSTNp", Desc: "increase to prevent repeated gating",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.2", // note: this was contaminating prjns to GPe etc too
			}},
		{Sel: ".GPeTAToMtx", Desc: "nonspecific gating activity surround inhibition -- wta",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "2", // 2 nominal
			}},
		{Sel: ".GPeInToMtx", Desc: "provides weak counterbalance for GPeTA -> Mtx to reduce oscillations",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5",
			}},
	},
}
