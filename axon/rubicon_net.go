// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/emergent/v2/relpos"
)

// AddLDTLayer adds a LDTLayer
func (net *Network) AddLDTLayer(prefix string) *Layer {
	ldt := net.AddLayer2D(prefix+"LDT", 1, 1, LDTLayer)
	return ldt
}

// AddBLALayers adds two BLA layers, acquisition / extinction / D1 / D2,
// for positive or negative valence
func (net *Network) AddBLALayers(prefix string, pos bool, nUs, nNeurY, nNeurX int, rel relpos.Relations, space float32) (acq, ext *Layer) {
	if pos {
		d1 := net.AddLayer4D(prefix+"BLAposAcqD1", 1, nUs, nNeurY, nNeurX, BLALayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Positive")
		d2 := net.AddLayer4D(prefix+"BLAposExtD2", 1, nUs, nNeurY, nNeurX, BLALayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Positive")
		acq = d1
		ext = d2
	} else {
		d1 := net.AddLayer4D(prefix+"BLAnegExtD1", 1, nUs, nNeurY, nNeurX, BLALayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Negative")
		d2 := net.AddLayer4D(prefix+"BLAnegAcqD2", 1, nUs, nNeurY, nNeurX, BLALayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Negative")
		d2.DefParams = params.Params{
			"Layer.Inhib.Layer.Gi": "1.2", // weaker
		}
		acq = d2
		ext = d1
	}

	pj := net.ConnectLayers(ext, acq, prjn.NewPoolOneToOne(), InhibPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "0.5", // key param for efficacy of inhibition -- may need to tweak
	}
	pj.AddClass("BLAExtToAcq")

	pj = net.ConnectLayers(acq, ext, prjn.NewOneToOne(), CTCtxtPrjn)
	pj.AddClass("BLAAcqToExt")

	if rel == relpos.Behind {
		ext.PlaceBehind(acq, space)
	} else {
		ext.PlaceRightOf(acq, space)
	}
	acq.AddClass("BLA")
	ext.AddClass("BLA")
	return
}

// AddAmygdala adds a full amygdala complex including BLA,
// CeM, and LDT.  Inclusion of negative valence is optional with neg
// arg -- neg* layers are nil if not included.
// Uses the network Rubicon.NPosUSs and NNegUSs for number of pools --
// must be configured prior to calling this.
func (net *Network) AddAmygdala(prefix string, neg bool, nNeurY, nNeurX int, space float32) (blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov *Layer) {
	nUSpos := int(net.Rubicon.NPosUSs)
	nUSneg := int(net.Rubicon.NNegUSs)

	blaPosAcq, blaPosExt = net.AddBLALayers(prefix, true, nUSpos, nNeurY, nNeurX, relpos.Behind, space)
	if neg {
		blaNegAcq, blaNegExt = net.AddBLALayers(prefix, false, nUSneg, nNeurY, nNeurX, relpos.Behind, space)
		blaPosAcq.SetBuildConfig("LayInhib1Name", blaNegAcq.Name())
		blaNegAcq.SetBuildConfig("LayInhib1Name", blaPosAcq.Name())
	}
	cemPos = net.AddLayer4D(prefix+"CeMPos", 1, nUSpos, 1, nNeurX, CeMLayer)
	cemPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	cemPos.SetBuildConfig("Valence", "Positive")

	if neg {
		cemNeg = net.AddLayer4D(prefix+"CeMNeg", 1, nUSneg, 1, nNeurX, CeMLayer)
		cemNeg.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
		cemNeg.SetBuildConfig("Valence", "Negative")
	}

	blaNov = net.AddLayer4D(prefix+"BLANovelCS", 1, 1, 4, 4, BLALayer)
	blaNov.SetBuildConfig("DAMod", "D1Mod")
	blaNov.SetBuildConfig("Valence", "Positive")
	blaNov.DefParams = params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.05",
		"Layer.Inhib.Layer.Gi":       "0.8",
		"Layer.Inhib.Pool.On":        "false",
	}

	p1to1 := prjn.NewPoolOneToOne()

	net.ConnectLayers(blaPosAcq, cemPos, p1to1, ForwardPrjn).AddClass("BLAToCeM_Excite")
	net.ConnectLayers(blaPosExt, cemPos, p1to1, InhibPrjn).AddClass("BLAToCeM_Inhib")
	// default Abs = 1 works for both of these

	if neg {
		net.ConnectLayers(blaNegAcq, cemNeg, p1to1, ForwardPrjn).AddClass("BLAToCeM_Excite")
		net.ConnectLayers(blaNegExt, cemNeg, p1to1, InhibPrjn).AddClass("BLAToCeM_Inhib")
	}

	pj := net.ConnectLayers(blaNov, blaPosAcq, p1to1, ForwardPrjn)
	pj.DefParams = params.Params{ // dilutes everyone else, so make it weaker Rel, compensate with Abs
		"Prjn.Learn.Learn":    "false",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.PrjnScale.Rel":  "0.1",
		"Prjn.PrjnScale.Abs":  "2", // 3 competes with CS too strongly
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.5",
		"Prjn.SWts.Init.Var":  "0.4",
	}
	pj.AddClass("BLAFromNovel")

	cemPos.PlaceBehind(blaPosExt, space)
	if neg {
		blaNegAcq.PlaceBehind(blaPosExt, space)
		cemPos.PlaceBehind(blaNegExt, space)
		cemNeg.PlaceBehind(cemPos, space)
		blaNov.PlaceBehind(cemNeg, space)
	} else {
		blaNov.PlaceBehind(cemPos, space)
	}

	return
}

// ConnectToBLAAcq adds a BLAPrjn from given sending layer to a BLA layer,
// and configures it for acquisition parameters. Sets class to BLAAcqPrjn.
func (net *Network) ConnectToBLAAcq(send, recv *Layer, pat prjn.Pattern) *Prjn {
	pj := net.ConnectLayers(send, recv, pat, BLAPrjn)
	pj.DefParams = params.Params{
		"Prjn.Learn.LRate.Base":  "0.02",
		"Prjn.Learn.Trace.Tau":   "1",    // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "0.01", // slow for acq -- could be 0
	}
	pj.AddClass("BLAAcqPrjn")
	return pj
}

// ConnectToBLAExt adds a BLAPrjn from given sending layer to a BLA layer,
// and configures it for extinctrion parameters.  Sets class to BLAExtPrjn.
func (net *Network) ConnectToBLAExt(send, recv *Layer, pat prjn.Pattern) *Prjn {
	pj := net.ConnectLayers(send, recv, pat, BLAPrjn)
	pj.DefParams = params.Params{
		"Prjn.Learn.LRate.Base":  "0.005", // 0.02 for pvlv CS 50% balance
		"Prjn.Learn.Trace.Tau":   "1",     // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "1",     // fast for extinction unlearning -- could be slower
	}
	pj.AddClass("BLAExtPrjn")
	return pj
}

// ConnectCSToBLApos connects the CS input to BLAposAcqD1, BLANovelCS layers
// using fixed, higher-variance weights, full projection.
// Sets classes to: CSToBLApos, CSToBLANovel with default params
func (net *Network) ConnectCSToBLApos(cs, blaAcq, blaNov *Layer) (toAcq, toNov *Prjn) {
	toAcq = net.ConnectLayers(cs, blaAcq, prjn.NewFull(), BLAPrjn)
	toAcq.DefParams = params.Params{ // stronger..
		"Prjn.PrjnScale.Abs":     "1.5",
		"Prjn.Learn.LRate.Base":  "0.1",  // faster learning
		"Prjn.Learn.Trace.Tau":   "1",    // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "0.01", // slow for acq -- could be 0
	}
	toAcq.AddClass("CSToBLApos")

	toNov = net.ConnectLayers(cs, blaNov, prjn.NewFull(), BLAPrjn)
	toNov.DefParams = params.Params{ // dilutes everyone else, so make it weaker Rel, compensate with Abs
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.75",
		"Prjn.SWts.Init.Var":  "0.25",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.Learn.Learn":    "false",
	}
	toNov.AddClass("CSToBLANovel")
	return
}

// ConnectUSToBLA connects the US input to BLApos(Neg)AcqD1(D2) and
// BLApos(Neg)ExtD2(D1) layers,
// using fixed, higher-variance weights, full projection.
// Sets classes to: USToBLAAcq and USToBLAExt
func (net *Network) ConnectUSToBLA(us, blaAcq, blaExt *Layer) (toAcq, toExt *Prjn) {
	toAcq = net.ConnectLayers(us, blaAcq, prjn.NewPoolOneToOne(), BLAPrjn)
	toAcq.DefParams = params.Params{
		"Prjn.PrjnScale.Rel":     "0.5",
		"Prjn.PrjnScale.Abs":     "6",
		"Prjn.SWts.Init.SPct":    "0",
		"Prjn.SWts.Init.Mean":    "0.75",
		"Prjn.SWts.Init.Var":     "0.25",
		"Prjn.Learn.LRate.Base":  "0.001", // could be 0
		"Prjn.Learn.Trace.Tau":   "1",     // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "0.01",  // slow for acq -- could be 0
	}
	toAcq.AddClass("USToBLAAcq")

	toExt = net.ConnectLayers(us, blaExt, prjn.NewPoolOneToOne(), InhibPrjn)
	toExt.DefParams = params.Params{ // actual US inhibits exinction -- must be strong enough to block ACh enh Ge
		"Prjn.PrjnScale.Abs":  "0.5", // note: key param
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.Learn.Learn":    "false",
	}
	toExt.AddClass("USToBLAExtInhib")
	return
}

// AddUSLayers adds USpos, USneg, and Cost layers for positive or negative valence
// unconditioned stimuli (USs), using a pop-code representation of US magnitude.
// These track the Global USpos, USneg, Cost for visualization and predictive learning.
// Actual US inputs are set in Rubicon.
// Uses the network Rubicon.NPosUSs, NNegUSs, and NCosts for number of pools --
// must be configured prior to calling this.
func (net *Network) AddUSLayers(popY, popX int, rel relpos.Relations, space float32) (usPos, usNeg, cost, costFinal *Layer) {
	nUSpos := int(net.Rubicon.NPosUSs)
	nUSneg := int(net.Rubicon.NNegUSs)
	nCost := int(net.Rubicon.NCosts)
	usPos = net.AddLayer4D("USpos", 1, nUSpos, popY, popX, USLayer)
	usPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	usPos.SetBuildConfig("Valence", "Positive")
	usNeg = net.AddLayer4D("USneg", 1, nUSneg, popY, popX, USLayer)
	usNeg.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
	usNeg.SetBuildConfig("Valence", "Negative")
	cost = net.AddLayer4D("Cost", 1, nCost, popY, popX, USLayer)
	cost.SetBuildConfig("DAMod", "D1Mod") // d1mod = incremental current
	cost.SetBuildConfig("Valence", "Cost")
	costFinal = net.AddLayer4D("CostFin", 1, nCost, popY, popX, USLayer)
	costFinal.SetBuildConfig("DAMod", "D2Mod") // d2mod = final
	costFinal.SetBuildConfig("Valence", "Cost")

	cost.PlaceRightOf(usNeg, space*2)
	costFinal.PlaceBehind(cost, space)
	if rel == relpos.Behind {
		usNeg.PlaceBehind(usPos, space)
	} else {
		usNeg.PlaceRightOf(usPos, space)
	}
	return
}

// AddUSPulvLayers adds USpos, USneg, and Cost layers for positive or negative valence
// unconditioned stimuli (USs), using a pop-code representation of US magnitude.
// These track the Global USpos, USneg, Cost, for visualization and predictive learning.
// Actual US inputs are set in Rubicon.
// Adds Pulvinar predictive layers for each.
func (net *Network) AddUSPulvLayers(popY, popX int, rel relpos.Relations, space float32) (usPos, usNeg, cost, costFinal, usPosP, usNegP, costP *Layer) {
	usPos, usNeg, cost, costFinal = net.AddUSLayers(popY, popX, rel, space)
	usPosP = net.AddPulvForLayer(usPos, space)
	usPosP.SetBuildConfig("Valence", "Positive")
	usNegP = net.AddPulvForLayer(usNeg, space)
	usNegP.SetBuildConfig("Valence", "Negative")
	costP = net.AddPulvForLayer(cost, space)
	costP.SetBuildConfig("Valence", "Cost")
	if rel == relpos.Behind {
		costFinal.PlaceBehind(costP, space)
		usNeg.PlaceBehind(usPosP, space)
	}
	usParams := params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.05",
		"Layer.Inhib.Layer.On":       "false",
		"Layer.Inhib.Pool.On":        "true",
		"Layer.Inhib.Pool.Gi":        "0.5",
	}
	usPosP.DefParams = usParams
	usPosP.AddClass("USLayer")
	usNegP.DefParams = usParams
	usNegP.AddClass("USLayer")
	costP.DefParams = usParams
	costP.AddClass("USLayer")
	costFinal.DefParams = params.Params{
		"Layer.Inhib.Pool.Gi":   "1",
		"Layer.Acts.PopCode.Ge": "1.0",
	}
	return
}

// AddPVLayers adds PVpos and PVneg layers for positive or negative valence
// primary value representations, representing the total drive and effort weighted
// USpos outcome, or total USneg outcome.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
func (net *Network) AddPVLayers(nNeurY, nNeurX int, rel relpos.Relations, space float32) (pvPos, pvNeg *Layer) {
	pvPos = net.AddLayer2D("PVpos", nNeurY, nNeurX, PVLayer)
	pvPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	pvPos.SetBuildConfig("Valence", "Positive")
	pvNeg = net.AddLayer2D("PVneg", nNeurY, nNeurX, PVLayer)
	pvNeg.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
	pvNeg.SetBuildConfig("Valence", "Negative")
	if rel == relpos.Behind {
		pvNeg.PlaceBehind(pvPos, space)
	} else {
		pvNeg.PlaceRightOf(pvPos, space)
	}
	return
}

// AddPVLayers adds PVpos and PVneg layers for positive or negative valence
// primary value representations, representing the total drive and effort weighted
// USpos outcomes, or total USneg outcomes.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
// Adds Pulvinar predictive layers for each.
func (net *Network) AddPVPulvLayers(nNeurY, nNeurX int, rel relpos.Relations, space float32) (pvPos, pvNeg, pvPosP, pvNegP *Layer) {
	pvPos, pvNeg = net.AddPVLayers(nNeurX, nNeurY, rel, space)
	pvPosP = net.AddPulvForLayer(pvPos, space)
	pvNegP = net.AddPulvForLayer(pvNeg, space)
	if rel == relpos.Behind {
		pvNeg.PlaceBehind(pvPosP, space)
	}
	pvParams := params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.2",
		"Layer.Inhib.Layer.Gi":       "0.5",
	}
	pvPosP.DefParams = pvParams
	pvPosP.AddClass("PVLayer")
	pvNegP.DefParams = pvParams
	pvNegP.AddClass("PVLayer")
	return
}

// AddVSPatchLayers adds VSPatch (Pos, D1, D2)
func (net *Network) AddVSPatchLayers(prefix string, nUs, nNeurY, nNeurX int, space float32) (d1, d2 *Layer) {
	d1 = net.AddLayer4D(prefix+"VsPatchD1", 1, nUs, nNeurY, nNeurX, VSPatchLayer)
	d1.SetBuildConfig("DAMod", "D1Mod")
	d1.SetBuildConfig("Valence", "Positive")
	d2 = net.AddLayer4D(prefix+"VsPatchD2", 1, nUs, nNeurY, nNeurX, VSPatchLayer)
	d2.SetBuildConfig("DAMod", "D2Mod")
	d2.SetBuildConfig("Valence", "Positive")
	d2.PlaceBehind(d1, space)
	return
}

// ConnectToVSPatch adds a VSPatchPrjn from given sending layer to VSPatchD1, D2 layers
func (net *Network) ConnectToVSPatch(send, vspD1, vspD2 *Layer, pat prjn.Pattern) (*Prjn, *Prjn) {
	d1 := net.ConnectLayers(send, vspD1, pat, VSPatchPrjn)
	d2 := net.ConnectLayers(send, vspD2, pat, VSPatchPrjn)
	return d1, d2
}

// AddVTALHbLDTLayers adds VTA dopamine, LHb DA dipping, and LDT ACh layers
// which are driven by corresponding values in Global
func (net *Network) AddVTALHbLDTLayers(rel relpos.Relations, space float32) (vta, lhb, ldt *Layer) {
	vta = net.AddLayer2D("VTA", 1, 1, VTALayer)
	lhb = net.AddLayer2D("LHb", 1, 2, LHbLayer)
	ldt = net.AddLDTLayer("")
	if rel == relpos.Behind {
		lhb.PlaceBehind(vta, space)
		ldt.PlaceBehind(lhb, space)
	} else {
		lhb.PlaceRightOf(vta, space)
		ldt.PlaceRightOf(lhb, space)
	}
	return
}

// AddSCLayer2D adds superior colliculcus 2D layer
// which computes stimulus onset via trial-delayed inhibition
// (Inhib.FFPrv) -- connect with fixed random input from sensory
// input layers.  Sets base name and class name to SC.
// Must set Inhib.FFPrv > 0 and Act.Decay.* = 0
func (net *Network) AddSCLayer2D(prefix string, nNeurY, nNeurX int) *Layer {
	sc := net.AddLayer2D(prefix+"SC", nNeurY, nNeurX, SuperLayer)
	sc.DefParams = params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.1",
		"Layer.Inhib.Layer.On":       "true",
		"Layer.Inhib.Layer.Gi":       "1.2",
		"Layer.Inhib.Pool.On":        "false",
		"Layer.Acts.Decay.Act":       "1", // key for rapid updating
		"Layer.Acts.Decay.Glong":     "0.0",
		"Layer.Acts.Decay.LearnCa":   "1.0", // uses CaSpkD as a readout -- clear
		"Layer.Acts.Decay.OnRew":     "true",
		"Layer.Acts.KNa.TrialSlow":   "true",
		"Layer.Acts.KNa.Slow.Max":    "0.05", // 0.1 enough to fully inhibit over several trials
	}
	sc.AddClass("SC")
	return sc
}

// AddSCLayer4D adds superior colliculcus 4D layer
// which computes stimulus onset via trial-delayed inhibition
// (Inhib.FFPrv) -- connect with fixed random input from sensory
// input layers.  Sets base name and class name to SC.
// Must set Inhib.FFPrv > 0 and Act.Decay.* = 0
func (net *Network) AddSCLayer4D(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	sc := net.AddLayer4D(prefix+"SC", nPoolsY, nPoolsX, nNeurY, nNeurX, SuperLayer)
	sc.DefParams = params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.1",
		"Layer.Inhib.Layer.On":       "true",
		"Layer.Inhib.Layer.Gi":       "1.2",
		"Layer.Inhib.Pool.On":        "true",
		"Layer.Inhib.Pool.Gi":        "1.2",
		"Layer.Acts.Decay.Act":       "1", // key for rapid updating
		"Layer.Acts.Decay.Glong":     "0.0",
		"Layer.Acts.Decay.LearnCa":   "1.0", // uses CaSpkD as a readout -- clear
		"Layer.Acts.Decay.OnRew":     "true",
		"Layer.Acts.KNa.TrialSlow":   "true",
		"Layer.Acts.KNa.Slow.Max":    "1",
	}
	sc.AddClass("SC")
	return sc
}

// ConnectToSC adds a ForwardPrjn from given sending layer to
// a SC layer, setting class as ToSC -- should set params
// as fixed random with more variance than usual.
func (net *Network) ConnectToSC(send, recv *Layer, pat prjn.Pattern) *Prjn {
	pj := net.ConnectLayers(send, recv, pat, ForwardPrjn)
	pj.AddClass("ToSC")
	return pj
}

// ConnectToSC1to1 adds a 1to1 ForwardPrjn from given sending layer to
// a SC layer, copying the geometry of the sending layer,
// setting class as ToSC.  The conection weights are set to uniform.
func (net *Network) ConnectToSC1to1(send, recv *Layer) *Prjn {
	recv.Shp.CopyShape(&send.Shp)
	pj := net.ConnectLayers(send, recv, prjn.NewOneToOne(), ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.Learn.Learn":    "false",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0.0",
	}
	pj.AddClass("ToSC")
	return pj
}

// AddDrivesLayer adds Rubicon layer representing current drive activity,
// from Global Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions, per drive pool.
func (net *Network) AddDrivesLayer(ctx *Context, nNeurY, nNeurX int) *Layer {
	drv := net.AddLayer4D("Drives", 1, int(ctx.NetIndexes.RubiconNPosUSs), nNeurY, nNeurX, DrivesLayer)
	return drv
}

// AddDrivesPulvLayer adds Rubicon layer representing current drive activity,
// from Global Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions, per drive pool.
// Adds Pulvinar predictive layers for Drives.
func (net *Network) AddDrivesPulvLayer(ctx *Context, nNeurY, nNeurX int, space float32) (drv, drvP *Layer) {
	drv = net.AddDrivesLayer(ctx, nNeurY, nNeurX)
	drvP = net.AddPulvForLayer(drv, space)
	drvP.DefParams = params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.01",
		"Layer.Inhib.Layer.On":       "false",
		"Layer.Inhib.Pool.On":        "true",
		"Layer.Inhib.Pool.Gi":        "0.5",
	}
	drvP.AddClass("DrivesLayer")
	return
}

// AddUrgencyLayer adds Rubicon layer representing current urgency factor,
// from Global Urgency.Urge
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
func (net *Network) AddUrgencyLayer(nNeurY, nNeurX int) *Layer {
	urge := net.AddLayer2D("Urgency", nNeurY, nNeurX, UrgencyLayer)
	return urge
}

// AddRubiconPulvLayers adds Rubicon layers for PV-related information visualizing
// the internal states of the Global state, with Pulvinar prediction
// layers for training PFC layers.
// Uses the network Rubicon.NPosUSs, NNegUSs, NCosts for number of pools --
// must be configured prior to calling this.
// * drives = popcode representation of drive strength (no activity for 0)
// number of active drives comes from Context; popY, popX neurons per pool.
// * urgency = popcode representation of urgency Go bias factor, popY, popX neurons.
// * us = popcode per US, positive & negative, cost
// * pv = popcode representation of final primary value on positive and negative
// valences -- this is what the dopamine value ends up conding (pos - neg).
// Layers are organized in depth per type: USs in one column, PVs in the next,
// with Drives in the back; urgency behind that.
func (net *Network) AddRubiconPulvLayers(ctx *Context, nYneur, popY, popX int, space float32) (drives, drivesP, urgency, usPos, usNeg, cost, costFinal, usPosP, usNegP, costP, pvPos, pvNeg, pvPosP, pvNegP *Layer) {
	rel := relpos.Behind

	usPos, usNeg, cost, costFinal, usPosP, usNegP, costP = net.AddUSPulvLayers(popY, popX, rel, space)
	pvPos, pvNeg, pvPosP, pvNegP = net.AddPVPulvLayers(popY, popX, rel, space)
	drives, drivesP = net.AddDrivesPulvLayer(ctx, popY, popX, space)
	urgency = net.AddUrgencyLayer(popY, popX)

	pvPos.PlaceRightOf(usPos, space)
	drives.PlaceBehind(usNegP, space)
	urgency.PlaceBehind(usNegP, space)
	return
}

// AddOFCpos adds orbital frontal cortex positive US-coding layers,
// for given number of pos US pools (first is novelty / curiosity pool),
// with given number of units per pool.
func (net *Network) AddOFCpos(ctx *Context, nUSs, nY, ofcY, ofcX int, space float32) (ofc, ofcCT, ofcPT, ofcPTp, ofcMD *Layer) {
	ofc, ofcCT, ofcPT, ofcPTp, ofcMD = net.AddPFC4D("OFCpos", "MD", 1, nUSs, ofcY, ofcX, true, true, space)
	ofc.DefParams["Layer.Inhib.Pool.Gi"] = "1"
	ofcPT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.02"
	ofcPT.DefParams["Layer.Inhib.Pool.On"] = "true"
	// ofcPT.DefParams["Layer.Inhib.Pool.Gi"] = "2.0"
	ofcPT.DefParams["Layer.Acts.Dend.ModACh"] = "true"
	ofcPTp.DefParams["Layer.Inhib.Pool.Gi"] = "1.0"
	ofcPTp.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.1"

	return
}

// AddOFCneg adds orbital frontal cortex negative US-coding layers,
// for given number of neg US pools with given number of units per pool.
func (net *Network) AddOFCneg(ctx *Context, nUSs, ofcY, ofcX int, space float32) (ofc, ofcCT, ofcPT, ofcPTp, ofcMD *Layer) {
	ofc, ofcCT, ofcPT, ofcPTp, ofcMD = net.AddPFC4D("OFCneg", "MD", 1, nUSs, ofcY, ofcX, true, true, space)

	ofc.DefParams["Layer.Inhib.Pool.Gi"] = "1"
	ofc.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.1"
	ofc.DefParams["Layer.Inhib.Layer.Gi"] = "1.2"
	// ofcPT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.2"
	// ofcPT.DefParams["Layer.Inhib.Pool.Gi"] = "3.0"
	ofcPT.DefParams["Layer.Acts.Dend.ModACh"] = "true"
	ofcPTp.DefParams["Layer.Inhib.Pool.Gi"] = "1.4"
	ofcPTp.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.1"

	return
}

// AddACCost adds anterior cingulate cost coding layers,
// for given number of cost pools (typically 2: time, effort),
// with given number of units per pool.
func (net *Network) AddACCost(ctx *Context, nCosts, accY, accX int, space float32) (acc, accCT, accPT, accPTp, accMD *Layer) {
	acc, accCT, accPT, accPTp, accMD = net.AddPFC4D("ACCcost", "MD", 1, nCosts, accY, accX, true, true, space)

	acc.DefParams["Layer.Inhib.Layer.On"] = "false" // no match
	acc.DefParams["Layer.Inhib.Pool.Gi"] = "1"
	acc.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.1"
	acc.DefParams["Layer.Inhib.Layer.Gi"] = "1.2"

	accCT.DefParams["Layer.Inhib.Layer.On"] = "false"
	accCT.DefParams["Layer.Inhib.Pool.Gi"] = "1.8"

	// accPT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.2"
	// accPT.DefParams["Layer.Inhib.Pool.Gi"] = "3.0"
	accPT.DefParams["Layer.Inhib.Layer.On"] = "false"
	accPT.DefParams["Layer.Acts.Dend.ModACh"] = "true"

	accPTp.DefParams["Layer.Inhib.Layer.On"] = "false"
	accPTp.DefParams["Layer.Inhib.Pool.Gi"] = "1.2"
	accPTp.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.1"
	return
}

// AddRubiconOFCus builds a complete Rubicon network with OFCpos
// (orbital frontal cortex) US-coding layers,
// ILpos infralimbic abstract positive value,
// OFCneg for negative value inputs, and ILneg value layers,
// and ACCost cost prediction layers.
// Uses the network Rubicon.NPosUSs, NNegUSs, NCosts for number of pools --
// must be configured prior to calling this.  Calls:
// * AddVTALHbLDTLayers
// * AddRubiconPulvLayers
// * AddVS
// * AddAmygdala
// * AddOFCpos
// * AddOFCneg
// Makes all appropriate interconnections and sets default parameters.
// Needs CS -> BLA, OFC connections to be made.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddRubiconOFCus(ctx *Context, nYneur, popY, popX, bgY, bgX, ofcY, ofcX int, space float32) (vSgpi, vSmtxGo, vSmtxNo, vSpatchD1, vSpatchD2, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, ilPos, ilPosCT, ilPosPT, ilPosPTp, ilPosMD, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, accCost, accCostCT, accCostPT, accCostPTp, accCostMD, ilNeg, ilNegCT, ilNegPT, ilNegPTp, ilNegMD, sc *Layer) {
	nUSpos := int(net.Rubicon.NPosUSs)
	nUSneg := int(net.Rubicon.NNegUSs)
	nCosts := int(net.Rubicon.NCosts)

	vta, lhb, ldt := net.AddVTALHbLDTLayers(relpos.Behind, space)
	_ = lhb
	_ = ldt

	drives, drivesP, urgency, usPos, usNeg, cost, costFinal, usPosP, usNegP, costP, pvPos, pvNeg, pvPosP, pvNegP := net.AddRubiconPulvLayers(ctx, nYneur, popY, popX, space)
	_ = urgency

	vSmtxGo, vSmtxNo, vSgpePr, vSgpeAk, vSstn, vSgpi := net.AddVBG("", 1, nUSpos, bgY, bgX, bgY, bgX, space)
	_, _ = vSgpeAk, vSgpePr
	vSgated := net.AddVSGatedLayer("", nYneur)
	vSpatchD1, vSpatchD2 = net.AddVSPatchLayers("", nUSpos, bgY, bgX, space)
	vSpatchD1.PlaceRightOf(vSstn, space)

	sc = net.AddSCLayer2D("", ofcY, ofcX)
	vSgated.PlaceRightOf(sc, space)
	ldt.SetBuildConfig("SrcLay1Name", sc.Name())

	blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov := net.AddAmygdala("", true, ofcY, ofcX, space)
	_, _, _, _, _ = blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov

	ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, ofcPosMD := net.AddOFCpos(ctx, nUSpos, nYneur, ofcY, ofcX, space)
	_ = ofcPosPT

	ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, ofcNegMD := net.AddOFCneg(ctx, nUSneg, ofcY, ofcX, space)
	_ = ofcNegPT

	ilPos, ilPosCT, ilPosPT, ilPosPTp, ilPosMD = net.AddPFC2D("ILpos", "MD", ofcY, ofcX, true, true, space)
	_ = ilPosPT

	ilNeg, ilNegCT, ilNegPT, ilNegPTp, ilNegMD = net.AddPFC2D("ILneg", "MD", ofcY, ofcX, true, true, space)
	_ = ilNegPT

	ilPosPT.DefParams["Layer.Acts.Dend.ModACh"] = "true"
	ilNegPT.DefParams["Layer.Acts.Dend.ModACh"] = "true"

	accCost, accCostCT, accCostPT, accCostPTp, accCostMD = net.AddACCost(ctx, nCosts, ofcY, ofcX, space)
	_ = accCostPT

	p1to1 := prjn.NewPoolOneToOne()
	// p1to1rnd := prjn.NewPoolUnifRnd()
	// p1to1rnd.PCon = 0.5
	full := prjn.NewFull()
	var pj, bpj *Prjn
	prjnClass := "PFCPrjn"

	vSmtxGo.SetBuildConfig("ThalLay1Name", ofcPosMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay1Name", ofcPosMD.Name())
	net.ConnectLayers(vSgpi, ofcPosMD, full, InhibPrjn) // BGThal sets defaults for this

	vSmtxGo.SetBuildConfig("ThalLay2Name", ofcNegMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay2Name", ofcNegMD.Name())
	net.ConnectLayers(vSgpi, ofcNegMD, full, InhibPrjn)

	vSmtxGo.SetBuildConfig("ThalLay3Name", ilPosMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay3Name", ilPosMD.Name())
	net.ConnectLayers(vSgpi, ilPosMD, full, InhibPrjn)

	vSmtxGo.SetBuildConfig("ThalLay4Name", ilNegMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay4Name", ilNegMD.Name())
	net.ConnectLayers(vSgpi, ilNegMD, full, InhibPrjn) // BGThal configs

	vSmtxGo.SetBuildConfig("ThalLay5Name", accCostMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay5Name", accCostMD.Name())
	net.ConnectLayers(vSgpi, accCostMD, full, InhibPrjn) // BGThal configs

	pfc2m := params.Params{ // contextual, not driving -- weaker
		"Prjn.PrjnScale.Rel": "1", // 0.1", todo was
	}

	// neg val goes to nogo
	pj = net.ConnectToVSMatrix(ilNeg, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")

	net.ConnectToVSPatch(ilNegPTp, vSpatchD1, vSpatchD2, full)

	///////////////////////////////////////////
	//  BLA

	net.ConnectUSToBLA(usPos, blaPosAcq, blaPosExt)
	net.ConnectUSToBLA(usNeg, blaNegAcq, blaNegExt)

	pj = net.ConnectLayers(blaPosAcq, ofcPos, p1to1, ForwardPrjn) // main driver strong input
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs":  "2",
		"Prjn.SWts.Init.Mean": "0.5",
		"Prjn.SWts.Init.Var":  "0.4",
	}
	pj.AddClass("BLAToOFC", prjnClass)

	pj = net.ConnectLayers(blaNegAcq, ofcNeg, p1to1, ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs":  "2",
		"Prjn.SWts.Init.Mean": "0.5",
		"Prjn.SWts.Init.Var":  "0.4",
	}
	pj.AddClass("BLAToOFC", prjnClass)

	pj = net.ConnectToBLAExt(ofcPosPTp, blaPosExt, p1to1)
	pj.DefParams["Prjn.Com.GType"] = "ModulatoryG"
	pj.DefParams["Prjn.PrjnScale.Abs"] = "0.5"
	pj.DefParams["Prjn.SWts.Init.Mean"] = "0.5"
	pj.DefParams["Prjn.SWts.Init.Var"] = "0.4"
	pj.AddClass("PTpToBLAExt", prjnClass)

	///////////////////////////////////////////
	// VS

	d1, d2 := net.ConnectToVSPatch(drives, vSpatchD1, vSpatchD2, p1to1)
	// modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act
	driveToVsp := params.Params{
		"Prjn.Learn.Learn":    "false",
		"Prjn.PrjnScale.Abs":  "1",
		"Prjn.PrjnScale.Rel":  "1",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0.0",
		"Prjn.Com.GType":      "ModulatoryG",
	}
	d1.DefParams = driveToVsp
	d2.DefParams = driveToVsp
	d1.AddClass("DrivesToVSPatch")
	d2.AddClass("DrivesToVSPatch")

	net.ConnectToVSPatch(ofcPosPTp, vSpatchD1, vSpatchD2, p1to1)
	net.ConnectToVSPatch(ilPosPTp, vSpatchD1, vSpatchD2, full)
	net.ConnectToVSPatch(ofcNegPTp, vSpatchD1, vSpatchD2, full)
	net.ConnectToVSPatch(ilNegPTp, vSpatchD1, vSpatchD2, full)
	net.ConnectToVSPatch(pvPosP, vSpatchD1, vSpatchD2, full)

	// same prjns to stn as mtxgo
	net.ConnectToVSMatrix(usPos, vSmtxGo, p1to1)
	// net.ConnectToVSMatrix(usPos, vSmtxNo, p1to1)
	// pj.DefParams = params.Params{
	// 	"Prjn.PrjnScale.Abs": "2", // strong
	// 	"Prjn.PrjnScale.Rel": ".2",
	// }
	net.ConnectToVSMatrix(blaPosAcq, vSmtxNo, p1to1)
	pj = net.ConnectToVSMatrix(blaPosAcq, vSmtxGo, p1to1)
	pj.AddClass("BLAAcqToGo")
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "2", // key strength driver
		"Prjn.PrjnScale.Rel": "1",
	}

	// The usPos version is needed for US gating to clear goal.
	// it is not clear that direct usNeg should drive nogo directly.
	// pj = net.ConnectToVSMatrix(usNeg, vSmtxNo, full)
	// pj.DefParams = params.Params{
	// 	"Prjn.PrjnScale.Abs": "2", // strong
	// 	"Prjn.PrjnScale.Rel": ".2",
	// }
	net.ConnectToVSMatrix(blaNegAcq, vSmtxNo, full).AddClass("BLAAcqToGo") // neg -> nogo
	// pj.DefParams = params.Params{
	// 	"Prjn.PrjnScale.Abs": "2",
	// 	"Prjn.PrjnScale.Rel": "1",
	// }

	net.ConnectLayers(blaPosAcq, vSstn, full, ForwardPrjn)
	net.ConnectLayers(blaNegAcq, vSstn, full, ForwardPrjn)

	// todo: ofc -> STN?

	pj = net.ConnectToVSMatrix(blaPosExt, vSmtxNo, p1to1)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "0.1", // extinction is mostly within BLA
		"Prjn.PrjnScale.Rel": "1",
	}
	pj.AddClass("BLAExtToNo")
	// pj = net.ConnectToVSMatrix(blaNegExt, vSmtxGo, full) // no neg -> go
	// Note: this impairs perf in basic examples/boa, and is questionable functionally
	// pj.DefParams = params.Params{
	// 	"Prjn.PrjnScale.Abs": "0.1", // extinction is mostly within BLA
	// 	"Prjn.PrjnScale.Rel": "1",
	// }
	// pj.AddClass("BLAExtToNo")

	// modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act
	d2m := params.Params{
		"Prjn.Learn.Learn":    "false",
		"Prjn.PrjnScale.Abs":  "1",
		"Prjn.PrjnScale.Rel":  "1",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0.0",
		"Prjn.Com.GType":      "ModulatoryG",
	}
	pj = net.ConnectToVSMatrix(drives, vSmtxGo, p1to1)
	pj.DefParams = d2m
	pj.AddClass("DrivesToMtx")
	pj = net.ConnectToVSMatrix(drives, vSmtxNo, p1to1)
	pj.DefParams = d2m
	pj.AddClass("DrivesToMtx")

	pj = net.ConnectToVSMatrix(ofcPos, vSmtxGo, p1to1)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")
	pj = net.ConnectToVSMatrix(ofcPos, vSmtxNo, p1to1)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")
	net.ConnectLayers(ofcPos, vSstn, full, ForwardPrjn)

	pj = net.ConnectToVSMatrix(ilPos, vSmtxGo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")
	pj = net.ConnectToVSMatrix(ilPos, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")
	net.ConnectLayers(ilPos, vSstn, full, ForwardPrjn)

	pj = net.ConnectToVSMatrix(ofcNeg, vSmtxGo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")
	pj = net.ConnectToVSMatrix(ofcNeg, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")

	pj = net.ConnectToVSMatrix(ilNeg, vSmtxGo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")
	pj = net.ConnectToVSMatrix(ilNeg, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")

	pj = net.ConnectToVSMatrix(accCost, vSmtxGo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")
	pj = net.ConnectToVSMatrix(accCost, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")

	// pj = net.ConnectToVSMatrix(urgency, vSmtxGo, full)
	// pj.DefParams = params.Params{
	// 	"Prjn.PrjnScale.Rel":  "0.1", // don't dilute from others
	// 	"Prjn.PrjnScale.Abs":  "4",   // but make it strong
	// 	"Prjn.SWts.Init.SPct": "0",
	// 	"Prjn.SWts.Init.Mean": "0.5",
	// 	"Prjn.SWts.Init.Var":  "0.4",
	// 	"Prjn.Learn.Learn":    "false",
	// }

	///////////////////////////////////////////
	// OFCpos

	// Drives -> ofcPos then activates ofcPos -> VS -- ofcPos needs to be strongly BLA dependent
	// to reflect either current CS or maintained CS but not just echoing drive state.
	// and not adding drives -> deep layers
	pj = net.ConnectLayers(drives, ofcPos, p1to1, ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Rel": "0.2", // weaker to not drive in absence of BLA
	}
	pj.AddClass("DrivesToOFC ", prjnClass)

	// net.ConnectCTSelf(ilPosCT, full, prjnClass) // todo: test

	net.ConnectLayers(pvPos, ofcPos, full, BackPrjn).AddClass("OFCPrjn", prjnClass)
	net.ConnectLayers(usPos, ofcPos, p1to1, BackPrjn).AddClass("OFCPrjn", prjnClass)

	// note: these are all very static, lead to static PT reps:
	// need a more dynamic US / value representation to predict.

	net.ConnectToPulv(ofcPos, ofcPosCT, drivesP, p1to1, p1to1, "OFCPrjn")
	net.ConnectToPulv(ofcPos, ofcPosCT, usPosP, p1to1, p1to1, "OFCPrjn")
	net.ConnectToPulv(ofcPos, ofcPosCT, pvPosP, full, full, "OFCPrjn")

	net.ConnectPTpToPulv(ofcPosPTp, drivesP, p1to1, p1to1, "OFCPrjn")
	net.ConnectPTToPulv(ofcPosPT, ofcPosPTp, usPosP, p1to1, p1to1, "OFCPrjn")
	net.ConnectPTpToPulv(ofcPosPTp, pvPosP, p1to1, p1to1, "OFCPrjn")

	net.ConnectLayers(ofcPosPT, pvPos, full, ForwardPrjn)

	///////////////////////////////////////////
	// ILpos

	// net.ConnectCTSelf(ilPosCT, full, prjnClass) // todo: test

	pj, bpj = net.BidirConnectLayers(ofcPos, ilPos, full)
	pj.AddClass("ILPrjn", prjnClass)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // val needs stronger input
	}
	bpj.AddClass("ILPrjn", prjnClass)

	// note: do *not* bidirectionally connect PTp layers -- too much sustained activity

	net.ConnectToPFC(pvPos, pvPosP, ilPos, ilPosCT, nil, ilPosPTp, full, "ILPrjn")
	net.ConnectPTpToPulv(ilPosPTp, pvPosP, full, full, "ILPrjn")
	net.BidirConnectLayers(ilPosPT, pvPos, full)

	// note: not connecting deeper CT and PT layers to vSmtxGo at this point
	// could explore that later

	///////////////////////////////////////////
	// OFCneg

	// net.ConnectCTSelf(ofcNegValCT, full, prjnClass) // todo: test

	net.ConnectLayers(pvNeg, ofcNeg, full, BackPrjn).AddClass("OFCPrjn", prjnClass)
	net.ConnectLayers(usNeg, ofcNeg, p1to1, BackPrjn).AddClass("OFCPrjn", prjnClass)

	// note: these are all very static, lead to static PT reps:
	// need a more dynamic US / value representation to predict.

	net.ConnectToPulv(ofcNeg, ofcNegCT, usNegP, p1to1, p1to1, "OFCPrjn")
	net.ConnectToPulv(ofcNeg, ofcNegCT, pvNegP, full, full, "OFCPrjn")
	net.ConnectPTToPulv(ofcNegPT, ofcNegPTp, usNegP, p1to1, p1to1, "OFCPrjn")
	net.ConnectPTpToPulv(ofcNegPTp, pvNegP, full, full, "OFCPrjn")

	net.ConnectLayers(ofcNegPT, pvNeg, full, ForwardPrjn)

	///////////////////////////////////////////
	// Costs

	net.ConnectLayers(pvNeg, accCost, full, BackPrjn).AddClass("ACCPrjn", prjnClass)
	net.ConnectLayers(cost, accCost, p1to1, BackPrjn).AddClass("ACCPrjn", prjnClass)

	net.ConnectToPulv(accCost, accCostCT, costP, p1to1, p1to1, "ACCPrjn")
	net.ConnectPTpToPulv(accCostPTp, costP, p1to1, p1to1, "ACCPrjn")
	pj = net.ConnectLayers(accCostPT, costFinal, p1to1, ForwardPrjn)
	// pj, _ = net.BidirConnectLayers(accCostPT, costFinal, p1to1)
	pj.AddClass("ACCCostToFinal")
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": ".2", // PT is too strong
	}

	///////////////////////////////////////////
	// ILneg

	// net.ConnectCTSelf(ilNegCT, full, "ILPrjn") // todo: test

	pj, bpj = net.BidirConnectLayers(ofcNeg, ilNeg, full)
	pj.AddClass("ILPrjn", prjnClass)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // val needs stronger input
	}
	bpj.AddClass("ILPrjn", prjnClass)

	pj, bpj = net.BidirConnectLayers(accCost, ilNeg, full)
	pj.AddClass("ACCPrjn", prjnClass)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // val needs stronger input
	}
	bpj.AddClass("ILPrjn", prjnClass)

	// note: do *not* bidirectionally connect PTp layers -- too much sustained activity

	net.ConnectToPFC(pvNeg, pvNegP, ilNeg, ilNegCT, nil, ilNegPTp, full, "ILPrjn")
	net.ConnectPTpToPulv(ilNegPTp, pvNegP, full, full, "ILPrjn")
	net.BidirConnectLayers(ilNegPT, pvNeg, full)

	// note: not connecting deeper CT and PT layers to vSmtxGo at this point
	// could explore that later

	////////////////////////////////////////////////
	// position

	vSgpi.PlaceRightOf(vta, space)
	drives.PlaceBehind(vSmtxGo, space)
	drivesP.PlaceBehind(vSmtxNo, space)
	blaNov.PlaceBehind(vSgated, space)
	sc.PlaceRightOf(vSpatchD1, space)
	usPos.PlaceAbove(vta)
	blaPosAcq.PlaceAbove(usPos)
	ofcPos.PlaceRightOf(blaPosAcq, space)
	ofcNeg.PlaceRightOf(ofcPos, space)
	ilPos.PlaceRightOf(ofcNeg, space*3)
	ilNeg.PlaceRightOf(ilPos, space)
	accCost.PlaceRightOf(ilNeg, space)

	return
}

// AddRubicon builds a complete Rubicon model for goal-driven decision making.
// Uses the network Rubicon.NPosUSs and NNegUSs for number of pools --
// must be configured prior to calling this.  Calls:
// * AddRubiconOFCus -- Rubicon, and OFC us coding
// Makes all appropriate interconnections and sets default parameters.
// Needs CS -> BLA, OFC connections to be made.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddRubicon(ctx *Context, nYneur, popY, popX, bgY, bgX, pfcY, pfcX int, space float32) (vSgpi, urgency, pvPos, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, ilPos, ilPosCT, ilPosPT, ilPosPTp, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, ilNeg, ilNegCT, ilNegPT, ilNegPTp, accCost, plUtil, sc *Layer) {

	full := prjn.NewFull()
	var pj *Prjn

	vSgpi, vSmtxGo, vSmtxNo, vSpatchD1, vSpatchD2, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, ilPos, ilPosCT, ilPosPT, ilPosPTp, ilPosMD, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, accCost, accCostCT, accCostPT, accCostPTp, accCostMD, ilNeg, ilNegCT, ilNegPT, ilNegPTp, ilNegMD, sc := net.AddRubiconOFCus(ctx, nYneur, popY, popX, bgY, bgX, pfcY, pfcX, space)
	_, _, _, _, _, _, _ = usPos, usNeg, usNegP, pvNeg, pvNegP, ilPosCT, ilNegMD
	_, _, _ = accCost, accCostCT, accCostPTp
	_, _ = blaNegAcq, blaNegExt
	_, _, _, _, _ = ofcPosPT, ofcNegPT, ilPosPT, ilNegPT, accCostPT

	// ILposP is what PLutil predicts, in order to learn about value (reward)
	ilPosP := net.AddPulvForSuper(ilPos, space)

	// ILnegP is what PLutil predicts, in order to learn about negative US
	ilNegP := net.AddPulvForSuper(ilNeg, space)

	// ACCcostP is what PLutil predicts, in order to learn about cost
	accCostP := net.AddPulvForSuper(accCost, space)

	pfc2m := params.Params{ // contextual, not driving -- weaker
		"Prjn.PrjnScale.Rel": "0.1",
	}

	plUtil, plUtilCT, plUtilPT, plUtilPTp, plUtilMD := net.AddPFC2D("PLutil", "MD", pfcY, pfcX, true, true, space)
	vSmtxGo.SetBuildConfig("ThalLay5Name", plUtilMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay5Name", plUtilMD.Name())
	net.ConnectLayers(vSgpi, plUtilMD, full, InhibPrjn)

	plUtilPT.DefParams["Layer.Acts.Dend.ModACh"] = "true"

	pj = net.ConnectToVSMatrix(plUtil, vSmtxGo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")

	pj = net.ConnectToVSMatrix(plUtil, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.AddClass("PFCToVSMtx")

	net.ConnectToVSPatch(plUtilPTp, vSpatchD1, vSpatchD2, full)

	///////////////////////////////////////////
	// ILneg

	// net.ConnectCTSelf(ilNegCT, full) // todo: test
	// todo: ofcNeg
	// net.ConnectToPFC(effort, effortP, ilNeg, ilNegCT, ilNegPT, ilNegPTp, full)
	// note: can provide input from *other* relevant inputs not otherwise being predicted
	// net.ConnectLayers(dist, ilNegPTPred, full, ForwardPrjn).AddClass("ToPTPred")

	///////////////////////////////////////////
	// PLutil

	// net.ConnectCTSelf(plUtilCT, full) // todo: test

	// util predicts OFCval and ILneg
	pj, _ = net.ConnectToPFCBidir(ilPos, ilPosP, plUtil, plUtilCT, plUtilPT, plUtilPTp, full, "ILToPL")
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "1", // not good to make this stronger actually
	}
	pj, _ = net.ConnectToPFCBidir(ilNeg, ilNegP, plUtil, plUtilCT, plUtilPT, plUtilPTp, full, "ILToPL")
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // drive pl stronger -- only this one works well
	}
	pj, _ = net.ConnectToPFCBidir(accCost, accCostP, plUtil, plUtilCT, plUtilPT, plUtilPTp, full, "ACCToPL")
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // drive pl stronger?
	}

	// todo: try PTPred predicting the input layers to PT

	ilPosP.PlaceBehind(ilPosMD, space)
	ilNegP.PlaceBehind(ilNegMD, space)
	accCostP.PlaceBehind(accCostMD, space)
	plUtil.PlaceRightOf(accCost, space)

	return
}
