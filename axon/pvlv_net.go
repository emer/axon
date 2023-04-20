// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"golang.org/x/exp/maps"
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
		d1 := net.AddLayer4D(prefix+"BLAPosAcqD1", 1, nUs, nNeurY, nNeurX, BLALayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Positive")
		d2 := net.AddLayer4D(prefix+"BLAPosExtD2", 1, nUs, nNeurY, nNeurX, BLALayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Positive")
		acq = d1
		ext = d2
	} else {
		d1 := net.AddLayer4D(prefix+"BLANegExtD1", 1, nUs, nNeurY, nNeurX, BLALayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Negative")
		d2 := net.AddLayer4D(prefix+"BLANegAcqD2", 1, nUs, nNeurY, nNeurX, BLALayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Negative")
		acq = d2
		ext = d1
	}

	pj := net.ConnectLayers(ext, acq, prjn.NewPoolOneToOne(), InhibPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "0.4", // key param for efficacy of inhibition -- may need to tweak
	}
	pj.SetClass("BLAExtToAcq")

	pj = net.ConnectLayers(acq, ext, prjn.NewOneToOne(), CTCtxtPrjn)
	pj.SetClass("BLAAcqToExt")

	if rel == relpos.Behind {
		ext.PlaceBehind(acq, space)
	} else {
		ext.PlaceRightOf(acq, space)
	}
	acq.SetClass("BLA")
	ext.SetClass("BLA")
	return
}

// AddAmygdala adds a full amygdala complex including BLA,
// CeM, and LDT.  Inclusion of negative valence is optional with neg
// arg -- neg* layers are nil if not included.
func (net *Network) AddAmygdala(prefix string, neg bool, nUs, nNeurY, nNeurX int, space float32) (blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov *Layer) {
	blaPosAcq, blaPosExt = net.AddBLALayers(prefix, true, nUs, nNeurY, nNeurX, relpos.Behind, space)
	if neg {
		blaNegAcq, blaNegExt = net.AddBLALayers(prefix, false, nUs, nNeurY, nNeurX, relpos.Behind, space)
		blaPosAcq.SetBuildConfig("LayInhib1Name", blaNegAcq.Name())
		blaNegAcq.SetBuildConfig("LayInhib1Name", blaPosAcq.Name())
	}
	cemPos = net.AddLayer4D(prefix+"CeMPos", 1, nUs, 1, nNeurX, CeMLayer)
	cemPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	cemPos.SetBuildConfig("Valence", "Positive")

	if neg {
		cemNeg = net.AddLayer4D(prefix+"CeMNeg", 1, nUs, 1, nNeurX, CeMLayer)
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

	net.ConnectLayers(blaPosAcq, cemPos, p1to1, ForwardPrjn).SetClass("BLAToCeM_Excite")
	net.ConnectLayers(blaPosExt, cemPos, p1to1, InhibPrjn).SetClass("BLAToCeM_Inhib")
	// default Abs = 1 works for both of these

	if neg {
		net.ConnectLayers(blaNegAcq, cemNeg, p1to1, ForwardPrjn).SetClass("BLAToCeM_Excite")
		net.ConnectLayers(blaNegExt, cemNeg, p1to1, InhibPrjn).SetClass("BLAToCeM_Inhib")
	}

	pj := net.ConnectLayers(blaNov, blaPosAcq, p1to1, ForwardPrjn)
	pj.DefParams = params.Params{ // dilutes everyone else, so make it weaker Rel, compensate with Abs
		"Prjn.Learn.Learn":   "false",
		"Prjn.PrjnScale.Rel": "0.1",
		"Prjn.PrjnScale.Abs": "5",
		"Prjn.SWt.Init.SPct": "0",
		"Prjn.SWt.Adapt.On":  "false",
		"Prjn.SWt.Init.Mean": "0.5",
		"Prjn.SWt.Init.Var":  "0.4",
	}
	pj.SetClass("BLAFromNovel")

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
	pj.SetClass("BLAAcqPrjn")
	return pj
}

// ConnectToBLAExt adds a BLAPrjn from given sending layer to a BLA layer,
// and configures it for extinctrion parameters.  Sets class to BLAExtPrjn.
func (net *Network) ConnectToBLAExt(send, recv *Layer, pat prjn.Pattern) *Prjn {
	pj := net.ConnectLayers(send, recv, pat, BLAPrjn)
	pj.DefParams = params.Params{
		"Prjn.Learn.LRate.Base":  "0.02",
		"Prjn.Learn.Trace.Tau":   "1", // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "1", // fast for extinction unlearning -- could be slower
	}
	pj.SetClass("BLAExtPrjn")
	return pj
}

// ConnectCSToBLAPos connects the CS input to BLAPosAcqD1, BLANovelCS layers
// using fixed, higher-variance weights, full projection.
// Sets classes to: CSToBLAPos, CSToBLANovel with default params
func (net *Network) ConnectCSToBLAPos(cs, blaAcq, blaNov *Layer) (toAcq, toNov *Prjn) {
	toAcq = net.ConnectLayers(cs, blaAcq, prjn.NewFull(), BLAPrjn)
	toAcq.DefParams = params.Params{ // stronger..
		"Prjn.PrjnScale.Abs":     "1.5",
		"Prjn.Learn.LRate.Base":  "0.02",
		"Prjn.Learn.Trace.Tau":   "1",    // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "0.01", // slow for acq -- could be 0
	}
	toAcq.SetClass("CSToBLAPos")

	toNov = net.ConnectLayers(cs, blaNov, prjn.NewFull(), BLAPrjn)
	toNov.DefParams = params.Params{ // dilutes everyone else, so make it weaker Rel, compensate with Abs
		"Prjn.SWt.Init.SPct": "0",
		"Prjn.SWt.Init.Mean": "0.75",
		"Prjn.SWt.Init.Var":  "0.25",
		"Prjn.SWt.Adapt.On":  "false",
		"Prjn.Learn.Learn":   "false",
	}
	toNov.SetClass("CSToBLANovel")
	return
}

// ConnectUSToBLAPos connects the US input to BLAPosAcqD1 and
// BLAPosExtD2 layers,
// using fixed, higher-variance weights, full projection.
// Sets classes to: USToBLAAcq and USToBLAExt
func (net *Network) ConnectUSToBLAPos(us, blaAcq, blaExt *Layer) (toAcq, toExt *Prjn) {
	toAcq = net.ConnectLayers(us, blaAcq, prjn.NewPoolOneToOne(), BLAPrjn)
	toAcq.DefParams = params.Params{
		"Prjn.PrjnScale.Rel":     "0.5",
		"Prjn.PrjnScale.Abs":     "6",
		"Prjn.SWt.Init.SPct":     "0",
		"Prjn.SWt.Init.Mean":     "0.75",
		"Prjn.SWt.Init.Var":      "0.25",
		"Prjn.Learn.LRate.Base":  "0.001", // could be 0
		"Prjn.Learn.Trace.Tau":   "1",     // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "0.01",  // slow for acq -- could be 0
	}
	toAcq.SetClass("USToBLAAcq")

	toExt = net.ConnectLayers(us, blaExt, prjn.NewPoolOneToOne(), InhibPrjn)
	toExt.DefParams = params.Params{ // actual US inhibits exinction -- must be strong enough to block ACh enh Ge
		"Prjn.PrjnScale.Abs": "2",
		"Prjn.SWt.Init.SPct": "0",
		"Prjn.SWt.Init.Mean": "0.8",
		"Prjn.SWt.Init.Var":  "0",
		"Prjn.SWt.Adapt.On":  "false",
		"Prjn.Learn.Learn":   "false",
	}
	toExt.SetClass("USToBLAExtInhib")
	return
}

// AddUSLayers adds USpos and USneg layers for positive or negative valence
// unconditioned stimuli (USs).
// These track the ContextPVLV.USpos or USneg, for visualization purposes.
// Actual US inputs are set in PVLV.
func (net *Network) AddUSLayers(nUSpos, nUSneg, nYneur int, rel relpos.Relations, space float32) (usPos, usNeg *Layer) {
	usPos = net.AddLayer4D("USpos", 1, nUSpos, nYneur, 1, USLayer)
	usPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	usPos.SetBuildConfig("Valence", "Positive")
	usNeg = net.AddLayer4D("USneg", 1, nUSneg, nYneur, 1, USLayer)
	usNeg.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
	usNeg.SetBuildConfig("Valence", "Negative")
	if rel == relpos.Behind {
		usNeg.PlaceBehind(usPos, space)
	} else {
		usNeg.PlaceRightOf(usPos, space)
	}
	return
}

// AddUSPulvLayers adds USpos and USneg layers for positive or negative valence
// unconditioned stimuli (USs).
// These track the ContextPVLV.USpos or USneg, for visualization purposes.
// Actual US inputs are set in PVLV.
// Adds Pulvinar predictive layers for each.
func (net *Network) AddUSPulvLayers(nUSpos, nUSneg, nYneur int, rel relpos.Relations, space float32) (usPos, usNeg, usPosP, usNegP *Layer) {
	usPos, usNeg = net.AddUSLayers(nUSpos, nUSneg, nYneur, rel, space)
	usPosP = net.AddPulvForLayer(usPos, space)
	usNegP = net.AddPulvForLayer(usNeg, space)
	if rel == relpos.Behind {
		usNeg.PlaceBehind(usPosP, space)
	}
	usParams := params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.2",
		"Layer.Inhib.Layer.Gi":       "0.5",
	}
	usPosP.DefParams = usParams
	usPosP.SetClass("USLayer")
	usNegP.DefParams = usParams
	usNegP.SetClass("USLayer")
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
	pvPosP.SetClass("PVLayer")
	pvNegP.DefParams = pvParams
	pvNegP.SetClass("PVLayer")
	return
}

// AddVSPatchLayer adds VSPatch (Pos, D1)
func (net *Network) AddVSPatchLayer(prefix string, nUs, nNeurY, nNeurX int) *Layer {
	d1 := net.AddLayer4D(prefix+"VSPatch", 1, nUs, nNeurY, nNeurX, VSPatchLayer)
	d1.SetBuildConfig("DAMod", "D1Mod")
	d1.SetBuildConfig("Valence", "Positive")
	return d1
}

// ConnectToVSPatch adds a VSPatchPrjn from given sending layer to a VSPatch layer
func (net *Network) ConnectToVSPatch(send, recv *Layer, pat prjn.Pattern) *Prjn {
	return net.ConnectLayers(send, recv, pat, VSPatchPrjn)
}

// AddVTALHbLDTLayers adds VTA dopamine, LHb DA dipping, and LDT ACh layers
// which are driven by corresponding values in ContextPVLV
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
	sc.SetClass("SC")
	return sc
}

// AddSCLayer4D adds superior colliculcus 4D layer
// which computes stimulus onset via trial-delayed inhibition
// (Inhib.FFPrv) -- connect with fixed random input from sensory
// input layers.  Sets base name and class name to SC.
// Must set Inhib.FFPrv > 0 and Act.Decay.* = 0
func (net *Network) AddSCLayer4D(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	sc := net.AddLayer4D(prefix+"SC", nPoolsY, nPoolsX, nNeurY, nNeurX, SuperLayer)
	sc.SetClass("SC")
	return sc
}

// ConnectToSC adds a ForwardPrjn from given sending layer to
// a SC layer, setting class as ToSC -- should set params
// as fixed random with more variance than usual.
func (net *Network) ConnectToSC(send, recv *Layer, pat prjn.Pattern) *Prjn {
	pj := net.ConnectLayers(send, recv, pat, ForwardPrjn)
	pj.SetClass("ToSC")
	return pj
}

// AddDrivesLayer adds PVLV layer representing current drive activity,
// from ContextPVLV.Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions, per drive pool.
func (net *Network) AddDrivesLayer(ctx *Context, nNeurY, nNeurX int) *Layer {
	drv := net.AddLayer4D("Drives", 1, int(ctx.PVLV.Drive.NActive), nNeurY, nNeurX, DrivesLayer)
	return drv
}

// AddDrivesPulvLayer adds PVLV layer representing current drive activity,
// from ContextPVLV.Drive.Drives.
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
	drvP.SetClass("DrivesLayer")
	return
}

// AddEffortLayer adds PVLV layer representing current effort factor,
// from ContextPVLV.Effort.Disc
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
func (net *Network) AddEffortLayer(nNeurY, nNeurX int) *Layer {
	eff := net.AddLayer2D("Effort", nNeurY, nNeurX, EffortLayer)
	return eff
}

// AddEffortPulvLayer adds PVLV layer representing current effort factor,
// from ContextPVLV.Effort.Disc
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
// Adds Pulvinar predictive layers for Effort.
func (net *Network) AddEffortPulvLayer(nNeurY, nNeurX int, space float32) (eff, effP *Layer) {
	eff = net.AddEffortLayer(nNeurY, nNeurX)
	effP = net.AddPulvForLayer(eff, space)
	effP.SetClass("EffortLayer")
	return
}

// AddUrgencyLayer adds PVLV layer representing current urgency factor,
// from ContextPVLV.Urgency.Urge
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
func (net *Network) AddUrgencyLayer(nNeurY, nNeurX int) *Layer {
	urge := net.AddLayer2D("Urgency", nNeurY, nNeurX, UrgencyLayer)
	return urge
}

// AddUrgencyPulvLayer adds PVLV layer representing current urgency factor,
// from ContextPVLV.Urgency.Urge
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
// Adds Pulvinar predictive layers for Urgency.
func (net *Network) AddUrgencyPulvLayer(nNeurY, nNeurX int, space float32) (urge, urgeP *Layer) {
	urge = net.AddUrgencyLayer(nNeurY, nNeurX)
	urgeP = net.AddPulvForLayer(urge, space)
	urgeP.SetClass("UrgencyLayer")
	return
}

// AddVS adds a Ventral Striatum (VS, mostly Nucleus Accumbens = NAcc) set of layers
// including extensive Ventral Pallidum (VP) using the pcore BG framework,
// via the AddBG method.  Also adds VSPatch and VSGated layers.
// vSmtxGo and No have VSMatrixLayer class set and default params
// appropriate for multi-pool etc
func (net *Network) AddVS(nUSs, nNeurY, nNeurX, nY int, space float32) (vSmtxGo, vSmtxNo, vSstnp, vSstns, vSgpi, vSpatch *Layer) {
	vSmtxGo, vSmtxNo, vSgpeTA, vSstnp, vSstns, vSgpi := net.AddBG("Vs", 1, nUSs, nNeurY, nNeurX, nNeurY, nNeurX, space)

	mp := params.Params{
		"Layer.Matrix.IsVS":          "true",
		"Layer.Inhib.ActAvg.Nominal": ".03",   // def .25
		"Layer.Inhib.Layer.On":       "false", // def true
		"Layer.Inhib.Pool.On":        "true",  // def false
		"Layer.Inhib.Pool.Gi":        "0.5",
	}
	vSmtxGo.DefParams = mp
	vSmtxGo.SetClass("VSMatrixLayer")

	vSmtxNo.DefParams = mp
	vSmtxGo.SetClass("VSMatrixLayer")

	vSgated := net.AddVSGatedLayer("", nY)
	vSpatch = net.AddVSPatchLayer("", nUSs, nNeurY, nNeurX)
	vSpatch.PlaceRightOf(vSstns, space)
	vSgated.PlaceRightOf(vSgpeTA, space)
	return
}

// AddPVLVPulvLayers adds PVLV layers for PV-related information visualizing
// the internal states of the ContextPVLV state, with Pulvinar prediction
// layers for training PFC layers.
// * drives = popcode representation of drive strength (no activity for 0)
// number of active drives comes from Context; popY, popX neurons per pool.
// * effort = popcode representation of effort discount factor, popY, popX neurons.
// * urgency = popcode representation of urgency Go bias factor, popY, popX neurons.
// * us = nYneur per US, represented as present or absent
// * pv = popcode representation of final primary value on positive and negative
// valences -- this is what the dopamine value ends up conding (pos - neg).
// Layers are organized in depth per type: USs in one column, PVs in the next,
// with Drives in the back; effort and urgency behind that.
func (net *Network) AddPVLVPulvLayers(ctx *Context, nUSneg, nYneur, popY, popX int, space float32) (drives, drivesP, effort, effortP, urgency, urgencyP, usPos, usNeg, usPosP, usNegP, pvPos, pvNeg, pvPosP, pvNegP *Layer) {
	rel := relpos.Behind
	nUSpos := int(ctx.PVLV.Drive.NActive)
	usPos, usNeg, usPosP, usNegP = net.AddUSPulvLayers(nUSpos, nUSneg, nYneur, rel, space)
	pvPos, pvNeg, pvPosP, pvNegP = net.AddPVPulvLayers(popY, popX, rel, space)
	drives, drivesP = net.AddDrivesPulvLayer(ctx, popY, popX, space)
	effort, effortP = net.AddEffortPulvLayer(popY, popX, space)
	urgency, urgencyP = net.AddUrgencyPulvLayer(popY, popX, space)

	pvPos.PlaceRightOf(usPos, space)
	drives.PlaceBehind(usNegP, space)
	effort.PlaceBehind(drivesP, space)
	urgency.PlaceRightOf(effort, space)
	return
}

// AddOFCus adds orbital frontal cortex US-coding layers,
// for given number of US pools (first is novelty / curiosity pool),
// with given number of units per pool.  Also adds a PTNotMaintLayer
// called NotMaint with nY units.
func (net *Network) AddOFCus(ctx *Context, nUSs, nY, ofcY, ofcX int, space float32) (ofc, ofcCT, ofcPT, ofcPTp, ofcMD, notMaint *Layer) {
	p1to1 := prjn.NewPoolOneToOne()
	one2one := prjn.NewOneToOne()

	ofc, ofcCT = net.AddSuperCT4D("OFCus", 1, nUSs, ofcY, ofcX, space, one2one)
	// prjns are: super->PT, PT self, CT-> thal
	ofcPT, ofcMD = net.AddPTMaintThalForSuper(ofc, ofcCT, "MD", one2one, p1to1, p1to1, space)
	ofcCT.SetClass("OFCus CTCopy")
	ofcPTp = net.AddPTPredLayer(ofcPT, ofcCT, ofcMD, p1to1, p1to1, p1to1, space)
	ofcPTp.SetClass("OFCus")
	notMaint = net.AddPTNotMaintLayer(ofcPT, nY, 1, space)
	notMaint.Nm = "NotMaint"

	ofcParams := params.Params{
		"Layer.Act.Decay.Act":        "0",
		"Layer.Act.Decay.Glong":      "0",
		"Layer.Act.Decay.OnRew":      "true", // everything clears
		"Layer.Inhib.ActAvg.Nominal": "0.025",
		"Layer.Inhib.Layer.Gi":       "2.2",
		"Layer.Inhib.Pool.On":        "true",
		"Layer.Inhib.Pool.Gi":        "1.2",
		"Layer.Act.Dend.SSGi":        "0",
	}
	ofc.DefParams = ofcParams

	ofcCT.CTDefParamsMedium()
	ofcCT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.025"
	ofcCT.DefParams["Layer.Inhib.Layer.Gi"] = "2.8"
	ofcCT.DefParams["Layer.Inhib.Pool.On"] = "true"
	ofcCT.DefParams["Layer.Inhib.Pool.Gi"] = "1.2"
	ofcCT.DefParams["Layer.Act.Decay.OnRew"] = "true"

	ofcPT.DefParams = maps.Clone(ofcParams)
	ofcPT.DefParams["Layer.Inhib.Layer.Gi"] = "1.8"
	ofcPT.DefParams["Layer.Inhib.Pool.Gi"] = "2.0"

	ofcPTp.DefParams = maps.Clone(ofcParams)
	ofcPTp.DefParams["Layer.Inhib.Layer.Gi"] = "0.8"
	ofcPTp.DefParams["Layer.Inhib.Pool.Gi"] = "0.8"

	ofcMD.DefParams = maps.Clone(ofcParams)
	ofcMD.DefParams["Layer.Inhib.Layer.Gi"] = "1.1"
	ofcMD.DefParams["Layer.Inhib.Pool.Gi"] = "0.6"

	return
}

// AddPVLVOFCus builds a complete PVLV network with OFCus
// (orbital frontal cortex) US-coding layers, calling:
// * AddPVLVPulvLayers
// * AddVS
// * AddAmygdala
// * AddOFCus
// Makes all appropriate interconnections and sets default parameters.
// Needs CS -> BLA, OFC connections to be made.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddPVLVOFCus(ctx *Context, nUSneg, nYneur, popY, popX, bgY, bgX, ofcY, ofcX int, space float32) (vSgpi, usPos, pvPos, ofc, ofcCT, ofcPTp, blaPosAcq, blaPosExt, blaNov *Layer) {
	nUSs := int(ctx.PVLV.Drive.NActive)

	drives, drivesP, effort, effortP, urgency, urgencyP, usPos, usNeg, usPosP, usNegP, pvPos, pvNeg, pvPosP, pvNegP := net.AddPVLVPulvLayers(ctx, nUSneg, nYneur, popY, popX, space)
	_, _, _ = effort, urgency, urgencyP
	_, _, _, _ = usNeg, usNegP, pvNeg, pvNegP

	vSmtxGo, vSmtxNo, vSstnp, vSstns, vSgpi, vSpatch := net.AddVS(nUSs, bgY, bgX, nYneur, space)

	blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov := net.AddAmygdala("", true, nUSs, ofcY, ofcX, space)
	_, _, _, _, _ = blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov

	ofc, ofcCT, ofcPT, ofcPTp, ofcMD, notMaint := net.AddOFCus(ctx, nUSs, nYneur, ofcY, ofcX, space)
	_ = notMaint

	vSmtxGo.SetBuildConfig("ThalLay1Name", ofcMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay1Name", ofcMD.Name())

	p1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()
	var pj *Prjn

	// nt.ConnectToPulv(ofc, ofcCT, usPulv, p1to1, p1to1)
	// Drives -> OFC then activates OFC -> VS -- OFC needs to be strongly BLA dependent
	// to reflect either current CS or maintained CS but not just echoing drive state.
	pj = net.ConnectLayers(drives, ofc, p1to1, ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Rel": "0.2", // weaker to not drive in absence of BLA
	}
	pj.SetClass("DrivesToOFC")

	// nt.ConnectLayers(drives, ofcCT, p1to1, ForwardPrjn).SetClass("DrivesToOFC")
	pj = net.ConnectLayers(vSgpi, ofcMD, full, InhibPrjn)
	pj.DefParams = params.Params{ // BgFixed
		"Prjn.SWt.Init.SPct": "0",
		"Prjn.SWt.Init.Mean": "0.8",
		"Prjn.SWt.Init.Var":  "0.0",
		"Prjn.SWt.Adapt.On":  "false",
		"Prjn.Learn.Learn":   "false",
	}
	net.ConnectLayers(pvPos, ofc, full, BackPrjn)
	net.ConnectLayers(usPos, ofc, p1to1, BackPrjn)
	net.ConnectLayers(ofcPT, ofcCT, p1to1, ForwardPrjn)

	net.ConnectToPulv(ofc, ofcCT, drivesP, p1to1, p1to1)
	net.ConnectToPulv(ofc, ofcCT, effortP, full, full)
	net.ConnectToPulv(ofc, ofcCT, urgencyP, full, full)
	net.ConnectToPulv(ofc, ofcCT, usPosP, p1to1, p1to1)
	net.ConnectToPulv(ofc, ofcCT, pvPosP, p1to1, p1to1)

	net.ConnectPTPredToPulv(ofcPTp, effortP, full, full)
	net.ConnectPTPredToPulv(ofcPTp, urgencyP, full, full)
	// these are all very static, lead to static PT reps:
	// net.ConnectPTPredToPulv(ofcPTp, drivesP, p1to1, p1to1)
	// net.ConnectPTPredToPulv(ofcPTp, usPosP, p1to1, p1to1)
	// net.ConnectPTPredToPulv(ofcPTp, pvPosP, p1to1, p1to1)
	// net.ConnectPTPredToPulv(ofcPTp, csP, full, full)

	net.ConnectUSToBLAPos(usPos, blaPosAcq, blaPosExt)

	pj = net.ConnectLayers(blaPosAcq, ofc, p1to1, ForwardPrjn) // main driver strong input
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "6",
		"Prjn.SWt.Init.Mean": "0.5",
		"Prjn.SWt.Init.Var":  "0.4",
	}

	pj = net.ConnectToBLAExt(ofcPTp, blaPosExt, p1to1)
	pj.DefParams["Prjn.Com.GType"] = "ModulatoryG"
	pj.DefParams["Prjn.PrjnScale.Abs"] = "0.5"
	pj.DefParams["Prjn.SWt.Init.Mean"] = "0.5"
	pj.DefParams["Prjn.SWt.Init.Var"] = "0.4"
	pj.SetClass("PTpToBLAExt")

	pj = net.ConnectToVSPatch(drives, vSpatch, p1to1)
	// modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act
	pj.DefParams = params.Params{
		"Prjn.Learn.Learn":   "false",
		"Prjn.PrjnScale.Abs": "2",
		"Prjn.PrjnScale.Rel": "1",
		"Prjn.SWt.Init.SPct": "0",
		"Prjn.SWt.Init.Mean": "0.8",
		"Prjn.SWt.Init.Var":  "0.0",
		"Prjn.Com.GType":     "ModulatoryG",
	}
	pj.SetClass("DrivesToVSPatch")

	net.ConnectToVSPatch(ofcPTp, vSpatch, p1to1)

	// same prjns to stn as mtxgo
	pj = net.ConnectToMatrix(usPos, vSmtxGo, p1to1)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "2", // strong
		"Prjn.PrjnScale.Rel": ".2",
	}
	pj = net.ConnectToMatrix(blaPosAcq, vSmtxGo, p1to1)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "4", // key strength driver
		"Prjn.PrjnScale.Rel": "1",
	}
	pj.SetClass("BLAAcqToGo")
	net.ConnectLayers(blaPosAcq, vSstnp, full, ForwardPrjn)
	net.ConnectLayers(blaPosAcq, vSstns, full, ForwardPrjn)

	pj = net.ConnectToMatrix(blaPosExt, vSmtxNo, p1to1)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "0.1", // extinction is mostly within BLA
		"Prjn.PrjnScale.Rel": "1",
	}
	pj.SetClass("BLAExtToNo")

	// modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act
	d2m := params.Params{
		"Prjn.Learn.Learn":   "false",
		"Prjn.PrjnScale.Abs": "2",
		"Prjn.PrjnScale.Rel": "1",
		"Prjn.SWt.Init.SPct": "0",
		"Prjn.SWt.Init.Mean": "0.8",
		"Prjn.SWt.Init.Var":  "0.0",
		"Prjn.Com.GType":     "ModulatoryG",
	}
	pj = net.ConnectToMatrix(drives, vSmtxGo, p1to1)
	pj.DefParams = d2m
	pj.SetClass("DrivesToMtx")
	pj = net.ConnectToMatrix(drives, vSmtxNo, p1to1)
	pj.DefParams = d2m
	pj.SetClass("DrivesToMtx")

	net.ConnectToMatrix(ofc, vSmtxGo, p1to1)
	net.ConnectToMatrix(ofc, vSmtxNo, p1to1)
	pj = net.ConnectToMatrix(urgency, vSmtxGo, full)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Rel": "0.1", // don't dilute from others
		"Prjn.PrjnScale.Abs": "20",
		"Prjn.SWt.Init.SPct": "0",
		"Prjn.SWt.Init.Mean": "0.5",
		"Prjn.SWt.Init.Var":  "0.4",
		"Prjn.Learn.Learn":   "false",
	}

	// net.ConnectToMatrix(ofcCT, vSmtxGo, p1to1) // important for matrix to mainly use CS & BLA
	// net.ConnectToMatrix(ofcCT, vSmtxNo, p1to1)
	// net.ConnectToMatrix(ofcPT, vSmtxGo, p1to1)
	// net.ConnectToMatrix(ofcPT, vSmtxNo, p1to1)

	////////////////////////////////////////////////
	// position

	blaPosAcq.PlaceAbove(usPos)
	ofc.PlaceRightOf(blaPosAcq, space)
	ofcMD.PlaceBehind(ofcPTp, space)

	return
}
