// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
)

// AddPPTgLayer adds a PPTgLayer
func (nt *Network) AddPPTgLayer(prefix string, nUs, unY, unX int) *Layer {
	pptg := nt.AddLayer4D(prefix+"PPTg", 1, nUs, unY, unX, PPTgLayer)
	return pptg
}

// AddBLALayers adds two BLA layers, acquisition / extinction / D1 / D2,
// for positive or negative valence
func (nt *Network) AddBLALayers(prefix string, pos bool, nUs, unY, unX int, rel relpos.Relations, space float32) (acq, ext *Layer) {
	if pos {
		d1 := nt.AddLayer4D(prefix+"BLAPosAcqD1", 1, nUs, unY, unX, BLALayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Positive")
		d2 := nt.AddLayer4D(prefix+"BLAPosExtD2", 1, nUs, unY, unX, BLALayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Positive")
		acq = d1
		ext = d2
	} else {
		d1 := nt.AddLayer4D(prefix+"BLANegExtD1", 1, nUs, unY, unX, BLALayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Negative")
		d2 := nt.AddLayer4D(prefix+"BLANegAcqD2", 1, nUs, unY, unX, BLALayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Negative")
		acq = d2
		ext = d1
	}

	nt.ConnectLayers(ext, acq, prjn.NewPoolOneToOne(), emer.Inhib).SetClass("BLAExtToAcq")

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
// CeM, and PPTg.  Inclusion of negative valence is optional with neg
// arg -- neg* layers are nil if not included.
func (nt *Network) AddAmygdala(prefix string, neg bool, nUs, unY, unX int, space float32) (blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, pptg *Layer) {
	blaPosAcq, blaPosExt = nt.AddBLALayers(prefix, true, nUs, unY, unX, relpos.Behind, space)
	if neg {
		blaNegAcq, blaNegExt = nt.AddBLALayers(prefix, false, nUs, unY, unX, relpos.Behind, space)
	}
	cemPos = nt.AddLayer4D(prefix+"CeMPos", 1, nUs, 1, unX, CeMLayer)
	if neg {
		cemNeg = nt.AddLayer4D(prefix+"CeMNeg", 1, nUs, 1, unX, CeMLayer)
	}

	pptg = nt.AddPPTgLayer(prefix, nUs, 1, unX)

	p1to1 := prjn.NewPoolOneToOne()

	nt.ConnectLayers(blaPosAcq, cemPos, p1to1, emer.Forward).SetClass("BLAToCeM_Excite")
	nt.ConnectLayers(blaPosExt, cemPos, p1to1, emer.Inhib).SetClass("BLAToCeM_Inhib")
	nt.ConnectLayers(cemPos, pptg, p1to1, emer.Forward).SetClass("CeMToPPTg")

	if neg {
		nt.ConnectLayers(blaNegAcq, cemNeg, p1to1, emer.Forward).SetClass("BLAToCeM_Excite")
		nt.ConnectLayers(blaNegExt, cemNeg, p1to1, emer.Inhib).SetClass("BLAToCeM_Inhib")
		// nt.ConnectLayers(cemNeg, pptg, p1to1, emer.Forward).SetClass("CeMToPPTg")
	}

	cemPos.PlaceBehind(blaPosExt, space)
	if neg {
		cemNeg.PlaceBehind(cemPos, space)
		pptg.PlaceBehind(cemNeg, space)
	} else {
		pptg.PlaceBehind(cemPos, space)
	}

	return
}

// ConnectToBLA adds a BLAAcqPrjn from given sending layer to a BLA layer
func (nt *Network) ConnectToBLA(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayers(send, recv, pat, emer.PrjnType(BLAAcqPrjn))
}

// AddUSLayers adds USpos and USneg layers for positive or negative valence
// unconditioned stimuli (USs).
// These track the Context.DrivePVLV.USpos or USneg, for visualization purposes.
// Actual US inputs are set in DrivePVLV.
func (nt *Network) AddUSLayers(nUSpos, nUSneg, nYunits int, rel relpos.Relations, space float32) (usPos, usNeg *Layer) {
	usPos = nt.AddLayer4D("USpos", 1, nUSpos, nYunits, 1, USLayer)
	usPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	usPos.SetBuildConfig("Valence", "Positive")
	usNeg = nt.AddLayer4D("USneg", 1, nUSneg, nYunits, 1, USLayer)
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
// These track the Context.DrivePVLV.USpos or USneg, for visualization purposes.
// Actual US inputs are set in DrivePVLV.
// Adds Pulvinar predictive layers for each.
func (nt *Network) AddUSPulvLayers(nUSpos, nUSneg, nYunits int, rel relpos.Relations, space float32) (usPos, usNeg, usPosP, usNegP *Layer) {
	usPos, usNeg = nt.AddUSLayers(nUSpos, nUSneg, nYunits, rel, space)
	usPosP = nt.AddPulvForLayer(usPos, space)
	usNegP = nt.AddPulvForLayer(usNeg, space)
	if rel == relpos.Behind {
		usNeg.PlaceBehind(usPosP, space)
	}
	usPosP.SetClass("USLayer")
	usNegP.SetClass("USLayer")
	return
}

// AddPVLayers adds PVpos and PVneg layers for positive or negative valence
// primary value representations, representing the total drive and effort weighted
// USpos outcome, or total USneg outcome.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of units in the X and Y dimensions.
func (nt *Network) AddPVLayers(unY, unX int, rel relpos.Relations, space float32) (pvPos, pvNeg *Layer) {
	pvPos = nt.AddLayer2D("PVpos", unY, unX, PVLayer)
	pvPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	pvPos.SetBuildConfig("Valence", "Positive")
	pvNeg = nt.AddLayer2D("PVneg", unY, unX, PVLayer)
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
// given numbers of units in the X and Y dimensions.
// Adds Pulvinar predictive layers for each.
func (nt *Network) AddPVPulvLayers(unY, unX int, rel relpos.Relations, space float32) (pvPos, pvNeg, pvPosP, pvNegP *Layer) {
	pvPos, pvNeg = nt.AddPVLayers(unX, unY, rel, space)
	pvPosP = nt.AddPulvForLayer(pvPos, space)
	pvNegP = nt.AddPulvForLayer(pvNeg, space)
	if rel == relpos.Behind {
		pvNeg.PlaceBehind(pvPosP, space)
	}
	pvPosP.SetClass("PVLayer")
	pvNegP.SetClass("PVLayer")
	return
}

// AddVSPatchLayer adds VSPatch (Pos, D1)
func (nt *Network) AddVSPatchLayer(prefix string, nUs, unY, unX int) *Layer {
	d1 := nt.AddLayer4D(prefix+"VSPatch", 1, nUs, unY, unX, VSPatchLayer)
	d1.SetBuildConfig("DAMod", "D1Mod")
	d1.SetBuildConfig("Valence", "Positive")
	return d1

	// d2 = nt.AddLayer4D(prefix+"VSPatchPosD2", 1, nUs, unY, unX, VSPatchLayer)
	// d2.SetBuildConfig("DAMod", "D2Mod")
	// d2.SetBuildConfig("Valence", "Positive")
	// if rel == relpos.Behind {
	// 	d2.PlaceBehind(d1, space)
	// } else {
	// 	d2.PlaceRightOf(d1, space)
	// }
	//  else {
	// d2 = nt.AddLayer4D(prefix+"VSPatchNegD2", 1, nUs, unY, unX, VSPatchLayer)
	// d2.SetBuildConfig("DAMod", "D2Mod")
	// d2.SetBuildConfig("Valence", "Negative")
	// d1 = nt.AddLayer4D(prefix+"VSPatchNegD1", 1, nUs, unY, unX, VSPatchLayer)
	// d1.SetBuildConfig("DAMod", "D1Mod")
	// d1.SetBuildConfig("Valence", "Negative")
	// if rel == relpos.Behind {
	// 	d1.PlaceBehind(d2, space)
	// } else {
	// 	d1.PlaceRightOf(d2, space)
	// }
}

// ConnectToVSPatch adds a VSPatchPrjn from given sending layer to a VSPatch layer
func (nt *Network) ConnectToVSPatch(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayers(send, recv, pat, emer.PrjnType(VSPatchPrjn))
}

// AddVTALHbLayers adds VTA dopamine and LHb dipping layers
// which are driven by corresponding VTA and LHb values in Context.DrivePVLV
func (nt *Network) AddVTALHbLayers(rel relpos.Relations, space float32) (vta, lhb *Layer) {
	vta = nt.AddLayer2D("VTA", 1, 1, VTALayer)
	lhb = nt.AddLayer2D("LHb", 1, 2, LHbLayer)
	if rel == relpos.Behind {
		lhb.PlaceBehind(vta, space)
	} else {
		lhb.PlaceRightOf(vta, space)
	}
	return
}

// AddDrivesLayer adds DrivePVLV layer representing current drive activity,
// which are driven by corresponding Context.DrivePVLV.Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of units in the X and Y dimensions, per drive pool.
func (nt *Network) AddDrivesLayer(ctx *Context, unY, unX int) *Layer {
	drv := nt.AddLayer4D("Drives", 1, int(ctx.DrivePVLV.Drive.NActive), unY, unX, DrivesLayer)
	return drv
}

// AddDrivesPulvLayer adds DrivePVLV layer representing current drive activity,
// which are driven by corresponding Context.DrivePVLV.Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of units in the X and Y dimensions, per drive pool.
// Adds Pulvinar predictive layers for Drives.
func (nt *Network) AddDrivesPulvLayer(ctx *Context, unY, unX int, space float32) (drv, drvP *Layer) {
	drv = nt.AddDrivesLayer(ctx, unY, unX)
	drvP = nt.AddPulvForLayer(drv, space)
	drvP.SetClass("DrivesLayer")
	return
}
