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
		ext.SetRelPos(relpos.Rel{Rel: rel, Other: acq.Name(), XAlign: relpos.Left, Space: space})
	} else {
		ext.SetRelPos(relpos.Rel{Rel: rel, Other: acq.Name(), YAlign: relpos.Front, Space: space})
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

	cemPos.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: blaPosExt.Name(), XAlign: relpos.Left, Space: space})
	if neg {
		cemNeg.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: cemPos.Name(), XAlign: relpos.Left, Space: space})
		pptg.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: cemNeg.Name(), XAlign: relpos.Left, Space: space})
	} else {
		pptg.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: cemPos.Name(), XAlign: relpos.Left, Space: space})
	}

	return
}

// ConnectToBLA adds a BLAPrjn from given sending layer to a BLA layer
func (nt *Network) ConnectToBLA(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayers(send, recv, pat, emer.PrjnType(BLAPrjn))
}

// AddPVLayers adds PosPV and NegPV layers for positive or negative valence
// primary value outcomes (USs)
func (nt *Network) AddPVLayers(nPosPV, nNegPV, nYunits int, rel relpos.Relations, space float32) (pospv, negpv *Layer) {
	pospv = nt.AddLayer4D("PosPV", 1, nPosPV, nYunits, 1, PVLayer)
	pospv.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	pospv.SetBuildConfig("Valence", "Positive")
	negpv = nt.AddLayer4D("NegPV", 1, nNegPV, nYunits, 1, PVLayer)
	negpv.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
	negpv.SetBuildConfig("Valence", "Negative")
	if rel == relpos.Behind {
		negpv.SetRelPos(relpos.Rel{Rel: rel, Other: pospv.Name(), XAlign: relpos.Left, Space: space})
	} else {
		negpv.SetRelPos(relpos.Rel{Rel: rel, Other: pospv.Name(), YAlign: relpos.Front, Space: space})
	}
	return
}

// AddVSPatchLayers adds two VSPatch layers: D1 / D2 for positive or negative valence
func (nt *Network) AddVSPatchLayers(prefix string, pos bool, nUs, unY, unX int, rel relpos.Relations, space float32) (d1, d2 *Layer) {
	if pos {
		d1 = nt.AddLayer4D(prefix+"VSPatchPosD1", 1, nUs, unY, unX, VSPatchLayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Positive")
		d2 = nt.AddLayer4D(prefix+"VSPatchPosD2", 1, nUs, unY, unX, VSPatchLayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Positive")
		if rel == relpos.Behind {
			d2.SetRelPos(relpos.Rel{Rel: rel, Other: d1.Name(), XAlign: relpos.Left, Space: space})
		} else {
			d2.SetRelPos(relpos.Rel{Rel: rel, Other: d1.Name(), YAlign: relpos.Front, Space: space})
		}
	} else {
		d2 = nt.AddLayer4D(prefix+"VSPatchNegD2", 1, nUs, unY, unX, VSPatchLayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Negative")
		d1 = nt.AddLayer4D(prefix+"VSPatchNegD1", 1, nUs, unY, unX, VSPatchLayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Negative")
		if rel == relpos.Behind {
			d1.SetRelPos(relpos.Rel{Rel: rel, Other: d2.Name(), XAlign: relpos.Left, Space: space})
		} else {
			d1.SetRelPos(relpos.Rel{Rel: rel, Other: d2.Name(), YAlign: relpos.Front, Space: space})
		}
	}
	return
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
		lhb.SetRelPos(relpos.Rel{Rel: rel, Other: vta.Name(), XAlign: relpos.Left, Space: space})
	} else {
		lhb.SetRelPos(relpos.Rel{Rel: rel, Other: vta.Name(), YAlign: relpos.Front, Space: space})
	}
	return
}

// AddDrivesLayer adds DrivePVLV layer representing current drive activity,
// which are driven by corresponding Context.DrivePVLV.Drive.Drives
func (nt *Network) AddDrivesLayer(ctx *Context, nYunits int) *Layer {
	drv := nt.AddLayer4D("Drives", 1, int(ctx.DrivePVLV.Drive.NActive), nYunits, 1, DrivesLayer)
	return drv
}
