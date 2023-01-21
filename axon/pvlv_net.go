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
func (nt *Network) AddPPTgLayer(prefix string, nUs, unY, unX int) AxonLayer {
	pptg := &Layer{}
	nt.AddLayerInit(pptg, prefix+"PPTg", []int{1, nUs, unY, unX}, emer.LayerType(PPTgLayer))
	return pptg
}

// AddBLALayers adds two BLA layers, acquisition / extinction / D1 / D2,
// for positive or negative valence
func (nt *Network) AddBLALayers(prefix string, pos bool, nUs, unY, unX int, rel relpos.Relations, space float32) (acq, ext AxonLayer) {
	if pos {
		d1 := &Layer{}
		nt.AddLayerInit(d1, prefix+"BLAPosAcqD1", []int{1, nUs, unY, unX}, emer.LayerType(BLALayer))
		d1.Params.Learn.NeuroMod.DAMod = D1Mod
		d2 := &Layer{}
		nt.AddLayerInit(d2, prefix+"BLAPosExtD2", []int{1, nUs, unY, unX}, emer.LayerType(BLALayer))
		d2.Params.Learn.NeuroMod.DAMod = D2Mod
		acq = d1
		ext = d2
	} else {
		d1 := &Layer{}
		nt.AddLayerInit(d1, prefix+"BLANegExtD1", []int{1, nUs, unY, unX}, emer.LayerType(BLALayer))
		d1.Params.Learn.NeuroMod.DAMod = D1Mod
		d2 := &Layer{}
		nt.AddLayerInit(d2, prefix+"BLANegAcqD2", []int{1, nUs, unY, unX}, emer.LayerType(BLALayer))
		d2.Params.Learn.NeuroMod.DAMod = D2Mod
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
func (nt *Network) AddAmygdala(prefix string, neg bool, nUs, unY, unX int, space float32) (blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, pptg AxonLayer) {
	blaPosAcq, blaPosExt = nt.AddBLALayers(prefix, true, nUs, unY, unX, relpos.Behind, space)
	if neg {
		blaNegAcq, blaNegExt = nt.AddBLALayers(prefix, false, nUs, unY, unX, relpos.Behind, space)
	}
	cemPos = nt.AddLayer4D(prefix+"CeMPos", 1, nUs, 1, unX, emer.LayerType(CeMLayer)).(AxonLayer)
	if neg {
		cemNeg = nt.AddLayer4D(prefix+"CeMNeg", 1, nUs, 1, unX, emer.LayerType(CeMLayer)).(AxonLayer)
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
