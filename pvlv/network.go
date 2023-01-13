// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
)

// AddBLALayers adds two BLA layers, acquisition / extinction / D1 / D2,
// for positive or negative valence
func AddBLALayers(nt *axon.Network, prefix string, pos bool, nUs, unY, unX int, rel relpos.Relations, space float32) (acq, ext axon.AxonLayer) {
	if pos {
		d1 := &BLALayer{}
		nt.AddLayerInit(d1, prefix+"BLAPosAcqD1", []int{1, nUs, unY, unX}, BLA)
		d1.DaMod.DAR = axon.D1R
		d2 := &BLALayer{}
		nt.AddLayerInit(d2, prefix+"BLAPosExtD2", []int{1, nUs, unY, unX}, BLA)
		d2.DaMod.DAR = axon.D2R
		acq = d1
		ext = d2
	} else {
		d1 := &BLALayer{}
		nt.AddLayerInit(d1, prefix+"BLANegExtD1", []int{1, nUs, unY, unX}, BLA)
		d1.DaMod.DAR = axon.D1R
		d2 := &BLALayer{}
		nt.AddLayerInit(d2, prefix+"BLANegAcqD2", []int{1, nUs, unY, unX}, BLA)
		d2.DaMod.DAR = axon.D2R
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
func AddAmygdala(nt *axon.Network, prefix string, neg bool, nUs, unY, unX int, space float32) (blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, pptg axon.AxonLayer) {
	blaPosAcq, blaPosExt = AddBLALayers(nt, prefix, true, nUs, unY, unX, relpos.Behind, space)
	if neg {
		blaNegAcq, blaNegExt = AddBLALayers(nt, prefix, false, nUs, unY, unX, relpos.Behind, space)
	}
	cemPos = nt.AddLayer4D(prefix+"CeMPos", 1, nUs, 1, unX, CeM).(axon.AxonLayer)
	if neg {
		cemNeg = nt.AddLayer4D(prefix+"CeMNeg", 1, nUs, 1, unX, CeM).(axon.AxonLayer)
	}
	pptg = &PPTgLayer{}
	nt.AddLayerInit(pptg, prefix+"PPTg", []int{1, nUs, 1, unX}, PPTg)

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
