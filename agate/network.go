// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

package agate

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
)

// AddMaintLayer adds a MaintLayer using 4D shape with pools,
// and lateral NMDAMaint PoolOneToOne connectivity.
func AddMaintLayer(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *MaintLayer {
	ly := &MaintLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	// chans.ConnectNMDA(nt, ly, ly, prjn.NewPoolOneToOne())
	return ly
}

// AddOutLayer adds a OutLayer using 4D shape with pools,
// and lateral PoolOneToOne connectivity.
func AddOutLayer(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *OutLayer {
	ly := &OutLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	return ly
}

// AddPFC adds a PFC system including SuperLayer, CT with CTCtxtPrjn, MaintLayer,
// and OutLayer which is gated by BG.
// Name is set to "PFC" if empty.  Other layers have appropriate suffixes.
// Optionally creates a TRC Pulvinar for Super.
// Standard Deep CTCtxtPrjn PoolOneToOne Super -> CT projection, and
// 1to1 projections Super -> Maint and Maint -> Out class PFCFixed are created by default.
// CT is placed Behind Super, then Out and Maint, and Pulvinar behind CT if created.
func AddPFC(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay bool) (super, ct, maint, out, pulv emer.Layer) {
	if name == "" {
		name = "PFC"
	}
	super = deep.AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = deep.AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	mainti := AddMaintLayer(nt, name+"Mnt", nPoolsY, nPoolsX, nNeurY, nNeurX)
	maint = mainti
	outi := AddOutLayer(nt, name+"Out", nPoolsY, nPoolsX, nNeurY, nNeurX)
	out = outi

	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	out.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: name, YAlign: relpos.Front, Space: 2})
	maint.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: out.Name(), XAlign: relpos.Left, Space: 2})

	one2one := prjn.NewOneToOne()
	deep.ConnectCtxtToCT(nt, super, ct, prjn.NewPoolOneToOne())

	pj := nt.ConnectLayers(super, maint, one2one, emer.Forward)
	pj.SetClass("PFCFixed")
	pj = nt.ConnectLayers(maint, out, one2one, emer.Forward)
	pj.SetClass("PFCFixed")

	if pulvLay {
		pulvi := deep.AddTRCLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
		pulv = pulvi
		pulvi.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: ct.Name(), XAlign: relpos.Left, Space: 2})
		// pulvi.DriverLay = name
	}
	return
}

// AddPFCPy adds a PFC system including SuperLayer, CT with CTCtxtPrjn, MaintLayer,
// and OutLayer which is gated by BG.
// Name is set to "PFC" if empty.  Other layers have appropriate suffixes.
// Optionally creates a TRC Pulvinar for Super.
// Standard Deep CTCtxtPrjn PoolOneToOne Super -> CT projection, and
// 1to1 projections Super -> Maint and Maint -> Out class PFCFixed are created by default.
// CT is placed Behind Super, then Out and Maint, and Pulvinar behind CT if created.
// Py is Python version, returns layers as a slice
func AddPFCPy(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay bool) []emer.Layer {
	super, ct, maint, out, pulv := AddPFC(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX, pulvLay)
	return []emer.Layer{super, ct, maint, out, pulv}
}
