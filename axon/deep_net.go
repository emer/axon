// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
)

// AddSuperLayer2D adds a Super Layer of given size, with given name.
func (nt *Network) AddSuperLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.Hidden)
	return ly
}

// AddSuperLayer4D adds a Super Layer of given size, with given name.
func (nt *Network) AddSuperLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	return ly
}

// AddCTLayer2D adds a CT Layer of given size, with given name.
func (nt *Network) AddCTLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.LayerType(CTLayer))
	return ly
}

// AddCTLayer4D adds a CT Layer of given size, with given name.
func (nt *Network) AddCTLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.LayerType(CTLayer))
	return ly
}

// AddPulvLayer2D adds a Pulvinar Layer of given size, with given name.
func (nt *Network) AddPulvLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.LayerType(PulvinarLayer))
	return ly
}

// AddPulvLayer4D adds a Pulvinar Layer of given size, with given name.
func (nt *Network) AddPulvLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.LayerType(PulvinarLayer))
	return ly
}

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT2D(name string, shapeY, shapeX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	super = nt.AddSuperLayer2D(name, shapeY, shapeX)
	ct = nt.AddCTLayer2D(name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	nt.ConnectSuperToCT(super, ct, pat)
	super.SetClass(name)
	ct.SetClass(name)
	return
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	super = nt.AddSuperLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = nt.AddCTLayer4D(name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	nt.ConnectSuperToCT(super, ct, pat)
	super.SetClass(name)
	ct.SetClass(name)
	return
}

// AddPulvForSuper adds a Pulvinar for given superficial layer (SuperLayer)
// with a P suffix.  The Pulv.Driver is set to Super.
// The Pulv layer needs other CT connections from higher up to predict this layer.
// Pulvinar is positioned behind the CT layer.
func (nt *Network) AddPulvForSuper(super emer.Layer, space float32) emer.Layer {
	name := super.Name()
	shp := super.Shape()
	var plv emer.Layer
	if shp.NumDims() == 2 {
		plv = nt.AddPulvLayer2D(name+"P", shp.Dim(0), shp.Dim(1))
	} else {
		plv = nt.AddPulvLayer4D(name+"P", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	plv.(AxonLayer).AsAxon().Params.Pulv.DriveLayIdx = uint32(super.Index())
	plv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: space})
	return plv
}

// ConnectToPulv connects Super and CT with given Pulv: CT -> Pulv is class CTToPulv,
// From Pulv = type = Back, class = FmPulv
// toPulvPat is the prjn.Pattern CT -> Pulv and fmPulvPat is Pulv -> CT, Super
// Typically Pulv is a different shape than Super and CT, so use Full or appropriate
// topological pattern
func (nt *Network) ConnectToPulv(super, ct, pulv emer.Layer, toPulvPat, fmPulvPat prjn.Pattern) (toPulv, toSuper, toCT emer.Prjn) {
	toPulv = nt.ConnectLayers(ct, pulv, toPulvPat, emer.Forward).SetClass("CTToPulv")
	toSuper = nt.ConnectLayers(pulv, super, fmPulvPat, emer.Back).SetClass("FmPulv")
	toCT = nt.ConnectLayers(pulv, ct, fmPulvPat, emer.Back).SetClass("FmPulv")
	return
}

// ConnectCtxtToCT adds a CTCtxtPrjn from given sending layer to a CT layer
func (nt *Network) ConnectCtxtToCT(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayers(send, recv, pat, emer.PrjnType(CTCtxtPrjn))
}

// ConnectCTSelf adds a Self (Lateral) CTCtxtPrjn projection within a CT layer,
// in addition to a regular lateral projection, which supports active maintenance.
// The CTCtxtPrjn has a Class label of CTSelfCtxt, and the regular one is CTSelfMaint
func (nt *Network) ConnectCTSelf(ly emer.Layer, pat prjn.Pattern) (ctxt, maint emer.Prjn) {
	ctxt = nt.ConnectLayers(ly, ly, pat, emer.PrjnType(CTCtxtPrjn)).SetClass("CTSelfCtxt")
	maint = nt.LateralConnectLayer(ly, pat).SetClass("CTSelfMaint")
	return
}

// ConnectSuperToCT adds a CTCtxtPrjn from given sending Super layer to a CT layer
// This automatically sets the FmSuper flag to engage proper defaults,
// Uses given projection pattern -- e.g., Full, OneToOne, or PoolOneToOne
func (nt *Network) ConnectSuperToCT(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	pj := nt.ConnectLayers(send, recv, pat, emer.PrjnType(CTCtxtPrjn))
	pj.SetClass("CTFmSuper")
	return pj
}

// AddInputPulv2D adds an Input and Layer of given size, with given name.
// The Input layer is set as the Driver of the Layer.
// Both layers have SetClass(name) called to allow shared params.
func (nt *Network) AddInputPulv2D(name string, nNeurY, nNeurX int, space float32) (emer.Layer, *Layer) {
	in := nt.AddLayer2D(name, nNeurY, nNeurX, emer.Input)
	pulv := nt.AddPulvLayer2D(name+"P", nNeurY, nNeurX)
	pulv.Params.Pulv.DriveLayIdx = uint32(in.Index())
	in.SetClass(name)
	pulv.SetClass(name)
	pulv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	return in, pulv
}

// AddInputPulv4D adds an Input and Layer of given size, with given name.
// The Input layer is set as the Driver of the Layer.
// Both layers have SetClass(name) called to allow shared params.
func (nt *Network) AddInputPulv4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32) (emer.Layer, *Layer) {
	in := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, emer.Input)
	pulv := nt.AddPulvLayer4D(name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	pulv.Params.Pulv.DriveLayIdx = uint32(in.Index())
	in.SetClass(name)
	pulv.SetClass(name)
	pulv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	return in, pulv
}

/*
// AddPulvAttnLayer2D adds a PulvAttnLayer of given size, with given name.
func (nt *Network) AddPulvAttnLayer2D(name string, nNeurY, nNeurX int) *PulvAttnLayer {
	ly := &PulvAttnLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, Pulv)
	return ly
}

// AddPulvAttnLayer4D adds a Layer of given size, with given name.
func (nt *Network) AddPulvAttnLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *PulvAttnLayer {
	ly := &PulvAttnLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, Pulv)
	return ly
}
*/
