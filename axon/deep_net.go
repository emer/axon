// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"strings"

	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
)

// AddSuperLayer2D adds a Super Layer of given size, with given name.
func (nt *Network) AddSuperLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, SuperLayer)
	return ly
}

// AddSuperLayer4D adds a Super Layer of given size, with given name.
func (nt *Network) AddSuperLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, SuperLayer)
	return ly
}

// AddCTLayer2D adds a CT Layer of given size, with given name.
func (nt *Network) AddCTLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, CTLayer)
	return ly
}

// AddCTLayer4D adds a CT Layer of given size, with given name.
func (nt *Network) AddCTLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, CTLayer)
	return ly
}

// AddPulvLayer2D adds a Pulvinar Layer of given size, with given name.
func (nt *Network) AddPulvLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, PulvinarLayer)
	return ly
}

// AddPulvLayer4D adds a Pulvinar Layer of given size, with given name.
func (nt *Network) AddPulvLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, PulvinarLayer)
	return ly
}

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT2D(name string, shapeY, shapeX int, space float32, pat prjn.Pattern) (super, ct *Layer) {
	super = nt.AddSuperLayer2D(name, shapeY, shapeX)
	ct = nt.AddCTLayer2D(name+"CT", shapeY, shapeX)
	ct.PlaceBehind(super, space)
	nt.ConnectSuperToCT(super, ct, pat)
	super.SetClass(name)
	ct.SetClass(name)
	return
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat prjn.Pattern) (super, ct *Layer) {
	super = nt.AddSuperLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = nt.AddCTLayer4D(name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.PlaceBehind(super, space)
	nt.ConnectSuperToCT(super, ct, pat)
	super.SetClass(name)
	ct.SetClass(name)
	return
}

// AddPulvForSuper adds a Pulvinar for given superficial layer (SuperLayer)
// with a P suffix.  The Pulv.Driver is set to Super.
// The Pulv layer needs other CT connections from higher up to predict this layer.
// Pulvinar is positioned behind the CT layer.
func (nt *Network) AddPulvForSuper(super *Layer, space float32) *Layer {
	name := super.Name()
	shp := super.Shape()
	var plv *Layer
	if shp.NumDims() == 2 {
		plv = nt.AddPulvLayer2D(name+"P", shp.Dim(0), shp.Dim(1))
	} else {
		plv = nt.AddPulvLayer4D(name+"P", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	plv.SetBuildConfig("DriveLayName", name)
	plv.SetRelPos(relpos.NewBehind(name+"CT", space))
	return plv
}

// AddPulvForLayer adds a Pulvinar for given Layer (typically an Input type layer)
// with a P suffix.  The Pulv.Driver is set to given Layer.
// The Pulv layer needs other CT connections from higher up to predict this layer.
// Pulvinar is positioned behind the given Layer.
func (nt *Network) AddPulvForLayer(lay *Layer, space float32) *Layer {
	name := lay.Name()
	shp := lay.Shape()
	var plv *Layer
	if shp.NumDims() == 2 {
		plv = nt.AddPulvLayer2D(name+"P", shp.Dim(0), shp.Dim(1))
	} else {
		plv = nt.AddPulvLayer4D(name+"P", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	plv.SetBuildConfig("DriveLayName", name)
	plv.PlaceBehind(lay, space)
	return plv
}

// ConnectToPulv connects Super and CT with given Pulv: CT -> Pulv is class CTToPulv,
// From Pulv = type = Back, class = FmPulv
// toPulvPat is the prjn.Pattern CT -> Pulv and fmPulvPat is Pulv -> CT, Super
// Typically Pulv is a different shape than Super and CT, so use Full or appropriate
// topological pattern
func (nt *Network) ConnectToPulv(super, ct, pulv *Layer, toPulvPat, fmPulvPat prjn.Pattern) (toPulv, toSuper, toCT *Prjn) {
	toPulv = nt.ConnectLayers(ct, pulv, toPulvPat, ForwardPrjn).SetClass("CTToPulv").(AxonPrjn).AsAxon()
	toSuper = nt.ConnectLayers(pulv, super, fmPulvPat, BackPrjn).SetClass("FmPulv").(AxonPrjn).AsAxon()
	toCT = nt.ConnectLayers(pulv, ct, fmPulvPat, BackPrjn).SetClass("FmPulv").(AxonPrjn).AsAxon()
	return
}

// ConnectCtxtToCT adds a CTCtxtPrjn from given sending layer to a CT layer
func (nt *Network) ConnectCtxtToCT(send, recv *Layer, pat prjn.Pattern) *Prjn {
	return nt.ConnectLayers(send, recv, pat, CTCtxtPrjn)
}

// ConnectCTSelf adds a Self (Lateral) CTCtxtPrjn projection within a CT layer,
// in addition to a regular lateral projection, which supports active maintenance.
// The CTCtxtPrjn has a Class label of CTSelfCtxt, and the regular one is CTSelfMaint
func (nt *Network) ConnectCTSelf(ly *Layer, pat prjn.Pattern) (ctxt, maint *Prjn) {
	ctxt = nt.ConnectLayers(ly, ly, pat, CTCtxtPrjn).SetClass("CTSelfCtxt").(AxonPrjn).AsAxon()
	maint = nt.LateralConnectLayer(ly, pat).SetClass("CTSelfMaint").(AxonPrjn).AsAxon()
	return
}

// ConnectSuperToCT adds a CTCtxtPrjn from given sending Super layer to a CT layer
// This automatically sets the FmSuper flag to engage proper defaults,
// Uses given projection pattern -- e.g., Full, OneToOne, or PoolOneToOne
func (nt *Network) ConnectSuperToCT(send, recv *Layer, pat prjn.Pattern) *Prjn {
	pj := nt.ConnectLayers(send, recv, pat, CTCtxtPrjn)
	pj.SetClass("CTFmSuper")
	return pj
}

// AddInputPulv2D adds an Input and Layer of given size, with given name.
// The Input layer is set as the Driver of the Layer.
// Both layers have SetClass(name) called to allow shared params.
func (nt *Network) AddInputPulv2D(name string, nNeurY, nNeurX int, space float32) (*Layer, *Layer) {
	in := nt.AddLayer2D(name, nNeurY, nNeurX, InputLayer)
	pulv := nt.AddPulvLayer2D(name+"P", nNeurY, nNeurX)
	pulv.SetBuildConfig("DriveLayName", name)
	in.SetClass(name)
	pulv.SetClass(name)
	pulv.PlaceBehind(in, space)
	return in, pulv
}

// AddInputPulv4D adds an Input and Layer of given size, with given name.
// The Input layer is set as the Driver of the Layer.
// Both layers have SetClass(name) called to allow shared params.
func (nt *Network) AddInputPulv4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32) (*Layer, *Layer) {
	in := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, InputLayer)
	pulv := nt.AddPulvLayer4D(name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	pulv.SetBuildConfig("DriveLayName", name)
	in.SetClass(name)
	pulv.SetClass(name)
	pulv.PlaceBehind(in, space)
	return in, pulv
}

//////////////////////////////////////////////////////////////////
//  PTMaintLayer

// AddPTMaintLayer2D adds a PTMaintLayer of given size, with given name.
func (nt *Network) AddPTMaintLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, PTMaintLayer)
	return ly
}

// AddPTMaintLayer4D adds a PTMaintLayer of given size, with given name.
func (nt *Network) AddPTMaintLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, PTMaintLayer)
	return ly
}

// ConnectPTMaintSelf adds a Self (Lateral) projection within a PTMaintLayer,
// which supports active maintenance, with a class of PTSelfMaint
func (nt *Network) ConnectPTMaintSelf(ly *Layer, pat prjn.Pattern) *Prjn {
	return nt.LateralConnectLayer(ly, pat).SetClass("PTSelfMaint").(AxonPrjn).AsAxon()
}

// AddPTNotMaintLayer adds a PTNotMaintLayer of given size, for given
// PTMaintLayer -- places it to the right of this layer, and calls
// ConnectPTNotMaint to connect the two, using full connectivity.
func (nt *Network) AddPTNotMaintLayer(ptMaint *Layer, nNeurY, nNeurX int, space float32) *Layer {
	name := ptMaint.Name()
	ly := nt.AddLayer2D(name+"Not", nNeurY, nNeurX, PTNotMaintLayer)
	nt.ConnectPTNotMaint(ptMaint, ly, prjn.NewFull())
	ly.PlaceRightOf(ptMaint, space)
	return ly
}

// ConnectPTNotMaint adds a projection from PTMaintLayer to PTNotMaintLayer,
// as fixed inhibitory connections, with class ToPTNotMaintInhib
func (nt *Network) ConnectPTNotMaint(ptMaint, ptNotMaint *Layer, pat prjn.Pattern) *Prjn {
	return nt.ConnectLayers(ptMaint, ptNotMaint, pat, CTCtxtPrjn).SetClass("ToPTNotMaintInhib").(AxonPrjn).AsAxon()
}

// AddPTMaintThalForSuper adds a PTMaint pyramidal tract active maintenance layer and a
// Thalamus layer for given superficial layer (SuperLayer) and associated CT
// with given suffix (e.g., MD, VM).
// PT and Thal have SetClass(super.Name()) called to allow shared params.
// Projections are made with given classes: SuperToPT, PTSelfMaint, CTtoThal,
// PTtoThal, ThalToPT
// The PT and Thal layers are positioned behind the CT layer.
func (nt *Network) AddPTMaintThalForSuper(super, ct *Layer, suffix string, superToPT, ptSelf, ctToThal prjn.Pattern, space float32) (pt, thal *Layer) {
	name := super.Name()
	shp := super.Shape()
	if shp.NumDims() == 2 {
		pt = nt.AddPTMaintLayer2D(name+"PT", shp.Dim(0), shp.Dim(1))
		thal = nt.AddThalLayer2D(name+suffix, shp.Dim(0), shp.Dim(1))
	} else {
		pt = nt.AddPTMaintLayer4D(name+"PT", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
		thal = nt.AddThalLayer4D(name+suffix, shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	pt.SetClass(name)
	thal.SetClass(name)
	pt.PlaceBehind(ct, space)
	thal.PlaceBehind(pt, space)
	one2one := prjn.NewOneToOne()
	pthal, thalpt := nt.BidirConnectLayers(pt, thal, one2one)
	pthal.SetClass("PTtoThal")
	thalpt.SetClass("ThalToPT")
	// note: cannot do this at this point -- need to have it in the params and overall Defaults
	// thalpt.(AxonPrjn).AsAxon().Params.Com.GType = ModulatoryG // thalamic projections are modulatory
	// on other inputs, multiplying their impact
	sthal, thals := nt.BidirConnectLayers(super, thal, superToPT) // shortcuts
	sthal.SetClass("SuperToThal")
	thals.SetClass("ThalToSuper")
	nt.ConnectLayers(super, pt, superToPT, ForwardPrjn).SetClass("SuperToPT")
	nt.LateralConnectLayer(pt, ptSelf).SetClass("PTSelfMaint")
	nt.ConnectLayers(ct, thal, ctToThal, ForwardPrjn).SetClass("CTtoThal")
	return
}

//////////////////////////////////////////////////////////////////
//  PTPredLayer

// AddPTPredLayer2D adds a PTPredLayer of given size, with given name.
func (nt *Network) AddPTPredLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, PTPredLayer)
	return ly
}

// AddPTPredLayer4D adds a PTPredLayer of given size, with given name.
func (nt *Network) AddPTPredLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, PTPredLayer)
	return ly
}

// ConnectPTPredSelf adds a Self (Lateral) projection within a PTPredLayer,
// which supports active maintenance, with a class of PTSelfMaint
func (nt *Network) ConnectPTPredSelf(ly *Layer, pat prjn.Pattern) *Prjn {
	return nt.LateralConnectLayer(ly, pat).SetClass("PTSelfMaint").(AxonPrjn).AsAxon()
}

// ConnectPTPredToPulv connects PTPred with given Pulv: PTPred -> Pulv is class PTPredToPulv,
// From Pulv = type = Back, class = FmPulv
// toPulvPat is the prjn.Pattern PTPred -> Pulv and fmPulvPat is Pulv -> PTPred
// Typically Pulv is a different shape than PTPred, so use Full or appropriate
// topological pattern
func (nt *Network) ConnectPTPredToPulv(ptPred, pulv *Layer, toPulvPat, fmPulvPat prjn.Pattern) (toPulv, toPTPred *Prjn) {
	toPulv = nt.ConnectLayers(ptPred, pulv, toPulvPat, ForwardPrjn).SetClass("PTPredToPulv").(AxonPrjn).AsAxon()
	toPTPred = nt.ConnectLayers(pulv, ptPred, fmPulvPat, BackPrjn).SetClass("FmPulv").(AxonPrjn).AsAxon()
	return
}

// AddPTPredLayer adds a PTPred pyramidal tract prediction layer
// for given PTMaint layer and associated CT.
// Sets SetClass(super.Name()) to allow shared params.
// Projections are made with given classes: SuperToPT, PTSelfMaint, CTtoThal,
// PTtoThal, ThalToPT
// The PTPred layer is positioned behind the PT layer.
func (nt *Network) AddPTPredLayer(ptMaint, ct, thal *Layer, ptToPredPrjn, ctToPredPrjn, toThalPrjn prjn.Pattern, space float32) (ptPred *Layer) {
	name := strings.TrimSuffix(ptMaint.Name(), "PT")
	shp := ptMaint.Shape()
	if shp.NumDims() == 2 {
		ptPred = nt.AddPTPredLayer2D(name+"PTp", shp.Dim(0), shp.Dim(1))
	} else {
		ptPred = nt.AddPTPredLayer4D(name+"PTp", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	ptPred.SetClass(name)
	ptPred.PlaceBehind(ptMaint, space)
	nt.ConnectCtxtToCT(ptMaint, ptPred, ptToPredPrjn).SetClass("PTtoPred")
	nt.ConnectLayers(ct, ptPred, ctToPredPrjn, ForwardPrjn).SetClass("CTtoPred")
	// note: ptpred does not connect to thalamus -- it is only active on trial *after* thal gating
	return
}

/*
// AddPulvAttnLayer2D adds a PulvAttnLayer of given size, with given name.
func (nt *Network) AddPulvAttnLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, PulvAttnLayer)
	return ly
}

// AddPulvAttnLayer4D adds a Layer of given size, with given name.
func (nt *Network) AddPulvAttnLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, PulvAttnLayer)
	return ly
}
*/
