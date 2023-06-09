// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"strings"

	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"golang.org/x/exp/maps"
)

// AddSuperLayer2D adds a Super Layer of given size, with given name.
func (net *Network) AddSuperLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, SuperLayer)
	return ly
}

// AddSuperLayer4D adds a Super Layer of given size, with given name.
func (net *Network) AddSuperLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, SuperLayer)
	return ly
}

// AddCTLayer2D adds a CT Layer of given size, with given name.
func (net *Network) AddCTLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, CTLayer)
	return ly
}

// AddCTLayer4D adds a CT Layer of given size, with given name.
func (net *Network) AddCTLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, CTLayer)
	return ly
}

// AddPulvLayer2D adds a Pulvinar Layer of given size, with given name.
func (net *Network) AddPulvLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, PulvinarLayer)
	return ly
}

// AddPulvLayer4D adds a Pulvinar Layer of given size, with given name.
func (net *Network) AddPulvLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, PulvinarLayer)
	return ly
}

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (net *Network) AddSuperCT2D(name, prjnClass string, shapeY, shapeX int, space float32, pat prjn.Pattern) (super, ct *Layer) {
	super = net.AddSuperLayer2D(name, shapeY, shapeX)
	ct = net.AddCTLayer2D(name+"CT", shapeY, shapeX)
	ct.PlaceBehind(super, space)
	net.ConnectSuperToCT(super, ct, pat, prjnClass)
	super.SetClass(name)
	ct.SetClass(name)
	return
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (net *Network) AddSuperCT4D(name, prjnClass string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat prjn.Pattern) (super, ct *Layer) {
	super = net.AddSuperLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = net.AddCTLayer4D(name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.PlaceBehind(super, space)
	net.ConnectSuperToCT(super, ct, pat, prjnClass)
	super.SetClass(name)
	ct.SetClass(name)
	return
}

// AddPulvForSuper adds a Pulvinar for given superficial layer (SuperLayer)
// with a P suffix.  The Pulv.Driver is set to Super, as is the Class on Pulv.
// The Pulv layer needs other CT connections from higher up to predict this layer.
// Pulvinar is positioned behind the CT layer.
func (net *Network) AddPulvForSuper(super *Layer, space float32) *Layer {
	name := super.Name()
	shp := super.Shape()
	var plv *Layer
	if shp.NumDims() == 2 {
		plv = net.AddPulvLayer2D(name+"P", shp.Dim(0), shp.Dim(1))
	} else {
		plv = net.AddPulvLayer4D(name+"P", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	plv.SetBuildConfig("DriveLayName", name)
	plv.SetRelPos(relpos.NewBehind(name+"CT", space))
	plv.SetClass(name)
	return plv
}

// AddPulvForLayer adds a Pulvinar for given Layer (typically an Input type layer)
// with a P suffix.  The Pulv.Driver is set to given Layer.
// The Pulv layer needs other CT connections from higher up to predict this layer.
// Pulvinar is positioned behind the given Layer.
func (net *Network) AddPulvForLayer(lay *Layer, space float32) *Layer {
	name := lay.Name()
	shp := lay.Shape()
	var plv *Layer
	if shp.NumDims() == 2 {
		plv = net.AddPulvLayer2D(name+"P", shp.Dim(0), shp.Dim(1))
	} else {
		plv = net.AddPulvLayer4D(name+"P", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	plv.SetBuildConfig("DriveLayName", name)
	plv.PlaceBehind(lay, space)
	return plv
}

// ConnectToPulv connects Super and CT with given Pulv: CT -> Pulv is class CTToPulv,
// From Pulv = type = Back, class = FmPulv
// toPulvPat is the prjn.Pattern CT -> Pulv and fmPulvPat is Pulv -> CT, Super
// Typically Pulv is a different shape than Super and CT, so use Full or appropriate
// topological pattern. adds optional class name to projection.
func (net *Network) ConnectToPulv(super, ct, pulv *Layer, toPulvPat, fmPulvPat prjn.Pattern, prjnClass string) (toPulv, toSuper, toCT *Prjn) {
	if prjnClass != "" {
		prjnClass = " " + prjnClass
	}
	toPulv = net.ConnectLayers(ct, pulv, toPulvPat, ForwardPrjn)
	toPulv.SetClass("CTToPulv" + prjnClass)
	toSuper = net.ConnectLayers(pulv, super, fmPulvPat, BackPrjn)
	toSuper.SetClass("FmPulv" + prjnClass)
	toCT = net.ConnectLayers(pulv, ct, fmPulvPat, BackPrjn)
	toCT.SetClass("FmPulv" + prjnClass)
	return
}

// ConnectCtxtToCT adds a CTCtxtPrjn from given sending layer to a CT layer
func (net *Network) ConnectCtxtToCT(send, recv *Layer, pat prjn.Pattern) *Prjn {
	return net.ConnectLayers(send, recv, pat, CTCtxtPrjn)
}

// ConnectCTSelf adds a Self (Lateral) CTCtxtPrjn projection within a CT layer,
// in addition to a regular lateral projection, which supports active maintenance.
// The CTCtxtPrjn has a Class label of CTSelfCtxt, and the regular one is CTSelfMaint
// with optional class added.
func (net *Network) ConnectCTSelf(ly *Layer, pat prjn.Pattern, prjnClass string) (ctxt, maint *Prjn) {
	if prjnClass != "" {
		prjnClass = " " + prjnClass
	}
	ctxt = net.ConnectLayers(ly, ly, pat, CTCtxtPrjn)
	ctxt.SetClass("CTSelfCtxt" + prjnClass)
	maint = net.LateralConnectLayer(ly, pat)
	maint.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "0.5", // normalized separately
		"Prjn.Com.GType":     "MaintG",
	}
	maint.SetClass("CTSelfMaint" + prjnClass)
	return
}

// ConnectSuperToCT adds a CTCtxtPrjn from given sending Super layer to a CT layer
// This automatically sets the FmSuper flag to engage proper defaults,
// Uses given projection pattern -- e.g., Full, OneToOne, or PoolOneToOne
func (net *Network) ConnectSuperToCT(send, recv *Layer, pat prjn.Pattern, prjnClass string) *Prjn {
	if prjnClass != "" {
		prjnClass = " " + prjnClass
	}
	pj := net.ConnectLayers(send, recv, pat, CTCtxtPrjn)
	pj.SetClass("CTFmSuper" + prjnClass)
	return pj
}

// AddInputPulv2D adds an Input and Layer of given size, with given name.
// The Input layer is set as the Driver of the Layer.
// Both layers have SetClass(name) called to allow shared params.
func (net *Network) AddInputPulv2D(name string, nNeurY, nNeurX int, space float32) (*Layer, *Layer) {
	in := net.AddLayer2D(name, nNeurY, nNeurX, InputLayer)
	pulv := net.AddPulvLayer2D(name+"P", nNeurY, nNeurX)
	pulv.SetBuildConfig("DriveLayName", name)
	in.SetClass(name)
	pulv.SetClass(name)
	pulv.PlaceBehind(in, space)
	return in, pulv
}

// AddInputPulv4D adds an Input and Layer of given size, with given name.
// The Input layer is set as the Driver of the Layer.
// Both layers have SetClass(name) called to allow shared params.
func (net *Network) AddInputPulv4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32) (*Layer, *Layer) {
	in := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, InputLayer)
	pulv := net.AddPulvLayer4D(name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	pulv.SetBuildConfig("DriveLayName", name)
	in.SetClass(name)
	pulv.SetClass(name)
	pulv.PlaceBehind(in, space)
	return in, pulv
}

//////////////////////////////////////////////////////////////////
//  PTMaintLayer

// AddPTMaintLayer2D adds a PTMaintLayer of given size, with given name.
func (net *Network) AddPTMaintLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, PTMaintLayer)
	return ly
}

// AddPTMaintLayer4D adds a PTMaintLayer of given size, with given name.
func (net *Network) AddPTMaintLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, PTMaintLayer)
	return ly
}

// ConnectPTMaintSelf adds a Self (Lateral) projection within a PTMaintLayer,
// which supports active maintenance, with a class of PTSelfMaint
func (net *Network) ConnectPTMaintSelf(ly *Layer, pat prjn.Pattern, prjnClass string) *Prjn {
	if prjnClass != "" && prjnClass[0] != ' ' {
		prjnClass = " " + prjnClass
	}
	pj := net.LateralConnectLayer(ly, pat)
	pj.DefParams = params.Params{
		"Prjn.Com.GType":        "MaintG",
		"Prjn.PrjnScale.Rel":    "1",      // use abs to manipulate
		"Prjn.PrjnScale.Abs":    "4",      // strong..
		"Prjn.Learn.LRate.Base": "0.0001", // slower > faster
		"Prjn.SWts.Init.Mean":   "0.5",
		"Prjn.SWts.Init.Var":    "0.5", // high variance so not just spreading out over time
	}
	pj.SetClass("PTSelfMaint" + prjnClass)
	return pj
}

// AddPTNotMaintLayer adds a PTNotMaintLayer of given size, for given
// PTMaintLayer -- places it to the right of this layer, and calls
// ConnectPTNotMaint to connect the two, using full connectivity.
func (net *Network) AddPTNotMaintLayer(ptMaint *Layer, nNeurY, nNeurX int, space float32) *Layer {
	name := ptMaint.Name()
	ly := net.AddLayer2D(name+"Not", nNeurY, nNeurX, PTNotMaintLayer)
	net.ConnectPTNotMaint(ptMaint, ly, prjn.NewFull())
	ly.PlaceRightOf(ptMaint, space)
	return ly
}

// ConnectPTNotMaint adds a projection from PTMaintLayer to PTNotMaintLayer,
// as fixed inhibitory connections, with class ToPTNotMaintInhib
func (net *Network) ConnectPTNotMaint(ptMaint, ptNotMaint *Layer, pat prjn.Pattern) *Prjn {
	pj := net.ConnectLayers(ptMaint, ptNotMaint, pat, CTCtxtPrjn)
	pj.SetClass("ToPTNotMaintInhib")
	return pj
}

// AddPTMaintThalForSuper adds a PTMaint pyramidal tract active maintenance layer
// and a BG gated Thalamus layer for given superficial layer (SuperLayer)
// and associated CT, with given thal suffix (e.g., MD, VM).
// PT and Thal have SetClass(super.Name()) called to allow shared params.
// Projections are made with given classes: SuperToPT, PTSelfMaint, PTtoThal, ThalToPT,
// with optional extra class
// The PT and BGThal layers are positioned behind the CT layer.
func (net *Network) AddPTMaintThalForSuper(super, ct *Layer, thalSuffix, prjnClass string, superToPT, ptSelf prjn.Pattern, space float32) (pt, thal *Layer) {
	if prjnClass != "" {
		prjnClass = " " + prjnClass
	}
	name := super.Name()
	shp := super.Shape()
	// is4D := false
	if shp.NumDims() == 2 {
		pt = net.AddPTMaintLayer2D(name+"PT", shp.Dim(0), shp.Dim(1))
		thal = net.AddBGThalLayer2D(name+thalSuffix, shp.Dim(0), shp.Dim(1))
	} else {
		// is4D = true
		pt = net.AddPTMaintLayer4D(name+"PT", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
		thal = net.AddBGThalLayer4D(name+thalSuffix, shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	pt.SetClass(name)
	thal.SetClass(name)

	// full := prjn.NewFull()
	one2one := prjn.NewOneToOne()
	pthal, thalpt := net.BidirConnectLayers(pt, thal, one2one)
	pthal.SetClass("PTtoThal" + prjnClass)
	thalpt.DefParams = params.Params{
		"Prjn.PrjnScale.Rel":  "1.0",
		"Prjn.Com.GType":      "ModulatoryG", // modulatory -- control with extra ModGain factor
		"Prjn.Learn.Learn":    "false",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0.0",
	}
	thalpt.SetClass("ThalToPT" + prjnClass)
	// if is4D {
	// fmThalInhib := params.Params{
	// 	"Prjn.PrjnScale.Rel": "1.0",
	// 	"Prjn.PrjnScale.Abs": "1.0",
	// 	"Prjn.Learn.Learn":   "false",
	// 	"Prjn.SWts.Adapt.On":  "false",
	// 	"Prjn.SWts.Init.SPct": "0",
	// 	"Prjn.SWts.Init.Mean": "0.8",
	// 	"Prjn.SWts.Init.Var":  "0.0",
	// }
	// note: holding off on these for now -- thal modulation should handle..
	// ti := net.ConnectLayers(thal, pt, full, InhibPrjn)
	// ti.DefParams = fmThalInhib
	// ti.SetClass("ThalToPFCInhib")
	// ti = net.ConnectLayers(thal, ct, full, InhibPrjn)
	// ti.DefParams = fmThalInhib
	// ti.SetClass("ThalToPFCInhib")

	sthal := net.ConnectLayers(super, thal, superToPT, ForwardPrjn) // shortcuts
	sthal.DefParams = params.Params{
		"Prjn.PrjnScale.Rel":  "1.0",
		"Prjn.PrjnScale.Abs":  "4.0", // key param for driving gating -- if too strong, premature gating
		"Prjn.Learn.Learn":    "false",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.8", // typically 1to1
		"Prjn.SWts.Init.Var":  "0.0",
	}
	sthal.SetClass("SuperToThal" + prjnClass)

	pj := net.ConnectLayers(super, pt, superToPT, ForwardPrjn)
	pj.DefParams = params.Params{
		// one-to-one from super -- just use fixed nonlearning prjn so can control behavior easily
		"Prjn.PrjnScale.Rel":  "1",   // irrelevant -- only normal prjn
		"Prjn.PrjnScale.Abs":  "0.5", // BGThal modulates this so strength doesn't cause wrong CS gating
		"Prjn.Learn.Learn":    "false",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0.0",
	}
	pj.SetClass("SuperToPT" + prjnClass)

	net.ConnectPTMaintSelf(pt, ptSelf, prjnClass)

	pt.PlaceBehind(ct, space)
	thal.PlaceBehind(pt, space)

	return
}

//////////////////////////////////////////////////////////////////
//  PTPredLayer

// AddPTPredLayer2D adds a PTPredLayer of given size, with given name.
func (net *Network) AddPTPredLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, PTPredLayer)
	return ly
}

// AddPTPredLayer4D adds a PTPredLayer of given size, with given name.
func (net *Network) AddPTPredLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, PTPredLayer)
	return ly
}

// ConnectPTPredSelf adds a Self (Lateral) projection within a PTPredLayer,
// which supports active maintenance, with a class of PTSelfMaint
func (net *Network) ConnectPTPredSelf(ly *Layer, pat prjn.Pattern) *Prjn {
	return net.LateralConnectLayer(ly, pat).SetClass("PTSelfMaint").(AxonPrjn).AsAxon()
}

// ConnectPTPredToPulv connects PTPred with given Pulv: PTPred -> Pulv is class PTPredToPulv,
// From Pulv = type = Back, class = FmPulv
// toPulvPat is the prjn.Pattern PTPred -> Pulv and fmPulvPat is Pulv -> PTPred
// Typically Pulv is a different shape than PTPred, so use Full or appropriate
// topological pattern. adds optional class name to projection.
func (net *Network) ConnectPTPredToPulv(ptPred, pulv *Layer, toPulvPat, fmPulvPat prjn.Pattern, prjnClass string) (toPulv, toPTPred *Prjn) {
	if prjnClass != "" {
		prjnClass = " " + prjnClass
	}
	toPulv = net.ConnectLayers(ptPred, pulv, toPulvPat, ForwardPrjn)
	toPulv.SetClass("PTPredToPulv" + prjnClass)
	toPTPred = net.ConnectLayers(pulv, ptPred, fmPulvPat, BackPrjn)
	toPTPred.SetClass("FmPulv" + prjnClass)
	return
}

// AddPTPredLayer adds a PTPred pyramidal tract prediction layer
// for given PTMaint layer and associated CT.
// Sets SetClass(super.Name()) to allow shared params.
// Projections are made with given classes: PTtoPred, CTtoPred
// The PTPred layer is positioned behind the PT layer.
func (net *Network) AddPTPredLayer(ptMaint, ct *Layer, ptToPredPrjn, ctToPredPrjn prjn.Pattern, prjnClass string, space float32) (ptPred *Layer) {
	if prjnClass != "" {
		prjnClass = " " + prjnClass
	}
	name := strings.TrimSuffix(ptMaint.Name(), "PT")
	shp := ptMaint.Shape()
	if shp.NumDims() == 2 {
		ptPred = net.AddPTPredLayer2D(name+"PTp", shp.Dim(0), shp.Dim(1))
	} else {
		ptPred = net.AddPTPredLayer4D(name+"PTp", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	ptPred.SetClass(name)
	ptPred.PlaceBehind(ptMaint, space)
	net.ConnectCtxtToCT(ptMaint, ptPred, ptToPredPrjn).SetClass("PTtoPred" + prjnClass)
	pj := net.ConnectLayers(ct, ptPred, ctToPredPrjn, ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Rel": "1", // 1 > 0.5
		// "Prjn.PrjnScale.Abs": "2.0", // 2?
	}
	pj.SetClass("CTtoPred" + prjnClass)

	// note: ptpred does not connect to thalamus -- it is only active on trial *after* thal gating
	return
}

// AddPFC4D adds a "full stack" of 4D PFC layers:
// * AddSuperCT4D (Super and CT)
// * AddPTMaintThal (PTMaint, BGThal)
// * AddPTPredLayer (PTPred)
// with given name prefix, which is also set as the Class for all layers,
// and suffix for the BGThal layer (e.g., "MD" or "VM" etc for different thalamic nuclei).
// OneToOne and PoolOneToOne connectivity is used between layers.
// decayOnRew determines the Act.Decay.OnRew setting (true of OFC, ACC type for sure).
// CT layer uses the Medium timescale params.
// use, e.g., pfcCT.DefParams["Layer.Inhib.Layer.Gi"] = "2.8" to change default params.
func (net *Network) AddPFC4D(name, thalSuffix string, nPoolsY, nPoolsX, nNeurY, nNeurX int, decayOnRew bool, space float32) (pfc, pfcCT, pfcPT, pfcPTp, pfcThal *Layer) {
	p1to1 := prjn.NewPoolOneToOne()
	one2one := prjn.NewOneToOne()
	prjnClass := "PFCPrjn"

	pfc, pfcCT = net.AddSuperCT4D(name, prjnClass, nPoolsY, nPoolsX, nNeurY, nNeurX, space, one2one)
	pfcCT.SetClass(name)
	// prjns are: super->PT, PT self
	pfcPT, pfcThal = net.AddPTMaintThalForSuper(pfc, pfcCT, thalSuffix, prjnClass, one2one, p1to1, space)
	pfcPTp = net.AddPTPredLayer(pfcPT, pfcCT, p1to1, p1to1, prjnClass, space)
	pfcPTp.SetClass(name)

	pfcThal.PlaceBehind(pfcPTp, space)

	net.ConnectLayers(pfcPT, pfcCT, p1to1, ForwardPrjn).SetClass(prjnClass)

	onRew := fmt.Sprintf("%v", decayOnRew)

	pfcParams := params.Params{
		"Layer.Acts.Decay.Act":       "0",
		"Layer.Acts.Decay.Glong":     "0",
		"Layer.Acts.Decay.OnRew":     onRew,
		"Layer.Inhib.ActAvg.Nominal": "0.025",
		"Layer.Inhib.Layer.On":       "true",
		"Layer.Inhib.Layer.Gi":       "2.2",
		"Layer.Inhib.Pool.On":        "true",
		"Layer.Inhib.Pool.Gi":        "0.8",
		"Layer.Acts.Dend.SSGi":       "0",
	}
	pfc.DefParams = pfcParams

	pfcCT.CTDefParamsMedium()
	pfcCT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.025"
	pfcCT.DefParams["Layer.Inhib.Layer.Gi"] = "4" // 4?  2.8 orig
	pfcCT.DefParams["Layer.Inhib.Pool.On"] = "true"
	pfcCT.DefParams["Layer.Inhib.Pool.Gi"] = "1.2"
	pfcCT.DefParams["Layer.Acts.Decay.OnRew"] = onRew

	pfcPT.DefParams = maps.Clone(pfcParams)
	pfcPT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.05" // more active
	pfcPT.DefParams["Layer.Inhib.Layer.Gi"] = "2.4"        // 2.4 orig
	pfcPT.DefParams["Layer.Inhib.Pool.Gi"] = "2.0"
	pfcPT.DefParams["Layer.Learn.NeuroMod.AChDisInhib"] = "1" // maybe better -- test further

	pfcPTp.DefParams = maps.Clone(pfcParams)
	pfcPTp.DefParams["Layer.Inhib.Layer.Gi"] = "1.2" // 0.8 orig
	pfcPTp.DefParams["Layer.Inhib.Pool.Gi"] = "0.8"

	pfcThal.DefParams = maps.Clone(pfcParams)
	pfcThal.DefParams["Layer.Inhib.Layer.Gi"] = "2.0" // 1.1 orig
	pfcThal.DefParams["Layer.Inhib.Pool.Gi"] = "0.6"

	return
}

// AddPFC2D adds a "full stack" of 2D PFC layers:
// * AddSuperCT2D (Super and CT)
// * AddPTMaintThal (PTMaint, BGThal)
// * AddPTPredLayer (PTPred)
// with given name prefix, which is also set as the Class for all layers,
// and suffix for the BGThal layer (e.g., "MD" or "VM" etc for different thalamic nuclei).
// OneToOne, full connectivity is used between layers.
// decayOnRew determines the Act.Decay.OnRew setting (true of OFC, ACC type for sure).
// CT layer uses the Medium timescale params.
func (net *Network) AddPFC2D(name, thalSuffix string, nNeurY, nNeurX int, decayOnRew bool, space float32) (pfc, pfcCT, pfcPT, pfcPTp, pfcThal *Layer) {
	one2one := prjn.NewOneToOne()
	full := prjn.NewFull()

	prjnClass := "PFCPrjn"
	pfc, pfcCT = net.AddSuperCT2D(name, prjnClass, nNeurY, nNeurX, space, one2one)
	pfcCT.SetClass(name)
	// prjns are: super->PT, PT self
	pfcPT, pfcThal = net.AddPTMaintThalForSuper(pfc, pfcCT, thalSuffix, prjnClass, one2one, full, space)
	pfcPTp = net.AddPTPredLayer(pfcPT, pfcCT, full, full, prjnClass, space)
	pfcPTp.SetClass(name)

	pfcThal.PlaceBehind(pfcPTp, space)

	net.ConnectLayers(pfcPT, pfcCT, full, ForwardPrjn).SetClass(prjnClass)

	onRew := fmt.Sprintf("%v", decayOnRew)

	pfcParams := params.Params{
		"Layer.Acts.Decay.Act":       "0",
		"Layer.Acts.Decay.Glong":     "0",
		"Layer.Acts.Decay.OnRew":     onRew,
		"Layer.Inhib.ActAvg.Nominal": "0.1",
		"Layer.Inhib.Layer.On":       "true",
		"Layer.Inhib.Layer.Gi":       "0.9",
		"Layer.Inhib.Pool.On":        "false",
		"Layer.Acts.Dend.SSGi":       "0",
	}
	pfc.DefParams = pfcParams

	pfcCT.CTDefParamsMedium()
	pfcCT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.1"
	pfcCT.DefParams["Layer.Inhib.Layer.On"] = "true"
	pfcCT.DefParams["Layer.Inhib.Layer.Gi"] = "1.4"
	pfcCT.DefParams["Layer.Inhib.Pool.On"] = "false"
	pfcCT.DefParams["Layer.Acts.Decay.OnRew"] = onRew

	pfcPT.DefParams = maps.Clone(pfcParams)
	pfcPT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.3" // more active
	pfcPT.DefParams["Layer.Inhib.Layer.Gi"] = "2.0"
	pfcPT.DefParams["Layer.Learn.NeuroMod.AChDisInhib"] = "1" // maybe better -- test further

	pfcPTp.DefParams = maps.Clone(pfcParams)
	pfcPTp.DefParams["Layer.Inhib.Layer.Gi"] = "0.8"

	pfcThal.DefParams = maps.Clone(pfcParams)
	pfcThal.DefParams["Layer.Inhib.Layer.Gi"] = "0.6"

	return
}

// ConnectToPFC connects given predictively learned input to all
// relevant PFC layers:
// lay -> pfc (skipped if lay == nil)
// layP -> pfc, layP <-> pfcCT
// pfcPTp <-> layP
// sets PFCPrjn class name for projections
func (net *Network) ConnectToPFC(lay, layP, pfc, pfcCT, pfcPTp *Layer, pat prjn.Pattern) {
	prjnClass := "PFCPrjn"
	if lay != nil {
		net.ConnectLayers(lay, pfc, pat, ForwardPrjn).SetClass(prjnClass)
		pj := net.ConnectLayers(lay, pfcPTp, pat, ForwardPrjn) // ptp needs more input
		pj.DefParams = params.Params{
			"Prjn.PrjnScale.Abs": "4",
		}
		pj.SetClass("ToPTp " + prjnClass)
	}
	net.ConnectToPulv(pfc, pfcCT, layP, pat, pat, prjnClass)
	net.ConnectPTPredToPulv(pfcPTp, layP, pat, pat, prjnClass)
}

// ConnectToPFCBack connects given predictively learned input to all
// relevant PFC layers:
// lay -> pfc using a BackPrjn -- weaker
// layP -> pfc, layP <-> pfcCT
// pfcPTp <-> layP
func (net *Network) ConnectToPFCBack(lay, layP, pfc, pfcCT, pfcPTp *Layer, pat prjn.Pattern) {
	prjnClass := "PFCPrjn"
	tp := net.ConnectLayers(lay, pfc, pat, BackPrjn)
	tp.SetClass(prjnClass)
	net.ConnectToPulv(pfc, pfcCT, layP, pat, pat, prjnClass)
	net.ConnectPTPredToPulv(pfcPTp, layP, pat, pat, prjnClass)
	pj := net.ConnectLayers(lay, pfcPTp, pat, ForwardPrjn) // ptp needs more input
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "4",
	}
	pj.SetClass("ToPTp " + prjnClass)
}

// ConnectToPFCBidir connects given predictively learned input to all
// relevant PFC layers, using bidirectional connections to super layers.
// lay <-> pfc bidirectional
// layP -> pfc, layP <-> pfcCT
// pfcPTp <-> layP
func (net *Network) ConnectToPFCBidir(lay, layP, pfc, pfcCT, pfcPTp *Layer, pat prjn.Pattern) (ff, fb *Prjn) {
	prjnClass := "PFCPrjn"
	ff, fb = net.BidirConnectLayers(lay, pfc, pat)
	ff.SetClass(prjnClass)
	fb.SetClass(prjnClass)
	net.ConnectToPulv(pfc, pfcCT, layP, pat, pat, prjnClass)
	net.ConnectPTPredToPulv(pfcPTp, layP, pat, pat, prjnClass)
	pj := net.ConnectLayers(lay, pfcPTp, pat, ForwardPrjn) // ptp needs more input
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "4",
	}
	pj.SetClass("ToPTp " + prjnClass)
	return
}
