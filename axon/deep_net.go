// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"strings"

	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
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
// with CTCtxtPath pathway from Super to CT using given pathway pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (net *Network) AddSuperCT2D(name, pathClass string, shapeY, shapeX int, space float32, pat paths.Pattern) (super, ct *Layer) {
	super = net.AddSuperLayer2D(name, shapeY, shapeX)
	ct = net.AddCTLayer2D(name+"CT", shapeY, shapeX)
	ct.PlaceBehind(super, space)
	net.ConnectSuperToCT(super, ct, pat, pathClass)
	super.AddClass(name)
	ct.AddClass(name)
	return
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath pathway from Super to CT using given pathway pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (net *Network) AddSuperCT4D(name, pathClass string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat paths.Pattern) (super, ct *Layer) {
	super = net.AddSuperLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = net.AddCTLayer4D(name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.PlaceBehind(super, space)
	net.ConnectSuperToCT(super, ct, pat, pathClass)
	super.AddClass(name)
	ct.AddClass(name)
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
		plv = net.AddPulvLayer2D(name+"P", shp.DimSize(0), shp.DimSize(1))
	} else {
		plv = net.AddPulvLayer4D(name+"P", shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
	}
	plv.SetBuildConfig("DriveLayName", name)
	plv.SetRelPos(relpos.NewBehind(name+"CT", space))
	plv.AddClass(name)
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
		plv = net.AddPulvLayer2D(name+"P", shp.DimSize(0), shp.DimSize(1))
	} else {
		plv = net.AddPulvLayer4D(name+"P", shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
	}
	plv.SetBuildConfig("DriveLayName", name)
	plv.PlaceBehind(lay, space)
	return plv
}

// ConnectToPulv adds the following pathways:
// layers      | class      | path type   | path pat
// ------------+------------+-------------+----------
// ct  ->pulv  | "CTToPulv" | ForwardPath | toPulvPat
// pulv->super | "FromPulv"   | BackPath    | fmPulvPat
// pulv->ct    | "FromPulv"   | BackPath    | fmPulvPat
//
// Typically pulv is a different shape than super and ct, so use Full or appropriate
// topological pattern. Adds optional pathClass name as a suffix.
func (net *Network) ConnectToPulv(super, ct, pulv *Layer, toPulvPat, fmPulvPat paths.Pattern, pathClass string) (toPulv, toSuper, toCT *Path) {
	pathClass = params.AddClass(pathClass, "PFCPath")
	toPulv = net.ConnectLayers(ct, pulv, toPulvPat, ForwardPath)
	toPulv.AddClass("CTToPulv", pathClass)
	toSuper = net.ConnectLayers(pulv, super, fmPulvPat, BackPath)
	toSuper.AddClass("FromPulv", pathClass)
	toCT = net.ConnectLayers(pulv, ct, fmPulvPat, BackPath)
	toCT.AddClass("FromPulv", pathClass)
	return
}

// ConnectCtxtToCT adds a CTCtxtPath from given sending layer to a CT layer
func (net *Network) ConnectCtxtToCT(send, recv *Layer, pat paths.Pattern) *Path {
	return net.ConnectLayers(send, recv, pat, CTCtxtPath)
}

// ConnectCTSelf adds a Self (Lateral) CTCtxtPath pathway within a CT layer,
// in addition to a regular lateral pathway, which supports active maintenance.
// The CTCtxtPath has a Class label of CTSelfCtxt, and the regular one is CTSelfMaint
// with optional class added.
func (net *Network) ConnectCTSelf(ly *Layer, pat paths.Pattern, pathClass string) (ctxt, maint *Path) {
	pathClass = params.AddClass(pathClass, "PFCPath")
	ctxt = net.ConnectLayers(ly, ly, pat, CTCtxtPath)
	ctxt.AddClass("CTSelfCtxt", pathClass)
	maint = net.LateralConnectLayer(ly, pat)
	maint.DefParams = params.Params{
		"Path.PathScale.Abs": "0.5", // normalized separately
		"Path.Com.GType":     "MaintG",
	}
	maint.AddClass("CTSelfMaint", pathClass)
	return
}

// ConnectSuperToCT adds a CTCtxtPath from given sending Super layer to a CT layer
// This automatically sets the FromSuper flag to engage proper defaults,
// Uses given pathway pattern -- e.g., Full, OneToOne, or PoolOneToOne
func (net *Network) ConnectSuperToCT(send, recv *Layer, pat paths.Pattern, pathClass string) *Path {
	pathClass = params.AddClass(pathClass, "PFCPath")
	pj := net.ConnectLayers(send, recv, pat, CTCtxtPath)
	pj.AddClass("CTFromSuper", pathClass)
	return pj
}

// AddInputPulv2D adds an Input and Layer of given size, with given name.
// The Input layer is set as the Driver of the Layer.
// Both layers have SetClass(name) called to allow shared params.
func (net *Network) AddInputPulv2D(name string, nNeurY, nNeurX int, space float32) (*Layer, *Layer) {
	in := net.AddLayer2D(name, nNeurY, nNeurX, InputLayer)
	pulv := net.AddPulvLayer2D(name+"P", nNeurY, nNeurX)
	pulv.SetBuildConfig("DriveLayName", name)
	in.AddClass(name)
	pulv.AddClass(name)
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
	in.AddClass(name)
	pulv.AddClass(name)
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

// ConnectPTMaintSelf adds a Self (Lateral) pathway within a PTMaintLayer,
// which supports active maintenance, with a class of PTSelfMaint
func (net *Network) ConnectPTMaintSelf(ly *Layer, pat paths.Pattern, pathClass string) *Path {
	pathClass = params.AddClass(pathClass, "PFCPath")
	pj := net.LateralConnectLayer(ly, pat)
	pj.DefParams = params.Params{
		"Path.Com.GType":        "MaintG",
		"Path.PathScale.Rel":    "1",      // use abs to manipulate
		"Path.PathScale.Abs":    "4",      // strong..
		"Path.Learn.LRate.Base": "0.0001", // slower > faster
		"Path.SWts.Init.Mean":   "0.5",
		"Path.SWts.Init.Var":    "0.5", // high variance so not just spreading out over time
	}
	pj.AddClass("PTSelfMaint", pathClass)
	return pj
}

// AddPTMaintThalForSuper adds a PTMaint pyramidal tract active maintenance layer
// and a BG gated Thalamus layer for given superficial layer (SuperLayer)
// and associated CT, with given thal suffix (e.g., MD, VM).
// PT and Thal have SetClass(super.Name()) called to allow shared params.
// Projections are made with given classes: SuperToPT, PTSelfMaint, PTtoThal, ThalToPT,
// with optional extra class.
// if selfMaint is true, the SMaint self-maintenance mechanism is used
// instead of lateral connections.
// The PT and BGThal layers are positioned behind the CT layer.
func (net *Network) AddPTMaintThalForSuper(super, ct *Layer, thalSuffix, pathClass string, superToPT, ptSelf, ptThal paths.Pattern, selfMaint bool, space float32) (pt, thal *Layer) {
	pathClass = params.AddClass(pathClass, "PFCPath")
	name := super.Name()
	shp := super.Shape()
	is4D := false
	ptExtra := 1 // extra size for pt layers
	if shp.NumDims() == 2 {
		pt = net.AddPTMaintLayer2D(name+"PT", shp.DimSize(0)*ptExtra, shp.DimSize(1)*ptExtra)
		thal = net.AddBGThalLayer2D(name+thalSuffix, shp.DimSize(0), shp.DimSize(1))
	} else {
		is4D = true
		pt = net.AddPTMaintLayer4D(name+"PT", shp.DimSize(0), shp.DimSize(1), shp.DimSize(2)*ptExtra, shp.DimSize(3)*ptExtra)
		thal = net.AddBGThalLayer4D(name+thalSuffix, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
	}
	pt.AddClass(name)
	thal.AddClass(name)
	if selfMaint {
		pt.DefParams = params.Params{
			"Layer.Acts.SMaint.On":  "true",
			"Layer.Acts.GabaB.Gbar": "0.015",
			"Layer.Inhib.Layer.Gi":  "0.5",
			"Layer.Inhib.Pool.Gi":   "0.5",
		}
		if is4D {
			pt.DefParams["Layer.Inhib.Pool.On"] = "true"
		}
	}

	pthal, thalpt := net.BidirConnectLayers(pt, thal, ptThal)
	pthal.AddClass("PTtoThal", pathClass)
	thalpt.DefParams = params.Params{
		"Path.PathScale.Rel":  "1.0",
		"Path.Com.GType":      "ModulatoryG", // modulatory -- control with extra ModGain factor
		"Path.Learn.Learn":    "false",
		"Path.SWts.Adapt.On":  "false",
		"Path.SWts.Init.SPct": "0",
		"Path.SWts.Init.Mean": "0.8",
		"Path.SWts.Init.Var":  "0.0",
	}
	thalpt.AddClass("ThalToPT", pathClass)
	// if is4D {
	// fmThalInhib := params.Params{
	// 	"Path.PathScale.Rel": "1.0",
	// 	"Path.PathScale.Abs": "1.0",
	// 	"Path.Learn.Learn":   "false",
	// 	"Path.SWts.Adapt.On":  "false",
	// 	"Path.SWts.Init.SPct": "0",
	// 	"Path.SWts.Init.Mean": "0.8",
	// 	"Path.SWts.Init.Var":  "0.0",
	// }
	// note: holding off on these for now -- thal modulation should handle..
	// ti := net.ConnectLayers(thal, pt, full, InhibPath)
	// ti.DefParams = fmThalInhib
	// ti.AddClass("ThalToPFCInhib")
	// ti = net.ConnectLayers(thal, ct, full, InhibPath)
	// ti.DefParams = fmThalInhib
	// ti.AddClass("ThalToPFCInhib")

	sthal := net.ConnectLayers(super, thal, superToPT, ForwardPath) // shortcuts
	sthal.DefParams = params.Params{
		"Path.PathScale.Rel":  "1.0",
		"Path.PathScale.Abs":  "4.0", // key param for driving gating -- if too strong, premature gating
		"Path.Learn.Learn":    "false",
		"Path.SWts.Adapt.On":  "false",
		"Path.SWts.Init.SPct": "0",
		"Path.SWts.Init.Mean": "0.8", // typically 1to1
		"Path.SWts.Init.Var":  "0.0",
	}
	sthal.AddClass("SuperToThal", pathClass)

	pj := net.ConnectLayers(super, pt, superToPT, ForwardPath)
	pj.DefParams = params.Params{
		// one-to-one from super -- just use fixed nonlearning path so can control behavior easily
		"Path.PathScale.Rel":  "1",   // irrelevant -- only normal path
		"Path.PathScale.Abs":  "0.5", // BGThal modulates this so strength doesn't cause wrong CS gating
		"Path.Learn.Learn":    "false",
		"Path.SWts.Adapt.On":  "false",
		"Path.SWts.Init.SPct": "0",
		"Path.SWts.Init.Mean": "0.8",
		"Path.SWts.Init.Var":  "0.0",
	}
	pj.AddClass("SuperToPT", pathClass)

	if !selfMaint {
		net.ConnectPTMaintSelf(pt, ptSelf, pathClass)
	}

	if ct != nil {
		pt.PlaceBehind(ct, space)
	} else {
		pt.PlaceBehind(super, space)
	}
	pt.Rel.Scale = float32(1) / float32(ptExtra)
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

// ConnectPTPredSelf adds a Self (Lateral) pathway within a PTPredLayer,
// which supports active maintenance, with a class of PTSelfMaint
func (net *Network) ConnectPTPredSelf(ly *Layer, pat paths.Pattern) *Path {
	return net.LateralConnectLayer(ly, pat).AddClass("PTSelfMaint").(AxonPath).AsAxon()
}

// ConnectPTToPulv connects PT, PTPred with given Pulv:
// PT -> Pulv is class PTToPulv; PT does NOT receive back from Pulv
// PTPred -> Pulv is class PTPredToPulv,
// From Pulv = type = Back, class = FromPulv
// toPulvPat is the paths.Pattern PT -> Pulv and fmPulvPat is Pulv -> PTPred
// Typically Pulv is a different shape than PTPred, so use Full or appropriate
// topological pattern. adds optional class name to pathway.
func (net *Network) ConnectPTToPulv(pt, ptPred, pulv *Layer, toPulvPat, fmPulvPat paths.Pattern, pathClass string) (ptToPulv, ptPredToPulv, toPTPred *Path) {
	pathClass = params.AddClass(pathClass, "PFCPath")
	ptToPulv = net.ConnectLayers(pt, pulv, toPulvPat, ForwardPath)
	ptToPulv.AddClass("PTToPulv", pathClass)
	ptPredToPulv = net.ConnectLayers(ptPred, pulv, toPulvPat, ForwardPath)
	ptPredToPulv.AddClass("PTPredToPulv", pathClass)
	toPTPred = net.ConnectLayers(pulv, ptPred, fmPulvPat, BackPath)
	toPTPred.AddClass("FromPulv", pathClass)
	return
}

// ConnectPTpToPulv connects PTPred with given Pulv:
// PTPred -> Pulv is class PTPredToPulv,
// From Pulv = type = Back, class = FromPulv
// toPulvPat is the paths.Pattern PT -> Pulv and fmPulvPat is Pulv -> PTPred
// Typically Pulv is a different shape than PTPred, so use Full or appropriate
// topological pattern. adds optional class name to pathway.
func (net *Network) ConnectPTpToPulv(ptPred, pulv *Layer, toPulvPat, fmPulvPat paths.Pattern, pathClass string) (ptToPulv, ptPredToPulv, toPTPred *Path) {
	pathClass = params.AddClass(pathClass, "PFCPath")
	ptPredToPulv = net.ConnectLayers(ptPred, pulv, toPulvPat, ForwardPath)
	ptPredToPulv.AddClass("PTPredToPulv", pathClass)
	toPTPred = net.ConnectLayers(pulv, ptPred, fmPulvPat, BackPath)
	toPTPred.AddClass("FromPulv", pathClass)
	return
}

// AddPTPredLayer adds a PTPred pyramidal tract prediction layer
// for given PTMaint layer and associated CT.
// Sets SetClass(super.Name()) to allow shared params.
// Projections are made with given classes: PTtoPred, CTtoPred
// The PTPred layer is positioned behind the PT layer.
func (net *Network) AddPTPredLayer(ptMaint, ct *Layer, ptToPredPath, ctToPredPath paths.Pattern, pathClass string, space float32) (ptPred *Layer) {
	pathClass = params.AddClass(pathClass, "PFCPath")
	name := strings.TrimSuffix(ptMaint.Name(), "PT")
	// shp := ptMaint.Shape()
	shp := ct.Shape()
	if shp.NumDims() == 2 {
		ptPred = net.AddPTPredLayer2D(name+"PTp", shp.DimSize(0), shp.DimSize(1))
	} else {
		ptPred = net.AddPTPredLayer4D(name+"PTp", shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
	}
	ptPred.AddClass(name)
	ptPred.PlaceBehind(ptMaint, space)
	pj := net.ConnectCtxtToCT(ptMaint, ptPred, ptToPredPath)
	pj.AddClass("PTtoPred", pathClass)

	pj = net.ConnectLayers(ct, ptPred, ctToPredPath, ForwardPath)
	pj.DefParams = params.Params{
		"Path.PathScale.Rel": "1",   // 1 > 0.5
		"Path.PathScale.Abs": "2.0", // 2?
	}
	pj.AddClass("CTtoPred", pathClass)

	// note: ptpred does not connect to thalamus -- it is only active on trial *after* thal gating
	return
}

// AddPFC4D adds a "full stack" of 4D PFC layers:
// * AddSuperCT4D (Super and CT)
// * AddPTMaintThal (PTMaint, BGThal)
// * AddPTPredLayer (PTPred)
// with given name prefix, which is also set as the Class for all layers & paths (+"Path"),
// and suffix for the BGThal layer (e.g., "MD" or "VM" etc for different thalamic nuclei).
// Sets PFCLayer as additional class for all cortical layers.
// OneToOne and PoolOneToOne connectivity is used between layers.
// decayOnRew determines the Act.Decay.OnRew setting (true of OFC, ACC type for sure).
// if selfMaint is true, the SMaint self-maintenance mechanism is used
// instead of lateral connections.
// CT layer uses the Medium timescale params.
// use, e.g., pfcCT.DefParams["Layer.Inhib.Layer.Gi"] = "2.8" to change default params.
func (net *Network) AddPFC4D(name, thalSuffix string, nPoolsY, nPoolsX, nNeurY, nNeurX int, decayOnRew, selfMaint bool, space float32) (pfc, pfcCT, pfcPT, pfcPTp, pfcThal *Layer) {
	p1to1 := paths.NewPoolOneToOne()
	// p1to1rnd := paths.NewPoolUnifRnd()
	// p1to1rnd.PCon = 0.5
	one2one := paths.NewOneToOne()
	pathClass := name + "Path"
	layClass := "PFCLayer"

	pfc, pfcCT = net.AddSuperCT4D(name, pathClass, nPoolsY, nPoolsX, nNeurY, nNeurX, space, one2one)
	pfcCT.AddClass(name)
	pfc.AddClass(layClass)
	pfcCT.AddClass(layClass)
	// paths are: super->PT, PT self
	pfcPT, pfcThal = net.AddPTMaintThalForSuper(pfc, pfcCT, thalSuffix, pathClass, one2one, p1to1, one2one, selfMaint, space)
	pfcPTp = net.AddPTPredLayer(pfcPT, pfcCT, p1to1, p1to1, pathClass, space)
	pfcPTp.AddClass(name)
	pfcPT.AddClass(layClass)
	pfcPTp.AddClass(layClass)

	pfcThal.PlaceBehind(pfcPTp, space)

	net.ConnectLayers(pfcPT, pfcCT, p1to1, ForwardPath).AddClass(pathClass)

	onRew := fmt.Sprintf("%v", decayOnRew)

	pfcParams := params.Params{
		"Layer.Acts.Decay.Act":               "0",
		"Layer.Acts.Decay.Glong":             "0",
		"Layer.Acts.Decay.OnRew":             onRew,
		"Layer.Inhib.ActAvg.Nominal":         "0.025",
		"Layer.Inhib.Layer.On":               "true",
		"Layer.Inhib.Layer.Gi":               "2.2",
		"Layer.Inhib.Pool.On":                "true",
		"Layer.Inhib.Pool.Gi":                "0.8",
		"Layer.Acts.Dend.SSGi":               "0",
		"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002",
	}
	pfc.DefParams = maps.Clone(pfcParams)

	pfcCT.CTDefParamsMedium()
	pfcCT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.025"
	pfcCT.DefParams["Layer.Inhib.Layer.Gi"] = "4" // 4?  2.8 orig
	pfcCT.DefParams["Layer.Inhib.Pool.On"] = "true"
	pfcCT.DefParams["Layer.Inhib.Pool.Gi"] = "1.2"
	pfcCT.DefParams["Layer.Acts.Decay.OnRew"] = onRew
	pfcCT.DefParams["Layer.Learn.TrgAvgAct.SynScaleRate"] = "0.0002"

	// pfcPT.DefParams = maps.Clone(pfcParams)
	// pfcPT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.05" // more active
	// pfcPT.DefParams["Layer.Inhib.Layer.Gi"] = "2.4"        // 2.4 orig
	// pfcPT.DefParams["Layer.Inhib.Pool.Gi"] = "2.4"
	// pfcPT.DefParams["Layer.Learn.NeuroMod.AChDisInhib"] = "0" // maybe better -- test further

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
// with given name prefix, which is also set as the Class for all layers & paths (+"Path"),
// and suffix for the BGThal layer (e.g., "MD" or "VM" etc for different thalamic nuclei).
// Sets PFCLayer as additional class for all cortical layers.
// OneToOne, full connectivity is used between layers.
// decayOnRew determines the Act.Decay.OnRew setting (true of OFC, ACC type for sure).
// if selfMaint is true, the SMaint self-maintenance mechanism is used
// instead of lateral connections.
// CT layer uses the Medium timescale params.
func (net *Network) AddPFC2D(name, thalSuffix string, nNeurY, nNeurX int, decayOnRew, selfMaint bool, space float32) (pfc, pfcCT, pfcPT, pfcPTp, pfcThal *Layer) {
	one2one := paths.NewOneToOne()
	full := paths.NewFull()
	// rnd := paths.NewUnifRnd()
	// rnd.PCon = 0.5
	pathClass := name + "Path"
	layClass := "PFCLayer"

	pfc, pfcCT = net.AddSuperCT2D(name, pathClass, nNeurY, nNeurX, space, one2one)
	pfcCT.AddClass(name)
	pfc.AddClass(layClass)
	pfcCT.AddClass(layClass)
	// paths are: super->PT, PT self
	pfcPT, pfcThal = net.AddPTMaintThalForSuper(pfc, pfcCT, thalSuffix, pathClass, one2one, full, one2one, selfMaint, space)
	pfcPTp = net.AddPTPredLayer(pfcPT, pfcCT, full, full, pathClass, space)
	pfcPTp.AddClass(name)
	pfcPT.AddClass(layClass)
	pfcPTp.AddClass(layClass)

	pfcThal.PlaceBehind(pfcPTp, space)

	net.ConnectLayers(pfcPT, pfcCT, full, ForwardPath).AddClass(pathClass)

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

	// pfcPT.DefParams = maps.Clone(pfcParams)
	// pfcPT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.3" // more active
	// pfcPT.DefParams["Layer.Inhib.Layer.Gi"] = "2.4"
	// pfcPT.DefParams["Layer.Inhib.Pool.Gi"] = "2.4"
	// pfcPT.DefParams["Layer.Learn.NeuroMod.AChDisInhib"] = "0" // maybe better -- test further

	pfcPTp.DefParams = maps.Clone(pfcParams)
	pfcPTp.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.1"
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
// if pfcPT != nil: pfcPT <-> layP
// sets PFCPath class name for pathways
func (net *Network) ConnectToPFC(lay, layP, pfc, pfcCT, pfcPT, pfcPTp *Layer, pat paths.Pattern, pathClass string) {
	if pathClass == "" {
		pathClass = "PFCPath"
	}
	if lay != nil {
		net.ConnectLayers(lay, pfc, pat, ForwardPath).AddClass(pathClass)
		pj := net.ConnectLayers(lay, pfcPTp, pat, ForwardPath) // ptp needs more input
		pj.DefParams = params.Params{
			"Path.PathScale.Abs": "4",
		}
		pj.AddClass("ToPTp ", pathClass)
	}
	net.ConnectToPulv(pfc, pfcCT, layP, pat, pat, pathClass)
	if pfcPT == nil {
		net.ConnectPTpToPulv(pfcPTp, layP, pat, pat, pathClass)
	} else {
		net.ConnectPTToPulv(pfcPT, pfcPTp, layP, pat, pat, pathClass)
	}
}

// ConnectToPFCBack connects given predictively learned input to all
// relevant PFC layers:
// lay -> pfc using a BackPath -- weaker
// layP -> pfc, layP <-> pfcCT
// pfcPTp <-> layP
func (net *Network) ConnectToPFCBack(lay, layP, pfc, pfcCT, pfcPT, pfcPTp *Layer, pat paths.Pattern, pathClass string) {
	if pathClass == "" {
		pathClass = "PFCPath"
	}
	tp := net.ConnectLayers(lay, pfc, pat, BackPath)
	tp.AddClass(pathClass)
	net.ConnectToPulv(pfc, pfcCT, layP, pat, pat, pathClass)
	net.ConnectPTToPulv(pfcPT, pfcPTp, layP, pat, pat, pathClass)
	pj := net.ConnectLayers(lay, pfcPTp, pat, ForwardPath) // ptp needs more input
	pj.DefParams = params.Params{
		"Path.PathScale.Abs": "4",
	}
	pj.AddClass("ToPTp ", pathClass)
}

// ConnectToPFCBidir connects given predictively learned input to all
// relevant PFC layers, using bidirectional connections to super layers.
// lay <-> pfc bidirectional
// layP -> pfc, layP <-> pfcCT
// pfcPTp <-> layP
func (net *Network) ConnectToPFCBidir(lay, layP, pfc, pfcCT, pfcPT, pfcPTp *Layer, pat paths.Pattern, pathClass string) (ff, fb *Path) {
	if pathClass == "" {
		pathClass = "PFCPath"
	}
	ff, fb = net.BidirConnectLayers(lay, pfc, pat)
	ff.AddClass(pathClass)
	fb.AddClass(pathClass)
	net.ConnectToPulv(pfc, pfcCT, layP, pat, pat, pathClass)
	net.ConnectPTToPulv(pfcPT, pfcPTp, layP, pat, pat, pathClass)
	pj := net.ConnectLayers(lay, pfcPTp, pat, ForwardPath) // ptp needs more input
	pj.DefParams = params.Params{
		"Path.PathScale.Abs": "4",
	}
	pj.AddClass("ToPTp ", pathClass)
	return
}
