// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/goki/ki/kit"
)

// deep.Network runs a Deep version of Axon network, for predictive learning
// via deep cortical layers interconnected with the thalamus (Pulvinar).
// It has special computational methods only for PlusPhase where deep layer
// context updating happens, corresponding to the bursting of deep layer 5IB
// neurons.  It also has methods for creating different specialized layer types.
type Network struct {
	axon.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = axon.NetworkProps

// NewNetwork returns a new deep Network
func NewNetwork(name string) *Network {
	net := &Network{}
	net.InitName(net, name)
	return net
}

// Defaults sets all the default parameters for all layers and projections
func (nt *Network) Defaults() {
	nt.Network.Defaults()
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and projections
func (nt *Network) UpdateParams() {
	nt.Network.UpdateParams()
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (nt *Network) UnitVarNames() []string {
	return NeuronVarsAll
}

//////////////////////////////////////////////////////////////////////////////////////
//  Basic Add Layer methods (independent of Network)

// AddSuperLayer2D adds a SuperLayer of given size, with given name.
func AddSuperLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *SuperLayer {
	ly := &SuperLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.Hidden)
	return ly
}

// AddSuperLayer4D adds a SuperLayer of given size, with given name.
func AddSuperLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *SuperLayer {
	ly := &SuperLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	return ly
}

// AddCTLayer2D adds a CTLayer of given size, with given name.
func AddCTLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *CTLayer {
	ly := &CTLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, CT)
	return ly
}

// AddCTLayer4D adds a CTLayer of given size, with given name.
func AddCTLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *CTLayer {
	ly := &CTLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, CT)
	return ly
}

// AddPTLayer2D adds a PTLayer of given size, with given name.
func AddPTLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *PTLayer {
	ly := &PTLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, PT)
	return ly
}

// AddPTLayer4D adds a PTLayer of given size, with given name.
func AddPTLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *PTLayer {
	ly := &PTLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, PT)
	return ly
}

// AddPulvLayer2D adds a PulvLayer of given size, with given name.
func AddPulvLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *PulvLayer {
	ly := &PulvLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, Pulv)
	return ly
}

// AddPulvLayer4D adds a PulvLayer of given size, with given name.
func AddPulvLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *PulvLayer {
	ly := &PulvLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, Pulv)
	return ly
}

// AddPulvAttnLayer2D adds a PulvAttnLayer of given size, with given name.
func AddPulvAttnLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *PulvAttnLayer {
	ly := &PulvAttnLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, Pulv)
	return ly
}

// AddPulvAttnLayer4D adds a PulvAttnLayer of given size, with given name.
func AddPulvAttnLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *PulvAttnLayer {
	ly := &PulvAttnLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, Pulv)
	return ly
}

// AddInputPulv2D adds an Input and PulvLayer of given size, with given name.
// The Input layer is set as the Driver of the PulvLayer
func AddInputPulv2D(nt *axon.Network, name string, nNeurY, nNeurX int, space float32) (emer.Layer, *PulvLayer) {
	in := nt.AddLayer2D(name, nNeurY, nNeurX, emer.Input)
	trc := AddPulvLayer2D(nt, name+"P", nNeurY, nNeurX)
	trc.Driver = name
	trc.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	return in, trc
}

// AddInputPulv4D adds an Input and PulvLayer of given size, with given name.
// The Input layer is set as the Driver of the PulvLayer
func AddInputPulv4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32) (emer.Layer, *PulvLayer) {
	in := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, emer.Input)
	trc := AddPulvLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	trc.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	trc.Driver = name
	return in, trc
}

// ConnectToPulv connects Super and CT with given Pulv: CT -> Pulv is class CTToPulv,
// From Pulv = type = Back, class = FmPulv.
// toTrcPat is the prjn.Pattern CT -> Pulv and fmTrcPat is Pulv -> CT, Super.
// Typically Pulv is a different shape than Super and CT, so use Full or appropriate
// topological pattern
func ConnectToPulv(nt *axon.Network, super, ct, trc emer.Layer, toTrcPat, fmTrcPat prjn.Pattern) {
	nt.ConnectLayers(ct, trc, toTrcPat, emer.Forward).SetClass("CTToPulv")
	nt.ConnectLayers(trc, super, fmTrcPat, emer.Back).SetClass("FmPulv")
	nt.ConnectLayers(trc, ct, fmTrcPat, emer.Back).SetClass("FmPulv")
}

// ConnectSuperToCT adds a CTCtxtPrjn from given sending Super layer to a CT layer
// This automatically sets the FmSuper flag to engage proper defaults,
// Uses given projection pattern -- e.g., Full, OneToOne, or PoolOneToOne
func ConnectSuperToCT(nt *axon.Network, send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	pj := nt.ConnectLayersPrjn(send, recv, pat, CTCtxt, &CTCtxtPrjn{}).(*CTCtxtPrjn)
	pj.SetClass("CTFmSuper")
	pj.FmSuper = true
	return pj
}

// ConnectCtxtToCT adds a CTCtxtPrjn from given sending layer to a CT layer
// Use ConnectSuperToCT for main projection from corresponding superficial layer.
func ConnectCtxtToCT(nt *axon.Network, send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayersPrjn(send, recv, pat, CTCtxt, &CTCtxtPrjn{})
}

// ConnectCTSelf adds a Self (Lateral) CTCtxtPrjn projection within a CT layer,
// in addition to a regular lateral projection, which supports active maintenance.
// The CTCtxtPrjn has a Class label of CTSelfCtxt, and the regular one is CTSelfMaint
func ConnectCTSelf(nt *axon.Network, ly emer.Layer, pat prjn.Pattern) (ctxt, maint emer.Prjn) {
	ctxt = nt.ConnectLayersPrjn(ly, ly, pat, CTCtxt, &CTCtxtPrjn{}).SetClass("CTSelfCtxt")
	maint = nt.LateralConnectLayer(ly, pat).SetClass("CTSelfMaint")
	return
}

// ConnectPTSelf adds a Self (Lateral) projection within a PT layer,
// which supports active maintenance, with a class of PTSelfMaint
func ConnectPTSelf(nt *axon.Network, ly emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.LateralConnectLayer(ly, pat).SetClass("PTSelfMaint")
}

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// super and ct have SetClass(name) called to allow shared params.
// CT is placed Behind Super.
func AddSuperCT2D(nt *axon.Network, name string, shapeY, shapeX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	super = AddSuperLayer2D(nt, name, shapeY, shapeX)
	ct = AddCTLayer2D(nt, name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	ConnectSuperToCT(nt, super, ct, pat)
	super.SetClass(name)
	ct.SetClass(name)
	return
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// super and ct have SetClass(name) called to allow shared params.
// CT is placed Behind Super.
func AddSuperCT4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	super = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	ConnectSuperToCT(nt, super, ct, pat)
	super.SetClass(name)
	ct.SetClass(name)
	return
}

// AddPulvForSuper adds a Pulvinar for given superficial layer (SuperLayer)
// with a P suffix.  The Pulv.Driver is set to Super.
// The Pulv layer needs other CT connections from higher up to predict this layer.
// Pulvinar is positioned behind the CT layer.
func AddPulvForSuper(nt *axon.Network, super emer.Layer, space float32) emer.Layer {
	name := super.Name()
	shp := super.Shape()
	var plv *PulvLayer
	if shp.NumDims() == 2 {
		plv = AddPulvLayer2D(nt, name+"P", shp.Dim(0), shp.Dim(1))
	} else {
		plv = AddPulvLayer4D(nt, name+"P", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	plv.Driver = name
	plv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: space})
	return plv
}

// AddPTThalForSuper adds a PT pyramidal tract layer and a
// Thalamus layer for given superficial layer (SuperLayer)
// with given suffix (e.g., MD, VM).
// The PT and Thal layers are positioned behind the CT layer.
func AddPTThalForSuper(nt *axon.Network, super emer.Layer, suffix string, space float32) (thal, pt emer.Layer) {
	name := super.Name()
	shp := super.Shape()
	if shp.NumDims() == 2 {
		pt = AddPTLayer2D(nt, name+"PT", shp.Dim(0), shp.Dim(1))
		thal = AddPulvLayer2D(nt, name+suffix, shp.Dim(0), shp.Dim(1))
	} else {
		pt = AddPTLayer4D(nt, name+"PT", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
		thal = AddPulvLayer4D(nt, name+suffix, shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	pt.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: space})
	thal.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: pt.Name(), XAlign: relpos.Left, Space: space})
	one2one := prjn.NewOneToOne()
	nt.ConnectLayers(super, thal, one2one, emer.Forward)
	nt.ConnectLayers(pt, thal, one2one, emer.Forward)
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network versions of Add Layer methods

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT2D(name string, shapeY, shapeX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	return AddSuperCT2D(&nt.Network, name, shapeY, shapeX, space, pat)
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO Pulv Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	return AddSuperCT4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX, space, pat)
}

// AddPulvForSuper adds a Pulvinar for given superficial layer (SuperLayer)
// with a P suffix.  The Pulv.Driver is set to Super.
// The Pulv layer needs other CT connections from higher up to predict this layer.
// Pulvinar is positioned behind the CT layer.
func (nt *Network) AddPulvForSuper(super emer.Layer, space float32) emer.Layer {
	return AddPulvForSuper(&nt.Network, super, space)
}

// ConnectToPulv connects Super and CT with given Pulv: CT -> Pulv is class CTToPulv,
// From Pulv = type = Back, class = FmPulv
// toTrcPat is the prjn.Pattern CT -> Pulv and fmTrcPat is Pulv -> CT, Super
// Typically Pulv is a different shape than Super and CT, so use Full or appropriate
// topological pattern
func (nt *Network) ConnectToPulv(super, ct, trc emer.Layer, toTrcPat, fmTrcPat prjn.Pattern) {
	ConnectToPulv(&nt.Network, super, ct, trc, toTrcPat, fmTrcPat)
}

// ConnectCtxtToCT adds a CTCtxtPrjn from given sending layer to a CT layer
func (nt *Network) ConnectCtxtToCT(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return ConnectCtxtToCT(&nt.Network, send, recv, pat)
}

// ConnectCTSelf adds a Self (Lateral) CTCtxtPrjn projection within a CT layer,
// in addition to a regular lateral projection, which supports active maintenance.
// The CTCtxtPrjn has a Class label of CTSelfCtxt, and the regular one is CTSelfMaint
func (nt *Network) ConnectCTSelf(ly emer.Layer, pat prjn.Pattern) (ctxt, maint emer.Prjn) {
	return ConnectCTSelf(&nt.Network, ly, pat)
}

// AddPTThalForSuper adds a PT pyramidal tract layer and a
// Thalamus layer for given superficial layer (SuperLayer)
// with given suffix (e.g., MD, VM).
// The PT and Thal layers are positioned behind the CT layer.
func (nt *Network) AddPTThalForSuper(super emer.Layer, suffix string, space float32) (thal, pt emer.Layer) {
	return AddPTThalForSuper(&nt.Network, super, suffix, space)
}

// AddPulvAttnLayer2D adds a PulvAttnLayer of given size, with given name.
func (nt *Network) AddPulvAttnLayer2D(name string, nNeurY, nNeurX int) *PulvAttnLayer {
	return AddPulvAttnLayer2D(&nt.Network, name, nNeurY, nNeurX)
}

// AddPulvAttnLayer4D adds a PulvLayer of given size, with given name.
func (nt *Network) AddPulvAttnLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *PulvAttnLayer {
	return AddPulvAttnLayer4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddInputPulv2D adds an Input and PulvLayer of given size, with given name.
// The Input layer is set as the Driver of the PulvLayer
func (nt *Network) AddInputPulv2D(name string, nNeurY, nNeurX int, space float32) (emer.Layer, *PulvLayer) {
	return AddInputPulv2D(&nt.Network, name, nNeurY, nNeurX, space)
}

// AddInputPulv4D adds an Input and PulvLayer of given size, with given name.
// The Input layer is set as the Driver of the PulvLayer
func (nt *Network) AddInputPulv4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32) (emer.Layer, *PulvLayer) {
	return AddInputPulv4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX, space)
}

// AddSuperLayer2D adds a SuperLayer of given size, with given name.
func (nt *Network) AddSuperLayer2D(name string, nNeurY, nNeurX int) *SuperLayer {
	return AddSuperLayer2D(&nt.Network, name, nNeurY, nNeurX)
}

// AddSuperLayer4D adds a PulvLayer of given size, with given name.
func (nt *Network) AddSuperLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *SuperLayer {
	return AddSuperLayer4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Compute methods

// PlusPhase does updating after end of plus phase
func (nt *Network) PlusPhaseImpl(ltime *axon.Time) {
	nt.Network.PlusPhaseImpl(ltime) // call base
	nt.CTCtxt(ltime)
}

// CTCtxt sends context to CT layers and integrates CtxtGe on CT layers
func (nt *Network) CTCtxt(ltime *axon.Time) {
	nt.ThrLayFun(func(ly axon.AxonLayer) {
		if dl, ok := ly.(CtxtSender); ok {
			dl.SendCtxtGe(ltime)
		} else {
			LayerSendCtxtGe(ly.AsAxon(), ltime)
		}
	}, "SendCtxtGe")

	nt.ThrLayFun(func(ly axon.AxonLayer) {
		if dl, ok := ly.(*CTLayer); ok {
			dl.CtxtFmGe(ltime)
		}
	}, "CtxtFmGe")
}

// LayerSendCtxtGe sends activation (CaSpkP) over CTCtxtPrjn projections to integrate
// CtxtGe excitatory conductance on CT layers.
// This should be called at the end of the 5IB Bursting phase via Network.CTCtxt
// Satisfies the CtxtSender interface.
func LayerSendCtxtGe(ly *axon.Layer, ltime *axon.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.CaSpkP < 0.1 {
			continue
		}
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			ptyp := sp.Type()
			if ptyp != CTCtxt {
				continue
			}
			pj, ok := sp.(*CTCtxtPrjn)
			if !ok {
				continue
			}
			pj.SendCtxtGe(ni, nrn.CaSpkP)
		}
	}
}
