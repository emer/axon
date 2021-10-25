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

// deep.Network has parameters for running a DeepAxon network
type Network struct {
	axon.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = axon.NetworkProps

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

// AddTRCLayer2D adds a TRCLayer of given size, with given name.
func AddTRCLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *TRCLayer {
	ly := &TRCLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, TRC)
	return ly
}

// AddTRCLayer4D adds a TRCLayer of given size, with given name.
func AddTRCLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *TRCLayer {
	ly := &TRCLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, TRC)
	return ly
}

// AddTRCALayer2D adds a TRCALayer of given size, with given name.
func AddTRCALayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *TRCALayer {
	ly := &TRCALayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, TRC)
	return ly
}

// AddTRCALayer4D adds a TRCALayer of given size, with given name.
func AddTRCALayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *TRCALayer {
	ly := &TRCALayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, TRC)
	return ly
}

// ConnectToTRC2D connects Super and CT with given TRC: CT -> TRC is class CTToPulv,
// From TRC = type = Back, class = FmPulv
// 2D version uses Full projections.
func ConnectToTRC2D(nt *axon.Network, super, ct, trc emer.Layer) {
	full := prjn.NewFull()
	nt.ConnectLayers(ct, trc, full, emer.Forward).SetClass("CTToPulv")
	nt.ConnectLayers(trc, super, full, emer.Back).SetClass("FmPulv")
	nt.ConnectLayers(trc, ct, full, emer.Back).SetClass("FmPulv")
}

// ConnectToTRC4D connects Super and CT with given TRC: CT -> TRC is class CTToPulv,
// From TRC = type = Back, class = FmPulv
// 4D version uses PoolOneToOne projections.
func ConnectToTRC4D(nt *axon.Network, super, ct, trc emer.Layer) {
	pone2one := prjn.NewPoolOneToOne()
	nt.ConnectLayers(ct, trc, pone2one, emer.Forward).SetClass("CTToPulv")
	nt.ConnectLayers(trc, super, pone2one, emer.Back).SetClass("FmPulv")
	nt.ConnectLayers(trc, ct, pone2one, emer.Back).SetClass("FmPulv")
}

// ConnectSuperToCT adds a CTCtxtPrjn from given sending Super layer to a CT layer
// This automatically sets the FmSuper flag to engage proper defaults,
// uses a OneToOne prjn pattern, and sets the class to CTFmSuper
func ConnectSuperToCT(nt *axon.Network, send, recv emer.Layer) emer.Prjn {
	pj := nt.ConnectLayersPrjn(send, recv, prjn.NewOneToOne(), CTCtxt, &CTCtxtPrjn{}).(*CTCtxtPrjn)
	pj.SetClass("CTFmSuper")
	pj.FmSuper = true
	return pj
}

// ConnectCtxtToCT adds a CTCtxtPrjn from given sending layer to a CT layer
// Use ConnectSuperToCT for main projection from corresponding superficial layer.
func ConnectCtxtToCT(nt *axon.Network, send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayersPrjn(send, recv, pat, CTCtxt, &CTCtxtPrjn{})
}

// AddSuperCTTRC2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn OneToOne projection from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC.Driver is set to Super -- needs other CT connections from higher up.
// CT is placed Behind Super, and Pulvinar behind CT.
func AddSuperCTTRC2D(nt *axon.Network, name string, shapeY, shapeX int) (super, ct, trc emer.Layer) {
	super = AddSuperLayer2D(nt, name, shapeY, shapeX)
	ct = AddCTLayer2D(nt, name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	ConnectSuperToCT(nt, super, ct)
	trci := AddTRCLayer2D(nt, name+"P", shapeY, shapeX)
	trc = trci
	trci.Driver = name
	trci.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: 2})
	return
}

// AddSuperCTTRC4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn OneToOne projection from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC.Driver is set to Super -- needs other CT connections from higher up.
// CT is placed Behind Super, and Pulvinar behind CT.
func AddSuperCTTRC4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct, trc emer.Layer) {
	super = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	ConnectSuperToCT(nt, super, ct)
	trci := AddTRCLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	trc = trci
	trci.Driver = name
	trci.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: 2})
	return
}

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn OneToOne projection from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
func AddSuperCT2D(nt *axon.Network, name string, shapeY, shapeX int) (super, ct emer.Layer) {
	super = AddSuperLayer2D(nt, name, shapeY, shapeX)
	ct = AddCTLayer2D(nt, name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	ConnectSuperToCT(nt, super, ct)
	return
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn OneToOne projection from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
func AddSuperCT4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct emer.Layer) {
	super = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	ConnectSuperToCT(nt, super, ct)
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Python versions

// AddSuperCTTRC2DPy adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn Full projection from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC projects back to Super and CT layers, type = Back, class = FmPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the TRC layer, and it must be sized appropriately for those drivers.
// Py is Python version, returns layers as a slice
func AddSuperCTTRC2DPy(nt *axon.Network, name string, shapeY, shapeX int) []emer.Layer {
	super, ct, trc := AddSuperCTTRC2D(nt, name, shapeY, shapeX)
	return []emer.Layer{super, ct, trc}
}

// AddSuperCTTRC4DPy adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn PoolOneToOne projection from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC projects back to Super and CT layers, also PoolOneToOne, class = FmPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the TRC layer, and it must be sized appropriately for those drivers.
// Py is Python version, returns layers as a slice
func AddSuperCTTRC4DPy(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) []emer.Layer {
	super, ct, trc := AddSuperCTTRC4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	return []emer.Layer{super, ct, trc}
}

// AddSuperCT2DPy adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn Full projection from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
// Py is Python version, returns layers as a slice
func AddSuperCT2DPy(nt *axon.Network, name string, shapeY, shapeX int) []emer.Layer {
	super, ct := AddSuperCT2D(nt, name, shapeY, shapeX)
	return []emer.Layer{super, ct}
}

// AddSuperCT4DPy adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn PoolOneToOne projection from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
// Py is Python version, returns layers as a slice
func AddSuperCT4DPy(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) []emer.Layer {
	super, ct := AddSuperCT4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	return []emer.Layer{super, ct}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network versions of Add Layer methods

// AddSuperCTTRC2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn OneToOne projection from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func (nt *Network) AddSuperCTTRC2D(name string, shapeY, shapeX int) (super, ct, pulv emer.Layer) {
	return AddSuperCTTRC2D(&nt.Network, name, shapeY, shapeX)
}

// AddSuperCTTRC4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn OneToOne projection from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func (nt *Network) AddSuperCTTRC4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct, pulv emer.Layer) {
	return AddSuperCTTRC4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn OneToOne projection from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT2D(name string, shapeY, shapeX int) (super, ct emer.Layer) {
	return AddSuperCT2D(&nt.Network, name, shapeY, shapeX)
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn PoolOneToOne projection from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct emer.Layer) {
	return AddSuperCT4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// ConnectToTRC2D connects Super and CT with given TRC: CT -> TRC is class CTToPulv,
// From TRC = type = Back, class = FmPulv
// 2D version uses Full projections.
func (nt *Network) ConnectToTRC2D(super, ct, trc emer.Layer) {
	ConnectToTRC2D(&nt.Network, super, ct, trc)
}

// ConnectToTRC4D connects Super and CT with given TRC: CT -> TRC is class CTToPulv,
// From TRC = type = Back, class = FmPulv
// 4D version uses PoolOneToOne projections.
func (nt *Network) ConnectToTRC4D(super, ct, trc emer.Layer) {
	ConnectToTRC4D(&nt.Network, super, ct, trc)
}

// ConnectCtxtToCT adds a CTCtxtPrjn from given sending layer to a CT layer
func (nt *Network) ConnectCtxtToCT(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return ConnectCtxtToCT(&nt.Network, send, recv, pat)
}

// AddTRCLayer2D adds a TRCLayer of given size, with given name.
func (nt *Network) AddTRCLayer2D(name string, nNeurY, nNeurX int) *TRCLayer {
	return AddTRCLayer2D(&nt.Network, name, nNeurY, nNeurX)
}

// AddTRCLayer4D adds a TRCLayer of given size, with given name.
func (nt *Network) AddTRCLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *TRCLayer {
	return AddTRCLayer4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddTRCALayer2D adds a TRCALayer of given size, with given name.
func (nt *Network) AddTRCALayer2D(name string, nNeurY, nNeurX int) *TRCALayer {
	return AddTRCALayer2D(&nt.Network, name, nNeurY, nNeurX)
}

// AddTRCALayer4D adds a TRCLayer of given size, with given name.
func (nt *Network) AddTRCALayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *TRCALayer {
	return AddTRCALayer4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddSuperLayer2D adds a SuperLayer of given size, with given name.
func (nt *Network) AddSuperLayer2D(name string, nNeurY, nNeurX int) *SuperLayer {
	return AddSuperLayer2D(&nt.Network, name, nNeurY, nNeurX)
}

// AddSuperLayer4D adds a TRCLayer of given size, with given name.
func (nt *Network) AddSuperLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *SuperLayer {
	return AddSuperLayer4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Compute methods

// CTCtxt sends context to CT layers and integrates CtxtGe on CT layers
func (nt *Network) CTCtxt(ltime *axon.Time) {
	nt.ThrLayFun(func(ly axon.AxonLayer) {
		if dl, ok := ly.(CtxtSender); ok {
			dl.SendCtxtGe(ltime)
		}
	}, "SendCtxtGe")

	nt.ThrLayFun(func(ly axon.AxonLayer) {
		if dl, ok := ly.(*CTLayer); ok {
			dl.CtxtFmGe(ltime)
		}
	}, "CtxtFmGe")
}
