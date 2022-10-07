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

// AddInputTRC2D adds an Input and TRCLayer of given size, with given name.
// The Input layer is set as the Driver of the TRCLayer
func AddInputTRC2D(nt *axon.Network, name string, nNeurY, nNeurX int, space float32) (emer.Layer, *TRCLayer) {
	in := nt.AddLayer2D(name, nNeurY, nNeurX, emer.Input)
	trc := AddTRCLayer2D(nt, name+"P", nNeurY, nNeurX)
	trc.Driver = name
	trc.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	return in, trc
}

// AddInputTRC4D adds an Input and TRCLayer of given size, with given name.
// The Input layer is set as the Driver of the TRCLayer
func AddInputTRC4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32) (emer.Layer, *TRCLayer) {
	in := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, emer.Input)
	trc := AddTRCLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	trc.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	trc.Driver = name
	return in, trc
}

// ConnectToTRC connects Super and CT with given TRC: CT -> TRC is class CTToPulv,
// From TRC = type = Back, class = FmPulv.
// toTrcPat is the prjn.Pattern CT -> TRC and fmTrcPat is TRC -> CT, Super.
// Typically TRC is a different shape than Super and CT, so use Full or appropriate
// topological pattern
func ConnectToTRC(nt *axon.Network, super, ct, trc emer.Layer, toTrcPat, fmTrcPat prjn.Pattern) {
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

// AddSuperCTTRC2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and TRC Pulvinar for Super (P suffix).
// TRC.Driver is set to Super -- needs other CT connections from higher up.
// CT is placed Behind Super, and Pulvinar behind CT.
func AddSuperCTTRC2D(nt *axon.Network, name string, shapeY, shapeX int, space float32, pat prjn.Pattern) (super, ct, trc emer.Layer) {
	super = AddSuperLayer2D(nt, name, shapeY, shapeX)
	ct = AddCTLayer2D(nt, name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	ConnectSuperToCT(nt, super, ct, pat)
	trci := AddTRCLayer2D(nt, name+"P", shapeY, shapeX)
	trc = trci
	trci.Driver = name
	trci.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: space})
	return
}

// AddSuperCTTRC4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and TRC Pulvinar for Super (P suffix).
// TRC.Driver is set to Super -- needs other CT connections from higher up.
// CT is placed Behind Super, and Pulvinar behind CT.
func AddSuperCTTRC4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat prjn.Pattern) (super, ct, trc emer.Layer) {
	super = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	ConnectSuperToCT(nt, super, ct, pat)
	trci := AddTRCLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	trc = trci
	trci.Driver = name
	trci.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: space})
	return
}

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO TRC Pulvinar.
// CT is placed Behind Super.
func AddSuperCT2D(nt *axon.Network, name string, shapeY, shapeX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	super = AddSuperLayer2D(nt, name, shapeY, shapeX)
	ct = AddCTLayer2D(nt, name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	ConnectSuperToCT(nt, super, ct, pat)
	return
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO TRC Pulvinar.
// CT is placed Behind Super.
func AddSuperCT4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	super = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: space})
	ConnectSuperToCT(nt, super, ct, pat)
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network versions of Add Layer methods

// AddSuperCTTRC2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and TRC Pulvinar for Super (P suffix).
// TRC.Driver is set to Super -- needs other CT connections from higher up.
// CT is placed Behind Super, and Pulvinar behind CT.
func (nt *Network) AddSuperCTTRC2D(name string, shapeY, shapeX int, space float32, pat prjn.Pattern) (super, ct, pulv emer.Layer) {
	return AddSuperCTTRC2D(&nt.Network, name, shapeY, shapeX, space, pat)
}

// AddSuperCTTRC4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and TRC Pulvinar for Super (P suffix).
// TRC.Driver is set to Super -- needs other CT connections from higher up.
// CT is placed Behind Super, and Pulvinar behind CT.
func (nt *Network) AddSuperCTTRC4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat prjn.Pattern) (super, ct, pulv emer.Layer) {
	return AddSuperCTTRC4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX, space, pat)
}

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO TRC Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT2D(name string, shapeY, shapeX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	return AddSuperCT2D(&nt.Network, name, shapeY, shapeX, space, pat)
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn projection from Super to CT using given projection pattern,
// and NO TRC Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddSuperCT4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32, pat prjn.Pattern) (super, ct emer.Layer) {
	return AddSuperCT4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX, space, pat)
}

// ConnectToTRC connects Super and CT with given TRC: CT -> TRC is class CTToPulv,
// From TRC = type = Back, class = FmPulv
// toTrcPat is the prjn.Pattern CT -> TRC and fmTrcPat is TRC -> CT, Super
// Typically TRC is a different shape than Super and CT, so use Full or appropriate
// topological pattern
func (nt *Network) ConnectToTRC(super, ct, trc emer.Layer, toTrcPat, fmTrcPat prjn.Pattern) {
	ConnectToTRC(&nt.Network, super, ct, trc, toTrcPat, fmTrcPat)
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

// AddInputTRC2D adds an Input and TRCLayer of given size, with given name.
// The Input layer is set as the Driver of the TRCLayer
func (nt *Network) AddInputTRC2D(name string, nNeurY, nNeurX int, space float32) (emer.Layer, *TRCLayer) {
	return AddInputTRC2D(&nt.Network, name, nNeurY, nNeurX, space)
}

// AddInputTRC4D adds an Input and TRCLayer of given size, with given name.
// The Input layer is set as the Driver of the TRCLayer
func (nt *Network) AddInputTRC4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32) (emer.Layer, *TRCLayer) {
	return AddInputTRC4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX, space)
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

// LayerSendCtxtGe sends activation over CTCtxtPrjn projections to integrate
// CtxtGe excitatory conductance on CT layers.
// This should be called at the end of the 5IB Bursting phase via Network.CTCtxt
// Satisfies the CtxtSender interface.
func LayerSendCtxtGe(ly *axon.Layer, ltime *axon.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.Act > 0.1 {
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
				pj.SendCtxtGe(ni, nrn.Act)
			}
		}
	}
}
