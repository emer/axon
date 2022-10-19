// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/goki/ki/kit"
)

// pcore.Network has methods for configuring specialized PCore network components
// PCore = Pallidal Core mode of BG
type Network struct {
	deep.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = axon.NetworkProps

// UnitVarNames returns a list of variable names available on the units in this layer
func (nt *Network) UnitVarNames() []string {
	return NeuronVarsAll
}

// SynVarNames returns the names of all the variables on the synapses in this network.
func (nt *Network) SynVarNames() []string {
	return SynVarsAll
}

// AddBG adds MtxGo, No, CIN, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi layers,
// with given optional prefix.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// All GP / STN layers have gpNeur neurons
// Appropriate PoolOneToOne connections are made between layers, using standard styles
// space is the spacing between layers (2 typical)
func (nt *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi axon.AxonLayer) {
	return AddBG(nt.AsAxon(), prefix, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX, space)
}

// AddThalLayer4D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 4D structure, with Pools representing separable gating domains.
func (nt *Network) AddThalLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *ThalLayer {
	return AddThalLayer4D(nt.AsAxon(), name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddThalLayer2D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 2D structure
func (nt *Network) AddThalLayer2D(name string, nNeurY, nNeurX int) *ThalLayer {
	return AddThalLayer2D(nt.AsAxon(), name, nNeurY, nNeurX)
}

// AddPTThalForSuper adds a PT pyramidal tract layer and a
// Thalamus layer for given superficial layer (SuperLayer)
// with given suffix (e.g., MD, VM).
// The PT and Thal layers are positioned behind the CT layer.
func (nt *Network) AddPTThalForSuper(super, ct emer.Layer, suffix string, space float32) (pt, thal emer.Layer) {
	return AddPTThalForSuper(nt.AsAxon(), super, ct, suffix, space)
}

// ConnectToMatrix adds a MatrixTracePrjn from given sending layer to a matrix layer
func (nt *Network) ConnectToMatrix(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return ConnectToMatrix(nt.AsAxon(), send, recv, pat)
}

////////////////////////////////////////////////////////////////////////
// Network functions available here as standalone functions
//         for mixing in to other models

// AddCINLayer adds a CINLayer, with a single neuron.
func AddCINLayer(nt *axon.Network, name string) *CINLayer {
	ly := &CINLayer{}
	nt.AddLayerInit(ly, name, []int{1, 1}, emer.Hidden)
	return ly
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func AddMatrixLayer(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DaReceptors) *MatrixLayer {
	ly := &MatrixLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	ly.DaR = da
	ly.SetClass("BG Matrix")
	return ly
}

// ConnectToMatrix adds a MatrixTracePrjn from given sending layer to a matrix layer
func ConnectToMatrix(nt *axon.Network, send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayersPrjn(send, recv, pat, emer.Forward, &MatrixPrjn{})
}

// AddGPLayer adds a GPLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func AddGPeLayer(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPLayer {
	ly := &GPLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	ly.SetClass("BG GP")
	return ly
}

// AddGPiLayer adds a GPiLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func AddGPiLayer(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPiLayer {
	ly := &GPiLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	ly.SetClass("BG GP")
	return ly
}

// AddSTNLayer adds a subthalamic nucleus Layer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func AddSTNLayer(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *STNLayer {
	ly := &STNLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	ly.SetClass("BG STN")
	return ly
}

// AddThalLayer2D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 2D structure
func AddThalLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *ThalLayer {
	ly := &ThalLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, Thal)
	ly.SetClass("BG Thal")
	return ly
}

// AddThalLayer4D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 4D structure, with Pools representing separable gating domains.
func AddThalLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *ThalLayer {
	ly := &ThalLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	ly.SetClass("BG Thal")
	return ly
}

// AddBG adds MtxGo, No, CIN, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi layers,
// with given optional prefix.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// All GP / STN layers have gpNeur neurons
// Appropriate PoolOneToOne connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
func AddBG(nt *axon.Network, prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi axon.AxonLayer) {
	gpi = AddGPiLayer(nt, prefix+"GPi", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeOuti := AddGPeLayer(nt, prefix+"GPeOut", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeOuti.GPLay = GPeOut
	gpeOut = gpeOuti
	gpeIni := AddGPeLayer(nt, prefix+"GPeIn", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeIni.GPLay = GPeIn
	gpeIn = gpeIni
	gpeTAi := AddGPeLayer(nt, prefix+"GPeTA", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeTAi.GPLay = GPeTA
	gpeTA = gpeTAi
	stnp = AddSTNLayer(nt, prefix+"STNp", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	stns = AddSTNLayer(nt, prefix+"STNs", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	mtxGo = AddMatrixLayer(nt, prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1R)
	mtxNo = AddMatrixLayer(nt, prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2R)
	cini := AddCINLayer(nt, prefix+"CIN")
	cin = cini

	cini.SendACh.Add(mtxGo.Name(), mtxNo.Name())

	one2one := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	nt.ConnectLayers(mtxGo, gpeOut, one2one, emer.Inhib).SetClass("BgFixed")

	nt.ConnectLayers(mtxNo, gpeIn, one2one, emer.Inhib)
	nt.ConnectLayers(gpeOut, gpeIn, one2one, emer.Inhib)

	nt.ConnectLayers(gpeIn, gpeTA, one2one, emer.Inhib).SetClass("BgFixed")
	nt.ConnectLayers(gpeIn, stnp, one2one, emer.Inhib).SetClass("BgFixed")

	// note: this projection exists in bio, but does weird things with Ca dynamics in STNs..
	// nt.ConnectLayers(gpeIn, stns, one2one, emer.Inhib).SetClass("BgFixed")

	nt.ConnectLayers(gpeIn, gpi, one2one, emer.Inhib)
	nt.ConnectLayers(mtxGo, gpi, one2one, emer.Inhib)

	nt.ConnectLayers(stnp, gpeOut, one2one, emer.Forward).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpeIn, one2one, emer.Forward).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpeTA, full, emer.Forward).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpi, one2one, emer.Forward).SetClass("FmSTNp")

	nt.ConnectLayers(stns, gpi, one2one, emer.Forward).SetClass("FmSTNs")

	nt.ConnectLayers(gpeTA, mtxGo, full, emer.Inhib).SetClass("GPeTAToMtx")
	nt.ConnectLayers(gpeTA, mtxNo, full, emer.Inhib).SetClass("GPeTAToMtx")

	nt.ConnectLayers(gpeIn, mtxGo, full, emer.Inhib).SetClass("GPeInToMtx")
	nt.ConnectLayers(gpeIn, mtxNo, full, emer.Inhib).SetClass("GPeInToMtx")

	gpeOut.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: gpi.Name(), XAlign: relpos.Left, Space: space})
	gpeIn.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpeOut.Name(), YAlign: relpos.Front, Space: space})
	gpeTA.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpeIn.Name(), YAlign: relpos.Front, Space: space})
	stnp.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpi.Name(), YAlign: relpos.Front, Space: space})
	stns.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: stnp.Name(), YAlign: relpos.Front, Space: space})

	mtxGo.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: gpeOut.Name(), XAlign: relpos.Left, Space: space})
	mtxNo.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxGo.Name(), YAlign: relpos.Front, Space: space})
	cin.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxNo.Name(), YAlign: relpos.Front, Space: space})

	return
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

// ConnectPTSelf adds a Self (Lateral) projection within a PT layer,
// which supports active maintenance, with a class of PTSelfMaint
func ConnectPTSelf(nt *axon.Network, ly emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.LateralConnectLayer(ly, pat).SetClass("PTSelfMaint")
}

// AddPTThalForSuper adds a PT pyramidal tract layer and a
// Thalamus layer for given superficial layer (deep.SuperLayer) and associated CT
// with given suffix (e.g., MD, VM).
// The PT and Thal layers are positioned behind the CT layer.
func AddPTThalForSuper(nt *axon.Network, super, ct emer.Layer, suffix string, space float32) (pt, thal emer.Layer) {
	name := super.Name()
	shp := super.Shape()
	if shp.NumDims() == 2 {
		pt = AddPTLayer2D(nt, name+"PT", shp.Dim(0), shp.Dim(1))
		thal = AddThalLayer2D(nt, name+suffix, shp.Dim(0), shp.Dim(1))
	} else {
		pt = AddPTLayer4D(nt, name+"PT", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
		thal = AddThalLayer4D(nt, name+suffix, shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	pt.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: ct.Name(), XAlign: relpos.Left, Space: space})
	thal.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: pt.Name(), XAlign: relpos.Left, Space: space})
	one2one := prjn.NewOneToOne()
	pthal, thalpt := nt.BidirConnectLayers(pt, thal, one2one)
	pthal.SetClass("PTtoThal")
	thalpt.SetClass("ThalToPT")
	nt.ConnectLayers(ct, thal, one2one, emer.Forward).SetClass("CTtoThal")
	return
}
