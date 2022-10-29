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
// Only the Matrix has pool-based 4D shape by default -- use pool for "role" like
// elements where matches need to be detected.
// All GP / STN layers have gpNeur neurons.
// Appropriate connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
func (nt *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi axon.AxonLayer) {
	return AddBG(nt.AsAxon(), prefix, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX, space)
}

// AddBG4D adds MtxGo, No, CIN, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi layers,
// with given optional prefix.
// This version makes 4D pools throughout the GP layers,
// with Pools representing separable gating domains.
// All GP / STN layers have gpNeur neurons.
// Appropriate PoolOneToOne connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
func (nt *Network) AddBG4D(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi axon.AxonLayer) {
	return AddBG4D(nt.AsAxon(), prefix, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX, space)
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
// Projections are made with given classes: SuperToPT, PTSelfMaint, CTtoThal.
// The PT and Thal layers are positioned behind the CT layer.
func (nt *Network) AddPTThalForSuper(super, ct emer.Layer, suffix string, superToPT, ptSelf, ctToThal prjn.Pattern, space float32) (pt, thal emer.Layer) {
	return AddPTThalForSuper(nt.AsAxon(), super, ct, suffix, superToPT, ptSelf, ctToThal, space)
}

// ConnectToMatrix adds a MatrixTracePrjn from given sending layer to a matrix layer
func (nt *Network) ConnectToMatrix(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return ConnectToMatrix(nt.AsAxon(), send, recv, pat)
}

// AddHypothalLayer adds a HypothalLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separate body states / drives.
func (nt *Network) AddHypothalLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *HypothalLayer {
	return AddHypothalLayer(nt.AsAxon(), name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

////////////////////////////////////////////////////////////////////////
// Network functions available here as standalone functions
//         for mixing in to other models

// AddCINLayer adds a CINLayer, with a single neuron.
func AddCINLayer(nt *axon.Network, name string) *CINLayer {
	ly := &CINLayer{}
	nt.AddLayerInit(ly, name, []int{1, 1}, CIN)
	return ly
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func AddMatrixLayer(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DaReceptors) *MatrixLayer {
	ly := &MatrixLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, Matrix)
	ly.DaR = da
	ly.SetClass("BG Matrix")
	return ly
}

// ConnectToMatrix adds a MatrixTracePrjn from given sending layer to a matrix layer
func ConnectToMatrix(nt *axon.Network, send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayersPrjn(send, recv, pat, emer.Forward, &MatrixPrjn{})
}

// AddGPLayer2D adds a GPLayer of given size, with given name.
func AddGPeLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *GPLayer {
	ly := &GPLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, GP)
	ly.SetClass("BG")
	return ly
}

// AddGPiLayer2D adds a GPiLayer of given size, with given name.
func AddGPiLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *GPiLayer {
	ly := &GPiLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, GP)
	ly.SetClass("BG")
	return ly
}

// AddSTNLayer2D adds a subthalamic nucleus Layer of given size, with given name.
func AddSTNLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *STNLayer {
	ly := &STNLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, STN)
	ly.SetClass("BG")
	return ly
}

// AddGPLayer4D adds a GPLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func AddGPeLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPLayer {
	ly := &GPLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, GP)
	ly.SetClass("BG")
	return ly
}

// AddGPiLayer4D adds a GPiLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func AddGPiLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPiLayer {
	ly := &GPiLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, GP)
	ly.SetClass("BG")
	return ly
}

// AddSTNLayer4D adds a subthalamic nucleus Layer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func AddSTNLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *STNLayer {
	ly := &STNLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, STN)
	ly.SetClass("BG")
	return ly
}

// AddThalLayer2D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 2D structure
func AddThalLayer2D(nt *axon.Network, name string, nNeurY, nNeurX int) *ThalLayer {
	ly := &ThalLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, Thal)
	ly.SetClass("BG")
	return ly
}

// AddThalLayer4D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 4D structure, with Pools representing separable gating domains.
func AddThalLayer4D(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *ThalLayer {
	ly := &ThalLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, Thal)
	ly.SetClass("BG")
	return ly
}

// AddHypothalLayer adds a HypothalLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separate body states / drives.
func AddHypothalLayer(nt *axon.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *HypothalLayer {
	ly := &HypothalLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, Hypothal)
	ly.SetClass("BG")
	return ly
}

// AddBG adds MtxGo, No, CIN, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi layers,
// with given optional prefix.
// Only the Matrix has pool-based 4D shape by default -- use pool for "role" like
// elements where matches need to be detected.
// All GP / STN layers have gpNeur neurons.
// Appropriate connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
func AddBG(nt *axon.Network, prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi axon.AxonLayer) {
	gpi = AddGPiLayer2D(nt, prefix+"GPi", gpNeurY, gpNeurX)
	gpeOuti := AddGPeLayer2D(nt, prefix+"GPeOut", gpNeurY, gpNeurX)
	gpeOuti.GPLay = GPeOut
	gpeOut = gpeOuti
	gpeIni := AddGPeLayer2D(nt, prefix+"GPeIn", gpNeurY, gpNeurX)
	gpeIni.GPLay = GPeIn
	gpeIn = gpeIni
	gpeTAi := AddGPeLayer2D(nt, prefix+"GPeTA", gpNeurY, gpNeurX)
	gpeTAi.GPLay = GPeTA
	gpeTA = gpeTAi
	stnp = AddSTNLayer2D(nt, prefix+"STNp", gpNeurY, gpNeurX)
	stns = AddSTNLayer2D(nt, prefix+"STNs", gpNeurY, gpNeurX)
	mtxGo = AddMatrixLayer(nt, prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1R)
	mtxNo = AddMatrixLayer(nt, prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2R)
	cini := AddCINLayer(nt, prefix+"CIN")
	cin = cini

	cini.SendACh.Add(mtxGo.Name(), mtxNo.Name())

	mtxGo.(*MatrixLayer).MtxThals.Add(mtxNo.Name())
	mtxNo.(*MatrixLayer).MtxThals.Add(mtxGo.Name())

	full := prjn.NewFull()

	nt.ConnectLayers(mtxGo, gpeOut, full, emer.Inhib).SetClass("BgFixed")

	nt.ConnectLayers(mtxNo, gpeIn, full, emer.Inhib)
	nt.ConnectLayers(gpeOut, gpeIn, full, emer.Inhib)

	nt.ConnectLayers(gpeIn, gpeTA, full, emer.Inhib).SetClass("BgFixed")
	nt.ConnectLayers(gpeIn, stnp, full, emer.Inhib).SetClass("BgFixed")

	// note: this projection exists in bio, but does weird things with Ca dynamics in STNs..
	// nt.ConnectLayers(gpeIn, stns, full, emer.Inhib).SetClass("BgFixed")

	nt.ConnectLayers(gpeIn, gpi, full, emer.Inhib)
	nt.ConnectLayers(mtxGo, gpi, full, emer.Inhib)

	nt.ConnectLayers(stnp, gpeOut, full, emer.Forward).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpeIn, full, emer.Forward).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpeTA, full, emer.Forward).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpi, full, emer.Forward).SetClass("FmSTNp")

	nt.ConnectLayers(stns, gpi, full, emer.Forward).SetClass("FmSTNs")

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

// AddBG4D adds MtxGo, No, CIN, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi layers,
// with given optional prefix.
// This version makes 4D pools throughout the GP layers,
// with Pools representing separable gating domains.
// All GP / STN layers have gpNeur neurons.
// Appropriate PoolOneToOne connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
func AddBG4D(nt *axon.Network, prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi axon.AxonLayer) {
	gpi = AddGPiLayer4D(nt, prefix+"GPi", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeOuti := AddGPeLayer4D(nt, prefix+"GPeOut", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeOuti.GPLay = GPeOut
	gpeOut = gpeOuti
	gpeIni := AddGPeLayer4D(nt, prefix+"GPeIn", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeIni.GPLay = GPeIn
	gpeIn = gpeIni
	gpeTAi := AddGPeLayer4D(nt, prefix+"GPeTA", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeTAi.GPLay = GPeTA
	gpeTA = gpeTAi
	stnp = AddSTNLayer4D(nt, prefix+"STNp", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	stns = AddSTNLayer4D(nt, prefix+"STNs", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
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
// PT and Thal have SetClass(super.Name()) called to allow shared params.
// Projections are made with given classes: SuperToPT, PTSelfMaint, CTtoThal.
// The PT and Thal layers are positioned behind the CT layer.
func AddPTThalForSuper(nt *axon.Network, super, ct emer.Layer, suffix string, superToPT, ptSelf, ctToThal prjn.Pattern, space float32) (pt, thal emer.Layer) {
	name := super.Name()
	shp := super.Shape()
	if shp.NumDims() == 2 {
		pt = AddPTLayer2D(nt, name+"PT", shp.Dim(0), shp.Dim(1))
		thal = AddThalLayer2D(nt, name+suffix, shp.Dim(0), shp.Dim(1))
	} else {
		pt = AddPTLayer4D(nt, name+"PT", shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
		thal = AddThalLayer4D(nt, name+suffix, shp.Dim(0), shp.Dim(1), shp.Dim(2), shp.Dim(3))
	}
	pt.SetClass(name)
	thal.SetClass(name)
	pt.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: ct.Name(), XAlign: relpos.Left, Space: space})
	thal.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: pt.Name(), XAlign: relpos.Left, Space: space})
	one2one := prjn.NewOneToOne()
	pthal, thalpt := nt.BidirConnectLayers(pt, thal, one2one)
	pthal.SetClass("PTtoThal")
	thalpt.SetClass("ThalToPT")
	sthal, thals := nt.BidirConnectLayers(super, thal, superToPT) // shortcuts
	sthal.SetClass("SuperToThal")
	thals.SetClass("ThalToSuper")
	nt.ConnectLayers(super, pt, superToPT, emer.Forward).SetClass("SuperToPT")
	nt.LateralConnectLayer(pt, ptSelf).SetClass("PTSelfMaint")
	nt.ConnectLayers(ct, thal, ctToThal, emer.Forward).SetClass("CTtoThal")
	return
}
