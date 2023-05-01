// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/prjn"
)

// AddBG adds MtxGo, MtxNo, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi layers,
// with given optional prefix. Doesn't return GPeOut, GpeIn which are purely internal.
// Only the Matrix has pool-based 4D shape by default -- use pool for "role" like
// elements where matches need to be detected.
// All GP / STN layers have gpNeur neurons.
// Appropriate connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical).
func (net *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpeTA, stnp, stns, gpi *Layer) {
	gpi = net.AddGPiLayer2D(prefix+"GPi", gpNeurY, gpNeurX)
	gpeOut := net.AddGPeLayer2D(prefix+"GPeOut", gpNeurY, gpNeurX)
	gpeOut.SetBuildConfig("GPType", "GPeOut")
	gpeIn := net.AddGPeLayer2D(prefix+"GPeIn", gpNeurY, gpNeurX)
	gpeIn.SetBuildConfig("GPType", "GPeIn")
	gpeTA = net.AddGPeLayer2D(prefix+"GPeTA", gpNeurY, gpNeurX)
	gpeTA.SetBuildConfig("GPType", "GPeTA")
	stnp = net.AddSTNLayer2D(prefix+"STNp", gpNeurY, gpNeurX)
	stns = net.AddSTNLayer2D(prefix+"STNs", gpNeurY, gpNeurX)
	mtxGo = net.AddMatrixLayer(prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = net.AddMatrixLayer(prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name())
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name())

	full := prjn.NewFull()

	net.ConnectLayers(mtxGo, gpeOut, full, InhibPrjn).SetClass("BgFixed")

	net.ConnectLayers(mtxNo, gpeIn, full, InhibPrjn)
	net.ConnectLayers(gpeOut, gpeIn, full, InhibPrjn)

	net.ConnectLayers(gpeIn, gpeTA, full, InhibPrjn).SetClass("BgFixed")
	net.ConnectLayers(gpeIn, stnp, full, InhibPrjn).SetClass("BgFixed")

	// note: this projection exists in bio, but does weird things with Ca dynamics in STNs..
	// nt.ConnectLayers(gpeIn, stns, full, InhibPrjn).SetClass("BgFixed")

	net.ConnectLayers(gpeIn, gpi, full, InhibPrjn)
	net.ConnectLayers(mtxGo, gpi, full, InhibPrjn)

	net.ConnectLayers(stnp, gpeOut, full, ForwardPrjn).SetClass("FmSTNp")
	net.ConnectLayers(stnp, gpeIn, full, ForwardPrjn).SetClass("FmSTNp")
	net.ConnectLayers(stnp, gpeTA, full, ForwardPrjn).SetClass("FmSTNp")
	net.ConnectLayers(stnp, gpi, full, ForwardPrjn).SetClass("FmSTNp")

	net.ConnectLayers(stns, gpi, full, ForwardPrjn).SetClass("FmSTNs")

	net.ConnectLayers(gpeTA, mtxGo, full, InhibPrjn).SetClass("GPeTAToMtx")
	net.ConnectLayers(gpeTA, mtxNo, full, InhibPrjn).SetClass("GPeTAToMtx")

	net.ConnectLayers(gpeIn, mtxGo, full, InhibPrjn).SetClass("GPeInToMtx")
	net.ConnectLayers(gpeIn, mtxNo, full, InhibPrjn).SetClass("GPeInToMtx")

	gpeOut.PlaceBehind(gpi, space)
	gpeIn.PlaceRightOf(gpeOut, space)
	gpeTA.PlaceRightOf(gpeIn, space)
	stnp.PlaceRightOf(gpi, space)
	stns.PlaceRightOf(stnp, space)

	mtxGo.PlaceBehind(gpeOut, space)
	mtxNo.PlaceRightOf(mtxGo, space)

	return
}

// AddBG4D adds MtxGo, MtxNo, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi layers,
// with given optional prefix. Doesn't return GPeOut, GpeIn which are purely internal.
// This version makes 4D pools throughout the GP layers,
// with Pools representing separable gating domains.
// All GP / STN layers have gpNeur neurons.
// Appropriate PoolOneToOne connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
// A CIN or more widely used RSalienceLayer should be added and
// project ACh to the MtxGo, No layers.
func (net *Network) AddBG4D(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpeTA, stnp, stns, gpi *Layer) {
	gpi = net.AddGPiLayer4D(prefix+"GPi", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeOut := net.AddGPeLayer4D(prefix+"GPeOut", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeOut.SetBuildConfig("GPType", "GPeOut")
	gpeIn := net.AddGPeLayer4D(prefix+"GPeIn", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeIn.SetBuildConfig("GPType", "GPeIn")
	gpeTA = net.AddGPeLayer4D(prefix+"GPeTA", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeTA.SetBuildConfig("GPType", "GPeTA")
	stnp = net.AddSTNLayer4D(prefix+"STNp", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	stnp.SetClass("STNp")
	stns = net.AddSTNLayer4D(prefix+"STNs", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	stns.SetClass("STNs")
	mtxGo = net.AddMatrixLayer(prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = net.AddMatrixLayer(prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name())
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name())

	p1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	net.ConnectLayers(mtxGo, gpeOut, p1to1, InhibPrjn).SetClass("BgFixed")

	net.ConnectLayers(mtxNo, gpeIn, p1to1, InhibPrjn)
	net.ConnectLayers(gpeOut, gpeIn, p1to1, InhibPrjn)

	net.ConnectLayers(gpeIn, gpeTA, p1to1, InhibPrjn).SetClass("BgFixed")
	net.ConnectLayers(gpeIn, stnp, p1to1, InhibPrjn).SetClass("BgFixed")

	// note: this projection exists in bio, but does weird things with Ca dynamics in STNs..
	// nt.ConnectLayers(gpeIn, stns, p1to1, InhibPrjn).SetClass("BgFixed")

	net.ConnectLayers(gpeIn, gpi, p1to1, InhibPrjn)
	net.ConnectLayers(mtxGo, gpi, p1to1, InhibPrjn)

	net.ConnectLayers(stnp, gpeOut, p1to1, ForwardPrjn).SetClass("FmSTNp")
	net.ConnectLayers(stnp, gpeIn, p1to1, ForwardPrjn).SetClass("FmSTNp")
	net.ConnectLayers(stnp, gpeTA, full, ForwardPrjn).SetClass("FmSTNp")
	net.ConnectLayers(stnp, gpi, p1to1, ForwardPrjn).SetClass("FmSTNp")

	net.ConnectLayers(stns, gpi, p1to1, ForwardPrjn).SetClass("FmSTNs")

	net.ConnectLayers(gpeTA, mtxGo, full, InhibPrjn).SetClass("GPeTAToMtx")
	net.ConnectLayers(gpeTA, mtxNo, full, InhibPrjn).SetClass("GPeTAToMtx")

	net.ConnectLayers(gpeIn, mtxGo, full, InhibPrjn).SetClass("GPeInToMtx")
	net.ConnectLayers(gpeIn, mtxNo, full, InhibPrjn).SetClass("GPeInToMtx")

	gpeOut.PlaceBehind(gpi, space)
	gpeIn.PlaceRightOf(gpeOut, space)
	gpeTA.PlaceRightOf(gpeIn, space)
	stnp.PlaceRightOf(gpi, space)
	stns.PlaceRightOf(stnp, space)

	mtxGo.PlaceBehind(gpeOut, space)
	mtxNo.PlaceRightOf(mtxGo, space)

	return
}

// AddBGThalLayer4D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 4D structure, with Pools representing separable gating domains.
func (net *Network) AddBGThalLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, BGThalLayer)
	ly.SetClass("BG")
	return ly
}

// AddBGThalLayer2D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 2D structure
func (net *Network) AddBGThalLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, BGThalLayer)
	ly.SetClass("BG")
	return ly
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (net *Network) AddMatrixLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DAModTypes) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, MatrixLayer)
	ly.SetBuildConfig("DAMod", da.String())
	ly.SetClass("BG")
	return ly
}

// ConnectToMatrix adds a MatrixPrjn from given sending layer to a matrix layer
func (net *Network) ConnectToMatrix(send, recv *Layer, pat prjn.Pattern) *Prjn {
	return net.ConnectLayers(send, recv, pat, MatrixPrjn)
}

// AddGPLayer2D adds a GPLayer of given size, with given name.
// Must set the GPType BuildConfig setting to appropriate GPLayerType
func (net *Network) AddGPeLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, GPLayer)
	ly.SetClass("BG")
	return ly
}

// AddGPiLayer2D adds a GPiLayer of given size, with given name.
func (net *Network) AddGPiLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, GPLayer)
	ly.SetBuildConfig("GPType", "GPi")
	ly.SetClass("BG")
	return ly
}

// AddSTNLayer2D adds a subthalamic nucleus Layer of given size, with given name.
func (net *Network) AddSTNLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, STNLayer)
	ly.SetClass("BG")
	return ly
}

// AddGPLayer4D adds a GPLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (net *Network) AddGPeLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, GPLayer)
	ly.SetClass("BG")
	return ly
}

// AddGPiLayer4D adds a GPiLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (net *Network) AddGPiLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, GPLayer)
	ly.SetClass("BG")
	return ly
}

// AddSTNLayer4D adds a subthalamic nucleus Layer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (net *Network) AddSTNLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, STNLayer)
	ly.SetClass("BG")
	return ly
}

// AddVSGatedLayer adds a VSGatedLayer with given number of Y units
// and 2 pools, first one represents JustGated, second is HasGated.
func (net *Network) AddVSGatedLayer(prefix string, nYunits int) *Layer {
	ly := net.AddLayer4D(prefix+"VSGated", 1, 2, nYunits, 1, VSGatedLayer)
	return ly
}
