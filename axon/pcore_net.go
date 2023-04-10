// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/prjn"
)

// AddBG adds MtxGo, MtxNo, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi layers,
// with given optional prefix.
// Only the Matrix has pool-based 4D shape by default -- use pool for "role" like
// elements where matches need to be detected.
// All GP / STN layers have gpNeur neurons.
// Appropriate connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical).
// A CIN or more widely used RSalienceLayer should be added and
// project ACh to the MtxGo, No layers.
func (nt *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpeOut, gpeIn, gpeTA, stnp, stns, gpi *Layer) {
	gpi = nt.AddGPiLayer2D(prefix+"GPi", gpNeurY, gpNeurX)
	gpeOuti := nt.AddGPeLayer2D(prefix+"GPeOut", gpNeurY, gpNeurX)
	gpeOuti.SetBuildConfig("GPType", "GPeOut")
	gpeOut = gpeOuti
	gpeIni := nt.AddGPeLayer2D(prefix+"GPeIn", gpNeurY, gpNeurX)
	gpeIni.SetBuildConfig("GPType", "GPeIn")
	gpeIn = gpeIni
	gpeTAi := nt.AddGPeLayer2D(prefix+"GPeTA", gpNeurY, gpNeurX)
	gpeTAi.SetBuildConfig("GPType", "GPeTA")
	gpeTA = gpeTAi
	stnp = nt.AddSTNLayer2D(prefix+"STNp", gpNeurY, gpNeurX)
	stns = nt.AddSTNLayer2D(prefix+"STNs", gpNeurY, gpNeurX)
	mtxGo = nt.AddMatrixLayer(prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = nt.AddMatrixLayer(prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name())
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name())

	full := prjn.NewFull()

	nt.ConnectLayers(mtxGo, gpeOut, full, InhibPrjn).SetClass("BgFixed")

	nt.ConnectLayers(mtxNo, gpeIn, full, InhibPrjn)
	nt.ConnectLayers(gpeOut, gpeIn, full, InhibPrjn)

	nt.ConnectLayers(gpeIn, gpeTA, full, InhibPrjn).SetClass("BgFixed")
	nt.ConnectLayers(gpeIn, stnp, full, InhibPrjn).SetClass("BgFixed")

	// note: this projection exists in bio, but does weird things with Ca dynamics in STNs..
	// nt.ConnectLayers(gpeIn, stns, full, InhibPrjn).SetClass("BgFixed")

	nt.ConnectLayers(gpeIn, gpi, full, InhibPrjn)
	nt.ConnectLayers(mtxGo, gpi, full, InhibPrjn)

	nt.ConnectLayers(stnp, gpeOut, full, ForwardPrjn).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpeIn, full, ForwardPrjn).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpeTA, full, ForwardPrjn).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpi, full, ForwardPrjn).SetClass("FmSTNp")

	nt.ConnectLayers(stns, gpi, full, ForwardPrjn).SetClass("FmSTNs")

	nt.ConnectLayers(gpeTA, mtxGo, full, InhibPrjn).SetClass("GPeTAToMtx")
	nt.ConnectLayers(gpeTA, mtxNo, full, InhibPrjn).SetClass("GPeTAToMtx")

	nt.ConnectLayers(gpeIn, mtxGo, full, InhibPrjn).SetClass("GPeInToMtx")
	nt.ConnectLayers(gpeIn, mtxNo, full, InhibPrjn).SetClass("GPeInToMtx")

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
// with given optional prefix.
// This version makes 4D pools throughout the GP layers,
// with Pools representing separable gating domains.
// All GP / STN layers have gpNeur neurons.
// Appropriate PoolOneToOne connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
// A CIN or more widely used RSalienceLayer should be added and
// project ACh to the MtxGo, No layers.
func (nt *Network) AddBG4D(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpeOut, gpeIn, gpeTA, stnp, stns, gpi *Layer) {
	gpi = nt.AddGPiLayer4D(prefix+"GPi", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeOuti := nt.AddGPeLayer4D(prefix+"GPeOut", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeOuti.SetBuildConfig("GPType", "GPeOut")
	gpeOut = gpeOuti
	gpeIni := nt.AddGPeLayer4D(prefix+"GPeIn", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeIni.SetBuildConfig("GPType", "GPeIn")
	gpeIn = gpeIni
	gpeTAi := nt.AddGPeLayer4D(prefix+"GPeTA", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeTAi.SetBuildConfig("GPType", "GPeTA")
	gpeTA = gpeTAi
	stnp = nt.AddSTNLayer4D(prefix+"STNp", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	stns = nt.AddSTNLayer4D(prefix+"STNs", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	mtxGo = nt.AddMatrixLayer(prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = nt.AddMatrixLayer(prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name())
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name())

	one2one := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	nt.ConnectLayers(mtxGo, gpeOut, one2one, InhibPrjn).SetClass("BgFixed")

	nt.ConnectLayers(mtxNo, gpeIn, one2one, InhibPrjn)
	nt.ConnectLayers(gpeOut, gpeIn, one2one, InhibPrjn)

	nt.ConnectLayers(gpeIn, gpeTA, one2one, InhibPrjn).SetClass("BgFixed")
	nt.ConnectLayers(gpeIn, stnp, one2one, InhibPrjn).SetClass("BgFixed")

	// note: this projection exists in bio, but does weird things with Ca dynamics in STNs..
	// nt.ConnectLayers(gpeIn, stns, one2one, InhibPrjn).SetClass("BgFixed")

	nt.ConnectLayers(gpeIn, gpi, one2one, InhibPrjn)
	nt.ConnectLayers(mtxGo, gpi, one2one, InhibPrjn)

	nt.ConnectLayers(stnp, gpeOut, one2one, ForwardPrjn).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpeIn, one2one, ForwardPrjn).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpeTA, full, ForwardPrjn).SetClass("FmSTNp")
	nt.ConnectLayers(stnp, gpi, one2one, ForwardPrjn).SetClass("FmSTNp")

	nt.ConnectLayers(stns, gpi, one2one, ForwardPrjn).SetClass("FmSTNs")

	nt.ConnectLayers(gpeTA, mtxGo, full, InhibPrjn).SetClass("GPeTAToMtx")
	nt.ConnectLayers(gpeTA, mtxNo, full, InhibPrjn).SetClass("GPeTAToMtx")

	nt.ConnectLayers(gpeIn, mtxGo, full, InhibPrjn).SetClass("GPeInToMtx")
	nt.ConnectLayers(gpeIn, mtxNo, full, InhibPrjn).SetClass("GPeInToMtx")

	gpeOut.PlaceBehind(gpi, space)
	gpeIn.PlaceRightOf(gpeOut, space)
	gpeTA.PlaceRightOf(gpeIn, space)
	stnp.PlaceRightOf(gpi, space)
	stns.PlaceRightOf(stnp, space)

	mtxGo.PlaceBehind(gpeOut, space)
	mtxNo.PlaceRightOf(mtxGo, space)

	return
}

// AddThalLayer4D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 4D structure, with Pools representing separable gating domains.
func (nt *Network) AddThalLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, VThalLayer)
	ly.SetClass("BG")
	return ly
}

// AddThalLayer2D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 2D structure
func (nt *Network) AddThalLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, VThalLayer)
	ly.SetClass("BG")
	return ly
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (nt *Network) AddMatrixLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DAModTypes) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, MatrixLayer)
	ly.SetBuildConfig("DAMod", da.String())
	ly.SetClass("BG")
	return ly
}

// ConnectToMatrix adds a MatrixPrjn from given sending layer to a matrix layer
func (nt *Network) ConnectToMatrix(send, recv *Layer, pat prjn.Pattern) *Prjn {
	return nt.ConnectLayers(send, recv, pat, MatrixPrjn)
}

// AddGPLayer2D adds a GPLayer of given size, with given name.
// Must set the GPType BuildConfig setting to appropriate GPLayerType
func (nt *Network) AddGPeLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, GPLayer)
	ly.SetClass("BG")
	return ly
}

// AddGPiLayer2D adds a GPiLayer of given size, with given name.
func (nt *Network) AddGPiLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, GPLayer)
	ly.SetBuildConfig("GPType", "GPi")
	ly.SetClass("BG")
	return ly
}

// AddSTNLayer2D adds a subthalamic nucleus Layer of given size, with given name.
func (nt *Network) AddSTNLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer2D(name, nNeurY, nNeurX, STNLayer)
	ly.SetClass("BG")
	return ly
}

// AddGPLayer4D adds a GPLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (nt *Network) AddGPeLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, GPLayer)
	ly.SetClass("BG")
	return ly
}

// AddGPiLayer4D adds a GPiLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (nt *Network) AddGPiLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, GPLayer)
	ly.SetClass("BG")
	return ly
}

// AddSTNLayer4D adds a subthalamic nucleus Layer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (nt *Network) AddSTNLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, STNLayer)
	ly.SetClass("BG")
	return ly
}

// AddVSGatedLayer adds a VSGatedLayer with given number of Y units
// and 2 pools, first one represents JustGated, second is HasGated.
func (nt *Network) AddVSGatedLayer(prefix string, nYunits int) *Layer {
	ly := nt.AddLayer4D(prefix+"VSGated", 1, 2, nYunits, 1, VSGatedLayer)
	return ly
}
