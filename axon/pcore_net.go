// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/prjn"
)

// AddBG adds MtxGo, MtxNo, GPePr, GPeAk, STN, GPi layers,
// with given optional prefix.
// Only the Matrix has pool-based 4D shape by default -- use pool for "role" like
// elements where matches need to be detected.
// All GP / STN layers have gpNeur neurons.
// Appropriate connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical).
func (net *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpePr, gpeAk, stn, gpi *Layer) {
	gpi = net.AddGPiLayer2D(prefix+"GPi", gpNeurY, gpNeurX)
	gpePr = net.AddGPeLayer2D(prefix+"GPePr", gpNeurY, gpNeurX)
	gpePr.SetBuildConfig("GPType", "GPePr")
	gpeAk = net.AddGPeLayer2D(prefix+"GPeAk", gpNeurY, gpNeurX)
	gpeAk.SetBuildConfig("GPType", "GPeAk")
	stn = net.AddSTNLayer2D(prefix+"STN", gpNeurY, gpNeurX)
	mtxGo = net.AddMatrixLayer(prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = net.AddMatrixLayer(prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name())
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name())

	full := prjn.NewFull()

	net.ConnectLayers(mtxNo, gpePr, full, InhibPrjn)

	net.ConnectLayers(gpePr, gpePr, full, InhibPrjn).SetClass("BgFixed")
	net.ConnectLayers(gpePr, gpeAk, full, InhibPrjn).SetClass("BgFixed")
	net.ConnectLayers(gpePr, stn, full, InhibPrjn).SetClass("BgFixed")

	// note: this projection exists in bio, but does weird things with Ca dynamics in STNs..
	// nt.ConnectLayers(gpePr, stns, full, InhibPrjn).SetClass("BgFixed")

	net.ConnectLayers(gpePr, gpi, full, InhibPrjn)
	net.ConnectLayers(mtxGo, gpi, full, InhibPrjn)

	net.ConnectLayers(stn, gpePr, full, ForwardPrjn).SetClass("STNToGPePr")
	net.ConnectLayers(stn, gpeAk, full, ForwardPrjn).SetClass("STNToGPeAk")
	net.ConnectLayers(stn, gpi, full, ForwardPrjn).SetClass("STNToGPi")

	net.ConnectLayers(mtxGo, gpeAk, full, InhibPrjn).SetClass("MtxToGPeAk")
	net.ConnectLayers(gpeAk, mtxGo, full, InhibPrjn).SetClass("GPeAkToMtx")
	net.ConnectLayers(gpeAk, mtxNo, full, InhibPrjn).SetClass("GPeAkToMtx")

	gpeAk.PlaceBehind(gpi, space)
	gpePr.PlaceRightOf(gpeAk, space)
	stn.PlaceRightOf(gpi, space)
	mtxGo.PlaceBehind(gpePr, space)
	mtxNo.PlaceRightOf(mtxGo, space)

	return
}

// AddBG4D adds MtxGo, MtxNo, GPePr, GPeAk, STN, GPi layers,
// with given optional prefix.
// This version makes 4D pools throughout the GP layers,
// with Pools representing separable gating domains.
// All GP / STN layers have gpNeur neurons.
// Appropriate PoolOneToOne connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
// A CIN or more widely used RSalienceLayer should be added and
// project ACh to the MtxGo, No layers.
func (net *Network) AddBG4D(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpePr, gpeAk, stn, gpi *Layer) {
	gpi = net.AddGPiLayer4D(prefix+"GPi", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpePr = net.AddGPeLayer4D(prefix+"GPePr", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpePr.SetBuildConfig("GPType", "GPePr")
	gpeAk = net.AddGPeLayer4D(prefix+"GPeAk", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeAk.SetBuildConfig("GPType", "GPeAk")
	stn = net.AddSTNLayer4D(prefix+"STN", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	stn.SetClass("STN")
	mtxGo = net.AddMatrixLayer(prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = net.AddMatrixLayer(prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name())
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name())

	p1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	net.ConnectLayers(mtxNo, gpePr, p1to1, InhibPrjn)

	net.ConnectLayers(gpePr, gpePr, p1to1, InhibPrjn).SetClass("BgFixed")
	net.ConnectLayers(gpePr, gpeAk, p1to1, InhibPrjn).SetClass("BgFixed")
	net.ConnectLayers(gpePr, stn, p1to1, InhibPrjn).SetClass("BgFixed")

	// note: this projection exists in bio, but does weird things with Ca dynamics in STNs..
	// nt.ConnectLayers(gpePr, stns, p1to1, InhibPrjn).SetClass("BgFixed")

	net.ConnectLayers(gpePr, gpi, p1to1, InhibPrjn)
	net.ConnectLayers(mtxGo, gpi, p1to1, InhibPrjn)

	net.ConnectLayers(stn, gpePr, p1to1, ForwardPrjn).SetClass("STNToGPePr")
	net.ConnectLayers(stn, gpeAk, full, ForwardPrjn).SetClass("STNToGPeAk")
	net.ConnectLayers(stn, gpi, p1to1, ForwardPrjn).SetClass("STNToGPi")

	net.ConnectLayers(mtxGo, gpeAk, full, InhibPrjn).SetClass("MtxToGPeAk")
	net.ConnectLayers(gpeAk, mtxGo, full, InhibPrjn).SetClass("GPeAkToMtx")
	net.ConnectLayers(gpeAk, mtxNo, full, InhibPrjn).SetClass("GPeAkToMtx")

	gpePr.PlaceBehind(gpi, space)
	gpeAk.PlaceRightOf(gpePr, space)
	stn.PlaceRightOf(gpi, space)

	mtxGo.PlaceBehind(gpePr, space)
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
