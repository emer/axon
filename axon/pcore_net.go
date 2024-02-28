// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/prjn"
)

// AddVBG adds Ventral Basal Ganglia layers, using the PCore Pallidal Core
// framework where GPe plays a central role.
// Returns VMtxGo, VMtxNo, VGPePr, VGPeAk, VSTN, VGPi layers,
// with given optional prefix.
// Only the Matrix has pool-based 4D shape by default -- use pool for "role" like
// elements where matches need to be detected.
// All GP / STN layers have gpNeur neurons.
// Appropriate connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical).
func (net *Network) AddVBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpePr, gpeAk, stn, gpi *Layer) {
	gpi = net.AddGPiLayer2D(prefix+"VGPi", gpNeurY, gpNeurX)
	gpePr = net.AddGPeLayer2D(prefix+"VGPePr", gpNeurY, gpNeurX)
	gpePr.SetBuildConfig("GPType", "GPePr")
	gpeAk = net.AddGPeLayer2D(prefix+"VGPeAk", gpNeurY, gpNeurX)
	gpeAk.SetBuildConfig("GPType", "GPeAk")
	stn = net.AddSTNLayer2D(prefix+"VSTN", gpNeurY, gpNeurX)
	mtxGo = net.AddMatrixLayer(prefix+"VMtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = net.AddMatrixLayer(prefix+"VMtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name())
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name())

	mp := params.Params{
		"Layer.Matrix.IsVS":          "true",
		"Layer.Inhib.ActAvg.Nominal": ".03", // pooled, lower
	}
	mtxGo.DefParams = mp
	mtxGo.SetClass("VSMatrixLayer")
	mtxNo.DefParams = mp
	mtxNo.SetClass("VSMatrixLayer")

	full := prjn.NewFull()

	net.ConnectLayers(mtxNo, gpePr, full, InhibPrjn)

	bgclass := "VBGInhib"
	net.ConnectLayers(gpePr, gpePr, full, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpePr, gpeAk, full, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpePr, stn, full, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpePr, gpi, full, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(mtxGo, gpi, full, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(mtxGo, gpeAk, full, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpeAk, mtxGo, full, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpeAk, mtxNo, full, InhibPrjn).SetClass(bgclass)

	stnclass := "VSTNExcite"
	net.ConnectLayers(stn, gpePr, full, ForwardPrjn).SetClass(stnclass)
	net.ConnectLayers(stn, gpeAk, full, ForwardPrjn).SetClass(stnclass)
	net.ConnectLayers(stn, gpi, full, ForwardPrjn).SetClass(stnclass)

	gpeAk.PlaceBehind(gpi, space)
	gpePr.PlaceRightOf(gpeAk, space)
	stn.PlaceRightOf(gpi, space)
	mtxGo.PlaceBehind(gpePr, space)
	mtxNo.PlaceRightOf(mtxGo, space)

	return
}

// AddDBG adds Dorsal Basal Ganglia layers, using the PCore Pallidal Core
// framework where GPe plays a central role.
// Returns DMtxGo, DMtxNo, DGPePr, DGPeAk, DSTN, DGPi, PF layers, with given optional prefix.
// Makes 4D pools throughout the GP layers, with Pools representing separable
// gating domains, i.e., action domains.
// All GP / STN layers have gpNeur neurons.
// Appropriate PoolOneToOne connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
func (net *Network) AddDBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpePr, gpeAk, stn, gpi, pf *Layer) {
	gpi = net.AddGPiLayer4D(prefix+"DGPi", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpePr = net.AddGPeLayer4D(prefix+"DGPePr", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpePr.SetBuildConfig("GPType", "GPePr")
	gpeAk = net.AddGPeLayer4D(prefix+"DGPeAk", nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeAk.SetBuildConfig("GPType", "GPeAk")
	stn = net.AddSTNLayer2D(prefix+"DSTN", gpNeurY, gpNeurX)
	mtxGo = net.AddMatrixLayer(prefix+"DMtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = net.AddMatrixLayer(prefix+"DMtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)
	pf = net.AddLayer4D(prefix+"PF", nPoolsY, nPoolsX, nNeurY, 1, SuperLayer)

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name())
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name())

	p1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	net.ConnectLayers(mtxNo, gpePr, p1to1, InhibPrjn)

	bgclass := "DBGInhib"
	net.ConnectLayers(gpePr, gpePr, full, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpePr, gpeAk, p1to1, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpePr, stn, full, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpePr, gpi, p1to1, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(mtxGo, gpi, p1to1, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(mtxGo, gpeAk, p1to1, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpeAk, mtxGo, p1to1, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpeAk, mtxNo, p1to1, InhibPrjn).SetClass(bgclass)
	net.ConnectLayers(gpi, pf, p1to1, InhibPrjn).SetClass(bgclass) // todo: was full!

	stnclass := "DSTNExcite"
	net.ConnectLayers(stn, gpePr, full, ForwardPrjn).SetClass(stnclass)
	net.ConnectLayers(stn, gpeAk, full, ForwardPrjn).SetClass(stnclass)
	net.ConnectLayers(stn, gpi, full, ForwardPrjn).SetClass(stnclass)

	pfm := params.Params{
		"Prjn.Learn.Learn":   "false",
		"Prjn.Com.GType":     "ModulatoryG",
		"Prjn.PrjnScale.Abs": "1",
	}
	pj := net.ConnectLayers(pf, mtxGo, p1to1, ForwardPrjn).SetClass("PFToDMtx").(*Prjn)
	pj.DefParams = pfm
	pj = net.ConnectLayers(pf, mtxNo, p1to1, ForwardPrjn).SetClass("PFToDMtx").(*Prjn)
	pj.DefParams = pfm

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

// ConnectToVSMatrix adds a VSMatrixPrjn from given sending layer to a matrix layer
func (net *Network) ConnectToVSMatrix(send, recv *Layer, pat prjn.Pattern) *Prjn {
	return net.ConnectLayers(send, recv, pat, VSMatrixPrjn)
}

// ConnectToDSMatrix adds a DSMatrixPrjn from given sending layer to a matrix layer
func (net *Network) ConnectToDSMatrix(send, recv *Layer, pat prjn.Pattern) *Prjn {
	return net.ConnectLayers(send, recv, pat, DSMatrixPrjn)
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
	ly.SetBuildConfig("GPType", "GPi")
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
