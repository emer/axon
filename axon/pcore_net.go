// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"

	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
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
	bglay := "VBG"
	gpi = net.AddGPiLayer2D(prefix+"VGPi", bglay, gpNeurY, gpNeurX)
	gpePr = net.AddGPeLayer2D(prefix+"VGPePr", bglay, gpNeurY, gpNeurX)
	gpePr.SetBuildConfig("GPType", "GPePr")
	gpeAk = net.AddGPeLayer2D(prefix+"VGPeAk", bglay, gpNeurY, gpNeurX)
	gpeAk.SetBuildConfig("GPType", "GPeAk")
	stn = net.AddSTNLayer2D(prefix+"VSTN", "VSTNLayer", gpNeurY, gpNeurX)
	mtxGo = net.AddVMatrixLayer(prefix+"VMtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = net.AddVMatrixLayer(prefix+"VMtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name)
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name)

	mp := params.Params{
		"Layer.Matrix.IsVS":          "true",
		"Layer.Inhib.ActAvg.Nominal": fmt.Sprintf("%g", .1/float32(nPoolsX*nPoolsY)),
		"Layer.Acts.Dend.ModACh":     "true",
	}
	mtxGo.DefaultParams = mp
	mtxNo.DefaultParams = mp

	full := paths.NewFull()
	p1to1 := paths.NewPoolOneToOne()

	net.ConnectLayers(mtxNo, gpePr, full, InhibPath)
	pt := net.ConnectLayers(mtxNo, mtxGo, p1to1, InhibPath)
	pt.DefaultParams = params.Params{
		"Path.Learn.Learn":   "false",
		"Path.PathScale.Rel": "0.05",
	}

	bgclass := "VBGInhib"
	net.ConnectLayers(gpePr, gpePr, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, gpeAk, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, stn, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, gpi, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(mtxGo, gpi, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(mtxGo, gpeAk, full, InhibPath).AddClass(bgclass)
	// this doesn't make that much diff -- bit cleaner RT without:
	// net.ConnectLayers(mtxGo, gpePr, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpeAk, mtxGo, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpeAk, mtxNo, full, InhibPath).AddClass(bgclass)

	stnclass := "VSTNExcite"
	net.ConnectLayers(stn, gpePr, full, ForwardPath).AddClass(stnclass)
	net.ConnectLayers(stn, gpeAk, full, ForwardPath).AddClass(stnclass)
	net.ConnectLayers(stn, gpi, full, ForwardPath).AddClass(stnclass)

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
	bglay := "DBG"
	gpi = net.AddGPiLayer4D(prefix+"DGPi", bglay, nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpePr = net.AddGPeLayer4D(prefix+"DGPePr", bglay, nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpePr.SetBuildConfig("GPType", "GPePr")
	gpeAk = net.AddGPeLayer4D(prefix+"DGPeAk", bglay, nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeAk.SetBuildConfig("GPType", "GPeAk")
	stn = net.AddSTNLayer2D(prefix+"DSTN", "DSTNLayer", gpNeurY, gpNeurX)
	mtxGo = net.AddDMatrixLayer(prefix+"DMtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	mtxNo = net.AddDMatrixLayer(prefix+"DMtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	pfp := params.Params{
		"Layer.Inhib.Layer.On": "false",
		"Layer.Inhib.Pool.On":  "false",
	}
	pf = net.AddLayer4D(prefix+"PF", nPoolsY, nPoolsX, nNeurY, 1, SuperLayer)
	pf.DefaultParams = pfp

	mtxGo.SetBuildConfig("OtherMatrixName", mtxNo.Name)
	mtxNo.SetBuildConfig("OtherMatrixName", mtxGo.Name)

	p1to1 := paths.NewPoolOneToOne()
	full := paths.NewFull()

	net.ConnectLayers(mtxNo, gpePr, p1to1, InhibPath)
	pt := net.ConnectLayers(mtxNo, mtxGo, p1to1, InhibPath)
	pt.DefaultParams = params.Params{
		"Path.Learn.Learn":   "false",
		"Path.PathScale.Rel": "0.1",
	}

	bgclass := "DBGInhib"
	net.ConnectLayers(gpePr, gpePr, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, gpeAk, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, stn, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, gpi, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(mtxGo, gpi, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(mtxGo, gpeAk, p1to1, InhibPath).AddClass(bgclass)
	// not much diff with this: basically is an offset that can be learned
	// net.ConnectLayers(mtxGo, gpePr, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpeAk, mtxGo, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpeAk, mtxNo, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpi, pf, p1to1, InhibPath).AddClass(bgclass)

	stnclass := "DSTNExcite"
	net.ConnectLayers(stn, gpePr, full, ForwardPath).AddClass(stnclass)
	net.ConnectLayers(stn, gpeAk, full, ForwardPath).AddClass(stnclass)
	net.ConnectLayers(stn, gpi, full, ForwardPath).AddClass(stnclass)

	pfm := params.Params{
		"Path.Learn.Learn":   "false",
		"Path.Com.GType":     "ModulatoryG",
		"Path.PathScale.Abs": "1",
	}
	pt = net.ConnectLayers(pf, mtxGo, p1to1, ForwardPath).AddClass("PFToDMtx").EmerPath.(*Path)
	pt.DefaultParams = pfm
	pt = net.ConnectLayers(pf, mtxNo, p1to1, ForwardPath).AddClass("PFToDMtx").EmerPath.(*Path)
	pt.DefaultParams = pfm

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
	ly.AddClass("BG")
	return ly
}

// AddBGThalLayer2D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 2D structure
func (net *Network) AddBGThalLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, BGThalLayer)
	ly.AddClass("BG")
	return ly
}

// AddVMatrixLayer adds a Ventral MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (net *Network) AddVMatrixLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DAModTypes) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, MatrixLayer)
	ly.SetBuildConfig("DAMod", da.String())
	ly.AddClass("VSMatrixLayer")
	return ly
}

// AddDMatrixLayer adds a Dorsal MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (net *Network) AddDMatrixLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DAModTypes) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, MatrixLayer)
	ly.SetBuildConfig("DAMod", da.String())
	ly.AddClass("DSMatrixLayer")
	return ly
}

// ConnectToVSMatrix adds a VSMatrixPath from given sending layer to a matrix layer
func (net *Network) ConnectToVSMatrix(send, recv *Layer, pat paths.Pattern) *Path {
	return net.ConnectLayers(send, recv, pat, VSMatrixPath)
}

// ConnectToDSMatrix adds a DSMatrixPath from given sending layer to a matrix layer
func (net *Network) ConnectToDSMatrix(send, recv *Layer, pat paths.Pattern) *Path {
	return net.ConnectLayers(send, recv, pat, DSMatrixPath)
}

// AddGPLayer2D adds a GPLayer of given size, with given name.
// Must set the GPType BuildConfig setting to appropriate GPLayerType
func (net *Network) AddGPeLayer2D(name, class string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, GPLayer)
	ly.AddClass(class)
	return ly
}

// AddGPiLayer2D adds a GPiLayer of given size, with given name.
func (net *Network) AddGPiLayer2D(name, class string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, GPLayer)
	ly.SetBuildConfig("GPType", "GPi")
	ly.AddClass(class)
	return ly
}

// AddSTNLayer2D adds a subthalamic nucleus Layer of given size, with given name.
func (net *Network) AddSTNLayer2D(name, class string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, nNeurY, nNeurX, STNLayer)
	ly.AddClass(class)
	return ly
}

// AddGPLayer4D adds a GPLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (net *Network) AddGPeLayer4D(name, class string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, GPLayer)
	ly.AddClass(class)
	return ly
}

// AddGPiLayer4D adds a GPiLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (net *Network) AddGPiLayer4D(name, class string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, GPLayer)
	ly.SetBuildConfig("GPType", "GPi")
	ly.AddClass(class)
	return ly
}

// AddSTNLayer4D adds a subthalamic nucleus Layer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (net *Network) AddSTNLayer4D(name, class string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, STNLayer)
	ly.AddClass(class)
	return ly
}

// AddVSGatedLayer adds a VSGatedLayer with given number of Y units
// and 2 pools, first one represents JustGated, second is HasGated.
func (net *Network) AddVSGatedLayer(prefix string, nYunits int) *Layer {
	ly := net.AddLayer4D(prefix+"VSGated", 1, 2, nYunits, 1, VSGatedLayer)
	return ly
}
