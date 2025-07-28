// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/paths"
)

// AddVentralBG adds Ventral Basal Ganglia layers, using the PCore Pallidal Core
// framework where GPe plays a central role.
// Returns VMatrixGo, VMatrixNo, VGPePr, VGPeAk, VSTN, VGPi layers,
// with given optional prefix.
// Only the Matrix has pool-based 4D shape by default -- use pool for "role" like
// elements where matches need to be detected.
// All GP / STN layers have gpNeur neurons.
// Appropriate connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical).
func (net *Network) AddVentralBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (matrixGo, matrixNo, gpePr, gpeAk, stn, gpi *Layer) {
	bglay := "VBG"
	gpi = net.AddGPiLayer2D(prefix+"VGPi", bglay, gpNeurY, gpNeurX)
	gpePr = net.AddGPeLayer2D(prefix+"VGPePr", bglay, gpNeurY, gpNeurX)
	gpePr.SetBuildConfig("GPType", "GPePr")
	gpeAk = net.AddGPeLayer2D(prefix+"VGPeAk", bglay, gpNeurY, gpNeurX)
	gpeAk.SetBuildConfig("GPType", "GPeAk")
	stn = net.AddSTNLayer2D(prefix+"VSTN", "VSTNLayer", gpNeurY, gpNeurX)
	matrixGo = net.AddVMatrixLayer(prefix+"VMatrixGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	matrixNo = net.AddVMatrixLayer(prefix+"VMatrixNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)

	matrixGo.SetBuildConfig("OtherName", matrixNo.Name)
	matrixNo.SetBuildConfig("OtherName", matrixGo.Name)

	mp := func(ly *LayerParams) {
		ly.Striatum.IsVS.SetBool(true)
		ly.Inhib.ActAvg.Nominal = 0.1 / float32(nPoolsX*nPoolsY)
		ly.Acts.Dend.ModACh.SetBool(true)
	}
	matrixGo.AddDefaultParams(mp)
	matrixNo.AddDefaultParams(mp)

	full := paths.NewFull()
	p1to1 := paths.NewPoolOneToOne()

	net.ConnectLayers(matrixNo, gpePr, full, InhibPath)
	pt := net.ConnectLayers(matrixNo, matrixGo, p1to1, InhibPath)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Learn.Learn.SetBool(false)
		pt.PathScale.Rel = 0.05
	})

	bgclass := "VBGInhib"
	net.ConnectLayers(gpePr, gpePr, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, gpeAk, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, stn, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, gpi, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(matrixGo, gpi, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(matrixGo, gpeAk, full, InhibPath).AddClass(bgclass)
	// this doesn't make that much diff -- bit cleaner RT without:
	// net.ConnectLayers(matrixGo, gpePr, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpeAk, matrixGo, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpeAk, matrixNo, full, InhibPath).AddClass(bgclass)

	stnclass := "VSTNExcite"
	net.ConnectLayers(stn, gpePr, full, ForwardPath).AddClass(stnclass)
	net.ConnectLayers(stn, gpeAk, full, ForwardPath).AddClass(stnclass)
	net.ConnectLayers(stn, gpi, full, ForwardPath).AddClass(stnclass)

	gpeAk.PlaceBehind(gpi, space)
	gpePr.PlaceRightOf(gpeAk, space)
	stn.PlaceRightOf(gpi, space)
	matrixGo.PlaceBehind(gpePr, space)
	matrixNo.PlaceRightOf(matrixGo, space)

	return
}

// TODO: need to integrate Patch into net signal, send that to corresponding dorsal pools.
// The PF signal as GModSyn is also probably pretty variable relative to actual activity --
// would be good to have longer time-average signal.
// Probably just put in Pool directly instead of having these synaptic signals, or trying to
// stick them onto global.

// AddDorsalBG adds Dorsal Basal Ganglia layers, using the PCore Pallidal Core
// framework where GPe plays a central role.
// Returns DMatrixGo, DMatrixNo, DGPePr, DGPeAk, DSTN, DGPi, PF layers, with given optional prefix.
// Makes 4D pools throughout the GP layers, with Pools representing separable
// gating domains, i.e., action domains.
// All GP / STN layers have gpNeur neurons.
// Appropriate PoolOneToOne connections are made between layers, using standard styles.
// space is the spacing between layers (2 typical)
func (net *Network) AddDorsalBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (matrixGo, matrixNo, patchD1, patchD2, gpePr, gpeAk, stn, gpi, pf *Layer) {
	bglay := "DBG"
	gpi = net.AddGPiLayer4D(prefix+"SNr-GPi", bglay, nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpePr = net.AddGPeLayer4D(prefix+"DGPePr", bglay, nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpePr.SetBuildConfig("GPType", "GPePr")
	gpeAk = net.AddGPeLayer4D(prefix+"DGPeAk", bglay, nPoolsY, nPoolsX, gpNeurY, gpNeurX)
	gpeAk.SetBuildConfig("GPType", "GPeAk")
	stn = net.AddSTNLayer2D(prefix+"DSTN", "DSTNLayer", gpNeurY, gpNeurX)
	matrixGo = net.AddDMatrixLayer(prefix+"DMatrixGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1Mod)
	matrixNo = net.AddDMatrixLayer(prefix+"DMatrixNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2Mod)
	patchD1, patchD2 = net.AddDSPatchLayers(prefix, nPoolsY, nPoolsX, nNeurY, nNeurX, space)

	pfp := func(ly *LayerParams) {
		ly.Inhib.Layer.On.SetBool(false)
		ly.Inhib.Pool.On.SetBool(false)
	}
	pf = net.AddLayer4D(prefix+"PF", SuperLayer, nPoolsY, nPoolsX, nNeurY, 1)
	pf.AddDefaultParams(pfp)

	matrixGo.SetBuildConfig("OtherName", matrixNo.Name)
	matrixNo.SetBuildConfig("OtherName", matrixGo.Name)
	matrixGo.SetBuildConfig("PFName", pf.Name)
	matrixNo.SetBuildConfig("PFName", pf.Name)
	matrixGo.SetBuildConfig("PatchD1Name", patchD1.Name)
	matrixGo.SetBuildConfig("PatchD2Name", patchD2.Name)
	matrixNo.SetBuildConfig("PatchD1Name", patchD1.Name)
	matrixNo.SetBuildConfig("PatchD2Name", patchD2.Name)

	patchD1.SetBuildConfig("OtherName", patchD2.Name)
	patchD2.SetBuildConfig("OtherName", patchD1.Name)
	patchD1.SetBuildConfig("PFName", pf.Name)
	patchD2.SetBuildConfig("PFName", pf.Name)

	p1to1 := paths.NewPoolOneToOne()
	full := paths.NewFull()

	net.ConnectLayers(matrixNo, gpePr, p1to1, InhibPath)
	pt := net.ConnectLayers(matrixNo, matrixGo, p1to1, InhibPath)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Learn.Learn.SetBool(false)
		pt.PathScale.Rel = 0.1
	})

	bgclass := "DBGInhib"
	net.ConnectLayers(gpePr, gpePr, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, gpeAk, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, stn, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpePr, gpi, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(matrixGo, gpi, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(matrixGo, gpeAk, p1to1, InhibPath).AddClass(bgclass)
	// not much diff with this: basically is an offset that can be learned
	// net.ConnectLayers(matrixGo, gpePr, full, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpeAk, matrixGo, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpeAk, matrixNo, p1to1, InhibPath).AddClass(bgclass)
	net.ConnectLayers(gpi, pf, p1to1, InhibPath).AddClass(bgclass)

	stnclass := "DSTNExcite"
	net.ConnectLayers(stn, gpePr, full, ForwardPath).AddClass(stnclass)
	net.ConnectLayers(stn, gpeAk, full, ForwardPath).AddClass(stnclass)
	net.ConnectLayers(stn, gpi, full, ForwardPath).AddClass(stnclass)

	gpePr.PlaceBehind(gpi, space)
	gpeAk.PlaceRightOf(gpePr, space)
	stn.PlaceRightOf(gpi, space)

	matrixGo.PlaceBehind(gpePr, space)
	matrixNo.PlaceRightOf(matrixGo, space)
	patchD1.PlaceBehind(matrixGo, space)

	return
}

// AddBGThalLayer4D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 4D structure, with Pools representing separable gating domains.
func (net *Network) AddBGThalLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, BGThalLayer, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ly.AddClass("BG")
	return ly
}

// AddBGThalLayer2D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 2D structure
func (net *Network) AddBGThalLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, BGThalLayer, nNeurY, nNeurX)
	ly.AddClass("BG")
	return ly
}

// AddVMatrixLayer adds a Ventral MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (net *Network) AddVMatrixLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DAModTypes) *Layer {
	ly := net.AddLayer4D(name, MatrixLayer, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ly.SetBuildConfig("DAMod", da.String())
	ly.AddClass("VSMatrixLayer")
	return ly
}

// AddDMatrixLayer adds a Dorsal MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (net *Network) AddDMatrixLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DAModTypes) *Layer {
	ly := net.AddLayer4D(name, MatrixLayer, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ly.SetBuildConfig("DAMod", da.String())
	ly.AddClass("DSMatrixLayer")
	return ly
}

// ConnectToVSMatrix adds a VSMatrixPath from given sending layer to a matrix layer
func (net *Network) ConnectToVSMatrix(send, recv *Layer, pat paths.Pattern) *Path {
	return net.ConnectLayers(send, recv, pat, VSMatrixPath)
}

// ConnectToDSMatrix adds a DSMatrixPath from given sending layer
// to matrix Go, No layers, adding given classes if present.
func (net *Network) ConnectToDSMatrix(send, matrixGo, matrixNo *Layer, pat paths.Pattern, class ...string) (*Path, *Path) {
	gp := net.ConnectLayers(send, matrixGo, pat, DSMatrixPath)
	np := net.ConnectLayers(send, matrixNo, pat, DSMatrixPath)
	if len(class) > 0 {
		gp.AddClass(class...)
		np.AddClass(class...)
	}
	return gp, np
}

// AddGPLayer2D adds a GPLayer of given size, with given name.
// Must set the GPType BuildConfig setting to appropriate GPLayerType
func (net *Network) AddGPeLayer2D(name, class string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, GPLayer, nNeurY, nNeurX)
	ly.AddClass(class)
	return ly
}

// AddGPiLayer2D adds an SNr / GPiLayer of given size, with given name.
func (net *Network) AddGPiLayer2D(name, class string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, GPLayer, nNeurY, nNeurX)
	ly.Doc = "SNr (substantia nigra pars reticulata) / GPi (globus pallidus interna) are the major output pathways from BG, with tonic levels of activity that can be inhibited to disinhibit the downstream targets of BG output"
	ly.SetBuildConfig("GPType", "GPi")
	ly.AddClass(class)
	return ly
}

// AddSTNLayer2D adds a subthalamic nucleus Layer of given size, with given name.
func (net *Network) AddSTNLayer2D(name, class string, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer2D(name, STNLayer, nNeurY, nNeurX)
	ly.AddClass(class)
	return ly
}

// AddGPLayer4D adds a GPLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (net *Network) AddGPeLayer4D(name, class string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, GPLayer, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ly.AddClass(class)
	return ly
}

// AddGPiLayer4D adds a GPiLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (net *Network) AddGPiLayer4D(name, class string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, GPLayer, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ly.Doc = "SNr (substantia nigra pars reticulata) / GPi (globus pallidus interna) are the major output pathways from BG, with tonic levels of activity that can be inhibited to disinhibit the downstream targets of BG output"
	ly.SetBuildConfig("GPType", "GPi")
	ly.AddClass(class)
	return ly
}

// AddSTNLayer4D adds a subthalamic nucleus Layer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (net *Network) AddSTNLayer4D(name, class string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := net.AddLayer4D(name, STNLayer, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ly.AddClass(class)
	return ly
}

// AddVSGatedLayer adds a VSGatedLayer with given number of Y units
// and 2 pools, first one represents JustGated, second is HasGated.
func (net *Network) AddVSGatedLayer(prefix string, nYunits int) *Layer {
	ly := net.AddLayer4D(prefix+"VSGated", VSGatedLayer, 1, 2, nYunits, 1)
	return ly
}

// AddDSPatchLayers adds DSPatch (Pos, D1, D2)
func (nt *Network) AddDSPatchLayers(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32) (d1, d2 *Layer) {
	d1 = nt.AddLayer4D(prefix+"DSPatchD1", DSPatchLayer, nPoolsY, nPoolsX, nNeurY, nNeurX)
	d1.SetBuildConfig("DAMod", "D1Mod")
	d1.SetBuildConfig("Valence", "Positive")
	d1.Doc = "DSPatch are dorsal striatum patch (striosome) neurons that provide a local critic reward-prediction-error (RPE) signal for the corresponding pool of Matrix neurons. D1 = learns from DA bursts."
	d2 = nt.AddLayer4D(prefix+"DSPatchD2", DSPatchLayer, nPoolsY, nPoolsX, nNeurY, nNeurX)
	d2.SetBuildConfig("DAMod", "D2Mod")
	d2.SetBuildConfig("Valence", "Positive")
	d2.Doc = "DSPatch are dorsal striatum patch (striosome) neurons that provide a local critic reward-prediction-error (RPE) signal for the corresponding pool of Matrix neurons. D2 = learns from DA dips."

	d2.PlaceBehind(d1, space)
	return
}

// ConnectToDSPatch adds a DSPatchPath from given sending layer
// to DSPatchD1, D2 layers, adding given classes if present.
func (nt *Network) ConnectToDSPatch(send, dspD1, dspD2 *Layer, pat paths.Pattern, class ...string) (*Path, *Path) {
	d1 := nt.ConnectLayers(send, dspD1, pat, DSPatchPath)
	d2 := nt.ConnectLayers(send, dspD2, pat, DSPatchPath)
	if len(class) > 0 {
		d1.AddClass(class...)
		d2.AddClass(class...)
	}
	return d1, d2
}
