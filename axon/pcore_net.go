// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
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
func (nt *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpeOut, gpeIn, gpeTA, stnp, stns, gpi AxonLayer) {
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
func (nt *Network) AddBG4D(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX, gpNeurY, gpNeurX int, space float32) (mtxGo, mtxNo, gpeOut, gpeIn, gpeTA, stnp, stns, gpi AxonLayer) {
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

	return
}

// AddCINLayer adds a RSalienceLayer unsigned reward salience coding ACh layer
// which sends ACh to given Matrix Go and No layers (names), and is default located
// to the right of the MtxNo layer with given spacing.
// CIN is a cholinergic interneuron interspersed in the striatum that shows these
// response properties and modulates learning in the striatum around US and CS events.
// If other ACh modulation is needed, a global RSalienceLayer can be used.
func (nt *Network) AddCINLayer(name, mtxGo, mtxNo string, space float32) *Layer {
	cin := nt.AddRSalienceAChLayer(name)
	cin.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxNo, YAlign: relpos.Front, Space: space})
	return cin
}

// AddThalLayer4D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 4D structure, with Pools representing separable gating domains.
func (nt *Network) AddThalLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.LayerType(VThalLayer))
	ly.SetClass("BG")
	return ly
}

// AddThalLayer2D adds a BG gated thalamus (e.g., VA/VL/VM, MD) Layer
// of given size, with given name.
// This version has a 2D structure
func (nt *Network) AddThalLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.LayerType(VThalLayer))
	ly.SetClass("BG")
	return ly
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (nt *Network) AddMatrixLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DAModTypes) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.LayerType(MatrixLayer))
	ly.SetBuildConfig("DAMod", da.String())
	ly.SetClass("BG")
	return ly
}

// ConnectToMatrix adds a MatrixPrjn from given sending layer to a matrix layer
func (nt *Network) ConnectToMatrix(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayers(send, recv, pat, emer.PrjnType(MatrixPrjn))
}

// AddGPLayer2D adds a GPLayer of given size, with given name.
// Must set the GPType BuildConfig setting to appropriate GPLayerType
func (nt *Network) AddGPeLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.LayerType(GPLayer))
	ly.SetClass("BG")
	return ly
}

// AddGPiLayer2D adds a GPiLayer of given size, with given name.
func (nt *Network) AddGPiLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.LayerType(GPLayer))
	ly.SetBuildConfig("GPType", "GPi")
	ly.SetClass("BG")
	return ly
}

// AddSTNLayer2D adds a subthalamic nucleus Layer of given size, with given name.
func (nt *Network) AddSTNLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.LayerType(STNLayer))
	ly.SetClass("BG")
	return ly
}

// AddGPLayer4D adds a GPLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (nt *Network) AddGPeLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.LayerType(GPLayer))
	ly.SetClass("BG")
	return ly
}

// AddGPiLayer4D adds a GPiLayer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (nt *Network) AddGPiLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.LayerType(GPLayer))
	ly.SetClass("BG")
	return ly
}

// AddSTNLayer4D adds a subthalamic nucleus Layer of given size, with given name.
// Makes a 4D structure with Pools representing separable gating domains.
func (nt *Network) AddSTNLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.LayerType(STNLayer))
	ly.SetClass("BG")
	return ly
}

// AddPTLayer2D adds a PTLayer of given size, with given name.
func (nt *Network) AddPTLayer2D(name string, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.LayerType(PTMaintLayer))
	return ly
}

// AddPTLayer4D adds a PTLayer of given size, with given name.
func (nt *Network) AddPTLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.LayerType(PTMaintLayer))
	return ly
}

// ConnectPTSelf adds a Self (Lateral) projection within a PT layer,
// which supports active maintenance, with a class of PTSelfMaint
func (nt *Network) ConnectPTSelf(ly emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.LateralConnectLayer(ly, pat).SetClass("PTSelfMaint")
}
