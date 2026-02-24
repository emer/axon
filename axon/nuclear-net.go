// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/emer/emergent/v2/paths"

// AddNuclearCNUp adds Nuclear model cerebellar upbound nucleus
// for adaptive filtering of given sensory input layer,
// from which they copy their shape. actEff layer is the efferent
// copy of the action layer, which sends a full modulatory projection.
// actEnv is the default ActionEnv environment timing value in cycles.
func (net *Network) AddNuclearCNUp(sense, actEff *Layer, actEnv int, space float32) (ioUp, cniIOUp, cniUp, cneUp *Layer) {
	name := sense.Name
	shp := sense.Shape
	if shp.NumDims() == 2 {
		ioUp = net.AddLayer2D(name+"IOUp", IOLayer, shp.DimSize(0), shp.DimSize(1))
		cniIOUp = net.AddLayer2D(name+"CNiIOup", CNiIOLayer, shp.DimSize(0), shp.DimSize(1))
		cniUp = net.AddLayer2D(name+"CNiUp", CNiUpLayer, shp.DimSize(0), shp.DimSize(1))
		cneUp = net.AddLayer2D(name+"CNeUp", CNeUpLayer, shp.DimSize(0), shp.DimSize(1))
	} else {
		ioUp = net.AddLayer4D(name+"IOUp", IOLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
		cniIOUp = net.AddLayer4D(name+"CNiIOUp", CNiIOLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
		cniUp = net.AddLayer4D(name+"CNiUp", CNiUpLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
		cneUp = net.AddLayer4D(name+"CNeUp", CNeUpLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
	}
	cniIOUp.SetBuildConfig("IOLayName", ioUp.Name)
	cniUp.SetBuildConfig("IOLayName", ioUp.Name)
	cneUp.SetBuildConfig("IOLayName", ioUp.Name)
	cniIOUp.AddClass("CNLayer", "CNiLayer")
	cniUp.AddClass("CNLayer", "CNiLayer")
	cneUp.AddClass("CNLayer")

	aep := func(ly *LayerParams) {
		ly.Nuclear.ActionEnv = int32(actEnv)
	}
	ioUp.AddDefaultParams(aep)
	cniIOUp.AddDefaultParams(aep)
	cniUp.AddDefaultParams(aep)
	cneUp.AddDefaultParams(aep)

	full := paths.NewFull()
	one2one := paths.NewPoolOneToOne()

	pt := net.ConnectLayers(actEff, ioUp, full, ForwardPath).AddClass("EffToIO")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.SetFixedWts()
		pt.Com.GType = ModulatoryG
	})
	pt = net.ConnectLayers(sense, ioUp, one2one, ForwardPath).AddClass("SenseToIO")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.SetFixedWts()
	})

	pt = net.ConnectLayers(cniIOUp, ioUp, one2one, InhibPath).AddClass("CNiIOToIO")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.SetFixedWts()
	})

	pt = net.ConnectLayers(cniUp, cneUp, one2one, InhibPath).AddClass("CNiToCNeUp")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.SetFixedWts()
	})

	// CNiIO in front, as most important learning
	cniUp.PlaceBehind(cniIOUp, space)
	cneUp.PlaceBehind(cniUp, space)
	ioUp.PlaceBehind(cneUp, space)
	return
}

// AddNuclearCNDn adds Nuclear model cerebellar downbound nucleus
// for forward model learning from given sensory input layer,
// from which they copy their shape. actEff layer is the efferent
// copy of the action layer, which sends a full modulatory projection.
// actEnv is the default ActionEnv environment timing value in cycles.
func (net *Network) AddNuclearCNDn(sense, actEff *Layer, actEnv int, space float32) (ioDn, cniIODn, cneDn *Layer) {
	name := sense.Name
	shp := sense.Shape
	if shp.NumDims() == 2 {
		ioDn = net.AddLayer2D(name+"IODn", IOLayer, shp.DimSize(0), shp.DimSize(1))
		cniIODn = net.AddLayer2D(name+"CNiIODn", CNiIOLayer, shp.DimSize(0), shp.DimSize(1))
		cneDn = net.AddLayer2D(name+"CNeDn", CNeDnLayer, shp.DimSize(0), shp.DimSize(1))
	} else {
		ioDn = net.AddLayer4D(name+"IODn", IOLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
		cniIODn = net.AddLayer4D(name+"CNiIODn", CNiIOLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
		cneDn = net.AddLayer4D(name+"CNeDn", CNeDnLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
	}
	cniIODn.SetBuildConfig("IOLayName", ioDn.Name)
	cneDn.SetBuildConfig("IOLayName", ioDn.Name)
	cniIODn.AddClass("CNLayer", "CNiLayer")
	cneDn.AddClass("CNLayer")

	aep := func(ly *LayerParams) {
		ly.Nuclear.ActionEnv = int32(actEnv)
	}
	ioDn.AddDefaultParams(aep)
	cniIODn.AddDefaultParams(aep)
	cneDn.AddDefaultParams(aep)

	full := paths.NewFull()
	one2one := paths.NewPoolOneToOne()

	pt := net.ConnectLayers(actEff, ioDn, full, ForwardPath).AddClass("EffToIO")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.SetFixedWts()
		pt.Com.GType = ModulatoryG
	})
	pt = net.ConnectLayers(sense, ioDn, one2one, ForwardPath).AddClass("SenseToIO")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.SetFixedWts()
	})

	pt = net.ConnectLayers(cniIODn, ioDn, one2one, InhibPath).AddClass("CNiIOToIO")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.SetFixedWts()
	})

	// CNiIO in front, as most important learning
	cneDn.PlaceBehind(cniIODn, space)
	ioDn.PlaceBehind(cneDn, space)
	return
}
