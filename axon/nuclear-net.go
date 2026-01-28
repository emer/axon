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
		ioUp = net.AddLayer2D(name+"IO", IOLayer, shp.DimSize(0), shp.DimSize(1))
		cniIOUp = net.AddLayer2D(name+"CNiIO", CNiIOLayer, shp.DimSize(0), shp.DimSize(1))
		cniUp = net.AddLayer2D(name+"CNiUp", CNiUpLayer, shp.DimSize(0), shp.DimSize(1))
		cneUp = net.AddLayer2D(name+"CNeUp", CNeUpLayer, shp.DimSize(0), shp.DimSize(1))
	} else {
		ioUp = net.AddLayer4D(name+"IO", IOLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
		cniIOUp = net.AddLayer4D(name+"CNiIO", CNiIOLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
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

	pt = net.ConnectLayers(cniUp, cneUp, one2one, CNeUpPath).AddClass("CNiToCNe")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Com.GType = InhibitoryG
	})

	// CNiIO in front, as most important learning
	cniUp.PlaceBehind(cniIOUp, space)
	cneUp.PlaceBehind(cniUp, space)
	ioUp.PlaceBehind(cneUp, space)
	return
}
