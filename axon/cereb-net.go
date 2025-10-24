// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/emer/emergent/v2/paths"

// AddCerebNucleus adds cerebellar nucleus layers that learn to cancel
// the given sensory input layer, from which they copy their shape.
// Returns cneUp, CNiPred layers, with given optional prefix.
func (net *Network) AddCerebellumNucleus(sense *Layer, space float32) (cneUp, cniPred *Layer) {
	name := sense.Name
	predName := name + "CNiPred"
	shp := sense.Shape
	if shp.NumDims() == 2 {
		cneUp = net.AddLayer2D(name+"CNeUp", CNeUpLayer, shp.DimSize(0), shp.DimSize(1))
		cniPred = net.AddLayer2D(predName, CNiPredLayer, shp.DimSize(0), shp.DimSize(1))
	} else {
		cneUp = net.AddLayer4D(name+"CNeUp", CNeUpLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
		cniPred = net.AddLayer4D(predName, CNiPredLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
	}
	cniPred.SetBuildConfig("DriveLayName", name)

	one2one := paths.NewOneToOne()
	net.ConnectLayers(sense, cneUp, one2one, ForwardPath).AddClass("CNeUpInput")
	net.ConnectLayers(cniPred, cneUp, one2one, CNiPredToOutPath).AddClass("CNiPredToOut")

	cneUp.SetBuildConfig("PredLayName", predName)
	cneUp.SetBuildConfig("SenseLayName", name)

	cniPred.PlaceBehind(cneUp, space)
	return
}
