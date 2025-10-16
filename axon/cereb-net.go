// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/emer/emergent/v2/paths"

// AddCerebNucleus adds cerebellar nucleus layers that learn to cancel
// the given sensory input layer, from which they copy their shape.
// Returns Cout, Cpred layers, with given optional prefix.
func (net *Network) AddCerebellumNucleus(sense *Layer, space float32) (cout, cpred *Layer) {
	name := sense.Name
	shp := sense.Shape
	if shp.NumDims() == 2 {
		cout = net.AddLayer2D(name+"Cout", CerebOutLayer, shp.DimSize(0), shp.DimSize(1))
		cpred = net.AddLayer2D(name+"Cpred", CerebPredLayer, shp.DimSize(0), shp.DimSize(1))
	} else {
		cout = net.AddLayer4D(name+"Cout", CerebOutLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
		cpred = net.AddLayer4D(name+"Cpred", CerebPredLayer, shp.DimSize(0), shp.DimSize(1), shp.DimSize(2), shp.DimSize(3))
	}
	cpred.SetBuildConfig("DriveLayName", name)

	one2one := paths.NewOneToOne()
	net.ConnectLayers(sense, cout, one2one, ForwardPath).AddClass("CerebOutInput")
	net.ConnectLayers(cpred, cout, one2one, InhibPath).AddClass("CerebPredToOut")
	// todo: initial params settings

	cpred.PlaceBehind(cout, space)
	return
}
