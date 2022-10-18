// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
)

// ThalLayer is a thalamus layer, which can be an MD mediodorsal thalamus
// or a VM / VL / VA ventral thalamic nucleus.
type ThalLayer struct {
	axon.Layer
}

var KiT_ThalLayer = kit.Types.AddType(&ThalLayer{}, LayerProps)

func (ly *ThalLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.Decay.AHP = 0
	ly.Learn.RLrate.SigmoidMin = 1 // don't use!
	ly.Typ = Thal
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *ThalLayer) UpdateParams() {
	ly.Layer.UpdateParams()
}

func (ly *ThalLayer) Class() string {
	return "Thal " + ly.Cls
}
