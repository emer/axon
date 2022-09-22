// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/relpos"
)

// AddBLALayers adds two BLA layers, acquisition / extinction / D1 / D2, for positive or negative valence
func AddBLALayers(nt *axon.Network, prefix string, pos bool, nUs, unY, unX int, rel relpos.Relations, space float32) (acq, ext axon.AxonLayer) {
	if pos {
		d1 := &BLALayer{}
		nt.AddLayerInit(d1, prefix+"PosAcqD1", []int{1, nUs, unY, unX}, emer.Hidden)
		d1.DaMod.DAR = D1R
		d2 := &BLALayer{}
		nt.AddLayerInit(d2, prefix+"PosExtD2", []int{1, nUs, unY, unX}, emer.Hidden)
		d2.DaMod.DAR = D2R
		acq = d1
		ext = d2
	} else {
		d1 := &BLALayer{}
		nt.AddLayerInit(d1, prefix+"NegExtD1", []int{1, nUs, unY, unX}, emer.Hidden)
		d1.DaMod.DAR = D1R
		d2 := &BLALayer{}
		nt.AddLayerInit(d2, prefix+"NegAcqD2", []int{1, nUs, unY, unX}, emer.Hidden)
		d2.DaMod.DAR = D2R
		acq = d2
		ext = d1
	}
	if rel == relpos.Behind {
		ext.SetRelPos(relpos.Rel{Rel: rel, Other: acq.Name(), XAlign: relpos.Left, Space: space})
	} else {
		ext.SetRelPos(relpos.Rel{Rel: rel, Other: acq.Name(), YAlign: relpos.Front, Space: space})
	}
	acq.SetClass("BLA")
	ext.SetClass("BLA")
	return
}
