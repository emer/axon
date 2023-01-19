// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hip

import (
	"github.com/emer/axon/axon"
)

// hip.EcCa1Prjn is for EC <-> CA1 projections, to perform error-driven
// learning of this encoder pathway according to the ThetaPhase algorithm
// uses Contrastive Hebbian Learning (CHL) on ActP - SpkSt1
// Q1: ECin -> CA1 -> ECout       : SpkSt1 = minus phase for auto-encoder
// Q2, 3: CA3 -> CA1 -> ECout     : ActM = minus phase for recall
// Q4: ECin -> CA1, ECin -> ECout : ActP = plus phase for everything
type EcCa1Prjn struct {
	axon.Prjn // access as .Prjn
}

func (pj *EcCa1Prjn) Defaults() {
	pj.Prjn.Defaults()
}

func (pj *EcCa1Prjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) -- on sending projections
// Delta version
func (pj *EcCa1Prjn) DWt(ctxt *axon.Context) {
	if !pj.Params.Learn.Learn {
		return
	}
	slay := pj.Send.(axon.AxonLayer).AsAxon()
	rlay := pj.Recv.(axon.AxonLayer).AsAxon()
	lr := pj.Params.Learn.LRate.Eff
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SendConN[si])
		st := int(pj.SendConIdxStart[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SendConIdx[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			err := pj.Params.Learn.CHLdWt(sn.CaP, sn.SpkSt1, rn.CaP, rn.SpkSt1)
			// err := (sn.ActP * rn.ActP) - (sn.SpkSt1 * rn.SpkSt1)
			if err > 0 {
				err *= (1 - sy.LWt)
			} else {
				err *= sy.LWt
			}
			sy.DWt += lr * err // rn.RLRate -- doesn't make sense here, b/c St1
		}
	}
}
