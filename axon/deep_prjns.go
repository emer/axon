// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

//gosl: start deep_prjns

//gosl: end deep_prjns

// GPU TODO: need to add special GPU function -- now just doing in CPU

// todo: use GMod for this?  Ignore until last cycle!?

// SendCtxtGe sends the full Burst activation from sending neuron index si,
// to integrate CtxtGe excitatory conductance on receivers
func (pj *Prjn) SendCtxtGe(si int, burst float32) {
	/*
		scdb := burst * pj.Params.GScale.Scale
		nc := pj.SendConN[si]
		st := pj.SendConIdxStart[si]
		syns := pj.Syns[st : st+nc]
		scons := pj.SendConIdx[st : st+nc]
		for ci := range syns {
			ri := scons[ci]
			pj.GVals[ri].GRaw += scdb * syns[ci].Wt
		}
	*/
}

// RecvCtxtGeInc increments the receiver's CtxtGe from that of all the projections
func (pj *Prjn) RecvCtxtGeInc() {
	/*
		rlay := pj.Recv.(AxonLayer).AsAxon()
		for ri := range rlay.Neurons {
			nrn := &rlay.Neurons[ri]
			nrn.CtxtGe += pj.GVals[ri].GRaw
			pj.GVals[ri].GRaw = 0
		}
	*/
}
