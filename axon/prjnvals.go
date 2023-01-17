// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/emer/emergent/ringidx"

//gosl: hlsl prjnvals
// #include "ringidx.hlsl"
//gosl: end prjnvals

//gosl: start prjnvals

// PrjnGVals contains projection-level conductance values,
// integrated by prjn before being integrated at the neuron level,
// which enables the neuron to perform non-linear integration as needed.
type PrjnGVals struct {
	GRaw float32 `desc:"raw conductance received from senders = current raw spiking drive"`
	GSyn float32 `desc:"time-integrated total synaptic conductance, with an instantaneous rise time from each spike (in GeRaw) and exponential decay"`
}

func (pv *PrjnGVals) Init() {
	pv.GRaw = 0
	pv.GSyn = 0
}

type PrjnVals struct {
	Gidx ringidx.FIx `inactive:"+" desc:"ring (circular) index for GBuf buffer of synaptically delayed conductance increments.  The current time is always at the zero index, which is read and then shifted.  Len is delay+1."`
}

//gosl: end prjnvals
