// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

// HebbPrjn is a simple hebbian learning projection, using the CPCA Hebbian rule
type HebbPrjn struct {
	Prjn            // access as .Prjn
	IncGain float32 `desc:"gain factor on increases relative to decreases -- lower = lower overall weights"`
}

func (pj *HebbPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.IncGain = 0.5
}

func (pj *HebbPrjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

// DWt computes the hebbian weight change
func (pj *HebbPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(AxonLayer).AsAxon()
	rlay := pj.Recv.(AxonLayer).AsAxon()
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		if sn.AvgS < pj.Learn.XCal.LrnThr && sn.AvgM < pj.Learn.XCal.LrnThr {
			continue
		}
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			ract := rn.AvgSLrn
			sact := sn.AvgSLrn
			wt := sy.LWt
			dwt := ract * (pj.IncGain*sact*(1-wt) - (1-sact)*wt)
			sy.DWt += pj.Learn.Lrate * dwt
		}
	}
}
