// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

// HebbPrjn is a simple hebbian learning projection, using the CPCA Hebbian rule.
// Note: when used with inhibitory projections, requires Learn.Trace.SubMean = 1
type HebbPrjn struct {
	Prjn            // access as .Prjn
	IncGain float32 `desc:"gain factor on increases relative to decreases -- lower = lower overall weights"`
}

func (pj *HebbPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.IncGain = 0.5
	pj.Learn.Trace.SubMean = 1 // this is critical!
}

func (pj *HebbPrjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

// DWt computes the hebbian weight change
func (pj *HebbPrjn) DWt(ctime *Time) {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(AxonLayer).AsAxon()
	rlay := pj.Recv.(AxonLayer).AsAxon()
	lr := pj.Learn.Lrate.Eff
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			ract := rn.CaP
			sact := sn.CaP
			wt := sy.LWt
			dwt := ract * (pj.IncGain*sact*(1-wt) - (1-sact)*wt)
			sy.DWt += lr * dwt
		}
	}
}
