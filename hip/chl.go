// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hip

import (
	"github.com/emer/axon/axon"
	"github.com/goki/mat32"
)

// Contrastive Hebbian Learning (CHL) parameters
type CHLParams struct {
	On      bool    `desc:"if true, use CHL learning instead of standard XCAL learning -- allows easy exploration of CHL vs. XCAL"`
	Hebb    float32 `def:"0.001" min:"0" max:"1" desc:"amount of hebbian learning (should be relatively small, can be effective at .0001)"`
	Err     float32 `def:"0.999" min:"0" max:"1" inactive:"+" desc:"amount of error driven learning, automatically computed to be 1-Hebb"`
	MinusQ1 bool    `desc:"if true, use SpkSt1 as the minus phase -- otherwise ActM"`
	SAvgCor float32 `def:"0.4:0.8" min:"0" max:"1" desc:"proportion of correction to apply to sending average activation for hebbian learning component (0=none, 1=all, .5=half, etc)"`
	SAvgThr float32 `def:"0.001" min:"0" desc:"threshold of sending average activation below which learning does not occur (prevents learning when there is no input)"`
}

func (ch *CHLParams) Defaults() {
	ch.On = true
	ch.Hebb = 0.001
	ch.SAvgCor = 0.4
	ch.SAvgThr = 0.001
	ch.Update()
}

func (ch *CHLParams) Update() {
	ch.Err = 1 - ch.Hebb
}

// MinusAct returns the minus-phase activation to use based on settings (ActM vs. SpkSt1)
func (ch *CHLParams) MinusAct(actM, actQ1 float32) float32 {
	if ch.MinusQ1 {
		return actQ1
	}
	return actM
}

// HebbDWt computes the hebbian DWt value from sending, recv acts, savgCor, and linear Wt
func (ch *CHLParams) HebbDWt(sact, ract, savgCor, linWt float32) float32 {
	return ract * (sact*(savgCor-linWt) - (1-sact)*linWt)
}

// ErrDWt computes the error-driven DWt value from sending,
// recv acts in both phases, and linear Wt, which is used
// for soft weight bounding (always applied here, separate from hebbian
// which has its own soft weight bounding dynamic).
func (ch *CHLParams) ErrDWt(sactP, sactM, ractP, ractM, linWt float32) float32 {
	err := (ractP * sactP) - (ractM * sactM)
	if err > 0 {
		err *= (1 - linWt)
	} else {
		err *= linWt
	}
	return err
}

// DWt computes the overall dwt from hebbian and error terms
func (ch *CHLParams) DWt(hebb, err float32) float32 {
	return ch.Hebb*hebb + ch.Err*err
}

////////////////////////////////////////////////////////////////////
//  CHLPrjn

// hip.CHLPrjn is a Contrastive Hebbian Learning (CHL) projection,
// based on basic rate-coded axon.Prjn, that implements a
// pure CHL learning rule, which works better in the hippocampus.
type CHLPrjn struct {
	axon.Prjn           // access as .Prjn
	CHL       CHLParams `view:"inline" desc:"parameters for CHL learning -- if CHL is On then WtSig.SoftBound is automatically turned off -- incompatible"`
}

func (pj *CHLPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.CHL.Defaults()
}

func (pj *CHLPrjn) UpdateParams() {
	pj.CHL.Update()
	// if pj.CHL.On {
	// 	pj.Prjn.Learn.WtSig.SoftBound = false
	// }
	pj.Prjn.UpdateParams()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) -- on sending projections
// CHL version supported if On
func (pj *CHLPrjn) DWt(ltime *axon.Time) {
	if !pj.Learn.Learn {
		return
	}
	if pj.CHL.On {
		pj.DWtCHL(ltime)
	} else {
		pj.Prjn.DWt(ltime)
	}
}

// SAvgCor computes the sending average activation, corrected according to the SAvgCor
// correction factor (typically makes layer appear more sparse than it is)
func (pj *CHLPrjn) SAvgCor(slay *axon.Layer) float32 {
	savg := .5 + pj.CHL.SAvgCor*(slay.Inhib.ActAvg.Init-0.5)
	savg = mat32.Max(pj.CHL.SAvgThr, savg) // keep this computed value within bounds
	return 0.5 / savg
}

// DWtCHL computes the weight change (learning) for CHL
func (pj *CHLPrjn) DWtCHL(ltime *axon.Time) {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(axon.AxonLayer).AsAxon()
	rlay := pj.Recv.(axon.AxonLayer).AsAxon()
	if slay.Pools[0].ActP.Avg < pj.CHL.SAvgThr { // inactive, no learn
		return
	}
	lr := pj.Learn.Lrate.Eff
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		snActM := pj.CHL.MinusAct(sn.ActM, sn.SpkSt1)

		savgCor := pj.SAvgCor(slay)

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			rnActM := pj.CHL.MinusAct(rn.ActM, rn.SpkSt1)

			hebb := pj.CHL.HebbDWt(sn.ActP, rn.ActP, savgCor, sy.LWt)
			err := pj.CHL.ErrDWt(sn.ActP, snActM, rn.ActP, rnActM, sy.LWt)

			dwt := pj.CHL.DWt(hebb, err)
			sy.DWt += rn.RLrate * lr * dwt
		}
	}
}
