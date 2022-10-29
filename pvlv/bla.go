// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/rl"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// BLAParams has parameters for basolateral amygdala
type BLAParams struct {
	NoDALrate float32 `desc:"baseline learning rate without any dopamine"`
	NegLrate  float32 `desc:"negative DWt learning rate multiplier -- weights go down much more slowly than up -- extinction is separate learning in extinction layer"`
}

func (bp *BLAParams) Defaults() {
	bp.NoDALrate = 0.0
	bp.NegLrate = 0.1
}

// BLALayer represents a basolateral amygdala layer
type BLALayer struct {
	rl.Layer
	DaMod DaModParams `view:"inline" desc:"dopamine modulation parameters"`
	BLA   BLAParams   `view:"inline" desc:"special BLA parameters"`
}

var KiT_BLALayer = kit.Types.AddType(&BLALayer{}, axon.LayerProps)

func (ly *BLALayer) Defaults() {
	ly.Layer.Defaults()
	ly.DaMod.Defaults()
	ly.BLA.Defaults()
}

func (ly *BLALayer) GFmSpike(ltime *axon.Time) {
	ly.GFmSpikePrjn(ltime)
	da := ly.DaMod.Gain(ly.DA)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.GFmSpikeNeuron(ltime, ni, nrn)
		daEff := da * nrn.CaSpkM // da effect interacts with spiking
		nrn.GeRaw += daEff
		nrn.GeSyn += ly.Act.Dt.GeSynFmRawSteady(daEff)
		ly.GFmRawSynNeuron(ltime, ni, nrn)
	}
}

func (ly *BLALayer) PlusPhase(ltime *axon.Time) {
	ly.Layer.PlusPhase(ltime)
	lrmod := ly.BLA.NoDALrate + mat32.Abs(ly.DA)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		mlr := ly.Learn.RLrate.RLrateSigDeriv(nrn.CaSpkP, ly.ActAvg.CaSpkP.Max)
		dlr := ly.Learn.RLrate.RLrateDiff(nrn.CaSpkP, nrn.SpkPrv) // delta on previous
		if nrn.CaSpkP-nrn.SpkPrv < 0 {
			dlr *= ly.BLA.NegLrate
		}
		nrn.RLrate = mlr * dlr * lrmod
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  BLAPrjn

// BLAPrjn implements the PVLV BLA learning rule:
// dW = X_t-1 * (Y_t - Y_t-1)
// The recv delta is across trials, where the US should activate on trial
// boundary, to enable sufficient time for gating through to OFC, so
// BLA initially learns based on US present - US absent.
// It can also learn based on CS onset if there is a prior CS that predicts that.
type BLAPrjn struct {
	axon.Prjn
}

var KiT_BLAPrjn = kit.Types.AddType(&BLAPrjn{}, axon.PrjnProps)

func (pj *BLAPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.Mean = 0.1
	pj.SWt.Init.Var = 0.05
	pj.SWt.Init.Sym = false
	pj.Learn.Trace.Tau = 1
	pj.Learn.Trace.Update()
}

func (pj *BLAPrjn) SendSynCa(ltime *axon.Time) {
	return
}

func (pj *BLAPrjn) RecvSynCa(ltime *axon.Time) {
	return
}

// DWt computes the weight change (learning) for BLA projections
func (pj *BLAPrjn) DWt(ltime *axon.Time) {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(axon.AxonLayer).AsAxon()
	rlay := pj.Recv.(axon.AxonLayer).AsAxon()
	lr := pj.Learn.Lrate.Eff
	for si := range slay.Neurons {
		sact := slay.Neurons[si].SpkPrv
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		for ci := range syns {
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			sy := &syns[ci]
			// not using the synaptic trace (yet)
			// kp.CurCa(ctime, sy.CaUpT, sy.CaM, sy.CaP, sy.CaD) // always update
			sy.Tr = pj.Learn.Trace.TrFmCa(sy.Tr, sact)
			if sy.Wt == 0 { // failed con, no learn
				continue
			}
			err := sy.Tr * (rn.CaSpkP - rn.SpkPrv)
			// sb immediately -- enters into zero sum
			if err > 0 {
				err *= (1 - sy.LWt)
			} else {
				err *= sy.LWt
			}
			sy.DWt += rn.RLrate * lr * err
		}
	}
}

// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
func (pj *BLAPrjn) WtFmDWt(ltime *axon.Time) {
	if !pj.Learn.Learn {
		return
	}
	for si := range pj.Syns {
		sy := &pj.Syns[si]
		if sy.DWt != 0 {
			sy.Wt += sy.DWt // straight update, no limits or anything
			if sy.Wt < 0 {
				sy.Wt = 0
			}
			sy.LWt = sy.Wt
			sy.DWt = 0
		}
	}
}
