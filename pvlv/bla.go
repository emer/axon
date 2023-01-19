// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/rl"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// BLAParams has parameters for basolateral amygdala
type BLAParams struct {
	NoDALRate float32 `desc:"baseline learning rate without any dopamine"`
	NoUSLRate float32 `desc:"learning rate outside of US active time window (i.e. for CSs)"`
	NegLRate  float32 `desc:"negative DWt learning rate multiplier -- weights go down much more slowly than up -- extinction is separate learning in extinction layer"`
}

func (bp *BLAParams) Defaults() {
	bp.NoDALRate = 0.0
	bp.NoUSLRate = 0.0
	bp.NegLRate = 0.1
}

// BLALayer represents a basolateral amygdala layer
type BLALayer struct {
	rl.Layer
	DaMod    DaModParams   `view:"inline" desc:"dopamine modulation parameters"`
	BLA      BLAParams     `view:"inline" desc:"special BLA parameters"`
	USLayers emer.LayNames `desc:"layer(s) that represent the presence of a US -- if the Max act of these layers is above .1, then USActive flag is set, which affects learning rate."`
	ACh      float32       `inactive:"+" desc:"acetylcholine value from rl.RSalience cholinergic layer reflecting the absolute value of reward or CS predictions thereof -- modulates BLA learning to restrict to US and CS times"`
	USActive bool          `inactive:"+" desc:"marks presence of US as a function of activity over USLayers -- affects learning rate."`
}

var KiT_BLALayer = kit.Types.AddType(&BLALayer{}, LayerProps)

func (ly *BLALayer) Defaults() {
	ly.Layer.Defaults()
	ly.DaMod.Defaults()
	ly.BLA.Defaults()
	ly.Typ = BLA

	// special inhib params
	ly.Params.Act.Decay.Act = 0
	ly.Params.Act.Decay.Glong = 0
	ly.Params.Act.Dend.SSGi = 0
	ly.Params.Inhib.Layer.On = true
	ly.Params.Inhib.Layer.Gi = 1.2
	ly.Params.Inhib.Pool.On = true
	ly.Params.Inhib.Pool.Gi = 1.0
	// ly.Params.Inhib.Layer.FB = 0
	// ly.Params.Inhib.Pool.FB = 0
	ly.Params.Inhib.ActAvg.Nominal = 0.025

}

// AChLayer interface:

func (ly *BLALayer) GetACh() float32    { return ly.ACh }
func (ly *BLALayer) SetACh(ach float32) { ly.ACh = ach }

func (ly *BLALayer) InitActs() {
	ly.Layer.InitActs()
	ly.ACh = 0
	ly.USActive = false
}

func (ly *BLALayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.USLayers.Validate(ly.Network, "BLALayer.USLayers")
	return err
}

// USActiveFmUS updates the USActive flag based on USLayers state
func (ly *BLALayer) USActiveFmUS(ctxt *axon.Context) {
	ly.USActive = false
	if len(ly.USLayers) == 0 {
		return
	}
	mx := rl.MaxAbsActFmLayers(ly.Network, ly.USLayers)
	if mx > 0.1 {
		ly.USActive = true
	}
}

func (ly *BLALayer) GInteg(ni uint32, nrn *axon.Neuron, ctxt *axon.Context) {
	da := ly.DaMod.Gain(ly.DA)
	ly.NeuronGatherSpikes(ni, nrn, ctxt)
	daEff := da * nrn.CaSpkM // da effect interacts with spiking
	nrn.GeRaw += daEff
	nrn.GeSyn += ly.Params.Act.Dt.GeSynFmRawSteady(daEff)
	ly.GFmRawSyn(ni, nrn, ctxt)
	ly.GiInteg(ni, nrn, ctxt)
}

func (ly *BLALayer) PlusPhase(ctxt *axon.Context) {
	ly.Layer.PlusPhase(ctxt)
	ly.USActiveFmUS(ctxt)
	lrmod := ly.BLA.NoDALRate + mat32.Abs(ly.DA)
	if !ly.USActive {
		lrmod *= ly.BLA.NoUSLRate
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		mlr := ly.Params.Learn.RLRate.RLRateSigDeriv(nrn.CaSpkP, ly.Vals.ActAvg.CaSpkP.Max)
		dlr := ly.Params.Learn.RLRate.RLRateDiff(nrn.CaSpkP, nrn.SpkPrv) // delta on previous
		if nrn.CaSpkP-nrn.SpkPrv < 0 {
			dlr *= ly.BLA.NegLRate
		}
		nrn.RLRate = mlr * dlr * lrmod
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  BLAPrjn

// BLAPrjn implements the PVLV BLA learning rule:
// dW = Ach * X_t-1 * (Y_t - Y_t-1)
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
	pj.Params.Learn.Trace.Tau = 1
	pj.Params.Learn.Trace.Update()
}

func (pj *BLAPrjn) SendSynCa(ctxt *axon.Context) {
	return
}

func (pj *BLAPrjn) RecvSynCa(ctxt *axon.Context) {
	return
}

// DWt computes the weight change (learning) for BLA projections
func (pj *BLAPrjn) DWt(ctxt *axon.Context) {
	if !pj.Params.Learn.Learn {
		return
	}
	rlay := pj.Recv.(*BLALayer)
	slay := pj.Send.(axon.AxonLayer).AsAxon()
	ach := rlay.ACh
	lr := ach * pj.Params.Learn.LRate.Eff
	for si := range slay.Neurons {
		sact := slay.Neurons[si].SpkPrv
		nc := int(pj.SendConN[si])
		st := int(pj.SendConIdxStart[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SendConIdx[st : st+nc]
		for ci := range syns {
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			sy := &syns[ci]
			// not using the synaptic trace (yet)
			// kp.CurCa(ctxt, sy.CaUpT, sy.CaM, sy.CaP, sy.CaD) // always update
			sy.Tr = pj.Params.Learn.Trace.TrFmCa(sy.Tr, sact)
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
			sy.DWt += rn.RLRate * lr * err
		}
	}
}

// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
func (pj *BLAPrjn) WtFmDWt(ctxt *axon.Context) {
	if !pj.Params.Learn.Learn {
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

///////////////////////////////////////////////////////////////////
// Unit var access

func (ly *BLALayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + 1
}

func (ly *BLALayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	nvi := 0
	switch varNm {
	case "ACh":
		nvi = 0
	default:
		return -1, fmt.Errorf("pvlv.BLALayer: variable named: %s not found", varNm)
	}
	nn := ly.Layer.UnitVarNum()
	return nn + nvi, nil
}

func (ly *BLALayer) UnitVal1D(varIdx int, idx int) float32 {
	if varIdx < 0 {
		return mat32.NaN()
	}
	nn := ly.Layer.UnitVarNum()
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return mat32.NaN()
	}
	varIdx -= nn
	switch varIdx {
	case 0:
		return ly.ACh
	default:
		return mat32.NaN()
	}
	return 0
}
