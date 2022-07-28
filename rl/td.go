// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"log"

	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
)

// TDRewPredLayer is the temporal differences reward prediction layer.
// It represents estimated value V(t) in the minus phase, and computes
// estimated V(t+1) based on its learned weights in plus phase.
// Use TDRewPredPrjn for DA modulated learning.
type TDRewPredLayer struct {
	Layer
}

var KiT_TDRewPredLayer = kit.Types.AddType(&TDRewPredLayer{}, axon.LayerProps)

func (ly *TDRewPredLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Act.Dt.GeTau = 40
}

func (ly *TDRewPredLayer) ActFmG(ltime *axon.Time) {
	ly.Layer.ActFmG(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Act = nrn.Ge
		nrn.ActInt = nrn.Act
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  TDRewIntegLayer

// TDRewIntegParams are params for reward integrator layer
type TDRewIntegParams struct {
	Discount    float32 `desc:"discount factor -- how much to discount the future prediction from RewPred"`
	RewPredGain float32 `desc:"gain factor on rew pred activations"`
	RewPred     string  `desc:"name of TDRewPredLayer to get reward prediction from "`
	Rew         string  `desc:"name of RewLayer to get current reward from "`
}

func (tp *TDRewIntegParams) Defaults() {
	tp.Discount = 0.9
	tp.RewPredGain = 1
	if tp.RewPred == "" {
		tp.RewPred = "RewPred"
		tp.Rew = "Rew"
	}
}

// TDRewIntegLayer is the temporal differences reward integration layer.
// It represents estimated value V(t) in the minus phase, and
// estimated V(t+1) + r(t) in the plus phase.
// It directly accesses (t) from Rew layer, and V(t) from RewPred layer.
type TDRewIntegLayer struct {
	Layer
	RewInteg TDRewIntegParams `desc:"parameters for reward integration"`
}

var KiT_TDRewIntegLayer = kit.Types.AddType(&TDRewIntegLayer{}, axon.LayerProps)

func (ly *TDRewIntegLayer) Defaults() {
	ly.Layer.Defaults()
	ly.RewInteg.Defaults()
	// ly.Inhib.Layer.Gi = 0.2
	ly.Inhib.ActAvg.Init = .5
}

// DALayer interface:

func (ly *TDRewIntegLayer) RewPredLayer() (*TDRewPredLayer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.RewInteg.RewPred)
	if err != nil {
		log.Printf("TDRewIntegLayer %s RewPredLayer: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(*TDRewPredLayer), nil
}

func (ly *TDRewIntegLayer) RewLayer() (*RewLayer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.RewInteg.Rew)
	if err != nil {
		log.Printf("TDRewIntegLayer %s RewLayer: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(*RewLayer), nil
}

func (ly *TDRewIntegLayer) RewPredAct(ltime *axon.Time) float32 {
	rply, _ := ly.RewPredLayer()
	if rply == nil {
		return 0
	}
	rly, _ := ly.RewLayer()
	if rly == nil {
		return 0
	}
	rew := rly.Neurons[0].Act
	rpn0 := rply.Neurons[0]
	rpn1 := rply.Neurons[1]
	rpAct := rew + ly.RewInteg.RewPredGain*(rpn0.Ge-rpn1.Ge) // linear
	rpActP := ly.RewInteg.RewPredGain * (rpn0.ActP - rpn1.ActP)
	var rpval float32
	if ltime.PlusPhase {
		rpval = ly.RewInteg.Discount * rpAct
	} else {
		rpval = rpActP
	}
	return rpval
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *TDRewIntegLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	_, err = ly.RewPredLayer()
	return err
}

func (ly *TDRewIntegLayer) GFmInc(ltime *axon.Time) {
	ly.RecvGInc(ltime)
	rpAct := ly.RewPredAct(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.SetFlag(axon.NeurHasExt)
		SetNeuronExtPosNeg(nrn, ni, rpAct)
		ly.GFmIncNeur(ltime, nrn, 0) // no extra
	}
}

func (ly *TDRewIntegLayer) ActFmG(ltime *axon.Time) {
	ly.Layer.ActFmG(ltime)
	rpAct := ly.RewPredAct(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Act = rpAct
		nrn.ActInt = nrn.Act
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  TDDaLayer

// TDDaLayer computes a dopamine (DA) signal as the temporal difference (TD)
// between the TDRewIntegLayer activations in the minus and plus phase.
type TDDaLayer struct {
	Layer
	SendDA   SendDA `desc:"list of layers to send dopamine to"`
	RewInteg string `desc:"name of TDRewIntegLayer from which this computes the temporal derivative"`
}

var KiT_TDDaLayer = kit.Types.AddType(&TDDaLayer{}, axon.LayerProps)

func (ly *TDDaLayer) Defaults() {
	ly.Layer.Defaults()
	if ly.RewInteg == "" {
		ly.RewInteg = "RewInteg"
	}
	// ly.Inhib.Layer.Gi = 0.2
	ly.Inhib.ActAvg.Init = .5
}

func (ly *TDDaLayer) RewIntegLayer() (*TDRewIntegLayer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.RewInteg)
	if err != nil {
		log.Printf("TDDaLayer %s RewIntegLayer: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(*TDRewIntegLayer), nil
}

func (ly *TDDaLayer) RewIntegDA(ltime *axon.Time) float32 {
	rily, _ := ly.RewIntegLayer()
	if rily == nil {
		return 0
	}
	rpActP := rily.Neurons[0].Act
	rpActM := rily.Neurons[0].ActM
	da := rpActP - rpActM
	if !ltime.PlusPhase {
		da = 0
	}
	return da
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *TDDaLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendDA.Validate(ly.Network, ly.Name()+" SendTo list")
	if err != nil {
		return err
	}
	_, err = ly.RewIntegLayer()
	return err
}

func (ly *TDDaLayer) GFmInc(ltime *axon.Time) {
	ly.RecvGInc(ltime)
	da := ly.RewIntegDA(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.SetFlag(axon.NeurHasExt)
		SetNeuronExtPosNeg(nrn, ni, da)
		ly.GFmIncNeur(ltime, nrn, 0) // no extra
	}
}

func (ly *TDDaLayer) ActFmG(ltime *axon.Time) {
	ly.Layer.ActFmG(ltime)
	da := ly.RewIntegDA(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Act = da
		nrn.ActInt = nrn.Act
	}
}

// CyclePost is called at end of Cycle
// We use it to send DA, which will then be active for the next cycle of processing.
func (ly *TDDaLayer) CyclePost(ltime *axon.Time) {
	act := ly.Neurons[0].Act
	ly.DA = act
	ly.SendDA.SendDA(ly.Network, act)
}

//////////////////////////////////////////////////////////////////////////////////////
//  TDRewPredPrjn

// TDRewPredPrjn does dopamine-modulated learning for reward prediction:
// DWt = Da * Send.ActPrv (activity on *previous* timestep)
// Use in TDRewPredLayer typically to generate reward predictions.
// If the Da sign is positive, the first recv unit learns fully;
// for negative, second one learns fully.  Lower lrate applies for
// opposite cases.  Weights are positive-only.
type TDRewPredPrjn struct {
	axon.Prjn
	OppSignLRate float32 `desc:"how much to learn on opposite DA sign coding neuron (0..1)"`
}

var KiT_TDRewPredPrjn = kit.Types.AddType(&TDRewPredPrjn{}, axon.PrjnProps)

func (pj *TDRewPredPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.OppSignLRate = 1.0
	pj.SWt.Adapt.SigGain = 1
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *TDRewPredPrjn) DWt(ltime *axon.Time) {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(axon.AxonLayer).AsAxon()
	// rlay := pj.Recv.(axon.AxonLayer).AsAxon()
	da := pj.Recv.(DALayer).GetDA()
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
			eff_lr := lr
			if ri == 0 {
				if da < 0 {
					eff_lr *= pj.OppSignLRate
				}
			} else {
				eff_lr = -eff_lr
				if da >= 0 {
					eff_lr *= pj.OppSignLRate
				}
			}

			dwt := da * sn.ActPrv // no recv unit activation, prior trial act
			sy.DWt += eff_lr * dwt
		}
	}
}

// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
func (pj *TDRewPredPrjn) WtFmDWt(ltime *axon.Time) {
	if !pj.Learn.Learn {
		return
	}
	for si := range pj.Syns {
		sy := &pj.Syns[si]
		if sy.DWt != 0 {
			sy.Wt += sy.DWt
			if sy.Wt < 0 {
				sy.Wt = 0
			}
			sy.LWt = sy.Wt
			sy.DWt = 0
		}
	}
}
