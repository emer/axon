// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"log"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
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

func (ly *TDRewPredLayer) GeFmInc(ltime *axon.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if ltime.PlusPhase {
			nrn.ClearFlag(axon.NeurHasExt)
		} else {
			nrn.SetFlag(axon.NeurHasExt)
			nrn.Ext = nrn.ActP // previous actP
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  TDRewIntegLayer

// TDRewIntegParams are params for reward integrator layer
type TDRewIntegParams struct {
	Discount float32 `desc:"discount factor -- how much to discount the future prediction from RewPred"`
	RewPred  string  `desc:"name of TDRewPredLayer to get reward prediction from "`
}

func (tp *TDRewIntegParams) Defaults() {
	tp.Discount = 0.9
	if tp.RewPred == "" {
		tp.RewPred = "RewPred"
	}
}

// TDRewIntegLayer is the temporal differences reward integration layer.
// It represents estimated value V(t) in the minus phase, and
// estimated V(t+1) + r(t) in the plus phase.
// It computes r(t) from (typically fixed) weights from a reward layer,
// and directly accesses values from RewPred layer.
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

// Build constructs the layer state, including calling Build on the projections.
func (ly *TDRewIntegLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	_, err = ly.RewPredLayer()
	return err
}

func (ly *TDRewIntegLayer) GeFmInc(ltime *axon.Time) {
	rply, _ := ly.RewPredLayer()
	if rply == nil {
		return
	}
	rpActP := rply.Neurons[0].ActP
	rpAct := rply.Neurons[0].Act
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if ltime.PlusPhase {
			nrn.SetFlag(axon.NeurHasExt)
			nrn.Ext = ly.RewInteg.Discount * rpAct
		} else {
			nrn.SetFlag(axon.NeurHasExt)
			nrn.Ext = rpActP // previous actP
		}
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

func (ly *TDDaLayer) GeFmInc(ltime *axon.Time) {
	rily, _ := ly.RewIntegLayer()
	if rily == nil {
		return
	}
	rpActP := rily.Neurons[0].Act
	rpActM := rily.Neurons[0].ActM
	da := rpActP - rpActM
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if ltime.PlusPhase {
			nrn.SetFlag(axon.NeurHasExt)
			nrn.Ext = da
		} else {
			nrn.SetFlag(axon.NeurHasExt)
			nrn.Ext = 0
		}
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
// DWt = Da * Send.ActQ0 (activity on *previous* timestep)
// Use in TDRewPredLayer typically to generate reward predictions.
// Has no weight bounds or limits on sign etc.
type TDRewPredPrjn struct {
	axon.Prjn
}

var KiT_TDRewPredPrjn = kit.Types.AddType(&TDRewPredPrjn{}, deep.PrjnProps)

func (pj *TDRewPredPrjn) Defaults() {
	pj.Prjn.Defaults()
	// no additional factors
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
		// scons := pj.SConIdx[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			// ri := scons[ci]

			dwt := da * sn.ActPrv // no recv unit activation, prior trial act
			sy.DWt += lr * dwt
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
			sy.Wt += sy.DWt // straight update, no limits or anything
			sy.LWt = sy.Wt
			sy.DWt = 0
		}
	}
}
