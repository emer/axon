// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"log"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/etable/minmax"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// RWPredLayer computes reward prediction for a simple Rescorla-Wagner
// learning dynamic (i.e., PV learning in the PVLV framework).
// Activity is computed as linear function of excitatory conductance
// (which can be negative -- there are no constraints).
// Use with RWPrjn which does simple delta-rule learning on minus-plus.
type RWPredLayer struct {
	Layer
	PredRange minmax.F32 `desc:"default 0.1..0.99 range of predictions that can be represented -- having a truncated range preserves some sensitivity in dopamine at the extremes of good or poor performance"`
}

var KiT_RWPredLayer = kit.Types.AddType(&RWPredLayer{}, axon.LayerProps)

func (ly *RWPredLayer) Defaults() {
	ly.Layer.Defaults()
	ly.PredRange.Set(0.01, 0.99)
	ly.Act.Decay.Act = 1
	ly.Act.Decay.Glong = 1
	ly.Act.Dt.GeTau = 40
}

func (ly *RWPredLayer) ActFmG(ltime *axon.Time) {
	ly.Layer.ActFmG(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Act = ly.PredRange.ClipVal(nrn.Ge) // clipped linear
		nrn.ActInt = nrn.Act
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  RWDaLayer

// RWDaLayer computes a dopamine (DA) signal based on a simple Rescorla-Wagner
// learning dynamic (i.e., PV learning in the PVLV framework).
// It computes difference between r(t) and RWPred values.
// r(t) is accessed directly from a Rew layer -- if no external input then no
// DA is computed -- critical for effective use of RW only for PV cases.
// RWPred prediction is also accessed directly from Rew layer to avoid any issues.
type RWDaLayer struct {
	Layer
	SendDA    SendDA `desc:"list of layers to send dopamine to"`
	RewLay    string `desc:"name of Reward-representing layer from which this computes DA -- if nothing clamped, no dopamine computed"`
	RWPredLay string `desc:"name of RWPredLayer layer that is subtracted from the reward value"`
}

var KiT_RWDaLayer = kit.Types.AddType(&RWDaLayer{}, deep.LayerProps)

func (ly *RWDaLayer) Defaults() {
	ly.Layer.Defaults()
	if ly.RewLay == "" {
		ly.RewLay = "Rew"
	}
	if ly.RWPredLay == "" {
		ly.RWPredLay = "RWPred"
	}
}

// RWLayers returns the reward and RWPred layers based on names
func (ly *RWDaLayer) RWLayers() (*axon.Layer, *RWPredLayer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.RewLay)
	if err != nil {
		log.Printf("RWDaLayer %s, RewLay: %v\n", ly.Name(), err)
		return nil, nil, err
	}
	ply, err := ly.Network.LayerByNameTry(ly.RWPredLay)
	if err != nil {
		log.Printf("RWDaLayer %s, RWPredLay: %v\n", ly.Name(), err)
		return nil, nil, err
	}
	return tly.(axon.AxonLayer).AsAxon(), ply.(*RWPredLayer), nil
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *RWDaLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendDA.Validate(ly.Network, ly.Name()+" SendTo list")
	if err != nil {
		return err
	}
	_, _, err = ly.RWLayers()
	return err
}

func (ly *RWDaLayer) ActFmG(ltime *axon.Time) {
	ly.Layer.ActFmG(ltime)
	rly, ply, _ := ly.RWLayers()
	if rly == nil || ply == nil {
		return
	}
	rnrn := &(rly.Neurons[0])
	hasRew := false
	if rnrn.HasFlag(axon.NeurHasExt) {
		hasRew = true
	}
	ract := rnrn.Act
	pnrn := &(ply.Neurons[0])
	pact := pnrn.Act
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if hasRew {
			nrn.Act = ract - pact
		} else {
			nrn.Act = 0 // nothing
		}
		nrn.ActInt = nrn.Act
	}
}

// CyclePost is called at end of Cycle
// We use it to send DA, which will then be active for the next cycle of processing.
func (ly *RWDaLayer) CyclePost(ltime *axon.Time) {
	act := ly.Neurons[0].Act
	ly.DA = act
	ly.SendDA.SendDA(ly.Network, act)
}

//////////////////////////////////////////////////////////////////////////////////////
//  RWPrjn

// RWPrjn does dopamine-modulated learning for reward prediction: Da * Send.Act
// Use in RWPredLayer typically to generate reward predictions.
// Has no weight bounds or limits on sign etc.
type RWPrjn struct {
	axon.Prjn
	DaTol        float32 `desc:"tolerance on DA -- if below this abs value, then DA goes to zero and there is no learning -- prevents prediction from exactly learning to cancel out reward value, retaining a residual valence of signal"`
	OppSignLRate float32 `desc:"how much to learn on opposite DA sign coding neuron (0..1)"`
}

var KiT_RWPrjn = kit.Types.AddType(&RWPrjn{}, deep.PrjnProps)

func (pj *RWPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.OppSignLRate = 1.0
	pj.SWt.Adapt.SigGain = 1
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *RWPrjn) DWt(ltime *axon.Time) {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(axon.AxonLayer).AsAxon()
	rlay := pj.Recv.(axon.AxonLayer).AsAxon()
	lda := pj.Recv.(DALayer).GetDA()
	lr := pj.Learn.Lrate.Eff
	if pj.DaTol > 0 {
		if mat32.Abs(lda) <= pj.DaTol {
			return // lda = 0 -- no learning
		}
	}
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

			da := lda
			if rn.Ge > rn.Act && da > 0 { // clipped at top, saturate up
				da = 0
			}
			if rn.Ge < rn.Act && da < 0 { // clipped at bottom, saturate down
				da = 0
			}
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

			dwt := da * sn.Act // no recv unit activation
			sy.DWt += eff_lr * dwt
		}
	}
}

// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
func (pj *RWPrjn) WtFmDWt(ltime *axon.Time) {
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
