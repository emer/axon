// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/Astera-org/axon/axon"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// BurstParams determine how the 5IB Burst activation is computed from
// standard Act activation values in SuperLayer -- thresholded.
type BurstParams struct {
	ThrRel float32 `max:"1" def:"0.1" desc:"Relative component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = CaSpkP).  This is the distance between the average and maximum activation values within layer (e.g., 0 = average, 1 = max).  Overall effective threshold is MAX of relative and absolute thresholds."`
	ThrAbs float32 `min:"0" max:"1" def:"0.1" desc:"Absolute component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = CaSpkP).  Overall effective threshold is MAX of relative and absolute thresholds."`
}

func (db *BurstParams) Defaults() {
	db.ThrRel = 0.1
	db.ThrAbs = 0.1
}

// SuperLayer is the DeepAxon superficial layer, based on basic rate-coded axon.Layer.
// Computes the Burst activation from regular activations.
type SuperLayer struct {
	axon.Layer               // access as .Layer
	Burst      BurstParams   `view:"inline" desc:"parameters for computing Burst from act, in Superficial layers (but also needed in Deep layers for deep self connections)"`
	SuperNeurs []SuperNeuron `desc:"slice of super neuron values -- same size as Neurons"`
}

var KiT_SuperLayer = kit.Types.AddType(&SuperLayer{}, LayerProps)

func (ly *SuperLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.Decay.AHP = 0
	ly.Burst.Defaults()
}

func (ly *SuperLayer) UpdateParams() {
	ly.Layer.UpdateParams()
}

func (ly *SuperLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.SuperNeurs = make([]SuperNeuron, len(ly.Neurons))
	return err
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *SuperLayer) InitActs() {
	ly.Layer.InitActs()
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.Burst = 0
		snr.BurstPrv = 0
	}
}

func (ly *SuperLayer) DecayState(decay, glong float32) {
	ly.Layer.DecayState(decay, glong)
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.Burst -= decay * (snr.Burst - ly.Act.Init.Act)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Burst -- computed in CyclePost

func (ly *SuperLayer) NewState() {
	ly.Layer.NewState()
	ly.BurstPrv()
}

// BurstPrv saves Burst as BurstPrv
func (ly *SuperLayer) BurstPrv() {
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.BurstPrv = snr.Burst
	}
}

// CyclePost calls BurstFmCaSpkP
func (ly *SuperLayer) CyclePost(ctime *axon.Time) {
	ly.Layer.CyclePost(ctime)
	ly.BurstFmCaSpkP(ctime)
}

// BurstFmCaSpkP updates Burst layer 5IB bursting value from current CaSpkP
// reflecting a time-integrated spiking value useful in learning,
// subject to thresholding.  Only updated during plus phase.
func (ly *SuperLayer) BurstFmCaSpkP(ctime *axon.Time) {
	if !ctime.PlusPhase {
		return
	}
	actMax := ly.ActAvg.CaSpkP.Max
	actAvg := ly.ActAvg.CaSpkP.Avg
	thr := actAvg + ly.Burst.ThrRel*(actMax-actAvg)
	thr = mat32.Max(thr, ly.Burst.ThrAbs)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		snr := &ly.SuperNeurs[ni]
		burst := float32(0)
		if nrn.CaSpkP > thr {
			burst = nrn.CaSpkP
		}
		snr.Burst = burst
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  DeepCtxt -- once after Burst quarter

// SendCtxtGe sends Burst activation over CTCtxtPrjn projections to integrate
// CtxtGe excitatory conductance on CT layers.
// This should be called at the end of the 5IB Bursting phase via Network.CTCtxt
// Satisfies the CtxtSender interface.
func (ly *SuperLayer) SendCtxtGe(ctime *axon.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		snr := &ly.SuperNeurs[ni]
		if snr.Burst == 0 {
			continue
		}
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			ptyp := sp.Type()
			if ptyp != CTCtxt {
				continue
			}
			pj, ok := sp.(*CTCtxtPrjn)
			if !ok {
				continue
			}
			pj.SendCtxtGe(ni, snr.Burst)
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Unit Vars

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *SuperLayer) UnitVarNames() []string {
	return NeuronVarsAll
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *SuperLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = SuperNeuronVarIdxByName(varNm)
	if err != nil {
		return vidx, err
	}
	vidx += ly.Layer.UnitVarNum()
	return vidx, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *SuperLayer) UnitVal1D(varIdx int, idx int) float32 {
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
	if varIdx >= len(SuperNeuronVars) {
		return mat32.NaN()
	}
	snr := &ly.SuperNeurs[idx]
	return snr.VarByIdx(varIdx)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *SuperLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + len(SuperNeuronVars)
}
