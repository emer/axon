// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// BurstParams determine how the 5IB Burst activation is computed from
// standard Act activation values in SuperLayer -- thresholded.
type BurstParams struct {
	ThrRel float32 `max:"1" def:"0.1,0.2,0.5" desc:"Relative component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  This is the distance between the average and maximum activation values within layer (e.g., 0 = average, 1 = max).  Overall effective threshold is MAX of relative and absolute thresholds."`
	ThrAbs float32 `min:"0" max:"1" def:"0.1,0.2,0.5" desc:"Absolute component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  Overall effective threshold is MAX of relative and absolute thresholds."`
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
	ly.Act.Decay.Act = 0 // deep doesn't decay!
	ly.Act.Decay.Glong = 0.5
	ly.Act.Decay.KNa = 0
	ly.Burst.Defaults()
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *SuperLayer) UpdateParams() {
	ly.Layer.UpdateParams()
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

func (ly *SuperLayer) DecayState(decay float32) {
	ly.Layer.DecayState(decay)
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.Burst -= decay * (snr.Burst - ly.Act.Init.Act)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Burst -- computed in CyclePost

// MinusPhase does updating after end of minus phase
func (ly *SuperLayer) MinusPhase(ltime *axon.Time) {
	ly.Layer.MinusPhase(ltime)
	ly.BurstPrv()
}

// BurstPrv saves Burst as BurstPrv
func (ly *SuperLayer) BurstPrv() {
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.BurstPrv = snr.Burst
	}
}

// CyclePost calls BurstFmAct
func (ly *SuperLayer) CyclePost(ltime *axon.Time) {
	ly.Layer.CyclePost(ltime)
	ly.BurstFmAct(ltime)
}

// BurstFmAct updates Burst layer 5IB bursting value from current Act
// (superficial activation), subject to thresholding.
// Updated during Time.PlusPhase
func (ly *SuperLayer) BurstFmAct(ltime *axon.Time) {
	if !ltime.PlusPhase {
		return
	}
	lpl := &ly.Pools[0]
	actMax := lpl.Inhib.Act.Max
	actAvg := lpl.Inhib.Act.Avg
	thr := actAvg + ly.Burst.ThrRel*(actMax-actAvg)
	thr = mat32.Max(thr, ly.Burst.ThrAbs)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		snr := &ly.SuperNeurs[ni]
		burst := float32(0)
		if nrn.Act > thr {
			burst = nrn.Act
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
func (ly *SuperLayer) SendCtxtGe(ltime *axon.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		snr := &ly.SuperNeurs[ni]
		if snr.Burst > 0.1 {
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
}

//////////////////////////////////////////////////////////////////////////////////////
//  Unit Vars

// Build constructs the layer state, including calling Build on the projections.
func (ly *SuperLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.SuperNeurs = make([]SuperNeuron, len(ly.Neurons))
	return err
}

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
