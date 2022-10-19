// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/rl"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// PCoreLayer exposes PCoreNeuron access and PhaseMax values
type PCoreLayer interface {
	// PCoreNeuronByIdx returns neuron at given index
	PCoreNeuronByIdx(idx int) *PCoreNeuron

	//	PhasicMaxAvgByPool returns the average PhasicMax value by given pool index
	PhasicMaxAvgByPool(pli int) float32

	//	PhasicMaxMaxByPool returns the max PhasicMax value by given pool index
	PhasicMaxMaxByPool(pli int) float32

	//	PhasicMaxMax returns the max PhasicMax value across layer
	PhasicMaxMax() float32
}

// Layer is the basic pcore layer, which has a DA dopamine value from rl.Layer
// and tracks the phasic maximum activation during the gating window.
type Layer struct {
	rl.Layer
	PhasicMaxCycMin int           `desc:"minimum cycle after which phasic maximum activity is recorded"`
	PCoreNeurs      []PCoreNeuron `desc:"slice of extra Neuron state for this layer -- flat list of len = Shape.Len(). You must iterate over index and use pointer to modify values."`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, axon.LayerProps)

func (ly *Layer) Defaults() {
	ly.Layer.Defaults()
	ly.PhasicMaxCycMin = 30
}

func (ly *Layer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	nn := len(ly.Neurons)
	ly.PCoreNeurs = make([]PCoreNeuron, nn)
	return nil
}

func (ly *Layer) InitActs() {
	ly.Layer.InitActs()
	ly.InitPhasicMax()
	ly.InitActLrn()
}

func (ly *Layer) NewState() {
	ly.Layer.NewState()
	ly.InitPhasicMax()
}

// InitPhasicMax initializes the PhasicMax to 0
func (ly *Layer) InitPhasicMax() {
	for ni := range ly.PCoreNeurs {
		pn := &ly.PCoreNeurs[ni]
		pn.PhasicMax = 0
	}
}

// InitActLrn initializes the ActLrn to 0
func (ly *Layer) InitActLrn() {
	for ni := range ly.PCoreNeurs {
		pn := &ly.PCoreNeurs[ni]
		pn.ActLrn = 0
	}
}

func (ly *Layer) ActFmG(ltime *axon.Time) {
	ly.Layer.ActFmG(ltime)
	ly.PhasicMaxFmAct(ltime)
}

// PhasicMaxFmAct computes PhasicMax from Activation (CaSpkP)
func (ly *Layer) PhasicMaxFmAct(ltime *axon.Time) {
	if ltime.Cycle < ly.PhasicMaxCycMin {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pn := &ly.PCoreNeurs[ni]
		if nrn.Act > pn.PhasicMax {
			pn.PhasicMax = nrn.CaSpkP
		}
	}
}

// ActLrnFmPhasicMax sets ActLrn to PhasicMax
func (ly *Layer) ActLrnFmPhasicMax() {
	for ni := range ly.PCoreNeurs {
		pn := &ly.PCoreNeurs[ni]
		pn.ActLrn = pn.PhasicMax
	}
}

// MaxPhasicMax returns the maximum PhasicMax across the layer
func (ly *Layer) MaxPhasicMax() float32 {
	mx := float32(0)
	for ni := range ly.PCoreNeurs {
		pn := &ly.PCoreNeurs[ni]
		if pn.PhasicMax > mx {
			mx = pn.PhasicMax
		}
	}
	return mx
}

// PCoreNeuronByIdx returns neuron at given index
func (ly *Layer) PCoreNeuronByIdx(idx int) *PCoreNeuron {
	return &ly.PCoreNeurs[idx]
}

// PhasicMaxAvgByPool returns the average PhasicMax value by given pool index
// Pool index 0 is whole layer, 1 is first sub-pool, etc
func (ly *Layer) PhasicMaxAvgByPool(pli int) float32 {
	pl := &ly.Pools[pli]
	sum := float32(0)
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		pn := &ly.PCoreNeurs[ni]
		sum += pn.PhasicMax
	}
	return sum / float32(pl.EdIdx-pl.StIdx)
}

// PhasicMaxMaxByPool returns the average PhasicMax value by given pool index
// Pool index 0 is whole layer, 1 is first sub-pool, etc
func (ly *Layer) PhasicMaxMaxByPool(pli int) float32 {
	pl := &ly.Pools[pli]
	max := float32(0)
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		pn := &ly.PCoreNeurs[ni]
		if pn.PhasicMax > max {
			max = pn.PhasicMax
		}
	}
	return max
}

/////////////////////////////////////////////////////
// Var access boilerplate below

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + len(PCoreNeuronVars)
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = PCoreNeuronVarIdxByName(varNm)
	if err != nil {
		return -1, err
	}
	nn := ly.Layer.UnitVarNum()
	return nn + vidx, err
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitVal1D(varIdx int, idx int) float32 {
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
	if varIdx > len(PCoreNeuronVars) {
		return mat32.NaN()
	}
	pn := &ly.PCoreNeurs[idx]
	return pn.VarByIndex(varIdx)
}
