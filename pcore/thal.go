// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"fmt"
	"strings"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/rl"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// ThalLayer represents a BG gated thalamic layer,
// e.g., the Ventral thalamus: VA / VM / VL or MD mediodorsal thalamus,
// which receives BG gating in the form of an inhibitory projection from GPi.
type ThalLayer struct {
	rl.Layer
	Gated []bool `inactive:"+" desc:"indicates whether each pool gated, based on Avg SpkMax being above threshold for Go Matrix and this thalamus -- computed by Matrix -- same size as Pools"`
}

var KiT_ThalLayer = kit.Types.AddType(&ThalLayer{}, LayerProps)

// Defaults in param.Sheet format
// Sel: "ThalLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.Pool.On":     "false",
// 		"Layer.Inhib.ActAvg.Nominal":  "0.25",
// }}

func (ly *ThalLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Typ = Thal

	// note: not tonically active

	ly.Act.Dend.SSGi = 0
	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.ActAvg.Nominal = 0.25

	for _, pji := range ly.RcvPrjns {
		pj := pji.(axon.AxonPrjn).AsAxon()
		pj.Learn.Learn = false
		pj.SWt.Adapt.SigGain = 1
		pj.SWt.Init.SPct = 0
		pj.SWt.Init.Mean = 0.75
		pj.SWt.Init.Var = 0.0
		pj.SWt.Init.Sym = false
		if strings.HasSuffix(pj.Send.Name(), "GPi") { // GPiToVThal
			pj.PrjnScale.Abs = 2 // was 2.5 for agate model..
		}
	}

	ly.UpdateParams()
}

func (ly *ThalLayer) Class() string {
	return "Thal " + ly.Cls
}

func (ly *ThalLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.Gated = make([]bool, len(ly.Pools))
	return nil
}

func (ly *ThalLayer) DecayState(decay, glong float32) {
	ly.Layer.DecayState(decay, glong)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Learn.DecayCaLrnSpk(nrn, glong)
	}
}

// GatedFmAvgSpk updates the Gated values based on Avg SpkMax
// using given threshold.  Called by Go Matrix layer.
// returns true if any gated.
func (ly *ThalLayer) GatedFmAvgSpk(thr float32) bool {
	anyGt := false
	for pi := range ly.Gated {
		smax := ly.AvgMaxVarByPool("SpkMax", pi).Avg
		gt := (smax > thr)
		ly.Gated[pi] = gt
		if gt {
			// fmt.Printf("thl %s gated spkavg: %g pool: %d\n", ly.Name(), smax, pi)
			anyGt = true
		}
	}
	return anyGt
}

// AnyGated returns true if any of the pools gated
func (ly *ThalLayer) AnyGated() bool {
	for _, gt := range ly.Gated {
		if gt {
			return true
		}
	}
	return false
}

///////////////////////////////////////////////////////////////////
// Unit var access

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *ThalLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + 1
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *ThalLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	nvi := 0
	switch varNm {
	case "Gated":
		nvi = 0
	default:
		return -1, fmt.Errorf("pcore.NeuronVars: variable named: %s not found", varNm)
	}
	nn := ly.Layer.UnitVarNum()
	return nn + nvi, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *ThalLayer) UnitVal1D(varIdx int, idx int) float32 {
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
		gt := ly.Gated[ly.Neurons[idx].SubPool]
		if gt {
			return 1.0
		}
		return 0
	default:
		return mat32.NaN()
	}
	return 0
}
