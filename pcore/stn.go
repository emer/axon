// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"strings"

	"github.com/Astera-org/axon/axon"
	"github.com/Astera-org/axon/chans"
	"github.com/Astera-org/axon/rl"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// CaParams control the calcium dynamics in STN neurons.
// The SKCa small-conductance calcium-gated potassium channel
// produces the pausing function as a consequence of rapid bursting.
type CaParams struct {
	SKCa      chans.SKCaParams `view:"inline" desc:"small-conductance calcium-activated potassium channel"`
	CaD       bool             `desc:"use CaD timescale (delayed) calcium signal -- for STNs -- else use CaP (faster) for STNp"`
	CaScale   float32          `desc:"scaling factor applied to input Ca to bring into proper range of these dynamics"`
	ThetaInit bool             `desc:"initialize Ca, KCa values at start of every ThetaCycle (i.e., behavioral trial)"`
}

func (kc *CaParams) Defaults() {
	kc.SKCa.Defaults()
	kc.SKCa.Gbar = 2
	kc.CaScale = 3
}

func (kc *CaParams) Update() {
	kc.SKCa.Update()
}

///////////////////////////////////////////////////////////////////////////
// STNLayer

// STNLayer represents STN neurons, with two subtypes:
// STNp are more strongly driven and get over bursting threshold, driving strong,
// rapid activation of the KCa channels, causing a long pause in firing, which
// creates a window during which GPe dynamics resolve Go vs. No balance.
// STNs are more weakly driven and thus more slowly activate KCa, resulting in
// a longer period of activation, during which the GPi is inhibited to prevent
// premature gating based only MtxGo inhibition -- gating only occurs when
// GPeIn signal has had a chance to integrate its MtxNo inputs.
type STNLayer struct {
	rl.Layer
	Ca       CaParams    `view:"inline" desc:"parameters for calcium and calcium-gated potassium channels that drive the afterhyperpolarization that open the gating window in STN neurons (Hallworth et al., 2003)"`
	STNNeurs []STNNeuron `desc:"slice of extra STNNeuron state for this layer -- flat list of len = Shape.Len(). You must iterate over index and use pointer to modify values."`
}

var KiT_STNLayer = kit.Types.AddType(&STNLayer{}, LayerProps)

// Defaults in param.Sheet format
// Sel: "STNLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.Pool.On":      "false",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// }}

func (ly *STNLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Ca.Defaults()
	ly.Typ = STN

	// STN is tonically self-active and has no FFFB inhibition

	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Inhib.Layer.On = true // was false
	ly.Inhib.Layer.Gi = 0.6
	ly.Inhib.Pool.On = false
	ly.Inhib.ActAvg.Init = 0.15

	if strings.HasSuffix(ly.Nm, "STNp") {
		ly.Ca.CaD = false
		ly.Ca.CaScale = 4
	} else {
		ly.Ca.CaD = true
		ly.Ca.CaScale = 3
		ly.Act.Init.Ge = 0.2
		ly.Act.Init.GeVar = 0.2
		ly.Inhib.Layer.Gi = 0.2
	}

	for _, pji := range ly.RcvPrjns {
		pj := pji.(axon.AxonPrjn).AsAxon()
		pj.Learn.Learn = false
		pj.SWt.Adapt.SigGain = 1
		pj.SWt.Init.SPct = 0
		pj.SWt.Init.Mean = 0.75
		pj.SWt.Init.Var = 0.25
		pj.SWt.Init.Sym = false
		if strings.HasSuffix(ly.Nm, "STNp") {
			if _, ok := pj.Send.(*GPLayer); ok { // GPeInToSTNp
				pj.PrjnScale.Abs = 0.1
			}
		} else { // STNs
			if _, ok := pj.Send.(*GPLayer); ok { // GPeInToSTNs
				pj.PrjnScale.Abs = 0.1 // note: not currently used -- interferes with threshold-based Ca self-inhib dynamics
			} else {
				pj.PrjnScale.Abs = 0.2 // weaker inputs
			}
		}
	}

	ly.UpdateParams()
}

func (ly *STNLayer) UpdateParams() {
	ly.Layer.UpdateParams()
	ly.Ca.Update()
}

func (ly *STNLayer) Class() string {
	return "STN " + ly.Cls
}

func (ly *STNLayer) InitActs() {
	ly.Layer.InitActs()
	for ni := range ly.STNNeurs {
		snr := &ly.STNNeurs[ni]
		snr.SKCai = 0
		snr.SKCaM = 0
		snr.Gsk = 0
	}
}

func (ly *STNLayer) NewState() {
	ly.Layer.NewState()
	if !ly.Ca.ThetaInit {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		snr := &ly.STNNeurs[ni]
		snr.SKCai = 0
		snr.SKCaM = 0
		snr.Gsk = 0
	}
}

func (ly *STNLayer) GInteg(ni int, nrn *axon.Neuron, ctime *axon.Time) {
	ly.Layer.GInteg(ni, nrn, ctime)
	snr := &ly.STNNeurs[ni]
	if ly.Ca.CaD {
		snr.SKCai = ly.Ca.CaScale * nrn.CaSpkD // todo: CaD?
	} else {
		snr.SKCai = ly.Ca.CaScale * nrn.CaSpkP // todo: CaP?
	}
	snr.SKCaM = ly.Ca.SKCa.MFmCa(snr.SKCai, snr.SKCaM)
	snr.Gsk = ly.Ca.SKCa.Gbar * snr.SKCaM
	nrn.Gk += snr.Gsk
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *STNLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.STNNeurs = make([]STNNeuron, len(ly.Neurons))
	return nil
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *STNLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = STNNeuronVarIdxByName(varNm)
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
func (ly *STNLayer) UnitVal1D(varIdx int, idx int) float32 {
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
	if varIdx > len(STNNeuronVars) {
		return mat32.NaN()
	}
	snr := &ly.STNNeurs[idx]
	return snr.VarByIndex(varIdx)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *STNLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + len(STNNeuronVars)
}
