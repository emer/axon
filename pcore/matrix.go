// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"fmt"
	"log"
	"strings"

	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// MatrixParams has parameters for Dorsal Striatum Matrix computation
// These are the main Go / NoGo gating units in BG driving updating of PFC WM in PBWM
type MatrixParams struct {
	ThalLay   string  `desc:"name of VThal layer -- needed to get overall gating output action"`
	ThalThr   float32 `def:"0.25" desc:"threshold for thal max activation (in pool) to be gated -- typically .25 or so to accurately reflect PFC output gating -- may need to adjust based on actual behavior"`
	Deriv     bool    `def:"true" desc:"use the sigmoid derivative factor 2 * Act * (1-Act) for matrix (recv) activity in modulating learning -- otherwise just multiply by activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)"`
	BurstGain float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
	DipGain   float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
}

func (mp *MatrixParams) Defaults() {
	if mp.ThalLay == "" {
		mp.ThalLay = "VThal"
	}
	mp.ThalThr = 0.25
	mp.Deriv = true
	mp.BurstGain = 1
	mp.DipGain = 1
}

// LrnFactor returns multiplicative factor for level of msn activation.  If Deriv
// is 2 * act * (1-act) -- the factor of 2 compensates for otherwise reduction in
// learning from these factors.  Otherwise is just act.
func (mp *MatrixParams) LrnFactor(act float32) float32 {
	if !mp.Deriv {
		return act
	}
	return 2 * act * (1 - act)
}

// MatrixLayer represents the dorsal matrisome MSN's that are the main
// Go / NoGo gating units in BG.  D1R = Go, D2R = NoGo.
type MatrixLayer struct {
	Layer
	DaR    DaReceptors  `desc:"dominant type of dopamine receptor -- D1R for Go pathway, D2R for NoGo"`
	Matrix MatrixParams `view:"inline" desc:"matrix parameters"`
	DALrn  float32      `inactive:"+" desc:"effective learning dopamine value for this layer: reflects DaR and Gains"`
	ACh    float32      `inactive:"+" desc:"acetylcholine value from CIN cholinergic interneurons reflecting the absolute value of reward or CS predictions thereof -- used for resetting the trace of matrix learning"`
}

var KiT_MatrixLayer = kit.Types.AddType(&MatrixLayer{}, axon.LayerProps)

// Defaults in param.Sheet format
// Sel: "MatrixLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Pool.On":      "false",
// 		"Layer.Inhib.Layer.On":     "true",
// 		"Layer.Inhib.Layer.Gi":     "1.5",
// 		"Layer.Inhib.Layer.FB":     "0.0",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.3", // 0.6 in localist -- expt
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 	}}

func (ly *MatrixLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Matrix.Defaults()

	// special inhib params
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Inhib.Pool.On = false
	ly.Inhib.Layer.On = true
	ly.Inhib.Layer.Gi = 1.5
	ly.Inhib.Layer.FB = 0
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.3 // 0.6 in localist one
	ly.Inhib.Self.Tau = 3.0
	ly.Inhib.ActAvg.Init = 0.25

	// important: user needs to adjust wt scale of some PFC inputs vs others:
	// drivers vs. modulators

	for _, pji := range ly.RcvPrjns {
		pj := pji.(axon.AxonPrjn).AsAxon()
		if _, ok := pj.Send.(*GPLayer); ok { // From GPe TA or In
			pj.PrjnScale.Abs = 3
			pj.Learn.Learn = false
			pj.SWt.Adapt.SigGain = 1
			pj.SWt.Init.Mean = 0.9
			pj.SWt.Init.Var = 0
			pj.SWt.Init.Sym = false
			if strings.HasSuffix(pj.Send.Name(), "GPeIn") { // GPeInToMtx
				pj.PrjnScale.Abs = 0.3 // counterbalance for GPeTA to reduce oscillations
			} else if strings.HasSuffix(pj.Send.Name(), "GPeTA") { // GPeTAToMtx
				if strings.HasSuffix(ly.Nm, "MtxGo") {
					pj.PrjnScale.Abs = 0.8
				} else {
					pj.PrjnScale.Abs = 0.3 // GPeTAToMtxNo must be weaker to prevent oscillations, even with GPeIn offset
				}
			}
		}
	}

	ly.UpdateParams()
}

// AChLayer interface:

func (ly *MatrixLayer) GetACh() float32    { return ly.ACh }
func (ly *MatrixLayer) SetACh(ach float32) { ly.ACh = ach }

func (ly *MatrixLayer) ThalLayer() (*VThalLayer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.Matrix.ThalLay)
	if err != nil {
		log.Printf("MatrixLayer %s ThalLayer: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(*VThalLayer), nil
}

func (ly *MatrixLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DALrn = 0
	ly.ACh = 0
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act.
// Matrix extends to call DALrnFmDA and updates PhasicMax -> ActLrn
func (ly *MatrixLayer) ActFmG(ltime *axon.Time) {
	ly.Layer.ActFmG(ltime)
	ly.DAActLrn(ltime)
}

// DAActLrn sets effective learning dopamine value from given raw DA value,
// applying Burst and Dip Gain factors, and then reversing sign for D2R.
// Also sets ActLrn based on whether corresponding VThal stripe fired
// above ThalThr -- flips sign of learning for stripe firing vs. not.
func (ly *MatrixLayer) DAActLrn(ltime *axon.Time) {
	da := ly.DA
	if da > 0 {
		da *= ly.Matrix.BurstGain
	} else {
		da *= ly.Matrix.DipGain
	}
	if ly.DaR == D2R {
		da *= -1
	}
	ly.DALrn = da
	if ltime.Cycle < ly.PhasicMaxCycMin {
		return
	}
	tly, err := ly.ThalLayer()
	if err != nil {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pn := &ly.PCoreNeurs[ni]
		amax := ly.Matrix.LrnFactor(pn.PhasicMax)
		tact := tly.PCoreNeurs[nrn.SubPool-1].PhasicMax
		if tact > ly.Matrix.ThalThr {
			pn.ActLrn = amax
		} else {
			pn.ActLrn = -amax
		}
	}
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *MatrixLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + 2
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *MatrixLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	nvi := 0
	switch varNm {
	case "DALrn":
		nvi = 0
	case "ACh":
		nvi = 1
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
func (ly *MatrixLayer) UnitVal1D(varIdx int, idx int) float32 {
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
		return ly.DALrn
	case 1:
		return ly.ACh
	default:
		return mat32.NaN()
	}
	return 0
}
