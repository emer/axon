// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"strings"

	"github.com/goki/gosl/slbool"
)

//gosl: start pcore_layers

// MatrixParams has parameters for BG Striatum Matrix MSN layers
// These are the main Go / NoGo gating units in BG.
// All of the learning modulation is pre-computed on the recv neuron
// RLRate variable via NeuroMod and LayerVals.Special.V1 = sign multiplier,
// in PlusPhase prior to DWt call.
type MatrixParams struct {
	InvertNoGate slbool.Bool `desc:"invert the direction of learning if not gated -- allows negative DA to increase gating when gating didn't happen."`
	GateThr      float32     `desc:"threshold on layer Avg SpkMax for Matrix Go and VThal layers to count as having gated"`
	NoGoGeLrn    float32     `desc:"multiplier on Ge in NoGo (D2) neurons to provide a baseline level of learning, so that if a negative DA outcome occurs, there is some activity in NoGo for learning to work on.  Strong values increase amount of NoGo learning.  Shows up in SpkMax value which is what drives learning."`

	pad float32
}

// todo: this is not currently supported or used, but might be relevant in the future:
// GPHasPools   bool    `desc:"do the GP pathways that we drive have separate pools that compete for selecting one out of multiple options in parallel (true) or is it a single big competition for Go vs. No (false)"`

func (mp *MatrixParams) Defaults() {
	mp.GateThr = 0.01
	mp.NoGoGeLrn = 1
}

func (mp *MatrixParams) Update() {
}

//gosl: end pcore_layers

// todo: original has this:

// func (ly *MatrixLayer) DecayState(decay, glong float32) {
// 	ly.Layer.DecayState(decay, glong)
// 	for ni := range ly.Neurons {
// 		nrn := &ly.Neurons[ni]
// 		if nrn.IsOff() {
// 			continue
// 		}
// 		ly.Params.Learn.DecayCaLrnSpk(nrn, glong) // ?
// 	}
// 	ly.InitMods()
// }

func (ly *Layer) MatrixLayerDefaults() {
	ly.Params.Act.Decay.Act = 0
	ly.Params.Act.Decay.Glong = 0
	ly.Params.Inhib.Pool.On.SetBool(false)
	ly.Params.Inhib.Layer.On.SetBool(true)
	ly.Params.Inhib.Layer.Gi = 0.5
	ly.Params.Inhib.Layer.FB = 0
	ly.Params.Inhib.Pool.FB = 0
	ly.Params.Inhib.Pool.Gi = 0.5
	ly.Params.Inhib.ActAvg.Nominal = 0.25

	// important: user needs to adjust wt scale of some PFC inputs vs others:
	// drivers vs. modulators

	for _, pji := range ly.RcvPrjns {
		pj := pji.(AxonPrjn).AsAxon()
		pj.Params.SWt.Init.SPct = 0
		if pj.Send.(AxonLayer).LayerType() == GPLayer { // From GPe TA or In
			pj.Params.PrjnScale.Abs = 1
			pj.Params.Learn.Learn.SetBool(false)
			pj.Params.SWt.Adapt.SigGain = 1
			pj.Params.SWt.Init.Mean = 0.75
			pj.Params.SWt.Init.Var = 0.0
			pj.Params.SWt.Init.Sym.SetBool(false)
			if strings.HasSuffix(pj.Send.Name(), "GPeIn") { // GPeInToMtx
				pj.Params.PrjnScale.Abs = 0.5 // counterbalance for GPeTA to reduce oscillations
			} else if strings.HasSuffix(pj.Send.Name(), "GPeTA") { // GPeTAToMtx
				if strings.HasSuffix(ly.Nm, "MtxGo") {
					pj.Params.PrjnScale.Abs = 2 // was .8
				} else {
					pj.Params.PrjnScale.Abs = 1 // was .3 GPeTAToMtxNo must be weaker to prevent oscillations, even with GPeIn offset
				}
			}
		}
	}

	ly.UpdateParams()
}

// MatrixPlusPhase is called on
func (ly *Layer) MatrixPlusPhase() {
}
