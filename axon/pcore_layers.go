// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"strings"

	"github.com/emer/axonold/axon"
)

//gosl: start pcore_layers

// MatrixParams has parameters for Dorsal Striatum Matrix computation
// These are the main Go / NoGo gating units in BG.
type MatrixParams struct {
	GPHasPools   bool    `desc:"do the GP pathways that we drive have separate pools that compete for selecting one out of multiple options in parallel (true) or is it a single big competition for Go vs. No (false)"`
	InvertNoGate bool    `desc:"invert the direction of learning if not gated -- allows negative DA to increase gating when gating didn't happen.  Does not work with GPHasPools at present."`
	GateThr      float32 `desc:"threshold on layer Avg SpkMax for Matrix Go and Thal layers to count as having gated"`
	BurstGain    float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
	DipGain      float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
	NoGoGeLrn    float32 `desc:"multiplier on Ge in NoGo (D2) neurons to provide a baseline level of learning, so that if a negative DA outcome occurs, there is some activity in NoGo for learning to work on.  Strong values increase amount of NoGo learning.  Shows up in SpkMax value which is what drives learning."`
	ModGain      float32 `desc:"gain factor multiplying the modulator input GeSyn conductances -- total modulation has a maximum of 1"`
	AChInhib     float32 `desc:"strength of extra Gi current multiplied by MaxACh-ACh (ACh > Max = 0) -- ACh is disinhibitory on striatal firing"`
	MaxACh       float32 `desc:"level of ACh at or above which AChInhib goes to 0 -- ACh typically ranges between 0-1"`
}

func (mp *MatrixParams) Defaults() {
	mp.GateThr = 0.01
	mp.BurstGain = 1
	mp.DipGain = 1
	mp.NoGoGeLrn = 1
	mp.ModGain = 1
	mp.AChInhib = 0
	mp.MaxACh = 0.5
}

// GiFmACh returns inhibitory conductance from ach value, where ACh is 0 at baseline
// and goes up to 1 at US or CS -- effect is disinhibitory on MSNs
func (mp *MatrixParams) GiFmACh(ach float32) float32 {
	ai := mp.MaxACh - ach
	if ai < 0 {
		ai = 0
	}
	return mp.AChInhib * ai
}

//gosl: end pcore_layers

func (ly *Layer) MatrixLayerDefaults() {
	ly.Params.Act.Decay.Act = 0
	ly.Params.Act.Decay.Glong = 0
	ly.Params.Inhib.Pool.On = false
	ly.Params.Inhib.Layer.On = true
	ly.Params.Inhib.Layer.Gi = 0.5
	ly.Params.Inhib.Layer.FB = 0
	ly.Params.Inhib.Pool.FB = 0
	ly.Params.Inhib.Pool.Gi = 0.5
	ly.Params.Inhib.ActAvg.Nominal = 0.25

	// important: user needs to adjust wt scale of some PFC inputs vs others:
	// drivers vs. modulators

	for _, pji := range ly.RcvPrjns {
		pj := pji.(axon.AxonPrjn).AsAxon()
		pj.SWt.Init.SPct = 0
		if _, ok := pj.Send.(*GPLayer); ok { // From GPe TA or In
			pj.PrjnScale.Abs = 1
			pj.Params.Learn.Learn = false
			pj.SWt.Adapt.SigGain = 1
			pj.SWt.Init.Mean = 0.75
			pj.SWt.Init.Var = 0.0
			pj.SWt.Init.Sym = false
			if strings.HasSuffix(pj.Send.Name(), "GPeIn") { // GPeInToMtx
				pj.PrjnScale.Abs = 0.5 // counterbalance for GPeTA to reduce oscillations
			} else if strings.HasSuffix(pj.Send.Name(), "GPeTA") { // GPeTAToMtx
				if strings.HasSuffix(ly.Nm, "MtxGo") {
					pj.PrjnScale.Abs = 2 // was .8
				} else {
					pj.PrjnScale.Abs = 1 // was .3 GPeTAToMtxNo must be weaker to prevent oscillations, even with GPeIn offset
				}
			}
		}
	}

	ly.UpdateParams()
}
