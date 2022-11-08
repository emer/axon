// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"strings"

	"github.com/Astera-org/axon/axon"
	"github.com/goki/ki/kit"
)

// GPiLayer represents the GPi / SNr output nucleus of the BG.
// It gets inhibited by the MtxGo and GPeIn layers, and its minimum
// activation during this inhibition is recorded in ActLrn, for learning.
// Typically just a single unit per Pool representing a given stripe.
type GPiLayer struct {
	GPLayer
}

var KiT_GPiLayer = kit.Types.AddType(&GPiLayer{}, LayerProps)

func (ly *GPiLayer) Defaults() {
	ly.GPLayer.Defaults()
	ly.GPLay = GPi

	ly.Act.Init.Ge = 0.6
	// note: GPLayer took care of STN input prjns

	for _, pji := range ly.RcvPrjns {
		pj := pji.(axon.AxonPrjn).AsAxon()
		pj.SWt.Adapt.SigGain = 1
		pj.SWt.Init.SPct = 0
		pj.SWt.Init.Mean = 0.75
		pj.SWt.Init.Var = 0.25
		pj.SWt.Init.Sym = false
		pj.Learn.Learn = false
		if _, ok := pj.Send.(*MatrixLayer); ok { // MtxGoToGPi
			pj.PrjnScale.Abs = 0.8 // slightly weaker than GPeIn
		} else if _, ok := pj.Send.(*GPLayer); ok { // GPeInToGPi
			pj.PrjnScale.Abs = 1 // stronger because integrated signal, also act can be weaker
		} else if strings.HasSuffix(pj.Send.Name(), "STNp") { // STNpToGPi
			pj.PrjnScale.Abs = 1
		} else if strings.HasSuffix(pj.Send.Name(), "STNs") { // STNsToGPi
			pj.PrjnScale.Abs = 0.3
		}
	}

	ly.UpdateParams()
}
