// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"strings"

	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
)

// VThalLayer represents the Ventral thalamus: VA / VM / VL,
// which receives BG gating in the form of inhibitory projection from GPi.
type VThalLayer struct {
	Layer
}

var KiT_VThalLayer = kit.Types.AddType(&VThalLayer{}, axon.LayerProps)

// Defaults in param.Sheet format
// Sel: "VThalLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.Pool.On":     "false",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3", // fastest
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *VThalLayer) Defaults() {
	ly.Layer.Defaults()

	// note: not tonically active

	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.4 // 0.4 in localist one
	ly.Inhib.Self.Tau = 3.0
	ly.Inhib.ActAvg.Init = 0.25
	// ly.Act.XX1.Gain = 20 // more graded -- still works with 40 but less Rt distrib
	// ly.Act.Dt.VmTau = 3.3
	// ly.Act.Dt.GTau = 3 // fastest
	// ly.Act.Init.Decay = 0

	for _, pji := range ly.RcvPrjns {
		pj := pji.(axon.AxonPrjn).AsAxon()
		pj.Learn.Learn = false
		pj.SWt.Adapt.SigGain = 1
		pj.SWt.Init.Mean = 0.9
		pj.SWt.Init.Var = 0
		pj.SWt.Init.Sym = false
		if strings.HasSuffix(pj.Send.Name(), "GPi") { // GPiToVThal
			pj.PrjnScale.Abs = 2.5 // 2.5 needed for agate model..
		}
	}

	ly.UpdateParams()
}
