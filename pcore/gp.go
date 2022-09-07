// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
)

// GPLayer represents a globus pallidus layer, including:
// GPeOut, GPeIn, GPeTA (arkypallidal), and GPi (see GPLay for type).
// Typically just a single unit per Pool representing a given stripe.
type GPLayer struct {
	Layer
	GPLay GPLays `desc:"type of GP layer"`
}

var KiT_GPLayer = kit.Types.AddType(&GPLayer{}, axon.LayerProps)

// Defaults in param.Sheet format
// Sel: "GPLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.Pool.On":      "false",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// }}

func (ly *GPLayer) Defaults() {
	ly.Layer.Defaults()
	ly.DA = 0

	// GP is tonically self-active and has no FFFB inhibition

	ly.Act.Init.Ge = 0.3
	ly.Act.Init.GeVar = 0.05
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.4 // 0.4 in localist one
	ly.Inhib.Self.Tau = 3.0
	ly.Inhib.ActAvg.Init = 0.25

	switch ly.GPLay {
	case GPeIn:
		ly.Act.Init.Act = 0.81
		ly.Act.Init.Vm = 0.60
	case GPeTA:
		ly.Act.Init.Act = 0.26
		ly.Act.Init.Vm = 0.50
	}

	for _, pjii := range ly.RcvPrjns {
		pji := pjii.(axon.AxonPrjn)
		pj := pji.AsAxon()
		pj.Learn.Learn = false
		pj.SWt.Adapt.SigGain = 1
		pj.SWt.Init.Mean = 0.9
		pj.SWt.Init.Var = 0
		pj.SWt.Init.Sym = false
		if _, ok := pj.Send.(*MatrixLayer); ok {
			pj.PrjnScale.Abs = 0.5
		} else if _, ok := pj.Send.(*STNLayer); ok {
			pj.PrjnScale.Abs = 0.1 // default level for GPeOut and GPeTA -- weaker to not oppose GPeIn surge
		}
		switch ly.GPLay {
		case GPeIn:
			if _, ok := pj.Send.(*MatrixLayer); ok { // MtxNoToGPeIn -- primary NoGo pathway
				pj.PrjnScale.Abs = 1
			} else if _, ok := pj.Send.(*GPLayer); ok { // GPeOutToGPeIn
				pj.PrjnScale.Abs = 0.5
			}
			if _, ok := pj.Send.(*STNLayer); ok { // STNpToGPeIn -- stronger to drive burst of activity
				pj.PrjnScale.Abs = 0.5
			}
		case GPeOut:
		case GPeTA:
			if _, ok := pj.Send.(*GPLayer); ok { // GPeInToGPeTA
				pj.PrjnScale.Abs = 0.9 // just enough to knock down to near-zero at baseline
			}
		}
	}

	ly.UpdateParams()
}

//////////////////////////////////////////////////////////////////////
//  GPLays

// GPLays for GPLayer type
type GPLays int

//go:generate stringer -type=GPLays

var KiT_GPLays = kit.Enums.AddEnum(GPLaysN, kit.NotBitFlag, nil)

func (ev GPLays) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *GPLays) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// GPeOut is Outer layer of GPe neurons, receiving inhibition from MtxGo
	GPeOut GPLays = iota

	// GPeIn is Inner layer of GPe neurons, receiving inhibition from GPeOut and MtxNo
	GPeIn

	// GPeTA is arkypallidal layer of GPe neurons, receiving inhibition from GPeIn
	// and projecting inhibition to Mtx
	GPeTA

	// GPi is the inner globus pallidus, functionally equivalent to SNr,
	// receiving from MtxGo and GPeIn, and sending inhibition to VThal
	GPi

	GPLaysN
)
