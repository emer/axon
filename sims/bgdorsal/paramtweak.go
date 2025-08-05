// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgdorsal

import (
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/params"
)

var (
	defSearch   = params.Increment
	defTweakPct = float32(0.2)
)

var PSearch = axon.PathSearches{
	// {Sel: "#DGPiToM1VM", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(2, defTweakPct)
	// }},

	// {Sel: "#DGPiToMotorBS", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(3, defTweakPct)
	// }},

	// {Sel: "#DGPiToPF", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	// return params.TweakPct(0.5, .1)
	// 	return []float32{1, 1.2, 1.5}
	// }},

	// {Sel: "#StateToM1", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(1, defTweakPct)
	// }},

	// {Sel: "#MotorBSToPF", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(1, defTweakPct)
	// }},

	// {Sel: ".M1ToMotorBS", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(2, defTweakPct)
	// }},

	// {Sel: "#M1PTToMotorBS", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(2, defTweakPct)
	// }},

	// {Sel: "#M1PTToVL", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(1, defTweakPct)
	// }},

	// {Sel: "#M1ToMotorBS", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(1.5, defTweakPct)
	// }},

	//////// basic BG 2nd batch

	// {Sel: "#DGPePrToDGPi", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(1, defTweakPct)
	// }},
	// {Sel: "#DMatrixGoToDGPi", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(1, defTweakPct)
	// }},
	// {Sel: "#DSTNToDGPi", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(0.2, defTweakPct)
	// }},

	// {Sel: "#DMatrixNoToDGPePr", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(1, defTweakPct)
	// }},
	// {Sel: "#DGPePrToDGPePr", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(4, defTweakPct)
	// }},
	// {Sel: "#DSTNToDGPePr", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(0.5, defTweakPct)
	// }},

	// {Sel: "#DGPePrToDGPeAk", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(1, defTweakPct)
	// }},
	// {Sel: "#DMatrixGoToDGPeAk", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return []float32{0.7, 0.8}
	// 	// params.TweakPct(0.5, defTweakPct)
	// }},
	// {Sel: "#DSTNToDGPeAk", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(0.1, defTweakPct)
	// }},

	// {Sel: "#DGPePrToDSTN", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(0.5, defTweakPct)
	// }},
	// {Sel: "#StateToDSTN", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(2, defTweakPct)
	// }},
	// {Sel: "#S1ToDSTN", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(2, defTweakPct)
	// }},

	// {Sel: "#DMatrixNoToDMatrixGo", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Rel = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(1, defTweakPct)
	// }},
	// {Sel: "#DGPeAkToDMatrixGo", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(3, defTweakPct)
	// }},

	// {Sel: "#DGPeAkToDMatrixNo", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return []float32{3, 2}
	// 	// .TweakPct(4, defTweakPct)
	// }},

}
