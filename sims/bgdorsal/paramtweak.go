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
	// 	return []float32{0.6, 0.7}
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

	// {Sel: "#DMatrixNoToDMatrixGo", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Rel = val
	// }, Vals: func() []float32 {
	// 	return params.TweakPct(0.1, defTweakPct)
	// }},

	// {Sel: "#DGPeAkToDMatrixNo", Set: func(pt *axon.PathParams, val float32) {
	// 	pt.PathScale.Abs = val
	// }, Vals: func() []float32 {
	// 	return []float32{3, 2}
	// 	// .TweakPct(5, defTweakPct)
	// }},
}
