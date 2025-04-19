// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNMDA_GFromV(t *testing.T) {
	const step = .01
	const maxRelativeDiff = .6
	prevG := float32(math.NaN())
	prevVM := float32(math.NaN())
	var params NMDAParams
	params.Defaults()
	nv := int(100.0 / step)
	for vi := range nv {
		vBio := float32(-100.0) + float32(vi)*step
		g := params.MgGFromV(vBio)
		assert.Greater(t, g, float32(0), "for input %v", vBio)
		assert.LessOrEqual(t, g, float32(8.8), "for input %v", vBio)
		if !math.IsNaN(float64(prevG)) {
			// check for discontinuities
			assert.InEpsilon(t, prevG, g, maxRelativeDiff, "for inputs %v and %v", prevVM, vBio)
			// check for monotonicity
			if vBio < -24.93 {
				assert.GreaterOrEqualf(t, g, prevG, "for inputs %v and %v", prevVM, vBio)
			} else if vBio > -24.87 {
				assert.LessOrEqualf(t, g, prevG, "for inputs %v and %v", prevVM, vBio)
			}
		}
		prevG = g
		prevVM = vBio
	}
}

func TestVGGC_GFromV(t *testing.T) {
	const step = .01
	const maxRelativeDiff = .1
	prevG := float32(math.NaN())
	prevVM := float32(math.NaN())
	var params VGCCParams
	params.Defaults()
	nv := int(100.0 / step)
	for vi := range nv {
		vBio := float32(-100.0) + float32(vi)*step
		g := params.GFromV(vBio)
		assert.Greater(t, g, float32(0), "for input %v", vBio)
		// should really be <= 100, but FastExp introduces some error
		assert.LessOrEqual(t, g, float32(100.1), "for input %v", vBio)
		if !math.IsNaN(float64(prevG)) {
			// check for discontinuities
			assert.InEpsilon(t, prevG, g, maxRelativeDiff, "for inputs %v and %v", prevVM, vBio)
			// check for monotonicity
			assert.LessOrEqual(t, g, prevG, "for inputs %v and %v", prevVM, vBio)
		}
		prevG = g
		prevVM = vBio
	}
}

func TestVGGC_MFromV(t *testing.T) {
	const step = .01
	const maxRelativeDiff = .1
	prevBio := float32(math.NaN())
	prevVBio := float32(math.NaN())
	var params VGCCParams
	params.Defaults()
	nv := int(100.0 / step)
	for vi := range nv {
		vBio := float32(-100.0) + float32(vi)*step
		m := params.MFromV(vBio)
		assert.GreaterOrEqual(t, m, float32(0), "for input %v", vBio)
		assert.LessOrEqual(t, m, float32(1), "for input %v", vBio)
		if !math.IsNaN(float64(prevBio)) {
			// check for discontinuities
			assert.InEpsilon(t, prevBio, m, maxRelativeDiff, "for inputs %v and %v", prevVBio, vBio)
			// check for monotonicity
			assert.LessOrEqual(t, prevBio, m, "for inputs %v and %v", prevVBio, vBio)
			prevBio = m
		}
		prevVBio = vBio
	}
}

func TestVGGC_HFromV(t *testing.T) {
	const step = .01
	const maxRelativeDiff = .1
	prevH := float32(math.NaN())
	prevVBio := float32(math.NaN())
	var params VGCCParams
	params.Defaults()
	nv := int(100.0 / step)
	for vi := range nv {
		vBio := float32(-100.0) + float32(vi)*step
		h := params.HFromV(vBio)
		assert.GreaterOrEqual(t, h, float32(0), "for input %v", vBio)
		assert.LessOrEqual(t, h, float32(1), "for input %v", vBio)
		if !math.IsNaN(float64(prevH)) {
			// check for discontinuities
			assert.InEpsilon(t, prevH, h, maxRelativeDiff, "for inputs %v and %v", prevVBio, vBio)
			// check for monotonicity
			assert.LessOrEqual(t, h, prevH, "for inputs %v and %v", prevVBio, vBio)
			prevH = h
		}
		prevVBio = vBio
	}
}
