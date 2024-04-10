package chans

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVGGC_GFromV(t *testing.T) {
	const step = .001
	const maxRelativeDiff = .1
	prevG := float32(math.NaN())
	prevVM := float32(math.NaN())
	var params VGCCParams
	for vM := float32(0); vM < 2; vM += step {
		g := params.GFromV(vM)
		assert.Greater(t, g, float32(0), "for input %v", vM)
		// should really be <= 100, but FastExp introduces some error
		assert.LessOrEqual(t, g, float32(100.1), "for input %v", vM)
		if !math.IsNaN(float64(prevG)) {
			// check for discontinuities
			assert.InEpsilon(t, prevG, g, maxRelativeDiff, "for inputs %v and %v", prevVM, vM)
			// check for monotonicity
			assert.LessOrEqual(t, g, prevG, "for inputs %v and %v", prevVM, vM)
		}
		prevG = g
		prevVM = vM
	}
}

func TestVGGC_MFromV(t *testing.T) {
	const step = .01
	const maxRelativeDiff = .1
	prevM := float32(math.NaN())
	prevVBio := float32(math.NaN())
	var params VGCCParams
	for vBio := float32(-100); vBio < 0; vBio += step {
		m := params.MFromV(vBio)
		assert.GreaterOrEqual(t, m, float32(0), "for input %v", vBio)
		assert.LessOrEqual(t, m, float32(1), "for input %v", vBio)
		if !math.IsNaN(float64(prevM)) {
			// check for discontinuities
			assert.InEpsilon(t, prevM, m, maxRelativeDiff, "for inputs %v and %v", prevVBio, vBio)
			// check for monotonicity
			assert.LessOrEqual(t, prevM, m, "for inputs %v and %v", prevVBio, vBio)
			prevM = m
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
	for vBio := float32(-100); vBio < 0; vBio += step {
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
