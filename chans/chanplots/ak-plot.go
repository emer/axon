// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chanplots

//go:generate core generate -add-types -gosl

import (
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/lab"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/chans"
)

type AKPlot struct {

	// AKs simplified function
	AKs chans.AKsParams

	// AK function
	AK AKParams

	// starting voltage
	Vstart float32 `default:"-100"`

	// ending voltage
	Vend float32 `default:"50"`

	// voltage increment
	Vstep float32 `default:"1"`

	// number of time steps
	TimeSteps int

	// do spiking instead of voltage ramp
	TimeSpike bool

	// spiking frequency
	SpikeFreq float32

	// time-run starting membrane potential
	TimeVstart float32

	// time-run ending membrane potential
	TimeVend float32

	Dir  *tensorfs.Node `display:"-"`
	Tabs lab.Tabber     `display:"-"`
}

// Config configures the plot
func (pl *AKPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("AK")
	pl.Tabs = tabs

	pl.AK.Defaults()
	pl.AK.Gk = 1
	pl.AKs.Defaults()
	pl.AKs.Gk = 1
	pl.Vstart = -100
	pl.Vend = 100
	pl.Vstep = 1
	pl.TimeSteps = 200
	pl.TimeSpike = true
	pl.SpikeFreq = 50
	pl.TimeVstart = -50
	pl.TimeVend = -20
	pl.Update()
}

func (pl *AKPlot) Update() {
	pl.AK.Update()
}

// GVRun plots the conductance G (and other variables) as a function of V.
func (pl *AKPlot) GVRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_V")

	ap := &pl.AK
	nv := int((pl.Vend - pl.Vstart) / pl.Vstep)
	for vi := range nv {
		v := pl.Vstart + float32(vi)*pl.Vstep
		k := ap.KFromV(v)
		a := ap.AlphaFromVK(v, k)
		b := ap.BetaFromVK(v, k)
		mt := ap.MTauFromAlphaBeta(a, b)
		ht := ap.HTauFromV(v)
		m := ap.MFromAlpha(a)
		h := ap.HFromV(v)
		g := ap.Gak(m, h)

		ms := pl.AKs.MFromV(v)
		gs := pl.AKs.Gak(v)

		dir.Float64("V", nv).SetFloat1D(float64(v), vi)
		dir.Float64("Gaks", nv).SetFloat1D(float64(gs), vi)
		dir.Float64("Gak", nv).SetFloat1D(float64(g), vi)
		dir.Float64("M", nv).SetFloat1D(float64(m), vi)
		dir.Float64("H", nv).SetFloat1D(float64(h), vi)
		dir.Float64("MTau", nv).SetFloat1D(float64(mt), vi)
		dir.Float64("HTau", nv).SetFloat1D(float64(ht), vi)
		dir.Float64("K", nv).SetFloat1D(float64(k), vi)
		dir.Float64("Alpha", nv).SetFloat1D(float64(a), vi)
		dir.Float64("Beta", nv).SetFloat1D(float64(b), vi)
		dir.Float64("Ms", nv).SetFloat1D(float64(ms), vi)
	}
	metadata.SetDoc(dir.Float64("Gaks"), "Gaks is the simplified AK conductance, actually used in models")
	metadata.SetDoc(dir.Float64("Ms"), "Ms is the simplified AK M gate, actually used in models")
	plot.SetFirstStyler(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gak", "Gaks"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "AK G(V)"
		})
	}
	if pl.Tabs != nil {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

// TimeRun runs the equations over time.
func (pl *AKPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Time")
	nv := pl.TimeSteps

	ap := &pl.AK
	m := float32(0)
	h := float32(1)
	msdt := float32(0.001)
	v := pl.TimeVstart
	vinc := float32(2) * (pl.TimeVend - pl.TimeVstart) / float32(pl.TimeSteps)

	isi := int(1000 / pl.SpikeFreq)
	var g float32
	for ti := range nv {
		t := float32(ti) * msdt

		k := ap.KFromV(v)
		a := ap.AlphaFromVK(v, k)
		b := ap.BetaFromVK(v, k)
		mt := ap.MTauFromAlphaBeta(a, b)
		ht := ap.HTauFromV(v)
		g = ap.Gak(m, h)

		dm, dh := pl.AK.DMHFromV(v, m, h)

		dir.Float64("Time", nv).SetFloat1D(float64(t), ti)
		dir.Float64("Gak", nv).SetFloat1D(float64(g), ti)
		dir.Float64("M", nv).SetFloat1D(float64(m), ti)
		dir.Float64("H", nv).SetFloat1D(float64(h), ti)
		dir.Float64("dM", nv).SetFloat1D(float64(dm), ti)
		dir.Float64("dH", nv).SetFloat1D(float64(dh), ti)
		dir.Float64("MTau", nv).SetFloat1D(float64(mt), ti)
		dir.Float64("HTau", nv).SetFloat1D(float64(ht), ti)
		dir.Float64("K", nv).SetFloat1D(float64(k), ti)
		dir.Float64("Alpha", nv).SetFloat1D(float64(a), ti)
		dir.Float64("Beta", nv).SetFloat1D(float64(b), ti)

		g = pl.AK.Gak(m, h)
		m += dm // already in msec time constants
		h += dh

		if pl.TimeSpike {
			if ti%isi < 3 {
				v = pl.TimeVend
			} else {
				v = pl.TimeVstart
			}
		} else {
			v += vinc
			if v > pl.TimeVend {
				v = pl.TimeVend
			}
		}
	}
	plot.SetFirstStyler(dir.Float64("Time"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gak", "M", "H"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "AK G(t)"
		})
	}
	if pl.Tabs != nil {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

func (pl *AKPlot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GVRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}

// AKParams control an A-type K+ channel, which is voltage gated with maximal
// activation around -37 mV.  It has two state variables, M (v-gated opening)
// and H (v-gated closing), which integrate with fast and slow time constants,
// respectively.  H relatively quickly hits an asymptotic level of inactivation
// for sustained activity patterns.
// It is particularly important for counteracting the excitatory effects of
// voltage gated calcium channels which can otherwise drive runaway excitatory currents.
// See AKsParams for a much simpler version that works fine when full AP-like spikes are
// not simulated, as in our standard axon models.
type AKParams struct {

	// Gk is the strength of the AK conductance contribution to Gk(t) factor
	// (which is then multiplied by Gbar.K that provides pA unit scaling).
	Gk float32 `default:"0.1,0.01,1"`

	// Beta multiplier for the beta term; 0.01446 for distal, 0.02039
	// for proximal dendrites.
	Beta float32 `default:"0.01446,02039"`

	// Dm factor: 0.5 for distal, 0.25 for proximal
	Dm float32 `default:"0.5,0.25"`

	// K is the offset for K, 1.8 for distal, 1.5 for proximal.
	Koff float32 `default:"1.8,1.5"`

	// Voff is the voltage offset for alpha and beta functions: 1 for distal,
	// 11 for proximal.
	Voff float32 `default:"1,11"`

	// Hf is the h multiplier factor, 0.1133 for distal, 0.1112 for proximal.
	Hf float32 `default:"0.1133,0.1112"`

	pad, pad1 float32
}

// Defaults sets the parameters for distal dendrites
func (ap *AKParams) Defaults() {
	ap.Gk = 0.01
	ap.Distal()
}

func (ap *AKParams) Update() {
}

func (ap *AKParams) ShouldDisplay(field string) bool {
	switch field {
	case "Gk":
		return true
	default:
		return ap.Gk > 0
	}
}

// Distal sets the parameters for distal dendrites
func (ap *AKParams) Distal() {
	ap.Beta = 0.01446
	ap.Dm = 0.5
	ap.Koff = 1.8
	ap.Voff = 1
	ap.Hf = 0.1133
}

// Proximal sets parameters for proximal dendrites
func (ap *AKParams) Proximal() {
	ap.Beta = 0.02039
	ap.Dm = 0.25
	ap.Koff = 1.5
	ap.Voff = 11
	ap.Hf = 0.1112
}

// AlphaFromVK returns the Alpha function from v (not normalized, must not exceed 0)
func (ap *AKParams) AlphaFromVK(v, k float32) float32 {
	return math32.FastExp(0.03707 * k * (v - ap.Voff))
}

// BetaFromVK returns the Beta function from v (not normalized, must not exceed 0)
func (ap *AKParams) BetaFromVK(v, k float32) float32 {
	return math32.FastExp(ap.Beta * k * (v - ap.Voff))
}

// KFromV returns the K value from v (not normalized, must not exceed 0)
func (ap *AKParams) KFromV(v float32) float32 {
	return -ap.Koff - 1.0/(1.0+math32.FastExp((v+40)/5))
}

// HFromV returns the H gate value from v (not normalized, must not exceed 0)
func (ap *AKParams) HFromV(v float32) float32 {
	return 1.0 / (1.0 + math32.FastExp(ap.Hf*(v+56)))
}

// HTauFromV returns the HTau rate constant in msec from v (clipped above 0)
func (ap *AKParams) HTauFromV(v float32) float32 {
	ve := min(v, 0)
	tau := 0.26 * (ve + 50)
	if tau < 2 {
		tau = 2
	}
	return tau
}

// MFromAlpha returns the M gate factor from alpha
func (ap *AKParams) MFromAlpha(alpha float32) float32 {
	return 1.0 / (1.0 + alpha)
}

// MTauFromAlphaBeta returns the MTau rate constant in msec from alpha, beta
func (ap *AKParams) MTauFromAlphaBeta(alpha, beta float32) float32 {
	return 1 + beta/(ap.Dm*(1+alpha)) // minimum of 1 msec
}

// DMHFromV returns the change at msec update scale in M, H factors
// as a function of V.
func (ap *AKParams) DMHFromV(v, m, h float32) (float32, float32) {
	k := ap.KFromV(v)
	a := ap.AlphaFromVK(v, k)
	b := ap.BetaFromVK(v, k)
	mt := ap.MTauFromAlphaBeta(a, b)
	ht := ap.HTauFromV(v)
	dm := (ap.MFromAlpha(a) - m) / mt
	dh := (ap.HFromV(v) - h) / ht
	return dm, dh
}

// Gak returns the AK net conductance from m, h gates.
func (ap *AKParams) Gak(m, h float32) float32 {
	return ap.Gk * m * h
}
