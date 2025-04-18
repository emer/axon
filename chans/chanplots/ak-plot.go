// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chanplots

//go:generate core generate -add-types

import (
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/lab"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/chans"
)

type AKPlot struct {

	// AK function
	AK chans.AKParams

	// AKs simplified function
	AKs chans.AKsParams

	// starting voltage
	Vstart float32 `default:"-100"`

	// ending voltage
	Vend float32 `default:"100"`

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
		dir.Float64("Gak", nv).SetFloat1D(float64(g), vi)
		dir.Float64("M", nv).SetFloat1D(float64(m), vi)
		dir.Float64("H", nv).SetFloat1D(float64(h), vi)
		dir.Float64("MTau", nv).SetFloat1D(float64(mt), vi)
		dir.Float64("HTau", nv).SetFloat1D(float64(ht), vi)
		dir.Float64("K", nv).SetFloat1D(float64(k), vi)
		dir.Float64("Alpha", nv).SetFloat1D(float64(a), vi)
		dir.Float64("Beta", nv).SetFloat1D(float64(b), vi)
		dir.Float64("Ms", nv).SetFloat1D(float64(ms), vi)
		dir.Float64("Gaks", nv).SetFloat1D(float64(gs), vi)
	}
	metadata.SetDoc(dir.Float64("Gaks"), "Gaks is the simplified AK conductance, actually used in models")
	metadata.SetDoc(dir.Float64("Ms"), "Ms is the simplified AK M gate, actually used in models")
	plot.SetFirstStyler(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gak", "M", "H", "Gaks"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "AK G(V)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
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
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
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
