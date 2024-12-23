// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chanplots

import (
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/lab"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/chans"
)

type VGCCPlot struct {

	// VGCC function
	VGCC chans.VGCCParams

	// starting voltage
	Vstart float32 `default:"-90"`

	// ending voltage
	Vend float32 `default:"0"`

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

// Config configures all the elements using the standard functions
func (pl *VGCCPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("VGCC")
	pl.Tabs = tabs

	pl.VGCC.Defaults()
	pl.VGCC.Gbar = 1
	pl.Vstart = -90
	pl.Vend = 2
	pl.Vstep = 0.01
	pl.TimeSteps = 200
	pl.TimeSpike = true
	pl.SpikeFreq = 50
	pl.TimeVstart = -70
	pl.TimeVend = -20
	pl.Update()
}

// Update updates computed values
func (pl *VGCCPlot) Update() {
}

// GVRun plots the conductance G (and other variables) as a function of V.
func (pl *VGCCPlot) GVRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_V")

	nv := int((pl.Vend - pl.Vstart) / pl.Vstep)
	for vi := range nv {
		v := pl.Vstart + float32(vi)*pl.Vstep
		vnorm := chans.VFromBio(v)
		g := pl.VGCC.GFromV(vnorm)
		m := pl.VGCC.MFromV(v)
		h := pl.VGCC.HFromV(v)
		var dm, dh float32
		pl.VGCC.DMHFromV(vnorm, m, h, &dm, &dh)

		dir.Float64("V", nv).SetFloat1D(float64(v), vi)
		dir.Float64("Gvgcc", nv).SetFloat1D(float64(g), vi)
		dir.Float64("M", nv).SetFloat1D(float64(m), vi)
		dir.Float64("H", nv).SetFloat1D(float64(h), vi)
		dir.Float64("dM", nv).SetFloat1D(float64(dm), vi)
		dir.Float64("dH", nv).SetFloat1D(float64(dh), vi)
	}
	plot.SetFirstStylerTo(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gvgcc", "M", "H"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "VGCC G(V)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

// TimeRun runs the equation over time.
func (pl *VGCCPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Time")
	nv := pl.TimeSteps

	m := float32(0)
	h := float32(1)
	msdt := float32(0.001)
	v := pl.TimeVstart
	vinc := float32(2) * (pl.TimeVend - pl.TimeVstart) / float32(pl.TimeSteps)

	isi := int(1000 / pl.SpikeFreq)
	var g float32

	for ti := range nv {
		vnorm := chans.VFromBio(v)
		t := float32(ti) * msdt
		g = pl.VGCC.Gvgcc(vnorm, m, h)
		var dm, dh float32
		pl.VGCC.DMHFromV(vnorm, m, h, &dm, &dh)
		m += dm
		h += dh

		dir.Float64("Time", nv).SetFloat1D(float64(t), ti)
		dir.Float64("V", nv).SetFloat1D(float64(v), ti)
		dir.Float64("Gvgcc", nv).SetFloat1D(float64(g), ti)
		dir.Float64("M", nv).SetFloat1D(float64(m), ti)
		dir.Float64("H", nv).SetFloat1D(float64(h), ti)
		dir.Float64("dM", nv).SetFloat1D(float64(dm), ti)
		dir.Float64("dH", nv).SetFloat1D(float64(dh), ti)

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
	plot.SetFirstStylerTo(dir.Float64("Time"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gvgcc", "V", "M", "H"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "VGCC G(t)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

func (pl *VGCCPlot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GVRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}
