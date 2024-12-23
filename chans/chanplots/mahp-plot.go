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

type MahpPlot struct {

	// mAHP function
	Mahp chans.MahpParams `display:"inline"`

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

// Config configures all the elements using the standard functions
func (pl *MahpPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("Mahp")
	pl.Tabs = tabs

	pl.Mahp.Defaults()
	pl.Mahp.Gbar = 1
	pl.Vstart = -100
	pl.Vend = 100
	pl.Vstep = 1
	pl.TimeSteps = 300
	pl.TimeSpike = true
	pl.SpikeFreq = 50
	pl.TimeVstart = -70
	pl.TimeVend = -50
	pl.Update()
}

// Update updates computed values
func (pl *MahpPlot) Update() {
}

// GVRun plots the conductance G (and other variables) as a function of V.
func (pl *MahpPlot) GVRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_V")

	mp := &pl.Mahp
	nv := int((pl.Vend - pl.Vstart) / pl.Vstep)
	for vi := range nv {
		vbio := pl.Vstart + float32(vi)*pl.Vstep
		var ninf, tau float32
		mp.NinfTauFromV(vbio, &ninf, &tau)

		dir.Float64("V", nv).SetFloat1D(float64(vbio), vi)
		dir.Float64("Ninf", nv).SetFloat1D(float64(ninf), vi)
		dir.Float64("Tau", nv).SetFloat1D(float64(tau), vi)
	}
	plot.SetFirstStylerTo(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Ninf", "Tau"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "Mahp G(V)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

// TimeRun runs the equation over time.
func (pl *MahpPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Time")
	nv := pl.TimeSteps

	mp := &pl.Mahp

	var n, tau float32
	mp.NinfTauFromV(pl.TimeVstart, &n, &tau)
	kna := float32(0)
	msdt := float32(0.001)
	v := pl.TimeVstart
	vinc := float32(2) * (pl.TimeVend - pl.TimeVstart) / float32(pl.TimeSteps)

	isi := int(1000 / pl.SpikeFreq)
	for ti := range nv {
		vnorm := chans.VFromBio(v)
		t := float32(ti+1) * msdt

		var ninf, tau float32
		mp.NinfTauFromV(v, &ninf, &tau)
		g := mp.GmAHP(vnorm, &n)

		dir.Float64("Time", nv).SetFloat1D(float64(t), ti)
		dir.Float64("V", nv).SetFloat1D(float64(v), ti)
		dir.Float64("GmAHP", nv).SetFloat1D(float64(g), ti)
		dir.Float64("N", nv).SetFloat1D(float64(n), ti)
		dir.Float64("Ninf", nv).SetFloat1D(float64(ninf), ti)
		dir.Float64("Tau", nv).SetFloat1D(float64(tau), ti)
		dir.Float64("Kna", nv).SetFloat1D(float64(kna), ti)

		if pl.TimeSpike {
			si := ti % isi
			if si == 0 {
				v = pl.TimeVend
				kna += 0.05 * (1 - kna)
			} else {
				v = pl.TimeVstart + (float32(si)/float32(isi))*(pl.TimeVend-pl.TimeVstart)
				kna -= kna / 50
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
	ons := []string{"GmAHP", "N"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "Mahp G(t)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

func (pl *MahpPlot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GVRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}
