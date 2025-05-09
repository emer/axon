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

type SahpPlot struct {

	// sAHP function
	Sahp chans.SahpParams `display:"add-fields"`

	// starting calcium
	CaStart float32 `default:"0"`

	// ending calcium
	CaEnd float32 `default:"1.5"`

	// calcium increment
	CaStep float32 `default:"0.01"`

	// number of time steps
	TimeSteps int

	// time-run starting calcium
	TimeCaStart float32

	// time-run CaD value at end of each theta cycle
	TimeCaD float32

	Dir  *tensorfs.Node `display:"-"`
	Tabs lab.Tabber     `display:"-"`
}

// Config configures all the elements using the standard functions
func (pl *SahpPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("sAHP")
	pl.Tabs = tabs

	pl.Sahp.Defaults()
	pl.Sahp.Gk = 1
	pl.CaStart = 0
	pl.CaEnd = 1.5
	pl.CaStep = 0.01
	pl.TimeSteps = 30
	pl.TimeCaStart = 0
	pl.TimeCaD = 1
	pl.Update()
}

// Update updates computed values
func (pl *SahpPlot) Update() {
}

// GCaRun plots the conductance G (and other variables) as a function of Ca.
func (pl *SahpPlot) GCaRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Ca")

	mp := &pl.Sahp
	nv := int((pl.CaEnd - pl.CaStart) / pl.CaStep)
	for vi := range nv {
		ca := pl.CaStart + float32(vi)*pl.CaStep
		var ninf, tau float32
		mp.NinfTauFromCa(ca, &ninf, &tau)

		dir.Float64("Ca", nv).SetFloat1D(float64(ca), vi)
		dir.Float64("Ninf", nv).SetFloat1D(float64(ninf), vi)
		dir.Float64("Tau", nv).SetFloat1D(float64(tau), vi)
	}
	plot.SetFirstStyler(dir.Float64("Ca"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Ninf"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "sAHP G(Ca)"
		})
	}
	plot.SetFirstStyler(dir.Float64("Tau"), func(s *plot.Style) {
		s.On = true
		s.RightY = true
	})
	if pl.Tabs != nil {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

// TimeRun runs the equation over time.
func (pl *SahpPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Time")
	nv := pl.TimeSteps

	mp := &pl.Sahp
	var n, tau float32
	mp.NinfTauFromCa(pl.TimeCaStart, &n, &tau)
	ca := pl.TimeCaStart
	for ti := range nv {
		t := float32(ti + 1)

		var ninf, tau float32
		mp.NinfTauFromCa(ca, &ninf, &tau)
		dn := mp.DNFromV(ca, n)
		g := mp.GsAHP(n)

		dir.Float64("Time", nv).SetFloat1D(float64(t), ti)
		dir.Float64("Ca", nv).SetFloat1D(float64(ca), ti)
		dir.Float64("Gsahp", nv).SetFloat1D(float64(g), ti)
		dir.Float64("N", nv).SetFloat1D(float64(n), ti)
		dir.Float64("dN", nv).SetFloat1D(float64(dn), ti)
		dir.Float64("Ninf", nv).SetFloat1D(float64(ninf), ti)
		dir.Float64("Tau", nv).SetFloat1D(float64(tau), ti)

		ca = mp.CaInt(ca, pl.TimeCaD)
		n += dn
	}
	plot.SetFirstStyler(dir.Float64("Time"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Ca", "Gsahp", "N"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "sAHP G(t)"
		})
	}
	if pl.Tabs != nil {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

func (pl *SahpPlot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GCaRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}
