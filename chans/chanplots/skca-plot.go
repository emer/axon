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
	"github.com/emer/axon/v2/kinase"
)

type SKCaPlot struct {

	// SKCa params
	SKCa chans.SKCaParams

	// time constants for integrating Ca from spiking across M, P and D cascading levels
	CaParams kinase.NeurCaParams

	// threshold of SK M gating factor above which the neuron cannot spike
	NoSpikeThr float32 `default:"0.5"`

	// Ca conc increment for M gating func plot
	CaStep float32 `default:"0.05"`

	// number of time steps
	TimeSteps int

	// do spiking instead of Ca conc ramp
	TimeSpike bool

	// spiking frequency
	SpikeFreq float32

	Dir  *tensorfs.Node `display:"-"`
	Tabs lab.Tabber     `display:"-"`
}

// Config configures all the elements using the standard functions
func (pl *SKCaPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("SKCa")
	pl.Tabs = tabs

	pl.SKCa.Defaults()
	pl.SKCa.Gbar = 1
	pl.CaParams.Defaults()
	pl.CaStep = .05
	pl.TimeSteps = 200 * 3
	pl.TimeSpike = true
	pl.NoSpikeThr = 0.5
	pl.SpikeFreq = 100
	pl.Update()
}

// Update updates computed values
func (pl *SKCaPlot) Update() {
}

// GCaRun plots the conductance G (and other variables) as a function of Ca.
func (pl *SKCaPlot) GCaRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_V")

	nv := int(1.0 / pl.CaStep)
	for vi := range nv {
		cai := float32(vi) * pl.CaStep
		mh := pl.SKCa.MAsympHill(cai)
		mg := pl.SKCa.MAsympGW06(cai)

		dir.Float64("Ca", nv).SetFloat1D(float64(cai), vi)
		dir.Float64("Mhill", nv).SetFloat1D(float64(mh), vi)
		dir.Float64("Mgw06", nv).SetFloat1D(float64(mg), vi)
	}
	plot.SetFirstStylerTo(dir.Float64("Ca"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Mhill", "Mgw06"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "sK Ca G(Ca)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

// TimeRun runs the equation over time.
func (pl *SKCaPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Time")
	nv := pl.TimeSteps

	caIn := float32(1)
	caR := float32(0)
	m := float32(0)
	spike := float32(0)
	msdt := float32(0.001)

	// caM := float32(0)
	// caP := float32(0)
	caD := float32(0)

	isi := int(1000 / pl.SpikeFreq)
	trial := 0
	for ti := range nv {
		trial = ti / 200
		t := float32(ti) * msdt
		m = pl.SKCa.MFromCa(caR, m)
		pl.SKCa.CaInRFromSpike(spike, caD, &caIn, &caR)

		dir.Float64("Time", nv).SetFloat1D(float64(t), ti)
		dir.Float64("Spike", nv).SetFloat1D(float64(spike), ti)
		// dir.Float64("CaM", nv).SetFloat1D(float64(caM), ti)
		// dir.Float64("CaP", nv).SetFloat1D(float64(caP), ti)
		// dir.Float64("CaD", nv).SetFloat1D(float64(caD), ti)
		dir.Float64("CaIn", nv).SetFloat1D(float64(caIn), ti)
		dir.Float64("CaR", nv).SetFloat1D(float64(caR), ti)
		dir.Float64("M", nv).SetFloat1D(float64(m), ti)

		if m < pl.NoSpikeThr && trial%2 == 0 && ti%isi == 0 { // spike on even trials
			spike = 1
		} else {
			spike = 0
		}
		// todo: update
		// ss.CaParams.FromSpike(spike, &caM, &caP, &caD)
	}
	plot.SetFirstStylerTo(dir.Float64("Time"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Spike", "CaIn", "CaR", "M"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "sK Ca G(t)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

func (pl *SKCaPlot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GCaRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}
