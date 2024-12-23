// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chanplots

import (
	"math"

	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/lab"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/chans"
)

type NMDAPlot struct {

	// standard NMDA implementation in chans
	NMDAStd chans.NMDAParams

	// multiplier on NMDA as function of voltage
	NMDAv float64 `default:"0.062"`

	// magnesium ion concentration -- somewhere between 1 and 1.5
	MgC float64

	// denominator of NMDA function
	NMDAd float64 `default:"3.57"`

	// NMDA reversal / driving potential
	NMDAerev float64 `default:"0"`

	// for old buggy NMDA: voff value to use
	BugVoff float64

	// starting voltage
	Vstart float64 `default:"-90"`

	// ending voltage
	Vend float64 `default:"10"`

	// voltage increment
	Vstep float64 `default:"1"`

	// decay time constant for NMDA current -- rise time is 2 msec and not worth extra effort for biexponential
	Tau float64 `default:"100"`

	// number of time steps
	TimeSteps int

	// voltage for TimeRun
	TimeV float64

	// NMDA Gsyn current input at every time step
	TimeGin float64

	Dir  *tensorfs.Node `display:"-"`
	Tabs lab.Tabber     `display:"-"`
}

// Config configures all the elements using the standard functions
func (pl *NMDAPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("NMDA")
	pl.Tabs = tabs

	pl.NMDAStd.Defaults()
	pl.NMDAStd.Voff = 0
	pl.BugVoff = 5
	pl.NMDAv = 0.062
	pl.MgC = 1
	pl.NMDAd = 3.57
	pl.NMDAerev = 0
	pl.Vstart = -1 // -90 // -90 -- use -1 1 to test val around 0
	pl.Vend = 1    // 2     // 50
	pl.Vstep = .01 // use 0.001 instead for testing around 0
	pl.Tau = 100
	pl.TimeSteps = 1000
	pl.TimeV = -50
	pl.TimeGin = .5
	pl.Update()
}

// Update updates computed values
func (pl *NMDAPlot) Update() {
}

// Equation here:
// https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html

// GVRun plots the conductance G (and other variables) as a function of V.
func (pl *NMDAPlot) GVRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_V")

	mgf := pl.MgC / pl.NMDAd
	nv := int((pl.Vend - pl.Vstart) / pl.Vstep)
	v := 0.0
	g := 0.0
	for vi := range nv {
		v = pl.Vstart + float64(vi)*pl.Vstep
		if v >= 0 {
			g = 0
		} else {
			g = float64(pl.NMDAStd.Gbar) * (pl.NMDAerev - v) / (1 + mgf*math.Exp(-pl.NMDAv*v))
		}

		gs := pl.NMDAStd.Gnmda(1, chans.VFromBio(float32(v)))
		ca := pl.NMDAStd.CaFromVbio(float32(v))

		dir.Float64("V", nv).SetFloat1D(v, vi)
		dir.Float64("Gnmda", nv).SetFloat1D(g, vi)
		dir.Float64("Gnmda_std", nv).SetFloat1D(float64(gs), vi)
		dir.Float64("Ca", nv).SetFloat1D(float64(ca), vi)
	}
	metadata.SetDoc(dir.Float64("Gnmda_std"), "std is the standard equations actually used in models")
	plot.SetFirstStylerTo(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gnmda"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "NMDA G(V)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

// TimeRun runs the equation over time.
func (pl *NMDAPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Time")
	nv := pl.TimeSteps

	v := pl.TimeV
	g := 0.0
	nmda := 0.0
	for ti := range nv {
		t := float64(ti) * .001
		gin := pl.TimeGin
		if ti < 10 || ti > pl.TimeSteps/2 {
			gin = 0
		}
		nmda += gin*(1-nmda) - (nmda / pl.Tau)
		g = nmda / (1 + math.Exp(-pl.NMDAv*v)/pl.NMDAd)

		dir.Float64("Time", nv).SetFloat1D(t, ti)
		dir.Float64("Gnmda", nv).SetFloat1D(g, ti)
		dir.Float64("NMDA", nv).SetFloat1D(nmda, ti)
	}
	plot.SetFirstStylerTo(dir.Float64("Time"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gnmda", "NMDA"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "NMDA G(t)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

func (pl *NMDAPlot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GVRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}
