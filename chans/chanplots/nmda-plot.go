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
	NMDA chans.NMDAParams `display:"add-fields"`

	// multiplier on NMDA as function of voltage
	Vgain float64 `default:"0.062"`

	// denominator of NMDA function
	Norm float64 `default:"3.57"`

	// reversal / driving potential
	Erev float64 `default:"0"`

	// starting voltage
	Vstart float64 `default:"-90"`

	// ending voltage
	Vend float64 `default:"10"`

	// voltage increment
	Vstep float64 `default:"1"`

	// number of 1msec time steps for time run
	TimeSteps int

	// clamped voltage for TimeRun
	TimeV float64

	// time in msec for inputs to remain on in TimeRun
	TimeIn int

	// frequency of spiking inputs at start of TimeRun
	TimeHz float64

	// proportion activation of NMDA channels per spike
	TimeGin float64

	Dir  *tensorfs.Node `display:"-"`
	Tabs lab.Tabber     `display:"-"`
}

// Config configures all the elements using the standard functions
func (pl *NMDAPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("NMDA")
	pl.Tabs = tabs

	pl.NMDA.Defaults()
	pl.NMDA.Ge = 1
	pl.NMDA.Voff = 0
	pl.Vgain = 0.062
	pl.Norm = 3.57
	pl.Erev = 0
	pl.Vstart = -90 // -90 -- use -1 1 to test val around 0
	pl.Vend = 10
	pl.Vstep = 1
	pl.TimeSteps = 500
	pl.TimeV = -50
	pl.TimeIn = 100
	pl.TimeHz = 50
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

	mgf := float64(pl.NMDA.MgC) / pl.Norm
	nv := int((pl.Vend - pl.Vstart) / pl.Vstep)
	for vi := range nv {
		v := pl.Vstart + float64(vi)*pl.Vstep
		g := float64(pl.NMDA.Ge) / (1 + mgf*math.Exp(-pl.Vgain*v))
		i := (pl.Erev - v) * g
		if v >= pl.Erev {
			i = 0
		}
		ca := pl.NMDA.CaFromV(float32(v))

		dir.Float64("V", nv).SetFloat1D(v, vi)
		dir.Float64("Gnmda", nv).SetFloat1D(g, vi)
		dir.Float64("Inmda", nv).SetFloat1D(i, vi)
		dir.Float64("Ca", nv).SetFloat1D(float64(ca), vi)
	}
	plot.SetFirstStyler(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gnmda", "Inmda"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "NMDA G(V)"
		})
	}
	metadata.SetDoc(dir.Float64("Gnmda_std"), "standard compute function used in axon sims")

	if pl.Tabs != nil {
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
	spikeInt := int(1000 / pl.TimeHz)
	for ti := range nv {
		t := float64(ti) * .001
		gin := 0.0
		if ti >= 10 && ti < (10+pl.TimeIn) && (ti-10)%spikeInt == 0 {
			gin = pl.TimeGin
		}
		nmda += gin*(1-nmda) - (nmda / float64(pl.NMDA.Tau))
		g = nmda / (1 + math.Exp(-pl.Vgain*v)/pl.Norm)

		dir.Float64("Time", nv).SetFloat1D(t, ti)
		dir.Float64("Gnmda", nv).SetFloat1D(g, ti)
		dir.Float64("NMDA", nv).SetFloat1D(nmda, ti)
	}
	plot.SetFirstStyler(dir.Float64("Time"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gnmda", "NMDA"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "NMDA G(t)"
		})
	}
	if pl.Tabs != nil {
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
