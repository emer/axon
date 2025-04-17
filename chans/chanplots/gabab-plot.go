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

type GABABPlot struct {
	// standard chans version of GABAB
	GABAB chans.GABABParams `display:"add-fields"`

	// multiplier on GABA-B as function of voltage
	Vgain float64 `default:"0.1"`

	// voltage offset for GABA-B exponential function
	Voff float64 `default:"10"`

	// GABAb reversal / driving potential
	Erev float64 `default:"-90"`

	// starting voltage
	Vstart float64 `default:"-90"`

	// ending voltage
	Vend float64 `default:"10"`

	// voltage increment
	Vstep float64 `default:"1"`

	// max number of spikes
	Smax int `default:"30"`

	// total number of time steps to take
	TimeSteps int

	// time increment per step
	TimeInc float64

	// time in msec for inputs to remain on in TimeRun
	TimeIn int

	// frequency of spiking inputs at start of TimeRun
	TimeHz float64

	Dir  *tensorfs.Node `display:"-"`
	Tabs lab.Tabber     `display:"-"`
}

// Config configures all the elements using the standard functions
func (pl *GABABPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("GabaB")
	pl.Tabs = tabs

	pl.GABAB.Defaults()
	pl.GABAB.GiSpike = 1
	pl.Vgain = 0.1
	pl.Voff = 10
	pl.Erev = -90
	pl.Vstart = -90
	pl.Vend = 10
	pl.Vstep = 1
	pl.Smax = 30
	pl.TimeSteps = 500
	pl.TimeInc = .001
	pl.TimeIn = 100
	pl.TimeHz = 50
	pl.Update()
}

// Update updates computed values
func (pl *GABABPlot) Update() {
	pl.GABAB.Update()
}

// GVRun plots the conductance G (and other variables) as a function of V.
func (pl *GABABPlot) GVRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_V")

	nv := int((pl.Vend - pl.Vstart) / pl.Vstep)
	for vi := range nv {
		v := pl.Vstart + float64(vi)*pl.Vstep
		g := float64(pl.GABAB.Gbar) / (1 + math.Exp(pl.Vgain*((v-pl.Erev)+pl.Voff)))
		i := (v - pl.Erev) * g

		dir.Float64("V", nv).SetFloat1D(v, vi)
		dir.Float64("Ggaba_b", nv).SetFloat1D(g, vi)
		dir.Float64("Igaba_b", nv).SetFloat1D(i, vi)
	}
	plot.SetFirstStyler(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Ggaba_b", "Igaba_b"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "GABA-B G(V)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

// GSRun plots conductance as function of spiking rate.
func (pl *GABABPlot) GSRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Spike")

	nv := int(float64(pl.Smax) / pl.Vstep)
	for si := range nv {
		s := float64(si) * pl.Vstep
		g := 1.0 / (1.0 + math.Exp(-(s-7.1)/1.4))

		dir.Float64("S", nv).SetFloat1D(s, si)
		dir.Float64("GgabaB_max", nv).SetFloat1D(g, si)
	}
	plot.SetFirstStyler(dir.Float64("S"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"GgabaB_max"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "GABAB G(spike)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

// TimeRun runs the equations over time.
func (pl *GABABPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Time")
	nv := pl.TimeSteps

	time := 0.0
	gs := 0.0
	x := 0.0
	spikeInt := int(1000 / pl.TimeHz)
	for ti := range nv {
		sin := 0.0
		if ti >= 10 && ti < (10+pl.TimeIn) && (ti-10)%spikeInt == 0 {
			sin = 1
		}

		// record starting state first, then update
		dir.Float64("Time", nv).SetFloat1D(time, ti)
		dir.Float64("GabaB", nv).SetFloat1D(gs, ti)
		dir.Float64("GabaBX", nv).SetFloat1D(x, ti)

		gis := 1.0 / (1.0 + math.Exp(-(sin-7.1)/1.4))
		dGs := (float64(pl.GABAB.TauFact)*x - gs) / float64(pl.GABAB.RiseTau)
		dXo := -x / float64(pl.GABAB.DecayTau)
		gs += dGs
		x += gis + dXo

		dir.Float64("dG", nv).SetFloat1D(dGs, ti)
		dir.Float64("dX", nv).SetFloat1D(dXo, ti)
		dir.Float64("Xmax", nv).SetFloat1D(gis, ti)

		time += pl.TimeInc
	}
	metadata.SetDoc(dir.Float64("GabaBstd"), "std is from code actually used in models")
	metadata.SetDoc(dir.Float64("GabaBXstd"), "std is from code actually used in models")
	plot.SetFirstStyler(dir.Float64("Time"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"GabaB", "GabaBX"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "GABAB G(t)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsLab().IsVisible() {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

func (pl *GABABPlot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GVRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GSRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}
