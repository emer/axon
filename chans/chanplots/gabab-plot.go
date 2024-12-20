// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chanplots

import (
	"math"

	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/plot"
	"cogentcore.org/core/tensor/databrowser"
	"cogentcore.org/core/tensor/tensorfs"
	"cogentcore.org/core/tree"
	"github.com/emer/axon/v2/chans"
)

type GababPlot struct {
	// standard chans version of GABAB
	GABAstd chans.GABABParams

	// multiplier on GABAb as function of voltage
	GABAbv float64 `default:"0.1"`

	// offset of GABAb function
	GABAbo float64 `default:"10"`

	// GABAb reversal / driving potential
	GABAberev float64 `default:"-90"`

	// starting voltage
	Vstart float64 `default:"-90"`

	// ending voltage
	Vend float64 `default:"0"`

	// voltage increment
	Vstep float64 `default:"1"`

	// max number of spikes
	Smax int `default:"15"`

	// rise time constant
	RiseTau float64

	// decay time constant -- must NOT be same as RiseTau
	DecayTau float64

	// initial value of GsX driving variable at point of synaptic input onset -- decays expoentially from this start
	GsXInit float64

	// time when peak conductance occurs, in TimeInc units
	MaxTime float64 `edit:"-"`

	// time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))
	TauFact float64 `edit:"-"`

	// total number of time steps to take
	TimeSteps int

	// time increment per step
	TimeInc float64

	Dir  *tensorfs.Node     `display:"-"`
	Tabs databrowser.Tabber `display:"-"`
}

// Config configures all the elements using the standard functions
func (pl *GababPlot) Config(parent *tensorfs.Node, tabs databrowser.Tabber) {
	pl.Dir = parent.Dir("GabaB")
	pl.Tabs = tabs

	pl.GABAstd.Defaults()
	pl.GABAstd.GiSpike = 1
	pl.GABAbv = 0.1
	pl.GABAbo = 10
	pl.GABAberev = -90
	pl.Vstart = -90
	pl.Vend = 0
	pl.Vstep = .01
	pl.Smax = 30
	pl.RiseTau = 45
	pl.DecayTau = 50
	pl.GsXInit = 1
	pl.TimeSteps = 200
	pl.TimeInc = .001
	pl.Update()
}

// Update updates computed values
func (pl *GababPlot) Update() {
	pl.TauFact = math.Pow(pl.DecayTau/pl.RiseTau, pl.RiseTau/(pl.DecayTau-pl.RiseTau))
	pl.MaxTime = ((pl.RiseTau * pl.DecayTau) / (pl.DecayTau - pl.RiseTau)) * math.Log(pl.DecayTau/pl.RiseTau)
}

// GVRun plots the conductance G (and other variables) as a function of V.
func (pl *GababPlot) GVRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_V")

	nv := int((pl.Vend - pl.Vstart) / pl.Vstep)
	v := 0.0
	g := 0.0
	for vi := range nv {
		v = pl.Vstart + float64(vi)*pl.Vstep
		g = float64(pl.GABAstd.Gbar) * (v - pl.GABAberev) / (1 + math.Exp(pl.GABAbv*((v-pl.GABAberev)+pl.GABAbo)))
		gs := pl.GABAstd.Gbar * pl.GABAstd.GFromV(chans.VFromBio(float32(v)))

		dir.Float64("V", nv).SetFloat1D(v, vi)
		dir.Float64("GgabaB", nv).SetFloat1D(g, vi)
		dir.Float64("GgabaBstd", nv).SetFloat1D(float64(gs), vi)
	}
	metadata.SetDoc(dir.Float64("GgabaBstd"), "std is from code actually used in models")
	plot.SetFirstStylerTo(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"GgabaB"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "GABA-B G(V)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsDataTabs().IsVisible() {
		pl.Tabs.PlotTensorFS(dir)
	}
}

// GSRun plots conductance over spiking.
func (pl *GababPlot) GSRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Spike")

	nv := int(float64(pl.Smax) / pl.Vstep)
	s := 0.0
	g := 0.0
	for si := range nv {
		s = float64(si) * pl.Vstep
		g = 1.0 / (1.0 + math.Exp(-(s-7.1)/1.4))
		gs := pl.GABAstd.GFromS(float32(s))

		dir.Float64("S", nv).SetFloat1D(s, si)
		dir.Float64("GgabaB_max", nv).SetFloat1D(g, si)
		dir.Float64("GgabaBstd_max", nv).SetFloat1D(float64(gs), si)
	}
	metadata.SetDoc(dir.Float64("GgabaBstd_max"), "std is from code actually used in models")
	plot.SetFirstStylerTo(dir.Float64("S"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"GgabaB_max"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "GABAB G(spike)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsDataTabs().IsVisible() {
		pl.Tabs.PlotTensorFS(dir)
	}
}

// TimeRun runs the equations over time.
func (pl *GababPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Time")
	nv := pl.TimeSteps

	time := 0.0
	gs := 0.0
	x := pl.GsXInit
	gabaBx := float32(pl.GsXInit)
	gabaB := float32(0.0)
	gi := 0.0 // just goes down
	for t := range nv {
		// record starting state first, then update
		dir.Float64("Time", nv).SetFloat1D(float64(time), t)
		dir.Float64("GabaB", nv).SetFloat1D(float64(gs), t)
		dir.Float64("GabaBX", nv).SetFloat1D(float64(x), t)
		dir.Float64("GabaBstd", nv).SetFloat1D(float64(gabaB), t)
		dir.Float64("GabaBXstd", nv).SetFloat1D(float64(gabaBx), t)

		gis := 1.0 / (1.0 + math.Exp(-(gi-7.1)/1.4))
		dGs := (pl.TauFact*x - gs) / pl.RiseTau
		dXo := -x / pl.DecayTau
		gs += dGs
		x += gis + dXo

		var dG, dX float32
		pl.GABAstd.BiExp(gabaB, gabaBx, &dG, &dX)
		dir.Float64("dG", nv).SetFloat1D(float64(dG), t)
		dir.Float64("dX", nv).SetFloat1D(float64(dX), t)

		pl.GABAstd.GABAB(float32(gi), &gabaB, &gabaBx)

		time += pl.TimeInc
	}
	metadata.SetDoc(dir.Float64("GabaBstd"), "std is from code actually used in models")
	metadata.SetDoc(dir.Float64("GabaBXstd"), "std is from code actually used in models")
	plot.SetFirstStylerTo(dir.Float64("Time"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"GabaB", "GabaBX"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "GABAB G(t)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsDataTabs().IsVisible() {
		pl.Tabs.PlotTensorFS(dir)
	}
}

func (pl *GababPlot) MakeToolbar(p *tree.Plan) {
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
