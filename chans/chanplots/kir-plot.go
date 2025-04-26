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

type KirPlot struct {

	// Kir function
	Kir chans.KirParams `display:"add-fields"`

	// Vstart is starting voltage
	Vstart float32 `default:"-100"`

	// Vend is ending voltage
	Vend float32 `default:"100"`

	// Vstep is voltage increment
	Vstep float32 `default:"1"`

	// TimeSteps is number of time steps
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
func (pl *KirPlot) Config(parent *tensorfs.Node, tabs lab.Tabber) {
	pl.Dir = parent.Dir("kIR")
	pl.Tabs = tabs

	pl.Kir.Defaults()
	pl.Kir.Gk = 1
	pl.Vstart = -100
	pl.Vend = 0
	pl.Vstep = 1
	pl.TimeSteps = 300
	pl.TimeSpike = true
	pl.SpikeFreq = 50
	pl.TimeVstart = -70
	pl.TimeVend = -50
	pl.Update()
}

// Update updates computed values
func (pl *KirPlot) Update() {
}

// VmRun plots the equation as a function of V
func (pl *KirPlot) GVRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_V")

	mp := &pl.Kir
	nv := int((pl.Vend - pl.Vstart) / pl.Vstep)
	m := mp.MinfRest()
	for vi := 0; vi < nv; vi++ {
		v := pl.Vstart + float32(vi)*pl.Vstep
		g := mp.Gkir(v, m)
		dm := mp.DM(v, m)
		m += dm
		minf := mp.Minf(v)
		mtau := mp.MTau(v)

		dir.Float64("V", nv).SetFloat1D(float64(v), vi)
		dir.Float64("Gkir", nv).SetFloat1D(float64(g), vi)
		dir.Float64("M", nv).SetFloat1D(float64(m), vi)
		dir.Float64("Minf", nv).SetFloat1D(float64(minf), vi)
		dir.Float64("Mtau", nv).SetFloat1D(float64(mtau), vi)
	}
	plot.SetFirstStyler(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"Gkir", "M"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "kIR G(V)"
		})
	}
	if pl.Tabs != nil {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

// TimeRun runs the equation over time.
func (pl *KirPlot) TimeRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_Time")
	nv := pl.TimeSteps

	mp := &pl.Kir

	m := mp.MinfRest()
	msdt := float32(0.001)
	v := pl.TimeVstart
	vinc := float32(2) * (pl.TimeVend - pl.TimeVstart) / float32(pl.TimeSteps)

	isi := int(1000 / pl.SpikeFreq)

	for ti := range nv {
		t := float32(ti+1) * msdt

		g := mp.Gkir(v, m)
		dm := mp.DM(v, m)
		m += dm
		minf := mp.Minf(v)
		mtau := mp.MTau(v)

		dir.Float64("Time", nv).SetFloat1D(float64(t), ti)
		dir.Float64("V", nv).SetFloat1D(float64(v), ti)
		dir.Float64("Gkir", nv).SetFloat1D(float64(g), ti)
		dir.Float64("M", nv).SetFloat1D(float64(m), ti)
		dir.Float64("Minf", nv).SetFloat1D(float64(minf), ti)
		dir.Float64("Mtau", nv).SetFloat1D(float64(mtau), ti)

		if pl.TimeSpike {
			si := ti % isi
			if si == 0 {
				v = pl.TimeVend
			} else {
				v = pl.TimeVstart + (float32(si)/float32(isi))*(pl.TimeVend-pl.TimeVstart)
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
	plot.SetFirstStyler(dir.Float64("V"), func(s *plot.Style) {
		s.On = true
		s.Plot.Title = "Gkir G(t)"
		s.RightY = true
	})
	ons := []string{"Gkir", "M"}
	for _, on := range ons {
		plot.SetFirstStyler(dir.Float64(on), func(s *plot.Style) {
			s.On = true
		})
	}
	if pl.Tabs != nil {
		pl.Tabs.AsLab().PlotTensorFS(dir)
	}
}

func (pl *KirPlot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GVRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}
