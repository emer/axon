// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// kir_plot plots the kIR inward rectifying potassium (K) channel equations.
package kir_plot

//go:generate core generate -add-types

import (
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/plot"
	"cogentcore.org/core/tensor/databrowser"
	"cogentcore.org/core/tensor/tensorfs"
	"cogentcore.org/core/tree"
	"github.com/emer/axon/v2/chans"
)

type Plot struct {

	// kIR function
	Kir chans.KirParams

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

	Dir  *tensorfs.Node     `display:"-"`
	Tabs databrowser.Tabber `display:"-"`
}

// Config configures all the elements using the standard functions
func (pl *Plot) Config(parent *tensorfs.Node, tabs databrowser.Tabber) {
	pl.Dir = parent.Dir("kIR")
	pl.Tabs = tabs

	pl.Kir.Defaults()
	pl.Kir.Gbar = 1
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
func (pl *Plot) Update() {
}

// VmRun plots the equation as a function of V
func (pl *Plot) GVRun() { //types:add
	pl.Update()
	dir := pl.Dir.Dir("G_V")

	mp := &pl.Kir
	nv := int((pl.Vend - pl.Vstart) / pl.Vstep)
	m := mp.MinfRest()
	for vi := 0; vi < nv; vi++ {
		vbio := pl.Vstart + float32(vi)*pl.Vstep
		v := chans.VFromBio(vbio)
		g := mp.Gkir(v, m)
		dm := mp.DM(vbio, m)
		m += dm
		minf := mp.Minf(vbio)
		mtau := mp.MTau(vbio)

		dir.Float64("V", nv).SetFloat1D(float64(vbio), vi)
		dir.Float64("GkIR", nv).SetFloat1D(float64(g), vi)
		dir.Float64("M", nv).SetFloat1D(float64(m), vi)
		dir.Float64("Minf", nv).SetFloat1D(float64(minf), vi)
		dir.Float64("Mtau", nv).SetFloat1D(float64(mtau), vi)
	}
	plot.SetFirstStylerTo(dir.Float64("V"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"GkIR", "M"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "kIR G(V)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsDataTabs().IsVisible() {
		pl.Tabs.PlotTensorFS(dir)
	}
}

// TimeRun runs the equation over time.
func (pl *Plot) TimeRun() { //types:add
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
		vnorm := chans.VFromBio(v)
		t := float32(ti+1) * msdt

		g := mp.Gkir(vnorm, m)
		dm := mp.DM(v, m)
		m += dm
		minf := mp.Minf(v)
		mtau := mp.MTau(v)

		dir.Float64("Time", nv).SetFloat1D(float64(t), ti)
		dir.Float64("V", nv).SetFloat1D(float64(v), ti)
		dir.Float64("GkIR", nv).SetFloat1D(float64(g), ti)
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
	plot.SetFirstStylerTo(dir.Float64("Time"), func(s *plot.Style) {
		s.Role = plot.X
	})
	ons := []string{"V", "GkIR", "M"}
	for _, on := range ons {
		plot.SetFirstStylerTo(dir.Float64(on), func(s *plot.Style) {
			s.On = true
			s.Plot.Title = "GkIR G(t)"
		})
	}
	if pl.Tabs != nil && pl.Tabs.AsDataTabs().IsVisible() {
		pl.Tabs.PlotTensorFS(dir)
	}
}

func (pl *Plot) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.GVRun).SetIcon(icons.PlayArrow)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(pl.TimeRun).SetIcon(icons.PlayArrow)
	})
}
