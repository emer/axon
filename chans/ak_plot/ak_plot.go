// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ak_plot plots an equation updating over time in a table.Table and Plot2D.
package main

//go:generate core generate -add-types

import (
	"strconv"

	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/plot/plotview"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/views"
	"github.com/emer/axon/v2/chans"
)

func main() {
	sim := &Sim{}
	sim.Config()
	sim.VmRun()
	b := sim.ConfigGUI()
	b.RunMainWindow()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// AK function
	AK chans.AKParams

	// AKs simplified function
	AKs chans.AKsParams

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

	// table for plot
	Table *table.Table `view:"no-inline"`

	// the plot
	Plot *plotview.PlotView `view:"-"`

	// table for plot
	TimeTable *table.Table `view:"no-inline"`

	// the plot
	TimePlot *plotview.PlotView `view:"-"`
}

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.AK.Defaults()
	ss.AK.Gbar = 1
	ss.AKs.Defaults()
	ss.AKs.Gbar = 1
	ss.Vstart = -100
	ss.Vend = 100
	ss.Vstep = .01
	ss.TimeSteps = 200
	ss.TimeSpike = true
	ss.SpikeFreq = 50
	ss.TimeVstart = -50
	ss.TimeVend = -20
	ss.Update()
	ss.Table = &table.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &table.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// VmRun plots the equation as a function of V
func (ss *Sim) VmRun() { //types:add
	ss.Update()
	dt := ss.Table

	ap := &ss.AK

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		vbio := ss.Vstart + float32(vi)*ss.Vstep
		vnorm := chans.VFromBio(vbio)
		k := ap.KFromV(vbio)
		a := ap.AlphaFromVK(vbio, k)
		b := ap.BetaFromVK(vbio, k)
		mt := ap.MTauFromAlphaBeta(a, b)
		ht := ap.HTauFromV(vbio)
		m := ap.MFromAlpha(a)
		h := ap.HFromV(vbio)
		g := ap.Gak(m, h)

		ms := ss.AKs.MFromV(vbio)
		gs := ss.AKs.Gak(vnorm)

		dt.SetFloat("V", vi, float64(vbio))
		dt.SetFloat("Gak", vi, float64(g))
		dt.SetFloat("M", vi, float64(m))
		dt.SetFloat("H", vi, float64(h))
		dt.SetFloat("MTau", vi, float64(mt))
		dt.SetFloat("HTau", vi, float64(ht))
		dt.SetFloat("K", vi, float64(k))
		dt.SetFloat("Alpha", vi, float64(a))
		dt.SetFloat("Beta", vi, float64(b))

		dt.SetFloat("Ms", vi, float64(ms))
		dt.SetFloat("Gaks", vi, float64(gs))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "AkplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"V", tensor.FLOAT64, nil, nil},
		{"Gak", tensor.FLOAT64, nil, nil},
		{"M", tensor.FLOAT64, nil, nil},
		{"H", tensor.FLOAT64, nil, nil},
		{"MTau", tensor.FLOAT64, nil, nil},
		{"HTau", tensor.FLOAT64, nil, nil},
		{"K", tensor.FLOAT64, nil, nil},
		{"Alpha", tensor.FLOAT64, nil, nil},
		{"Beta", tensor.FLOAT64, nil, nil},
		{"Ms", tensor.FLOAT64, nil, nil},
		{"Gaks", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "AK V-G Function Plot"
	plt.Params.XAxisColumn = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Gak", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Gaks", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("M", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Ms", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("H", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("MTau", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("HTau", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("K", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Alpha", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Beta", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //types:add
	ss.Update()
	dt := ss.TimeTable

	ap := &ss.AK

	m := float32(0)
	h := float32(1)
	msdt := float32(0.001)
	v := ss.TimeVstart
	vinc := float32(2) * (ss.TimeVend - ss.TimeVstart) / float32(ss.TimeSteps)

	isi := int(1000 / ss.SpikeFreq)

	dt.SetNumRows(ss.TimeSteps)
	var g float32
	for ti := 0; ti < ss.TimeSteps; ti++ {
		vnorm := chans.VFromBio(v)
		t := float32(ti) * msdt

		k := ap.KFromV(v)
		a := ap.AlphaFromVK(v, k)
		b := ap.BetaFromVK(v, k)
		mt := ap.MTauFromAlphaBeta(a, b)
		ht := ap.HTauFromV(v)
		g = ap.Gak(m, h)

		dm, dh := ss.AK.DMHFromV(vnorm, m, h)

		dt.SetFloat("Time", ti, float64(t))
		dt.SetFloat("V", ti, float64(v))
		dt.SetFloat("Gak", ti, float64(g))
		dt.SetFloat("M", ti, float64(m))
		dt.SetFloat("H", ti, float64(h))
		dt.SetFloat("dM", ti, float64(dm))
		dt.SetFloat("dH", ti, float64(dh))
		dt.SetFloat("MTau", ti, float64(mt))
		dt.SetFloat("HTau", ti, float64(ht))
		dt.SetFloat("K", ti, float64(k))
		dt.SetFloat("Alpha", ti, float64(a))
		dt.SetFloat("Beta", ti, float64(b))

		g = ss.AK.Gak(m, h)
		m += dm // already in msec time constants
		h += dh

		if ss.TimeSpike {
			if ti%isi < 3 {
				v = ss.TimeVend
			} else {
				v = ss.TimeVstart
			}
		} else {
			v += vinc
			if v > ss.TimeVend {
				v = ss.TimeVend
			}
		}
	}
	if ss.TimePlot != nil {
		ss.TimePlot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTimeTable(dt *table.Table) {
	dt.SetMetaData("name", "AkplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Time", tensor.FLOAT64, nil, nil},
		{"V", tensor.FLOAT64, nil, nil},
		{"Gak", tensor.FLOAT64, nil, nil},
		{"M", tensor.FLOAT64, nil, nil},
		{"H", tensor.FLOAT64, nil, nil},
		{"dM", tensor.FLOAT64, nil, nil},
		{"dH", tensor.FLOAT64, nil, nil},
		{"MTau", tensor.FLOAT64, nil, nil},
		{"HTau", tensor.FLOAT64, nil, nil},
		{"K", tensor.FLOAT64, nil, nil},
		{"Alpha", tensor.FLOAT64, nil, nil},
		{"Beta", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisColumn = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Gak", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("M", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("H", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("dM", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("dH", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("MTau", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("HTau", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("K", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Alpha", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Beta", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Body {
	b := core.NewBody("Ak Plot")

	split := core.NewSplits(b, "split")
	sv := views.NewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := core.NewTabs(split, "tv")

	ss.Plot = plotview.NewSubPlot(tv.NewTab("V-G Plot"))
	ss.ConfigPlot(ss.Plot, ss.Table)

	ss.TimePlot = plotview.NewSubPlot(tv.NewTab("TimePlot"))
	ss.ConfigTimePlot(ss.TimePlot, ss.TimeTable)

	split.SetSplits(.3, .7)

	b.AddAppBar(func(tb *core.Toolbar) {
		views.NewFuncButton(tb, ss.VmRun).SetIcon(icons.PlayArrow)
		views.NewFuncButton(tb, ss.TimeRun).SetIcon(icons.PlayArrow)
	})

	return b
}
