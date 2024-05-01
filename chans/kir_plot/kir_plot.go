// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// kir_plot plots an equation updating over time in a table.Table and Plot2D.
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
	ss.Kir.Defaults()
	ss.Kir.Gbar = 1
	ss.Vstart = -100
	ss.Vend = 0
	ss.Vstep = 1
	ss.TimeSteps = 300
	ss.TimeSpike = true
	ss.SpikeFreq = 50
	ss.TimeVstart = -70
	ss.TimeVend = -50
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

	mp := &ss.Kir

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	m := mp.MinfRest()
	for vi := 0; vi < nv; vi++ {
		vbio := ss.Vstart + float32(vi)*ss.Vstep
		v := chans.VFromBio(vbio)
		g := mp.Gkir(v, &m)
		var minf, mtau float32
		mp.MRates(vbio, &minf, &mtau)

		dt.SetFloat("V", vi, float64(vbio))
		dt.SetFloat("GkIR", vi, float64(g))
		dt.SetFloat("M", vi, float64(m))
		dt.SetFloat("Minf", vi, float64(minf))
		dt.SetFloat("Mtau", vi, float64(mtau))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "kIRplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"V", tensor.FLOAT64, nil, nil},
		{"GkIR", tensor.FLOAT64, nil, nil},
		{"M", tensor.FLOAT64, nil, nil},
		{"Minf", tensor.FLOAT64, nil, nil},
		{"Mtau", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "kIR V Function Plot"
	plt.Params.XAxisColumn = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GkIR", plotview.On, plotview.FixMin, 0, plotview.FixMax, 1)
	plt.SetColParams("M", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 1)
	plt.SetColParams("Minf", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 1)
	plt.SetColParams("Mtau", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 1)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //types:add
	ss.Update()
	dt := ss.TimeTable

	mp := &ss.Kir

	m := mp.MinfRest()
	msdt := float32(0.001)
	v := ss.TimeVstart
	vinc := float32(2) * (ss.TimeVend - ss.TimeVstart) / float32(ss.TimeSteps)

	isi := int(1000 / ss.SpikeFreq)

	dt.SetNumRows(ss.TimeSteps)
	for ti := 1; ti <= ss.TimeSteps; ti++ {
		vnorm := chans.VFromBio(v)
		t := float32(ti) * msdt

		g := mp.Gkir(vnorm, &m)
		var minf, mtau float32
		mp.MRates(v, &minf, &mtau)

		dt.SetFloat("Time", ti, float64(t))
		dt.SetFloat("V", ti, float64(v))
		dt.SetFloat("GkIR", ti, float64(g))
		dt.SetFloat("M", ti, float64(m))
		dt.SetFloat("Minf", ti, float64(minf))
		dt.SetFloat("Mtau", ti, float64(mtau))

		if ss.TimeSpike {
			si := ti % isi
			if si == 0 {
				v = ss.TimeVend
			} else {
				v = ss.TimeVstart + (float32(si)/float32(isi))*(ss.TimeVend-ss.TimeVstart)
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
	dt.SetMetaData("name", "kIRplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Time", tensor.FLOAT64, nil, nil},
		{"V", tensor.FLOAT64, nil, nil},
		{"GkIR", tensor.FLOAT64, nil, nil},
		{"M", tensor.FLOAT64, nil, nil},
		{"Minf", tensor.FLOAT64, nil, nil},
		{"Mtau", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisColumn = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("V", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GkIR", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("M", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Minf", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 1)
	plt.SetColParams("Mtau", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 1)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Body {
	b := core.NewBody("Kir Plot")

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
