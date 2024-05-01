// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// sahp_plot plots an equation updating over time in a table.Table and Plot2D.
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
	sim.CaRun()
	b := sim.ConfigGUI()
	b.RunMainWindow()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// sAHP function
	Sahp chans.SahpParams `view:"inline"`

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
	ss.Sahp.Defaults()
	ss.Sahp.Gbar = 1
	ss.CaStart = 0
	ss.CaEnd = 1.5
	ss.CaStep = 0.01
	ss.TimeSteps = 30
	ss.TimeCaStart = 0
	ss.TimeCaD = 1
	ss.Update()
	ss.Table = &table.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &table.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// CaRun plots the equation as a function of V
func (ss *Sim) CaRun() { //types:add
	ss.Update()
	dt := ss.Table

	mp := &ss.Sahp

	nv := int((ss.CaEnd - ss.CaStart) / ss.CaStep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		ca := ss.CaStart + float32(vi)*ss.CaStep
		var ninf, tau float32
		mp.NinfTauFromCa(ca, &ninf, &tau)

		dt.SetFloat("Ca", vi, float64(ca))
		dt.SetFloat("Ninf", vi, float64(ninf))
		dt.SetFloat("Tau", vi, float64(tau))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "sAHPplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Ca", tensor.FLOAT64, nil, nil},
		{"Ninf", tensor.FLOAT64, nil, nil},
		{"Tau", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "sAHP Ca Function Plot"
	plt.Params.XAxisColumn = "Ca"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Ca", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Ninf", plotview.On, plotview.FixMin, 0, plotview.FixMax, 1)
	plt.SetColParams("Tau", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 1)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //types:add
	ss.Update()
	dt := ss.TimeTable

	mp := &ss.Sahp

	var n, tau float32
	mp.NinfTauFromCa(ss.TimeCaStart, &n, &tau)
	ca := ss.TimeCaStart

	dt.SetNumRows(ss.TimeSteps)
	for ti := 1; ti <= ss.TimeSteps; ti++ {
		t := float32(ti)

		var ninf, tau float32
		mp.NinfTauFromCa(ca, &ninf, &tau)
		dn := mp.DNFromV(ca, n)
		g := mp.GsAHP(n)

		dt.SetFloat("Time", ti, float64(t))
		dt.SetFloat("Ca", ti, float64(ca))
		dt.SetFloat("GsAHP", ti, float64(g))
		dt.SetFloat("N", ti, float64(n))
		dt.SetFloat("dN", ti, float64(dn))
		dt.SetFloat("Ninf", ti, float64(ninf))
		dt.SetFloat("Tau", ti, float64(tau))

		ca = mp.CaInt(ca, ss.TimeCaD)
		n += dn
	}
	if ss.TimePlot != nil {
		ss.TimePlot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTimeTable(dt *table.Table) {
	dt.SetMetaData("name", "sAHPplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Time", tensor.FLOAT64, nil, nil},
		{"Ca", tensor.FLOAT64, nil, nil},
		{"GsAHP", tensor.FLOAT64, nil, nil},
		{"N", tensor.FLOAT64, nil, nil},
		{"dN", tensor.FLOAT64, nil, nil},
		{"Ninf", tensor.FLOAT64, nil, nil},
		{"Tau", tensor.FLOAT64, nil, nil},
		{"Kna", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisColumn = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Ca", plotview.On, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GsAHP", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("N", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("dN", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Ninf", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Tau", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Kna", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 1)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Body {
	b := core.NewBody("Sahp Plot")

	split := core.NewSplits(b, "split")
	sv := views.NewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := core.NewTabs(split, "tv")

	ss.Plot = plotview.NewSubPlot(tv.NewTab("Ca-G Plot"))
	ss.ConfigPlot(ss.Plot, ss.Table)

	ss.TimePlot = plotview.NewSubPlot(tv.NewTab("TimePlot"))
	ss.ConfigTimePlot(ss.TimePlot, ss.TimeTable)

	split.SetSplits(.3, .7)

	b.AddAppBar(func(tb *core.Toolbar) {
		views.NewFuncButton(tb, ss.CaRun).SetIcon(icons.PlayArrow)
		views.NewFuncButton(tb, ss.TimeRun).SetIcon(icons.PlayArrow)
	})

	return b
}
