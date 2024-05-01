// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// nmda_plot plots an equation updating over time in a table.Table and Plot2D.
package main

//go:generate core generate -add-types

import (
	"math"
	"strconv"

	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/plot/plotview"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/views"
	"github.com/emer/axon/v2/chans"
)

func main() {
	sim := &Sim{}
	sim.Config()
	sim.Run()
	b := sim.ConfigGUI()
	b.RunMainWindow()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

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
	ss.NMDAStd.Defaults()
	ss.NMDAStd.Voff = 0
	ss.BugVoff = 5
	ss.NMDAv = 0.062
	ss.MgC = 1
	ss.NMDAd = 3.57
	ss.NMDAerev = 0
	ss.Vstart = -1 // -90 // -90 -- use -1 1 to test val around 0
	ss.Vend = 1    // 2     // 50
	ss.Vstep = .01 // use 0.001 instead for testing around 0
	ss.Tau = 100
	ss.TimeSteps = 1000
	ss.TimeV = -50
	ss.TimeGin = .5
	ss.Update()
	ss.Table = &table.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &table.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// Equation here:
// https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html

// Run runs the equation.
func (ss *Sim) Run() { //types:add
	ss.Update()
	dt := ss.Table

	mgf := ss.MgC / ss.NMDAd

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	v := 0.0
	g := 0.0
	gbug := 0.0
	for vi := 0; vi < nv; vi++ {
		v = ss.Vstart + float64(vi)*ss.Vstep
		if v >= 0 {
			g = 0
		} else {
			g = float64(ss.NMDAStd.Gbar) * (ss.NMDAerev - v) / (1 + mgf*math.Exp(-ss.NMDAv*v))
		}
		bugv := float32(v + ss.BugVoff)
		if bugv >= 0 {
			gbug = 0
		} else {
			gbug = 0.15 / (1.0 + float64(ss.NMDAStd.MgFact*math32.FastExp(float32(-0.062*bugv))))
		}

		gs := ss.NMDAStd.Gnmda(1, chans.VFromBio(float32(v)))
		ca := ss.NMDAStd.CaFromVbio(float32(v))

		dt.SetFloat("V", vi, v)
		dt.SetFloat("Gnmda", vi, g)
		dt.SetFloat("Gnmda_std", vi, float64(gs))
		dt.SetFloat("Gnmda_bug", vi, float64(gbug))
		dt.SetFloat("Ca", vi, float64(ca))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "NmDaplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"V", tensor.FLOAT64, nil, nil},
		{"Gnmda", tensor.FLOAT64, nil, nil},
		{"Gnmda_std", tensor.FLOAT64, nil, nil},
		{"Gnmda_bug", tensor.FLOAT64, nil, nil},
		{"Ca", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "NMDA V-G Function Plot"
	plt.Params.XAxisColumn = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Gnmda", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Gnmda_std", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Gnmda_bug", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Ca", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //types:add
	ss.Update()
	dt := ss.TimeTable

	v := ss.TimeV

	g := 0.0
	nmda := 0.0
	dt.SetNumRows(ss.TimeSteps)
	for ti := 0; ti < ss.TimeSteps; ti++ {
		t := float64(ti) * .001
		gin := ss.TimeGin
		if ti < 10 || ti > ss.TimeSteps/2 {
			gin = 0
		}
		nmda += gin*(1-nmda) - (nmda / ss.Tau)
		g = nmda / (1 + math.Exp(-ss.NMDAv*v)/ss.NMDAd)

		dt.SetFloat("Time", ti, t)
		dt.SetFloat("Gnmda", ti, g)
		dt.SetFloat("NMDA", ti, nmda)
	}
	if ss.TimePlot != nil {
		ss.TimePlot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTimeTable(dt *table.Table) {
	dt.SetMetaData("name", "NmDaplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Time", tensor.FLOAT64, nil, nil},
		{"Gnmda", tensor.FLOAT64, nil, nil},
		{"NMDA", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisColumn = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Gnmda", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("NMDA", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Body {
	b := core.NewBody("Nmda plot")

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
		views.NewFuncButton(tb, ss.Run).SetIcon(icons.PlayArrow)
		views.NewFuncButton(tb, ss.TimeRun).SetIcon(icons.PlayArrow)
	})

	return b
}
