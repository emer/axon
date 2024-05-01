// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// mahp_plot plots an equation updating over time in a table.Table and Plot2D.
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

	// mAHP function
	Mahp chans.MahpParams `view:"inline"`

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
	ss.Mahp.Defaults()
	ss.Mahp.Gbar = 1
	ss.Vstart = -100
	ss.Vend = 100
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

	mp := &ss.Mahp

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		vbio := ss.Vstart + float32(vi)*ss.Vstep
		var ninf, tau float32
		mp.NinfTauFromV(vbio, &ninf, &tau)

		dt.SetFloat("V", vi, float64(vbio))
		dt.SetFloat("Ninf", vi, float64(ninf))
		dt.SetFloat("Tau", vi, float64(tau))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "mAHPplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"V", tensor.FLOAT64, nil, nil},
		{"Ninf", tensor.FLOAT64, nil, nil},
		{"Tau", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "mAHP V Function Plot"
	plt.Params.XAxisColumn = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Ninf", plotview.On, plotview.FixMin, 0, plotview.FixMax, 1)
	plt.SetColParams("Tau", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 1)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //types:add
	ss.Update()
	dt := ss.TimeTable

	mp := &ss.Mahp

	var n, tau float32
	mp.NinfTauFromV(ss.TimeVstart, &n, &tau)
	kna := float32(0)
	msdt := float32(0.001)
	v := ss.TimeVstart
	vinc := float32(2) * (ss.TimeVend - ss.TimeVstart) / float32(ss.TimeSteps)

	isi := int(1000 / ss.SpikeFreq)

	dt.SetNumRows(ss.TimeSteps)
	for ti := 1; ti <= ss.TimeSteps; ti++ {
		vnorm := chans.VFromBio(v)
		t := float32(ti) * msdt

		var ninf, tau float32
		mp.NinfTauFromV(v, &ninf, &tau)
		g := mp.GmAHP(vnorm, &n)

		dt.SetFloat("Time", ti, float64(t))
		dt.SetFloat("V", ti, float64(v))
		dt.SetFloat("GmAHP", ti, float64(g))
		dt.SetFloat("N", ti, float64(n))
		dt.SetFloat("Ninf", ti, float64(ninf))
		dt.SetFloat("Tau", ti, float64(tau))
		dt.SetFloat("Kna", ti, float64(kna))

		if ss.TimeSpike {
			si := ti % isi
			if si == 0 {
				v = ss.TimeVend
				kna += 0.05 * (1 - kna)
			} else {
				v = ss.TimeVstart + (float32(si)/float32(isi))*(ss.TimeVend-ss.TimeVstart)
				kna -= kna / 50
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
	dt.SetMetaData("name", "mAHPplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Time", tensor.FLOAT64, nil, nil},
		{"V", tensor.FLOAT64, nil, nil},
		{"GmAHP", tensor.FLOAT64, nil, nil},
		{"N", tensor.FLOAT64, nil, nil},
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
	plt.SetColParams("V", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GmAHP", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("N", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Ninf", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Tau", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Kna", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 1)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Body {
	b := core.NewBody("Mahp Plot")

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
