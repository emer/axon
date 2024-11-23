// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ska_plot plots an equation updating over time in a table.Table and PlotView.
package main

//go:generate core generate -add-types

import (
	"strconv"

	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/plot/plotcore"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tree"
	"github.com/emer/axon/v2/chans"
	"github.com/emer/axon/v2/kinase"
)

func main() {
	sim := &Sim{}
	sim.Config()
	sim.CamRun()
	b := sim.ConfigGUI()
	b.RunMainWindow()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// SKCa params
	SKCa chans.SKCaParams

	// time constants for integrating Ca from spiking across M, P and D cascading levels
	CaParams kinase.NeurCaParams

	// threshold of SK M gating factor above which the neuron cannot spike
	NoSpikeThr float32 `default:"0.5"`

	// Ca conc increment for M gating func plot
	CaStep float32 `default:"0.05"`

	// number of time steps
	TimeSteps int

	// do spiking instead of Ca conc ramp
	TimeSpike bool

	// spiking frequency
	SpikeFreq float32

	// table for plot
	Table *table.Table `display:"no-inline"`

	// the plot
	Plot *plotcore.PlotEditor `display:"-"`

	// table for plot
	TimeTable *table.Table `display:"no-inline"`

	// the plot
	TimePlot *plotcore.PlotEditor `display:"-"`
}

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.SKCa.Defaults()
	ss.SKCa.Gbar = 1
	ss.CaParams.Defaults()
	ss.CaStep = .05
	ss.TimeSteps = 200 * 3
	ss.TimeSpike = true
	ss.NoSpikeThr = 0.5
	ss.SpikeFreq = 100
	ss.Update()
	ss.Table = table.New()
	ss.ConfigTable(ss.Table)
	ss.TimeTable = table.New()
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// CamRun plots the equation as a function of Ca
func (ss *Sim) CamRun() { //types:add
	ss.Update()
	dt := ss.Table

	nv := int(1.0 / ss.CaStep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		cai := float32(vi) * ss.CaStep
		mh := ss.SKCa.MAsympHill(cai)
		mg := ss.SKCa.MAsympGW06(cai)

		dt.SetFloat("Ca", vi, float64(cai))
		dt.SetFloat("Mhill", vi, float64(mh))
		dt.SetFloat("Mgw06", vi, float64(mg))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "SKCaPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.AddFloat64Column("Ca")
	dt.AddFloat64Column("Mhill")
	dt.AddFloat64Column("Mgw06")
	dt.SetNumRows(0)
}

func (ss *Sim) ConfigPlot(plt *plotcore.PlotEditor, dt *table.Table) *plotcore.PlotEditor {
	plt.Options.Title = "SKCa Ca-G Function Plot"
	plt.Options.XAxis = "Ca"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColumnOptions("Ca", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColumnOptions("Mhill", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("Mgw06", plotcore.Off, plotcore.FixMin, 0, plotcore.FloatMax, 1)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //types:add
	ss.Update()
	dt := ss.TimeTable

	caIn := float32(1)
	caR := float32(0)
	m := float32(0)
	spike := float32(0)
	msdt := float32(0.001)

	caM := float32(0)
	caP := float32(0)
	caD := float32(0)

	isi := int(1000 / ss.SpikeFreq)
	trial := 0

	dt.SetNumRows(ss.TimeSteps)
	for ti := 0; ti < ss.TimeSteps; ti++ {
		trial = ti / 200
		t := float32(ti) * msdt
		m = ss.SKCa.MFromCa(caR, m)
		ss.SKCa.CaInRFromSpike(spike, caD, &caIn, &caR)

		dt.SetFloat("Time", ti, float64(t))
		dt.SetFloat("Spike", ti, float64(spike))
		dt.SetFloat("CaM", ti, float64(caM))
		dt.SetFloat("CaP", ti, float64(caP))
		dt.SetFloat("CaD", ti, float64(caD))
		dt.SetFloat("CaIn", ti, float64(caIn))
		dt.SetFloat("CaR", ti, float64(caR))
		dt.SetFloat("M", ti, float64(m))

		if m < ss.NoSpikeThr && trial%2 == 0 && ti%isi == 0 { // spike on even trials
			spike = 1
		} else {
			spike = 0
		}
		// todo: update
		// ss.CaParams.FromSpike(spike, &caM, &caP, &caD)
	}
	if ss.TimePlot != nil {
		ss.TimePlot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTimeTable(dt *table.Table) {
	dt.SetMetaData("name", "CagCcplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.AddFloat64Column("Time")
	dt.AddFloat64Column("Spike")
	dt.AddFloat64Column("CaM")
	dt.AddFloat64Column("CaP")
	dt.AddFloat64Column("CaD")
	dt.AddFloat64Column("CaIn")
	dt.AddFloat64Column("CaR")
	dt.AddFloat64Column("M")
	dt.SetNumRows(0)
}

func (ss *Sim) ConfigTimePlot(plt *plotcore.PlotEditor, dt *table.Table) *plotcore.PlotEditor {
	plt.Options.Title = "Time Function Plot"
	plt.Options.XAxis = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColumnOptions("Time", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColumnOptions("Spike", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("CaM", plotcore.Off, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("CaP", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("CaD", plotcore.Off, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("CaIn", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("CaR", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("M", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Body {
	b := core.NewBody("Skca Plot")

	split := core.NewSplits(b)
	core.NewForm(split).SetStruct(ss)

	tv := core.NewTabs(split)

	cgp, _ := tv.NewTab("Ca-G Plot")
	ss.Plot = plotcore.NewSubPlot(cgp)
	ss.ConfigPlot(ss.Plot, ss.Table)

	ttp, _ := tv.NewTab("TimePlot")
	ss.TimePlot = plotcore.NewSubPlot(ttp)
	ss.ConfigTimePlot(ss.TimePlot, ss.TimeTable)

	split.SetSplits(.3, .7)

	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.FuncButton) {
				w.SetFunc(ss.CamRun).SetIcon(icons.PlayArrow)
			})
			tree.Add(p, func(w *core.FuncButton) {
				w.SetFunc(ss.TimeRun).SetIcon(icons.PlayArrow)
			})
		})
	})

	return b
}
