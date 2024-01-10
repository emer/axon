// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ska_plot plots an equation updating over time in a etable.Table and Plot2D.
package main

//go:generate goki generate -add-types

import (
	"strconv"

	"github.com/emer/axon/v2/chans"
	"github.com/emer/axon/v2/kinase"
	"goki.dev/etable/v2/eplot"
	"goki.dev/etable/v2/etable"
	"goki.dev/etable/v2/etensor"
	_ "goki.dev/etable/v2/etview" // include to get gui views
	"goki.dev/gi/v2/gi"
	"goki.dev/gi/v2/giv"
	"goki.dev/icons"
)

func main() {
	sim := &Sim{}
	sim.Config()
	sim.CamRun()
	b := sim.ConfigGUI()
	b.NewWindow().Run().Wait()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// SKCa params
	SKCa chans.SKCaParams

	// time constants for integrating Ca from spiking across M, P and D cascading levels
	CaParams kinase.CaParams

	// threshold of SK M gating factor above which the neuron cannot spike
	NoSpikeThr float32 `def:"0.5"`

	// Ca conc increment for M gating func plot
	CaStep float32 `def:"0.05"`

	// number of time steps
	TimeSteps int

	// do spiking instead of Ca conc ramp
	TimeSpike bool

	// spiking frequency
	SpikeFreq float32

	// table for plot
	Table *etable.Table `view:"no-inline"`

	// the plot
	Plot *eplot.Plot2D `view:"-"`

	// table for plot
	TimeTable *etable.Table `view:"no-inline"`

	// the plot
	TimePlot *eplot.Plot2D `view:"-"`
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
	ss.Table = &etable.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &etable.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// CamRun plots the equation as a function of Ca
func (ss *Sim) CamRun() { //gti:add
	ss.Update()
	dt := ss.Table

	nv := int(1.0 / ss.CaStep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		cai := float32(vi) * ss.CaStep
		mh := ss.SKCa.MAsympHill(cai)
		mg := ss.SKCa.MAsympGW06(cai)

		dt.SetCellFloat("Ca", vi, float64(cai))
		dt.SetCellFloat("Mhill", vi, float64(mh))
		dt.SetCellFloat("Mgw06", vi, float64(mg))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "SKCaPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Ca", etensor.FLOAT64, nil, nil},
		{"Mhill", etensor.FLOAT64, nil, nil},
		{"Mgw06", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SKCa Ca-G Function Plot"
	plt.Params.XAxisCol = "Ca"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Ca", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Mhill", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Mgw06", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //gti:add
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
		m = ss.SKCa.MFmCa(caR, m)
		ss.SKCa.CaInRFmSpike(spike, caD, &caIn, &caR)

		dt.SetCellFloat("Time", ti, float64(t))
		dt.SetCellFloat("Spike", ti, float64(spike))
		dt.SetCellFloat("CaM", ti, float64(caM))
		dt.SetCellFloat("CaP", ti, float64(caP))
		dt.SetCellFloat("CaD", ti, float64(caD))
		dt.SetCellFloat("CaIn", ti, float64(caIn))
		dt.SetCellFloat("CaR", ti, float64(caR))
		dt.SetCellFloat("M", ti, float64(m))

		if m < ss.NoSpikeThr && trial%2 == 0 && ti%isi == 0 { // spike on even trials
			spike = 1
		} else {
			spike = 0
		}
		ss.CaParams.FmSpike(spike, &caM, &caP, &caD)
	}
	if ss.TimePlot != nil {
		ss.TimePlot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "CagCcplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"Spike", etensor.FLOAT64, nil, nil},
		{"CaM", etensor.FLOAT64, nil, nil},
		{"CaP", etensor.FLOAT64, nil, nil},
		{"CaD", etensor.FLOAT64, nil, nil},
		{"CaIn", etensor.FLOAT64, nil, nil},
		{"CaR", etensor.FLOAT64, nil, nil},
		{"M", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Spike", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CaM", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CaP", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CaD", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CaIn", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CaR", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("M", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

// ConfigGUI configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGUI() *gi.Body {
	b := gi.NewAppBody("skca_plot").SetTitle("Plotting Equations")

	split := gi.NewSplits(b, "split")
	sv := giv.NewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.NewTabs(split, "tv")

	ss.Plot = eplot.NewSubPlot(tv.NewTab("Ca-G Plot"))
	ss.ConfigPlot(ss.Plot, ss.Table)

	ss.TimePlot = eplot.NewSubPlot(tv.NewTab("TimePlot"))
	ss.ConfigTimePlot(ss.TimePlot, ss.TimeTable)

	split.SetSplits(.3, .7)

	b.AddAppBar(func(tb *gi.Toolbar) {
		giv.NewFuncButton(tb, ss.CamRun).SetIcon(icons.PlayArrow)
		giv.NewFuncButton(tb, ss.TimeRun).SetIcon(icons.PlayArrow)
	})

	return b
}
