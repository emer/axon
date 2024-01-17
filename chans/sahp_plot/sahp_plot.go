// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// sahp_plot plots an equation updating over time in a etable.Table and Plot2D.
package main

//go:generate core generate -add-types

import (
	"strconv"

	"cogentcore.org/core/gi"
	"cogentcore.org/core/giv"
	"cogentcore.org/core/icons"
	"github.com/emer/axon/v2/chans"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	_ "github.com/emer/etable/v2/etview" // include to get gui views
)

func main() {
	sim := &Sim{}
	sim.Config()
	sim.CaRun()
	b := sim.ConfigGUI()
	b.NewWindow().Run().Wait()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// sAHP function
	Sahp chans.SahpParams `view:"inline"`

	// starting calcium
	CaStart float32 `def:"0"`

	// ending calcium
	CaEnd float32 `def:"1.5"`

	// calcium increment
	CaStep float32 `def:"0.01"`

	// number of time steps
	TimeSteps int

	// time-run starting calcium
	TimeCaStart float32

	// time-run CaD value at end of each theta cycle
	TimeCaD float32

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
	ss.Sahp.Defaults()
	ss.Sahp.Gbar = 1
	ss.CaStart = 0
	ss.CaEnd = 1.5
	ss.CaStep = 0.01
	ss.TimeSteps = 30
	ss.TimeCaStart = 0
	ss.TimeCaD = 1
	ss.Update()
	ss.Table = &etable.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &etable.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// CaRun plots the equation as a function of V
func (ss *Sim) CaRun() { //gti:add
	ss.Update()
	dt := ss.Table

	mp := &ss.Sahp

	nv := int((ss.CaEnd - ss.CaStart) / ss.CaStep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		ca := ss.CaStart + float32(vi)*ss.CaStep
		var ninf, tau float32
		mp.NinfTauFmCa(ca, &ninf, &tau)

		dt.SetCellFloat("Ca", vi, float64(ca))
		dt.SetCellFloat("Ninf", vi, float64(ninf))
		dt.SetCellFloat("Tau", vi, float64(tau))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "sAHPplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Ca", etensor.FLOAT64, nil, nil},
		{"Ninf", etensor.FLOAT64, nil, nil},
		{"Tau", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "sAHP Ca Function Plot"
	plt.Params.XAxisCol = "Ca"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Ca", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Ninf", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Tau", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 1)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //gti:add
	ss.Update()
	dt := ss.TimeTable

	mp := &ss.Sahp

	var n, tau float32
	mp.NinfTauFmCa(ss.TimeCaStart, &n, &tau)
	ca := ss.TimeCaStart

	dt.SetNumRows(ss.TimeSteps)
	for ti := 1; ti <= ss.TimeSteps; ti++ {
		t := float32(ti)

		var ninf, tau float32
		mp.NinfTauFmCa(ca, &ninf, &tau)
		dn := mp.DNFmV(ca, n)
		g := mp.GsAHP(n)

		dt.SetCellFloat("Time", ti, float64(t))
		dt.SetCellFloat("Ca", ti, float64(ca))
		dt.SetCellFloat("GsAHP", ti, float64(g))
		dt.SetCellFloat("N", ti, float64(n))
		dt.SetCellFloat("dN", ti, float64(dn))
		dt.SetCellFloat("Ninf", ti, float64(ninf))
		dt.SetCellFloat("Tau", ti, float64(tau))

		ca = mp.CaInt(ca, ss.TimeCaD)
		n += dn
	}
	if ss.TimePlot != nil {
		ss.TimePlot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "sAHPplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"Ca", etensor.FLOAT64, nil, nil},
		{"GsAHP", etensor.FLOAT64, nil, nil},
		{"N", etensor.FLOAT64, nil, nil},
		{"dN", etensor.FLOAT64, nil, nil},
		{"Ninf", etensor.FLOAT64, nil, nil},
		{"Tau", etensor.FLOAT64, nil, nil},
		{"Kna", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Ca", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GsAHP", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("N", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("dN", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Ninf", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Tau", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Kna", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	return plt
}

// ConfigGUI configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGUI() *gi.Body {
	b := gi.NewAppBody("sahp_plot").SetTitle("Plotting Equations")

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
		giv.NewFuncButton(tb, ss.CaRun).SetIcon(icons.PlayArrow)
		giv.NewFuncButton(tb, ss.TimeRun).SetIcon(icons.PlayArrow)
	})

	return b
}
