// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// effort_plot plots the Rubicon effort cost equations.
package main

import (
	"math/rand"
	"strconv"

	"cogentcore.org/core/gi"
	"cogentcore.org/core/giv"
	"cogentcore.org/core/glop/num"
	"cogentcore.org/core/icons"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/etable/v2/minmax"
)

func DriveEffortGUI() {
	ep := &DrEffPlot{}
	ep.Config()
	b := ep.ConfigGUI()
	b.RunMainWindow()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// DrEffPlot holds the params, table, etc
type DrEffPlot struct {

	// context just for plotting
	Context axon.Context

	// Rubicon params
	Rubicon axon.Rubicon

	// total number of time steps to simulate
	TimeSteps int

	// range for number of time steps between US receipt
	USTime minmax.Int

	// range for random effort per step
	Effort minmax.F32

	// table for plot
	Table *etable.Table `view:"no-inline"`

	// the plot
	Plot *eplot.Plot2D `view:"-"`

	// table for plot
	TimeTable *etable.Table `view:"no-inline"`

	// the plot
	TimePlot *eplot.Plot2D `view:"-"`

	// random number generator
	Rand erand.SysRand `view:"-"`
}

// Config configures all the elements using the standard functions
func (ss *DrEffPlot) Config() {
	ss.Context.Defaults()
	pp := &ss.Rubicon
	pp.SetNUSs(&ss.Context, 1, 1)
	pp.Defaults()
	pp.Drive.DriveMin = 0
	pp.Drive.Base[0] = 1
	pp.Drive.Tau[0] = 100
	pp.Drive.Satisfaction[0] = 1
	pp.Drive.Update()
	ss.TimeSteps = 100
	ss.USTime.Set(2, 20)
	ss.Effort.Set(0.5, 1.5)
	ss.Update()
	ss.Table = &etable.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &etable.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *DrEffPlot) Update() {
}

// EffortPlot plots the equation as a function of effort / time
func (ss *DrEffPlot) EffortPlot() { //gti:add
	ss.Update()
	ctx := &ss.Context
	pp := &ss.Rubicon
	dt := ss.Table
	nv := 100
	dt.SetNumRows(nv)
	pp.TimeEffortReset(ctx, 0)
	for vi := 0; vi < nv; vi++ {
		ev := 1 - axon.RubiconNormFun(0.02)
		dt.SetCellFloat("X", vi, float64(vi))
		dt.SetCellFloat("Y", vi, float64(ev))

		pp.AddTimeEffort(ctx, 0, 1) // unit
	}
	ss.Plot.Update()
}

// UrgencyPlot plots the equation as a function of effort / time
func (ss *DrEffPlot) UrgencyPlot() { //gti:add
	ctx := &ss.Context
	pp := &ss.Rubicon
	ss.Update()
	dt := ss.Table
	nv := 100
	dt.SetNumRows(nv)
	pp.Urgency.Reset(ctx, 0)
	for vi := 0; vi < nv; vi++ {
		ev := pp.Urgency.Urge(ctx, 0)
		dt.SetCellFloat("X", vi, float64(vi))
		dt.SetCellFloat("Y", vi, float64(ev))

		pp.Urgency.AddEffort(ctx, 0, 1) // unit
	}
	ss.Plot.Update()
}

func (ss *DrEffPlot) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "PlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"X", etensor.FLOAT64, nil, nil},
		{"Y", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *DrEffPlot) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Effort Discount or Urgency Function Plot"
	plt.Params.XAxisCol = "X"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("X", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *DrEffPlot) TimeRun() { //gti:add
	ss.Update()
	dt := ss.TimeTable
	pp := &ss.Rubicon
	ctx := &ss.Context
	pp.TimeEffortReset(ctx, 0)
	pp.Urgency.Reset(ctx, 0)
	ut := ss.USTime.Min + rand.Intn(ss.USTime.Range())
	dt.SetNumRows(ss.TimeSteps)
	axon.SetGlbUSposV(ctx, 0, axon.GvUSpos, 1, 0)
	pp.Drive.ToBaseline(ctx, 0)
	// pv.Update()
	lastUS := 0
	for ti := 0; ti < ss.TimeSteps; ti++ {
		ev := 1 - axon.RubiconNormFun(0.02)
		urg := pp.Urgency.Urge(ctx, 0)
		ei := ss.Effort.Min + rand.Float32()*ss.Effort.Range()
		dr := axon.GlbUSposV(ctx, 0, axon.GvDrives, 0)
		usv := float32(0)
		if ti == lastUS+ut {
			ei = 0 // don't update on us trial
			lastUS = ti
			ut = ss.USTime.Min + rand.Intn(ss.USTime.Range())
			usv = 1
		}
		dt.SetCellFloat("T", ti, float64(ti))
		dt.SetCellFloat("Eff", ti, float64(ev))
		dt.SetCellFloat("EffInc", ti, float64(ei))
		dt.SetCellFloat("Urge", ti, float64(urg))
		dt.SetCellFloat("US", ti, float64(usv))
		dt.SetCellFloat("Drive", ti, float64(dr))

		axon.SetGlbUSposV(ctx, 0, axon.GvUSpos, 1, usv)
		axon.SetGlbV(ctx, 0, axon.GvHadRew, num.FromBool[float32](usv > 0))
		pp.EffortUrgencyUpdate(ctx, 0, 0)
		pp.DriveUpdate(ctx, 0)
	}
	ss.TimePlot.Update()
}

func (ss *DrEffPlot) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "TimeTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"T", etensor.FLOAT64, nil, nil},
		{"Eff", etensor.FLOAT64, nil, nil},
		{"EffInc", etensor.FLOAT64, nil, nil},
		{"Urge", etensor.FLOAT64, nil, nil},
		{"US", etensor.FLOAT64, nil, nil},
		{"Drive", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *DrEffPlot) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Effort / Drive over Time Plot"
	plt.Params.XAxisCol = "T"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("T", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Eff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("EffInc", eplot.Off, eplot.FixMin, 0, eplot.FixMax, float64(ss.Effort.Max))
	plt.SetColParams("Urge", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("US", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Drive", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *DrEffPlot) ConfigGUI() *gi.Body {
	b := gi.NewBody("Drive / Effort / Urgency Plotting")

	split := gi.NewSplits(b, "split")
	sv := giv.NewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.NewTabs(split, "tv")

	ss.Plot = eplot.NewSubPlot(tv.NewTab("Effort Plot"))
	ss.ConfigPlot(ss.Plot, ss.Table)

	ss.TimePlot = eplot.NewSubPlot(tv.NewTab("TimePlot"))
	ss.ConfigTimePlot(ss.TimePlot, ss.TimeTable)

	split.SetSplits(.3, .7)

	b.AddAppBar(func(tb *gi.Toolbar) {
		giv.NewFuncButton(tb, ss.EffortPlot).SetIcon(icons.PlayArrow)
		giv.NewFuncButton(tb, ss.UrgencyPlot).SetIcon(icons.PlayArrow)
		giv.NewFuncButton(tb, ss.TimeRun).SetIcon(icons.PlayArrow)
	})

	return b
}
