// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gabab_plot plots an equation updating over time in a table.Table and Plot2D.
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
	sim.VGRun()
	sim.SGRun()
	b := sim.ConfigGUI()
	b.RunMainWindow()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// standard chans version of GABAB
	GABAstd chans.GABABParams

	// multiplier on GABAb as function of voltage
	GABAbv float64 `default:"0.1"`

	// offset of GABAb function
	GABAbo float64 `default:"10"`

	// GABAb reversal / driving potential
	GABAberev float64 `default:"-90"`

	// starting voltage
	Vstart float64 `default:"-90"`

	// ending voltage
	Vend float64 `default:"0"`

	// voltage increment
	Vstep float64 `default:"1"`

	// max number of spikes
	Smax int `default:"15"`

	// rise time constant
	RiseTau float64

	// decay time constant -- must NOT be same as RiseTau
	DecayTau float64

	// initial value of GsX driving variable at point of synaptic input onset -- decays expoentially from this start
	GsXInit float64

	// time when peak conductance occurs, in TimeInc units
	MaxTime float64 `edit:"-"`

	// time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))
	TauFact float64 `edit:"-"`

	// total number of time steps to take
	TimeSteps int

	// time increment per step
	TimeInc float64

	// table for plot
	VGTable *table.Table `view:"no-inline"`

	// table for plot
	SGTable *table.Table `view:"no-inline"`

	// table for plot
	TimeTable *table.Table `view:"no-inline"`

	// the plot
	VGPlot *plotview.PlotView `view:"-"`

	// the plot
	SGPlot *plotview.PlotView `view:"-"`

	// the plot
	TimePlot *plotview.PlotView `view:"-"`
}

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.GABAstd.Defaults()
	ss.GABAstd.GiSpike = 1
	ss.GABAbv = 0.1
	ss.GABAbo = 10
	ss.GABAberev = -90
	ss.Vstart = -90
	ss.Vend = 0
	ss.Vstep = .01
	ss.Smax = 30
	ss.RiseTau = 45
	ss.DecayTau = 50
	ss.GsXInit = 1
	ss.TimeSteps = 200
	ss.TimeInc = .001
	ss.Update()

	ss.VGTable = &table.Table{}
	ss.ConfigVGTable(ss.VGTable)

	ss.SGTable = &table.Table{}
	ss.ConfigSGTable(ss.SGTable)

	ss.TimeTable = &table.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
	ss.TauFact = math.Pow(ss.DecayTau/ss.RiseTau, ss.RiseTau/(ss.DecayTau-ss.RiseTau))
	ss.MaxTime = ((ss.RiseTau * ss.DecayTau) / (ss.DecayTau - ss.RiseTau)) * math.Log(ss.DecayTau/ss.RiseTau)
}

// VGRun runs the V-G equation.
func (ss *Sim) VGRun() { //types:add
	ss.Update()
	dt := ss.VGTable

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	v := 0.0
	g := 0.0
	for vi := 0; vi < nv; vi++ {
		v = ss.Vstart + float64(vi)*ss.Vstep
		g = float64(ss.GABAstd.Gbar) * (v - ss.GABAberev) / (1 + math.Exp(ss.GABAbv*((v-ss.GABAberev)+ss.GABAbo)))
		gs := ss.GABAstd.Gbar * ss.GABAstd.GFromV(chans.VFromBio(float32(v)))

		gbug := 0.2 / (1.0 + math32.FastExp(float32(0.1*((v+90)+10))))

		dt.SetFloat("V", vi, v)
		dt.SetFloat("GgabaB", vi, g)
		dt.SetFloat("GgabaB_std", vi, float64(gs))
		dt.SetFloat("GgabaB_bug", vi, float64(gbug))
	}
	if ss.VGPlot != nil {
		ss.VGPlot.UpdatePlot()
	}
}

func (ss *Sim) ConfigVGTable(dt *table.Table) {
	dt.SetMetaData("name", "GABABplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"V", tensor.FLOAT64, nil, nil},
		{"GgabaB", tensor.FLOAT64, nil, nil},
		{"GgabaB_std", tensor.FLOAT64, nil, nil},
		{"GgabaB_bug", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigVGPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "V-G Function Plot"
	plt.Params.XAxisColumn = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GgabaB", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GgabaB_std", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////////

// SGRun runs the spike-g equation.
func (ss *Sim) SGRun() { //types:add
	ss.Update()
	dt := ss.SGTable

	nv := int(float64(ss.Smax) / ss.Vstep)
	dt.SetNumRows(nv)
	s := 0.0
	g := 0.0
	for si := 0; si < nv; si++ {
		s = float64(si) * ss.Vstep
		g = 1.0 / (1.0 + math.Exp(-(s-7.1)/1.4))
		gs := ss.GABAstd.GFromS(float32(s))

		dt.SetFloat("S", si, s)
		dt.SetFloat("GgabaB_max", si, g)
		dt.SetFloat("GgabaBstd_max", si, float64(gs))
	}
	if ss.SGPlot != nil {
		ss.SGPlot.UpdatePlot()
	}
}

func (ss *Sim) ConfigSGTable(dt *table.Table) {
	dt.SetMetaData("name", "SG_GABAplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"S", tensor.FLOAT64, nil, nil},
		{"GgabaB_max", tensor.FLOAT64, nil, nil},
		{"GgabaBstd_max", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigSGPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "S-G Function Plot"
	plt.Params.XAxisColumn = "S"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("S", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GgabaB_max", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GgabaBstd_max", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////////

// TimeRun runs the equation.
func (ss *Sim) TimeRun() { //types:add
	ss.Update()
	dt := ss.TimeTable

	dt.SetNumRows(ss.TimeSteps)
	time := 0.0
	gs := 0.0
	x := ss.GsXInit
	gabaBx := float32(ss.GsXInit)
	gabaB := float32(0.0)
	gi := 0.0 // just goes down
	for t := 0; t < ss.TimeSteps; t++ {
		// record starting state first, then update
		dt.SetFloat("Time", t, time)
		dt.SetFloat("Gs", t, gs)
		dt.SetFloat("GsX", t, x)
		dt.SetFloat("GABAB", t, float64(gabaB))
		dt.SetFloat("GABABx", t, float64(gabaBx))

		gis := 1.0 / (1.0 + math.Exp(-(gi-7.1)/1.4))
		dGs := (ss.TauFact*x - gs) / ss.RiseTau
		dXo := -x / ss.DecayTau
		gs += dGs
		x += gis + dXo

		var dG, dX float32
		ss.GABAstd.BiExp(gabaB, gabaBx, &dG, &dX)
		dt.SetFloat("dG", t, float64(dG))
		dt.SetFloat("dX", t, float64(dX))

		ss.GABAstd.GABAB(float32(gi), &gabaB, &gabaBx)

		time += ss.TimeInc
	}
	if ss.TimePlot != nil {
		ss.TimePlot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTimeTable(dt *table.Table) {
	dt.SetMetaData("name", "TimeGaBabplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Time", tensor.FLOAT64, nil, nil},
		{"Gs", tensor.FLOAT64, nil, nil},
		{"GsX", tensor.FLOAT64, nil, nil},
		{"GABAB", tensor.FLOAT64, nil, nil},
		{"GABABx", tensor.FLOAT64, nil, nil},
		{"dG", tensor.FLOAT64, nil, nil},
		{"dX", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "G Time Function Plot"
	plt.Params.XAxisColumn = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", plotview.Off, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Gs", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GsX", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GABAB", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("GABABx", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Body {
	b := core.NewBody("Gabab Plot")

	split := core.NewSplits(b, "split")
	sv := views.NewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := core.NewTabs(split, "tv")

	ss.VGPlot = plotview.NewSubPlot(tv.NewTab("V-G Plot"))
	ss.ConfigVGPlot(ss.VGPlot, ss.VGTable)

	ss.SGPlot = plotview.NewSubPlot(tv.NewTab("S-G Plot"))
	ss.ConfigSGPlot(ss.SGPlot, ss.SGTable)

	ss.TimePlot = plotview.NewSubPlot(tv.NewTab("TimePlot"))
	ss.ConfigTimePlot(ss.TimePlot, ss.TimeTable)

	split.SetSplits(.3, .7)

	b.AddAppBar(func(tb *core.Toolbar) {
		views.NewFuncButton(tb, ss.VGRun).SetIcon(icons.PlayArrow)
		views.NewFuncButton(tb, ss.SGRun).SetIcon(icons.PlayArrow)
		views.NewFuncButton(tb, ss.TimeRun).SetIcon(icons.PlayArrow)
	})

	return b
}
