// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// synca_plot plots kinase SynCa update equations
package main

//go:generate core generate -add-types

import (
	"math"
	"strconv"

	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/plot/plotview"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	_ "cogentcore.org/core/tensor/tensorview" // include to get gui views
	"cogentcore.org/core/views"
	"github.com/emer/axon/v2/kinase"
	_ "github.com/emer/gosl/v2/slboolview" // ditto
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

	// Ca time constants
	CaDt  kinase.CaParams `view:"inline"`
	Minit float64
	Pinit float64
	Dinit float64

	// adjustment to dt to account for discrete time updating
	MdtAdj float64 `default:"0,0.11"`

	// adjustment to dt to account for discrete time updating
	PdtAdj float64 `default:"0.0.03"`

	// adjustment to dt to account for discrete time updating
	DdtAdj float64 `default:"0.0.03"`

	// number of time steps
	TimeSteps int

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
	ss.CaDt.Defaults()
	ss.Minit = 0.7
	ss.Pinit = 0.5
	ss.Dinit = 0.3
	ss.MdtAdj = 0
	ss.PdtAdj = 0
	ss.DdtAdj = 0
	ss.TimeSteps = 1000
	ss.Update()
	ss.Table = &table.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &table.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// Run runs the equation.
func (ss *Sim) Run() { //types:add
	ss.Update()
	dt := ss.Table

	dt.SetNumRows(200)
	mi := ss.Minit
	pi := ss.Pinit
	di := ss.Dinit
	mdt := float64(ss.CaDt.Dt.MDt) * (1.0 + ss.MdtAdj)
	pdt := float64(ss.CaDt.Dt.PDt) * (1.0 + ss.PdtAdj)
	ddt := float64(ss.CaDt.Dt.DDt) * (1.0 + ss.DdtAdj)
	for ti := 0; ti < 200; ti++ {
		t := float64(ti)
		m := ss.Minit * math.Exp(-t*mdt)

		em := math.Exp(t * mdt)
		ep := math.Exp(t * pdt)

		p := ss.Pinit*math.Exp(-t*pdt) - (pdt*ss.Minit*math.Exp(-t*(mdt+pdt))*(em-ep))/(pdt-mdt)

		epd := math.Exp(t * (pdt + ddt))
		emd := math.Exp(t * (mdt + ddt))
		emp := math.Exp(t * (mdt + pdt))

		d := pdt*ddt*ss.Minit*math.Exp(-t*(mdt+pdt+ddt))*(ddt*(emd-epd)+(pdt*(epd-emp))+mdt*(emp-emd))/((mdt-pdt)*(mdt-ddt)*(pdt-ddt)) - ddt*ss.Pinit*math.Exp(-t*(pdt+ddt))*(ep-math.Exp(t*ddt))/(ddt-pdt) + ss.Dinit*math.Exp(-t*ddt)

		// test eqs:
		caM := float32(ss.Minit)
		caP := float32(ss.Pinit)
		caD := float32(ss.Dinit)
		ss.CaDt.Dt.CaAtT(int32(ti), &caM, &caP, &caD)
		m = float64(caM)
		p = float64(caP)
		d = float64(caD)

		caM = float32(ss.Minit)
		caP = float32(ss.Pinit)
		caD = float32(ss.Dinit)
		ss.CaDt.CurCa(float32(ti), 0, &caM, &caP, &caD)
		mi4 := float64(caM)
		pi4 := float64(caP)
		di4 := float64(caD)

		dt.SetFloat("t", ti, t)
		dt.SetFloat("mi", ti, mi)
		dt.SetFloat("pi", ti, pi)
		dt.SetFloat("di", ti, di)
		dt.SetFloat("mi4", ti, mi4)
		dt.SetFloat("pi4", ti, pi4)
		dt.SetFloat("di4", ti, di4)
		dt.SetFloat("m", ti, m)
		dt.SetFloat("p", ti, p)
		dt.SetFloat("d", ti, d)

		mi += float64(ss.CaDt.Dt.MDt) * (0 - mi)
		pi += float64(ss.CaDt.Dt.PDt) * (mi - pi)
		di += float64(ss.CaDt.Dt.DDt) * (pi - di)

	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "SynCa(t)")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"t", tensor.FLOAT64, nil, nil},
		{"mi", tensor.FLOAT64, nil, nil},
		{"pi", tensor.FLOAT64, nil, nil},
		{"di", tensor.FLOAT64, nil, nil},
		{"mi4", tensor.FLOAT64, nil, nil},
		{"pi4", tensor.FLOAT64, nil, nil},
		{"di4", tensor.FLOAT64, nil, nil},
		{"m", tensor.FLOAT64, nil, nil},
		{"p", tensor.FLOAT64, nil, nil},
		{"d", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "SynCa Exp Decay Plot"
	plt.Params.XAxisColumn = "t"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("t", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("mi", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("pi", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("di", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("mi4", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("pi4", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("di4", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("m", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("p", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("d", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //types:add
	ss.Update()
	/*
		dt := ss.TimeTable

			v := ss.TimeV

			g := 0.0
			synca := 0.0
			dt.SetNumRows(ss.TimeSteps)
			for ti := 0; ti < ss.TimeSteps; ti++ {
				t := float64(ti) * .001
				gin := ss.TimeGin
				if ti < 10 || ti > ss.TimeSteps/2 {
					gin = 0
				}
				synca += gin*(1-synca) - (synca / ss.Tau)
				g = synca / (1 + math.Exp(-ss.SYNCav*v)/ss.SYNCad)

				dt.SetFloat("Time", ti, t)
				dt.SetFloat("Gsynca", ti, g)
				dt.SetFloat("SYNCa", ti, synca)
			}
			ss.TimePlot.Update()
	*/
}

func (ss *Sim) ConfigTimeTable(dt *table.Table) {
	dt.SetMetaData("name", "SyNcaplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Time", tensor.FLOAT64, nil, nil},
		{"Gsynca", tensor.FLOAT64, nil, nil},
		{"SYNCa", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *plotview.PlotView, dt *table.Table) *plotview.PlotView {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisColumn = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", plotview.Off, plotview.FloatMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("Gsynca", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	plt.SetColParams("SYNCa", plotview.On, plotview.FixMin, 0, plotview.FloatMax, 0)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Body {
	b := core.NewBody("Synca Plot")

	split := core.NewSplits(b, "split")
	sv := views.NewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := core.NewTabs(split, "tv")

	ss.Plot = plotview.NewSubPlot(tv.NewTab("T Exp Plot"))
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
