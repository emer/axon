// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// synca_plot plots kinase SynCa update equations
package main

//go:generate goki generate -add-types

import (
	"math"
	"strconv"

	"github.com/emer/axon/v2/kinase"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	_ "github.com/emer/etable/v2/etview" // include to get gui views
	"goki.dev/gi"
	"goki.dev/gimain"
	"goki.dev/giv"
	_ "goki.dev/gosl/v2/slboolview" // ditto
	"goki.dev/icons"
)

func main() { gimain.Run(app) }

func app() {
	sim := &Sim{}
	sim.Config()
	sim.Run()
	b := sim.ConfigGUI()
	b.NewWindow().Run().Wait()
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
	MdtAdj float64 `def:"0,0.11"`

	// adjustment to dt to account for discrete time updating
	PdtAdj float64 `def:"0.0.03"`

	// adjustment to dt to account for discrete time updating
	DdtAdj float64 `def:"0.0.03"`

	// number of time steps
	TimeSteps int

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
	ss.CaDt.Defaults()
	ss.Minit = 0.7
	ss.Pinit = 0.5
	ss.Dinit = 0.3
	ss.MdtAdj = 0
	ss.PdtAdj = 0
	ss.DdtAdj = 0
	ss.TimeSteps = 1000
	ss.Update()
	ss.Table = &etable.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &etable.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// Run runs the equation.
func (ss *Sim) Run() { //gti:add
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

		dt.SetCellFloat("t", ti, t)
		dt.SetCellFloat("mi", ti, mi)
		dt.SetCellFloat("pi", ti, pi)
		dt.SetCellFloat("di", ti, di)
		dt.SetCellFloat("mi4", ti, mi4)
		dt.SetCellFloat("pi4", ti, pi4)
		dt.SetCellFloat("di4", ti, di4)
		dt.SetCellFloat("m", ti, m)
		dt.SetCellFloat("p", ti, p)
		dt.SetCellFloat("d", ti, d)

		mi += float64(ss.CaDt.Dt.MDt) * (0 - mi)
		pi += float64(ss.CaDt.Dt.PDt) * (mi - pi)
		di += float64(ss.CaDt.Dt.DDt) * (pi - di)

	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "SynCa(t)")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"t", etensor.FLOAT64, nil, nil},
		{"mi", etensor.FLOAT64, nil, nil},
		{"pi", etensor.FLOAT64, nil, nil},
		{"di", etensor.FLOAT64, nil, nil},
		{"mi4", etensor.FLOAT64, nil, nil},
		{"pi4", etensor.FLOAT64, nil, nil},
		{"di4", etensor.FLOAT64, nil, nil},
		{"m", etensor.FLOAT64, nil, nil},
		{"p", etensor.FLOAT64, nil, nil},
		{"d", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SynCa Exp Decay Plot"
	plt.Params.XAxisCol = "t"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("t", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("mi", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("pi", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("di", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("mi4", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("pi4", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("di4", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("m", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("p", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("d", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //gti:add
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

				dt.SetCellFloat("Time", ti, t)
				dt.SetCellFloat("Gsynca", ti, g)
				dt.SetCellFloat("SYNCa", ti, synca)
			}
			ss.TimePlot.Update()
	*/
}

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "SyNcaplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"Gsynca", etensor.FLOAT64, nil, nil},
		{"SYNCa", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gsynca", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SYNCa", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

// ConfigGUI configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGUI() *gi.Body {
	b := gi.NewAppBody("synca_plot").SetTitle("Plotting Equations")

	split := gi.NewSplits(b, "split")
	sv := giv.NewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.NewTabs(split, "tv")

	ss.Plot = eplot.NewSubPlot(tv.NewTab("T Exp Plot"))
	ss.ConfigPlot(ss.Plot, ss.Table)

	ss.TimePlot = eplot.NewSubPlot(tv.NewTab("TimePlot"))
	ss.ConfigTimePlot(ss.TimePlot, ss.TimeTable)

	split.SetSplits(.3, .7)

	b.AddAppBar(func(tb *gi.Toolbar) {
		giv.NewFuncButton(tb, ss.Run).SetIcon(icons.PlayArrow)
		giv.NewFuncButton(tb, ss.TimeRun).SetIcon(icons.PlayArrow)
	})

	return b
}
