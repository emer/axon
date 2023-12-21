// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// vgcc_plot plots an equation updating over time in a etable.Table and Plot2D.
package main

//go:generate goki generate -add-types

import (
	"strconv"

	"github.com/emer/axon/v2/chans"
	"goki.dev/etable/v2/eplot"
	"goki.dev/etable/v2/etable"
	"goki.dev/etable/v2/etensor"
	_ "goki.dev/etable/v2/etview" // include to get gui views
	"goki.dev/gi/v2/gi"
	"goki.dev/gi/v2/gimain"
	"goki.dev/gi/v2/giv"
	"goki.dev/icons"
)

func main() { gimain.Run(app) }

func app() {
	sim := &Sim{}
	sim.Config()
	sim.VmRun()
	b := sim.ConfigGUI()
	b.NewWindow().Run().Wait()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// VGCC function
	VGCC chans.VGCCParams

	// starting voltage
	Vstart float32 `def:"-90"`

	// ending voltage
	Vend float32 `def:"0"`

	// voltage increment
	Vstep float32 `def:"1"`

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
	ss.VGCC.Defaults()
	ss.VGCC.Gbar = 1
	ss.Vstart = -90
	ss.Vend = 2
	ss.Vstep = 0.01
	ss.TimeSteps = 200
	ss.TimeSpike = true
	ss.SpikeFreq = 50
	ss.TimeVstart = -70
	ss.TimeVend = -20
	ss.Update()
	ss.Table = &etable.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &etable.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// VmRun plots the equation as a function of V
func (ss *Sim) VmRun() { //gti:add
	ss.Update()
	dt := ss.Table

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		v := ss.Vstart + float32(vi)*ss.Vstep
		vnorm := chans.VFmBio(v)
		g := ss.VGCC.GFmV(vnorm)
		m := ss.VGCC.MFmV(v)
		h := ss.VGCC.HFmV(v)
		var dm, dh float32
		ss.VGCC.DMHFmV(vnorm, m, h, &dm, &dh)

		dt.SetCellFloat("V", vi, float64(v))
		dt.SetCellFloat("Gvgcc", vi, float64(g))
		dt.SetCellFloat("M", vi, float64(m))
		dt.SetCellFloat("H", vi, float64(h))
		dt.SetCellFloat("dM", vi, float64(dm))
		dt.SetCellFloat("dH", vi, float64(dh))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "VgCcplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"V", etensor.FLOAT64, nil, nil},
		{"Gvgcc", etensor.FLOAT64, nil, nil},
		{"M", etensor.FLOAT64, nil, nil},
		{"H", etensor.FLOAT64, nil, nil},
		{"dM", etensor.FLOAT64, nil, nil},
		{"dH", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "VGCC V-G Function Plot"
	plt.Params.XAxisCol = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gvgcc", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("M", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("H", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //gti:add
	ss.Update()
	dt := ss.TimeTable

	m := float32(0)
	h := float32(1)
	msdt := float32(0.001)
	v := ss.TimeVstart
	vinc := float32(2) * (ss.TimeVend - ss.TimeVstart) / float32(ss.TimeSteps)

	isi := int(1000 / ss.SpikeFreq)

	dt.SetNumRows(ss.TimeSteps)
	var g float32
	for ti := 0; ti < ss.TimeSteps; ti++ {
		vnorm := chans.VFmBio(v)
		t := float32(ti) * msdt
		g = ss.VGCC.Gvgcc(vnorm, m, h)
		var dm, dh float32
		ss.VGCC.DMHFmV(vnorm, m, h, &dm, &dh)
		m += dm
		h += dh

		dt.SetCellFloat("Time", ti, float64(t))
		dt.SetCellFloat("V", ti, float64(v))
		dt.SetCellFloat("Gvgcc", ti, float64(g))
		dt.SetCellFloat("M", ti, float64(m))
		dt.SetCellFloat("H", ti, float64(h))
		dt.SetCellFloat("dM", ti, float64(dm))
		dt.SetCellFloat("dH", ti, float64(dh))

		if ss.TimeSpike {
			if ti%isi < 3 {
				v = ss.TimeVend
			} else {
				v = ss.TimeVstart
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

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "VgCcplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"V", etensor.FLOAT64, nil, nil},
		{"Gvgcc", etensor.FLOAT64, nil, nil},
		{"M", etensor.FLOAT64, nil, nil},
		{"H", etensor.FLOAT64, nil, nil},
		{"dM", etensor.FLOAT64, nil, nil},
		{"dH", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gvgcc", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("M", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("H", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("dM", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("dH", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	return plt
}

// ConfigGUI configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGUI() *gi.Body {
	b := gi.NewAppBody("vgcc_plot").SetTitle("Plotting Equations")

	split := gi.NewSplits(b, "split")
	sv := giv.NewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.NewTabs(split, "tv")

	ss.Plot = eplot.NewSubPlot(tv.NewTab("V-G Plot"))
	ss.ConfigPlot(ss.Plot, ss.Table)

	ss.TimePlot = eplot.NewSubPlot(tv.NewTab("TimePlot"))
	ss.ConfigTimePlot(ss.TimePlot, ss.TimeTable)

	split.SetSplits(.3, .7)

	b.AddAppBar(func(tb *gi.Toolbar) {
		giv.NewFuncButton(tb, ss.VmRun).SetIcon(icons.PlayArrow)
		giv.NewFuncButton(tb, ss.TimeRun).SetIcon(icons.PlayArrow)
	})

	return b
}
