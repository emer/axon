// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ak_plot plots an equation updating over time in a table.Table and PlotView.
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

	// AK function
	AK chans.AKParams

	// AKs simplified function
	AKs chans.AKsParams

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
	ss.AK.Defaults()
	ss.AK.Gbar = 1
	ss.AKs.Defaults()
	ss.AKs.Gbar = 1
	ss.Vstart = -100
	ss.Vend = 100
	ss.Vstep = .01
	ss.TimeSteps = 200
	ss.TimeSpike = true
	ss.SpikeFreq = 50
	ss.TimeVstart = -50
	ss.TimeVend = -20
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

	ap := &ss.AK

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		vbio := ss.Vstart + float32(vi)*ss.Vstep
		vnorm := chans.VFromBio(vbio)
		k := ap.KFromV(vbio)
		a := ap.AlphaFromVK(vbio, k)
		b := ap.BetaFromVK(vbio, k)
		mt := ap.MTauFromAlphaBeta(a, b)
		ht := ap.HTauFromV(vbio)
		m := ap.MFromAlpha(a)
		h := ap.HFromV(vbio)
		g := ap.Gak(m, h)

		ms := ss.AKs.MFromV(vbio)
		gs := ss.AKs.Gak(vnorm)

		dt.SetFloat("V", vi, float64(vbio))
		dt.SetFloat("Gak", vi, float64(g))
		dt.SetFloat("M", vi, float64(m))
		dt.SetFloat("H", vi, float64(h))
		dt.SetFloat("MTau", vi, float64(mt))
		dt.SetFloat("HTau", vi, float64(ht))
		dt.SetFloat("K", vi, float64(k))
		dt.SetFloat("Alpha", vi, float64(a))
		dt.SetFloat("Beta", vi, float64(b))

		dt.SetFloat("Ms", vi, float64(ms))
		dt.SetFloat("Gaks", vi, float64(gs))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "AkplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.AddFloat64Column("V")
	dt.AddFloat64Column("Gak")
	dt.AddFloat64Column("M")
	dt.AddFloat64Column("H")
	dt.AddFloat64Column("MTau")
	dt.AddFloat64Column("HTau")
	dt.AddFloat64Column("K")
	dt.AddFloat64Column("Alpha")
	dt.AddFloat64Column("Beta")
	dt.AddFloat64Column("Ms")
	dt.AddFloat64Column("Gaks")
	dt.SetNumRows(0)
}

func (ss *Sim) ConfigPlot(plt *plotcore.PlotEditor, dt *table.Table) *plotcore.PlotEditor {
	plt.Params.Title = "AK V-G Function Plot"
	plt.Params.XAxisColumn = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("Gak", plotcore.On, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("Gaks", plotcore.On, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("M", plotcore.On, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("Ms", plotcore.On, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("H", plotcore.On, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("MTau", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("HTau", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("K", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("Alpha", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("Beta", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //types:add
	ss.Update()
	dt := ss.TimeTable

	ap := &ss.AK

	m := float32(0)
	h := float32(1)
	msdt := float32(0.001)
	v := ss.TimeVstart
	vinc := float32(2) * (ss.TimeVend - ss.TimeVstart) / float32(ss.TimeSteps)

	isi := int(1000 / ss.SpikeFreq)

	dt.SetNumRows(ss.TimeSteps)
	var g float32
	for ti := 0; ti < ss.TimeSteps; ti++ {
		vnorm := chans.VFromBio(v)
		t := float32(ti) * msdt

		k := ap.KFromV(v)
		a := ap.AlphaFromVK(v, k)
		b := ap.BetaFromVK(v, k)
		mt := ap.MTauFromAlphaBeta(a, b)
		ht := ap.HTauFromV(v)
		g = ap.Gak(m, h)

		dm, dh := ss.AK.DMHFromV(vnorm, m, h)

		dt.SetFloat("Time", ti, float64(t))
		dt.SetFloat("V", ti, float64(v))
		dt.SetFloat("Gak", ti, float64(g))
		dt.SetFloat("M", ti, float64(m))
		dt.SetFloat("H", ti, float64(h))
		dt.SetFloat("dM", ti, float64(dm))
		dt.SetFloat("dH", ti, float64(dh))
		dt.SetFloat("MTau", ti, float64(mt))
		dt.SetFloat("HTau", ti, float64(ht))
		dt.SetFloat("K", ti, float64(k))
		dt.SetFloat("Alpha", ti, float64(a))
		dt.SetFloat("Beta", ti, float64(b))

		g = ss.AK.Gak(m, h)
		m += dm // already in msec time constants
		h += dh

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

func (ss *Sim) ConfigTimeTable(dt *table.Table) {
	dt.SetMetaData("name", "AkplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.AddFloat64Column("Time")
	dt.AddFloat64Column("V")
	dt.AddFloat64Column("Gak")
	dt.AddFloat64Column("M")
	dt.AddFloat64Column("H")
	dt.AddFloat64Column("dM")
	dt.AddFloat64Column("dH")
	dt.AddFloat64Column("MTau")
	dt.AddFloat64Column("HTau")
	dt.AddFloat64Column("K")
	dt.AddFloat64Column("Alpha")
	dt.AddFloat64Column("Beta")
	dt.SetNumRows(0)
}

func (ss *Sim) ConfigTimePlot(plt *plotcore.PlotEditor, dt *table.Table) *plotcore.PlotEditor {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisColumn = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("Gak", plotcore.On, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("M", plotcore.Off, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("H", plotcore.Off, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("dM", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("dH", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("MTau", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("HTau", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("K", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("Alpha", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColParams("Beta", plotcore.Off, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Body {
	b := core.NewBody("Ak Plot")

	split := core.NewSplits(b)
	core.NewForm(split).SetStruct(ss)
	tv := core.NewTabs(split)

	ss.Plot = plotcore.NewSubPlot(tv.NewTab("V-G Plot"))
	ss.ConfigPlot(ss.Plot, ss.Table)

	ss.TimePlot = plotcore.NewSubPlot(tv.NewTab("TimePlot"))
	ss.ConfigTimePlot(ss.TimePlot, ss.TimeTable)

	split.SetSplits(.3, .7)

	b.AddAppBar(func(p *tree.Plan) {
		tree.Add(p, func(w *core.FuncButton) {
			w.SetFunc(ss.VmRun).SetIcon(icons.PlayArrow)
		})
		tree.Add(p, func(w *core.FuncButton) {
			w.SetFunc(ss.TimeRun).SetIcon(icons.PlayArrow)
		})
	})

	return b
}
