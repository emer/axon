// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ak_plot plots an equation updating over time in a etable.Table and Plot2D.
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
	sim.VmRun()
	b := sim.ConfigGUI()
	b.NewWindow().Run().Wait()
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

	ap := &ss.AK

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		vbio := ss.Vstart + float32(vi)*ss.Vstep
		vnorm := chans.VFmBio(vbio)
		k := ap.KFmV(vbio)
		a := ap.AlphaFmVK(vbio, k)
		b := ap.BetaFmVK(vbio, k)
		mt := ap.MTauFmAlphaBeta(a, b)
		ht := ap.HTauFmV(vbio)
		m := ap.MFmAlpha(a)
		h := ap.HFmV(vbio)
		g := ap.Gak(m, h)

		ms := ss.AKs.MFmV(vbio)
		gs := ss.AKs.Gak(vnorm)

		dt.SetCellFloat("V", vi, float64(vbio))
		dt.SetCellFloat("Gak", vi, float64(g))
		dt.SetCellFloat("M", vi, float64(m))
		dt.SetCellFloat("H", vi, float64(h))
		dt.SetCellFloat("MTau", vi, float64(mt))
		dt.SetCellFloat("HTau", vi, float64(ht))
		dt.SetCellFloat("K", vi, float64(k))
		dt.SetCellFloat("Alpha", vi, float64(a))
		dt.SetCellFloat("Beta", vi, float64(b))

		dt.SetCellFloat("Ms", vi, float64(ms))
		dt.SetCellFloat("Gaks", vi, float64(gs))
	}
	if ss.Plot != nil {
		ss.Plot.UpdatePlot()
	}
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "AkplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"V", etensor.FLOAT64, nil, nil},
		{"Gak", etensor.FLOAT64, nil, nil},
		{"M", etensor.FLOAT64, nil, nil},
		{"H", etensor.FLOAT64, nil, nil},
		{"MTau", etensor.FLOAT64, nil, nil},
		{"HTau", etensor.FLOAT64, nil, nil},
		{"K", etensor.FLOAT64, nil, nil},
		{"Alpha", etensor.FLOAT64, nil, nil},
		{"Beta", etensor.FLOAT64, nil, nil},
		{"Ms", etensor.FLOAT64, nil, nil},
		{"Gaks", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "AK V-G Function Plot"
	plt.Params.XAxisCol = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gak", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gaks", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("M", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Ms", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("H", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("MTau", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("HTau", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("K", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Alpha", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Beta", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() { //gti:add
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
		vnorm := chans.VFmBio(v)
		t := float32(ti) * msdt

		k := ap.KFmV(v)
		a := ap.AlphaFmVK(v, k)
		b := ap.BetaFmVK(v, k)
		mt := ap.MTauFmAlphaBeta(a, b)
		ht := ap.HTauFmV(v)
		g = ap.Gak(m, h)

		dm, dh := ss.AK.DMHFmV(vnorm, m, h)

		dt.SetCellFloat("Time", ti, float64(t))
		dt.SetCellFloat("V", ti, float64(v))
		dt.SetCellFloat("Gak", ti, float64(g))
		dt.SetCellFloat("M", ti, float64(m))
		dt.SetCellFloat("H", ti, float64(h))
		dt.SetCellFloat("dM", ti, float64(dm))
		dt.SetCellFloat("dH", ti, float64(dh))
		dt.SetCellFloat("MTau", ti, float64(mt))
		dt.SetCellFloat("HTau", ti, float64(ht))
		dt.SetCellFloat("K", ti, float64(k))
		dt.SetCellFloat("Alpha", ti, float64(a))
		dt.SetCellFloat("Beta", ti, float64(b))

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

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "AkplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"V", etensor.FLOAT64, nil, nil},
		{"Gak", etensor.FLOAT64, nil, nil},
		{"M", etensor.FLOAT64, nil, nil},
		{"H", etensor.FLOAT64, nil, nil},
		{"dM", etensor.FLOAT64, nil, nil},
		{"dH", etensor.FLOAT64, nil, nil},
		{"MTau", etensor.FLOAT64, nil, nil},
		{"HTau", etensor.FLOAT64, nil, nil},
		{"K", etensor.FLOAT64, nil, nil},
		{"Alpha", etensor.FLOAT64, nil, nil},
		{"Beta", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gak", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("M", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("H", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("dM", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("dH", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("MTau", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("HTau", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("K", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Alpha", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Beta", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *gi.Body {
	b := gi.NewBody("Ak Plot")

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
