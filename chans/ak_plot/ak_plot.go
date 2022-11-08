// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ak_plot plots an equation updating over time in a etable.Table and Plot2D.
package main

import (
	"strconv"

	"github.com/Astera-org/axon/chans"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

func main() {
	TheSim.Config()
	gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
		guirun()
	})
}

func guirun() {
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {
	AK         chans.AKParams  `desc:"AK function"`
	AKs        chans.AKsParams `desc:"AKs simplified function"`
	Vstart     float32         `def:"-100" desc:"starting voltage"`
	Vend       float32         `def:"100" desc:"ending voltage"`
	Vstep      float32         `def:"1" desc:"voltage increment"`
	TimeSteps  int             `desc:"number of time steps"`
	TimeSpike  bool            `desc:"do spiking instead of voltage ramp"`
	SpikeFreq  float32         `desc:"spiking frequency"`
	TimeVstart float32         `desc:"time-run starting membrane potential"`
	TimeVend   float32         `desc:"time-run ending membrane potential"`
	Table      *etable.Table   `view:"no-inline" desc:"table for plot"`
	Plot       *eplot.Plot2D   `view:"-" desc:"the plot"`
	TimeTable  *etable.Table   `view:"no-inline" desc:"table for plot"`
	TimePlot   *eplot.Plot2D   `view:"-" desc:"the plot"`
	Win        *gi.Window      `view:"-" desc:"main GUI window"`
	ToolBar    *gi.ToolBar     `view:"-" desc:"the master toolbar"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.AK.Defaults()
	ss.AK.Gbar = 1
	ss.AKs.Defaults()
	ss.AKs.Gbar = 1
	ss.Vstart = -100
	ss.Vend = 100
	ss.Vstep = 1
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
func (ss *Sim) VmRun() {
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
	ss.Plot.Update()
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
func (ss *Sim) TimeRun() {
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
	ss.TimePlot.Update()
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

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	// gi.WinEventTrace = true

	gi.SetAppName("ak_plot")
	gi.SetAppAbout(`This plots an equation. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("ak_plot", "Plotting Equations", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "V-G Plot").(*eplot.Plot2D)
	ss.Plot = ss.ConfigPlot(plt, ss.Table)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TimePlot").(*eplot.Plot2D)
	ss.TimePlot = ss.ConfigTimePlot(plt, ss.TimeTable)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "V-G Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.VmRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Time Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.TimeRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/Astera-org/axon/blob/master/chans/ak_plot/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	win.MainMenuUpdated()
	return win
}
