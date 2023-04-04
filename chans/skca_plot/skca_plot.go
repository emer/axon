// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ska_plot plots an equation updating over time in a etable.Table and Plot2D.
package main

import (
	"strconv"

	"github.com/emer/axon/chans"
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
	SKCa        chans.SKCaParams `desc:"SKCa function"`
	CaStart     float32          `def:"0.001" desc:"starting Ca conc"`
	CaEnd       float32          `def:"2" desc:"ending Ca conc"`
	CaStep      float32          `def:"0.05" desc:"Ca conc increment"`
	TimeSteps   int              `desc:"number of time steps"`
	TimeSpike   bool             `desc:"do spiking instead of Ca conc ramp"`
	SpikeFreq   float32          `desc:"spiking frequency"`
	SpikeCa     float32          `desc:"increment in Ca from spiking"`
	CaDecayTau  float32          `desc:"time constant for Ca Decay"`
	TimeCaStart float32          `desc:"time-run starting Ca conc"`
	Table       *etable.Table    `view:"no-inline" desc:"table for plot"`
	Plot        *eplot.Plot2D    `view:"-" desc:"the plot"`
	TimeTable   *etable.Table    `view:"no-inline" desc:"table for plot"`
	TimePlot    *eplot.Plot2D    `view:"-" desc:"the plot"`
	Win         *gi.Window       `view:"-" desc:"main GUI window"`
	ToolBar     *gi.ToolBar      `view:"-" desc:"the master toolbar"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.SKCa.Defaults()
	ss.SKCa.Gbar = 1
	ss.CaStart = 0.001
	ss.CaEnd = 5
	ss.CaStep = .1
	ss.TimeSteps = 200
	ss.TimeSpike = true
	ss.SpikeFreq = 50
	ss.SpikeCa = 0.1
	ss.CaDecayTau = 20 // .4ms in Fujita, 185.7 in Gillies & Willshaw;
	// AnwarRoomeNedelescuEtAl14 is definitive study -- looks like ~10-20.
	// old: HelmchenImotoSakmann96 -- says 100-ish
	ss.TimeCaStart = 0
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
func (ss *Sim) CamRun() {
	ss.Update()
	dt := ss.Table

	nv := int((ss.CaEnd - ss.CaStart) / ss.CaStep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		cai := ss.CaStart + float32(vi)*ss.CaStep
		mh := ss.SKCa.MAsympHill(cai)
		mg := ss.SKCa.MAsympGW06(cai)

		dt.SetCellFloat("Ca", vi, float64(cai))
		dt.SetCellFloat("Mhill", vi, float64(mh))
		dt.SetCellFloat("Mgw06", vi, float64(mg))
	}
	ss.Plot.Update()
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
	plt.SetColParams("Mhill", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Mgw06", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() {
	ss.Update()
	dt := ss.TimeTable

	ca := ss.TimeCaStart
	msdt := float32(0.001)
	m := float32(0)

	isi := int(1000 / ss.SpikeFreq)

	dt.SetNumRows(ss.TimeSteps)
	for ti := 0; ti < ss.TimeSteps; ti++ {
		t := float32(ti) * msdt
		m = ss.SKCa.MFmCa(ca, m)
		g := m * ss.SKCa.Gbar * m

		dt.SetCellFloat("Time", ti, float64(t))
		dt.SetCellFloat("Ca", ti, float64(ca))
		dt.SetCellFloat("Gsk", ti, float64(g))
		dt.SetCellFloat("M", ti, float64(m))

		if ss.TimeSpike {
			if ti%isi < 3 {
				ca += ss.SpikeCa
			}
			ca -= ca / ss.CaDecayTau
		}
	}
	ss.TimePlot.Update()
}

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "CagCcplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"Ca", etensor.FLOAT64, nil, nil},
		{"Gsk", etensor.FLOAT64, nil, nil},
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
	plt.SetColParams("Gsk", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("M", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	// gi.WinEventTrace = true

	gi.SetAppName("skca_plot")
	gi.SetAppAbout(`This plots an equation. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("skca_plot", "Plotting Equations", width, height)
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

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "Ca-G Plot").(*eplot.Plot2D)
	ss.Plot = ss.ConfigPlot(plt, ss.Table)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TimePlot").(*eplot.Plot2D)
	ss.TimePlot = ss.ConfigTimePlot(plt, ss.TimeTable)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Ca-G Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.CamRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Time Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.TimeRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/chans/skca_plot/README.md")
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
