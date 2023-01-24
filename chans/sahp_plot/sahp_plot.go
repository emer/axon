// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// mahp_plot plots an equation updating over time in a etable.Table and Plot2D.
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
	Sahp        chans.SahpParams `view:"inline" desc:"sAHP function"`
	CaStart     float32          `def:"0" desc:"starting calcium"`
	CaEnd       float32          `def:"1.5" desc:"ending calcium"`
	CaStep      float32          `def:"0.01" desc:"calcium increment"`
	TimeSteps   int              `desc:"number of time steps"`
	TimeCaStart float32          `desc:"time-run starting calcium"`
	TimeCaD     float32          `desc:"time-run CaD value at end of each theta cycle"`
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
func (ss *Sim) CaRun() {
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
	ss.Plot.Update()
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
func (ss *Sim) TimeRun() {
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
	ss.TimePlot.Update()
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

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	// gi.WinEventTrace = true

	gi.SetAppName("sahp_plot")
	gi.SetAppAbout(`This plots an equation. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("sahp_plot", "Plotting Equations", width, height)
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
		ss.CaRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Time Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.TimeRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/chans/sahp_plot/README.md")
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
