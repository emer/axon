// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// effort_plot plots the PVLV effort cost equations.
package main

import (
	"math/rand"
	"strconv"

	"github.com/emer/axon/axon"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/minmax"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

func DriveEffortGUI() {
	ep := &DrEffPlot{}
	ep.Config()
	win := ep.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// DrEffPlot holds the params, table, etc
type DrEffPlot struct {
	Context   axon.Context  `desc:"Drive, Effort params are under DrivePVLV"`
	TimeSteps int           `desc:"total number of time steps to simulate"`
	USTime    minmax.Int    `desc:"range for number of time steps between US receipt"`
	Effort    minmax.F32    `desc:"range for random effort per step"`
	Table     *etable.Table `view:"no-inline" desc:"table for plot"`
	Plot      *eplot.Plot2D `view:"-" desc:"the plot"`
	TimeTable *etable.Table `view:"no-inline" desc:"table for plot"`
	TimePlot  *eplot.Plot2D `view:"-" desc:"the plot"`
	Win       *gi.Window    `view:"-" desc:"main GUI window"`
	ToolBar   *gi.ToolBar   `view:"-" desc:"the master toolbar"`
}

// Config configures all the elements using the standard functions
func (ss *DrEffPlot) Config() {
	ss.Context.Defaults()
	pp := &ss.Context.PVLV
	pp.Drive.NActive = 1
	pp.Drive.DriveMin = 0
	pp.Drive.Base.Set(0, 1)
	pp.Drive.Tau.Set(0, 100)
	pp.Drive.USDec.Set(0, 1)
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

// EffortPlot plots the equation as a function of V
func (ss *DrEffPlot) EffortPlot() {
	ss.Update()
	dt := ss.Table
	nv := 100
	dt.SetNumRows(nv)
	pp := &ss.Context.PVLV
	pp.Effort.Reset()
	for vi := 0; vi < nv; vi++ {
		ev := pp.Effort.DiscFmEffort()
		dt.SetCellFloat("T", vi, float64(vi))
		dt.SetCellFloat("Eff", vi, float64(ev))

		pp.Effort.AddEffort(1) // unit
	}
	ss.Plot.Update()
}

func (ss *DrEffPlot) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "EffortTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"T", etensor.FLOAT64, nil, nil},
		{"Eff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *DrEffPlot) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Effort Discount Function Plot"
	plt.Params.XAxisCol = "T"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("T", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Eff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *DrEffPlot) TimeRun() {
	ss.Update()
	dt := ss.TimeTable

	pp := &ss.Context.PVLV
	pp.Effort.Reset()
	ut := ss.USTime.Min + rand.Intn(ss.USTime.Range())
	dt.SetNumRows(ss.TimeSteps)
	pp.USpos.Set(0, 0)
	pp.Drive.ToBaseline()
	pp.Update()
	lastUS := 0
	for ti := 0; ti < ss.TimeSteps; ti++ {
		ev := pp.Effort.DiscFmEffort()
		ei := ss.Effort.Min + rand.Float32()*ss.Effort.Range()
		dr := pp.Drive.Drives.Get(0)
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
		dt.SetCellFloat("US", ti, float64(usv))
		dt.SetCellFloat("Drive", ti, float64(dr))

		pp.USpos.Set(0, usv)
		pp.DriveEffortUpdt(ei, usv > 0, false)
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
	plt.SetColParams("US", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Drive", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *DrEffPlot) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	// gi.WinEventTrace = true

	win := gi.NewMainWindow("dreff_plot", "Drive / EFfort Plotting", width, height)
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

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "EffortPlot").(*eplot.Plot2D)
	ss.Plot = ss.ConfigPlot(plt, ss.Table)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TimePlot").(*eplot.Plot2D)
	ss.TimePlot = ss.ConfigTimePlot(plt, ss.TimeTable)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Effort Plot", Icon: "update", Tooltip: "plot basic effort equation."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.EffortPlot()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Time Run", Icon: "update", Tooltip: "Run a simulated time-evolution and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.TimeRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/pvlv/README.md")
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
