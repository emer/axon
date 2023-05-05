// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// nmda_plot plots an equation updating over time in a etable.Table and Plot2D.
package main

import (
	"math"
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
	gimain.Main(func() { // this starts gui
		guirun()
	})
}

func guirun() {
	TheSim.Run()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {
	NMDAStd   chans.NMDAParams `desc:"standard NMDA implementation in chans"`
	NMDAv     float64          `def:"0.062" desc:"multiplier on NMDA as function of voltage"`
	MgC       float64          `desc:"magnesium ion concentration -- somewhere between 1 and 1.5"`
	NMDAd     float64          `def:"3.57" desc:"denominator of NMDA function"`
	NMDAerev  float64          `def:"0" desc:"NMDA reversal / driving potential"`
	BugVoff   float64          `desc:"for old buggy NMDA: voff value to use"`
	Vstart    float64          `def:"-90" desc:"starting voltage"`
	Vend      float64          `def:"10" desc:"ending voltage"`
	Vstep     float64          `def:"1" desc:"voltage increment"`
	Tau       float64          `def:"100" desc:"decay time constant for NMDA current -- rise time is 2 msec and not worth extra effort for biexponential"`
	TimeSteps int              `desc:"number of time steps"`
	TimeV     float64          `desc:"voltage for TimeRun"`
	TimeGin   float64          `desc:"NMDA Gsyn current input at every time step"`
	Table     *etable.Table    `view:"no-inline" desc:"table for plot"`
	Plot      *eplot.Plot2D    `view:"-" desc:"the plot"`
	TimeTable *etable.Table    `view:"no-inline" desc:"table for plot"`
	TimePlot  *eplot.Plot2D    `view:"-" desc:"the plot"`
	Win       *gi.Window       `view:"-" desc:"main GUI window"`
	ToolBar   *gi.ToolBar      `view:"-" desc:"the master toolbar"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.NMDAStd.Defaults()
	ss.NMDAStd.Voff = 0
	ss.BugVoff = 5
	ss.NMDAv = 0.062
	ss.MgC = 1
	ss.NMDAd = 3.57
	ss.NMDAerev = 0
	ss.Vstart = -1 // -90 // -90 -- use -1 1 to test val around 0
	ss.Vend = 1    // 2     // 50
	ss.Vstep = .01 // use 0.001 instead for testing around 0
	ss.Tau = 100
	ss.TimeSteps = 1000
	ss.TimeV = -50
	ss.TimeGin = .5
	ss.Update()
	ss.Table = &etable.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &etable.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// Equation here:
// https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html

// Run runs the equation.
func (ss *Sim) Run() {
	ss.Update()
	dt := ss.Table

	mgf := ss.MgC / ss.NMDAd

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	v := 0.0
	g := 0.0
	gbug := 0.0
	for vi := 0; vi < nv; vi++ {
		v = ss.Vstart + float64(vi)*ss.Vstep
		if v >= 0 {
			g = 0
		} else {
			g = float64(ss.NMDAStd.Gbar) * (ss.NMDAerev - v) / (1 + mgf*math.Exp(-ss.NMDAv*v))
		}
		bugv := float32(v + ss.BugVoff)
		if bugv >= 0 {
			gbug = 0
		} else {
			gbug = 0.15 / (1.0 + float64(ss.NMDAStd.MgFact*mat32.FastExp(float32(-0.062*bugv))))
		}

		gs := ss.NMDAStd.Gnmda(1, chans.VFmBio(float32(v)))
		ca := ss.NMDAStd.CaFmVbio(float32(v))

		dt.SetCellFloat("V", vi, v)
		dt.SetCellFloat("Gnmda", vi, g)
		dt.SetCellFloat("Gnmda_std", vi, float64(gs))
		dt.SetCellFloat("Gnmda_bug", vi, float64(gbug))
		dt.SetCellFloat("Ca", vi, float64(ca))
	}
	ss.Plot.Update()
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "NmDaplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"V", etensor.FLOAT64, nil, nil},
		{"Gnmda", etensor.FLOAT64, nil, nil},
		{"Gnmda_std", etensor.FLOAT64, nil, nil},
		{"Gnmda_bug", etensor.FLOAT64, nil, nil},
		{"Ca", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "NMDA V-G Function Plot"
	plt.Params.XAxisCol = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gnmda", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gnmda_std", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gnmda_bug", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Ca", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() {
	ss.Update()
	dt := ss.TimeTable

	v := ss.TimeV

	g := 0.0
	nmda := 0.0
	dt.SetNumRows(ss.TimeSteps)
	for ti := 0; ti < ss.TimeSteps; ti++ {
		t := float64(ti) * .001
		gin := ss.TimeGin
		if ti < 10 || ti > ss.TimeSteps/2 {
			gin = 0
		}
		nmda += gin*(1-nmda) - (nmda / ss.Tau)
		g = nmda / (1 + math.Exp(-ss.NMDAv*v)/ss.NMDAd)

		dt.SetCellFloat("Time", ti, t)
		dt.SetCellFloat("Gnmda", ti, g)
		dt.SetCellFloat("NMDA", ti, nmda)
	}
	ss.TimePlot.Update()
}

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "NmDaplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"Gnmda", etensor.FLOAT64, nil, nil},
		{"NMDA", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gnmda", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("NMDA", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	// gi.WinEventTrace = true

	gi.SetAppName("nmda_plot")
	gi.SetAppAbout(`This plots an equation. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("nmdaplot", "Plotting Equations", width, height)
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
		ss.Run()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Time Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.TimeRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/chans/nmda_plot/README.md")
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
