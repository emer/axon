// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gabab_plot plots an equation updating over time in a etable.Table and Plot2D.
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
	gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
		guirun()
	})
}

func guirun() {
	TheSim.VGRun()
	TheSim.SGRun()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// standard chans version of GABAB
	GABAstd chans.GABABParams

	// multiplier on GABAb as function of voltage
	GABAbv float64 `def:"0.1"`

	// offset of GABAb function
	GABAbo float64 `def:"10"`

	// GABAb reversal / driving potential
	GABAberev float64 `def:"-90"`

	// starting voltage
	Vstart float64 `def:"-90"`

	// ending voltage
	Vend float64 `def:"0"`

	// voltage increment
	Vstep float64 `def:"1"`

	// max number of spikes
	Smax int `def:"15"`

	// rise time constant
	RiseTau float64

	// decay time constant -- must NOT be same as RiseTau
	DecayTau float64

	// initial value of GsX driving variable at point of synaptic input onset -- decays expoentially from this start
	GsXInit float64

	// time when peak conductance occurs, in TimeInc units
	MaxTime float64 `inactive:"+"`

	// time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))
	TauFact float64 `inactive:"+"`

	// total number of time steps to take
	TimeSteps int

	// time increment per step
	TimeInc float64

	// table for plot
	VGTable *etable.Table `view:"no-inline"`

	// table for plot
	SGTable *etable.Table `view:"no-inline"`

	// table for plot
	TimeTable *etable.Table `view:"no-inline"`

	// the plot
	VGPlot *eplot.Plot2D `view:"-"`

	// the plot
	SGPlot *eplot.Plot2D `view:"-"`

	// the plot
	TimePlot *eplot.Plot2D `view:"-"`

	// main GUI window
	Win *gi.Window `view:"-"`

	// the master toolbar
	ToolBar *gi.ToolBar `view:"-"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.GABAstd.Defaults()
	ss.GABAstd.GiSpike = 1
	ss.GABAbv = 0.1
	ss.GABAbo = 10
	ss.GABAberev = -90
	ss.Vstart = -90
	ss.Vend = 0
	ss.Vstep = .01
	ss.Smax = 30
	ss.RiseTau = 45
	ss.DecayTau = 50
	ss.GsXInit = 1
	ss.TimeSteps = 200
	ss.TimeInc = .001
	ss.Update()

	ss.VGTable = &etable.Table{}
	ss.ConfigVGTable(ss.VGTable)

	ss.SGTable = &etable.Table{}
	ss.ConfigSGTable(ss.SGTable)

	ss.TimeTable = &etable.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
	ss.TauFact = math.Pow(ss.DecayTau/ss.RiseTau, ss.RiseTau/(ss.DecayTau-ss.RiseTau))
	ss.MaxTime = ((ss.RiseTau * ss.DecayTau) / (ss.DecayTau - ss.RiseTau)) * math.Log(ss.DecayTau/ss.RiseTau)
}

// VGRun runs the V-G equation.
func (ss *Sim) VGRun() {
	ss.Update()
	dt := ss.VGTable

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	v := 0.0
	g := 0.0
	for vi := 0; vi < nv; vi++ {
		v = ss.Vstart + float64(vi)*ss.Vstep
		g = float64(ss.GABAstd.Gbar) * (v - ss.GABAberev) / (1 + math.Exp(ss.GABAbv*((v-ss.GABAberev)+ss.GABAbo)))
		gs := ss.GABAstd.Gbar * ss.GABAstd.GFmV(chans.VFmBio(float32(v)))

		gbug := 0.2 / (1.0 + mat32.FastExp(float32(0.1*((v+90)+10))))

		dt.SetCellFloat("V", vi, v)
		dt.SetCellFloat("GgabaB", vi, g)
		dt.SetCellFloat("GgabaB_std", vi, float64(gs))
		dt.SetCellFloat("GgabaB_bug", vi, float64(gbug))
	}
	ss.VGPlot.Update()
}

func (ss *Sim) ConfigVGTable(dt *etable.Table) {
	dt.SetMetaData("name", "GABABplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"V", etensor.FLOAT64, nil, nil},
		{"GgabaB", etensor.FLOAT64, nil, nil},
		{"GgabaB_std", etensor.FLOAT64, nil, nil},
		{"GgabaB_bug", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigVGPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "V-G Function Plot"
	plt.Params.XAxisCol = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GgabaB", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GgabaB_std", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////////

// SGRun runs the spike-g equation.
func (ss *Sim) SGRun() {
	ss.Update()
	dt := ss.SGTable

	nv := int(float64(ss.Smax) / ss.Vstep)
	dt.SetNumRows(nv)
	s := 0.0
	g := 0.0
	for si := 0; si < nv; si++ {
		s = float64(si) * ss.Vstep
		g = 1.0 / (1.0 + math.Exp(-(s-7.1)/1.4))
		gs := ss.GABAstd.GFmS(float32(s))

		dt.SetCellFloat("S", si, s)
		dt.SetCellFloat("GgabaB_max", si, g)
		dt.SetCellFloat("GgabaBstd_max", si, float64(gs))
	}
	ss.SGPlot.Update()
}

func (ss *Sim) ConfigSGTable(dt *etable.Table) {
	dt.SetMetaData("name", "SG_GABAplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"S", etensor.FLOAT64, nil, nil},
		{"GgabaB_max", etensor.FLOAT64, nil, nil},
		{"GgabaBstd_max", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigSGPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "S-G Function Plot"
	plt.Params.XAxisCol = "S"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("S", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GgabaB_max", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GgabaBstd_max", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////////

// TimeRun runs the equation.
func (ss *Sim) TimeRun() {
	ss.Update()
	dt := ss.TimeTable

	dt.SetNumRows(ss.TimeSteps)
	time := 0.0
	gs := 0.0
	x := ss.GsXInit
	gabaBx := float32(ss.GsXInit)
	gabaB := float32(0.0)
	gi := 0.0 // just goes down
	for t := 0; t < ss.TimeSteps; t++ {
		// record starting state first, then update
		dt.SetCellFloat("Time", t, time)
		dt.SetCellFloat("Gs", t, gs)
		dt.SetCellFloat("GsX", t, x)
		dt.SetCellFloat("GABAB", t, float64(gabaB))
		dt.SetCellFloat("GABABx", t, float64(gabaBx))

		gis := 1.0 / (1.0 + math.Exp(-(gi-7.1)/1.4))
		dGs := (ss.TauFact*x - gs) / ss.RiseTau
		dXo := -x / ss.DecayTau
		gs += dGs
		x += gis + dXo

		var dG, dX float32
		ss.GABAstd.BiExp(gabaB, gabaBx, &dG, &dX)
		dt.SetCellFloat("dG", t, float64(dG))
		dt.SetCellFloat("dX", t, float64(dX))

		ss.GABAstd.GABAB(float32(gi), &gabaB, &gabaBx)

		time += ss.TimeInc
	}
	ss.TimePlot.Update()
}

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "TimeGaBabplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"Gs", etensor.FLOAT64, nil, nil},
		{"GsX", etensor.FLOAT64, nil, nil},
		{"GABAB", etensor.FLOAT64, nil, nil},
		{"GABABx", etensor.FLOAT64, nil, nil},
		{"dG", etensor.FLOAT64, nil, nil},
		{"dX", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "G Time Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gs", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GsX", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GABAB", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GABABx", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	// gi.WinEventTrace = true

	gi.SetAppName("gabab_plot")
	gi.SetAppAbout(`This plots an equation. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("gababplot", "Plotting Equations", width, height)
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

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "VGPlot").(*eplot.Plot2D)
	ss.VGPlot = ss.ConfigVGPlot(plt, ss.VGTable)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "SGPlot").(*eplot.Plot2D)
	ss.SGPlot = ss.ConfigSGPlot(plt, ss.SGTable)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TimePlot").(*eplot.Plot2D)
	ss.TimePlot = ss.ConfigTimePlot(plt, ss.TimeTable)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Run VG", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.VGRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Run SG", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.SGRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Run Time", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.TimeRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/chans/gabab_plot/README.md")
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
