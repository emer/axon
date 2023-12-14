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
	TheSim.VmRun()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// mAHP function
	Mahp chans.MahpParams `view:"inline"`

	// starting voltage
	Vstart float32 `def:"-100"`

	// ending voltage
	Vend float32 `def:"100"`

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

	// main GUI window
	Win *gi.Window `view:"-"`

	// the master toolbar
	ToolBar *gi.ToolBar `view:"-"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.Mahp.Defaults()
	ss.Mahp.Gbar = 1
	ss.Vstart = -100
	ss.Vend = 100
	ss.Vstep = 1
	ss.TimeSteps = 300
	ss.TimeSpike = true
	ss.SpikeFreq = 50
	ss.TimeVstart = -70
	ss.TimeVend = -50
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

	mp := &ss.Mahp

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	for vi := 0; vi < nv; vi++ {
		vbio := ss.Vstart + float32(vi)*ss.Vstep
		var ninf, tau float32
		mp.NinfTauFmV(vbio, &ninf, &tau)

		dt.SetCellFloat("V", vi, float64(vbio))
		dt.SetCellFloat("Ninf", vi, float64(ninf))
		dt.SetCellFloat("Tau", vi, float64(tau))
	}
	ss.Plot.Update()
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "mAHPplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"V", etensor.FLOAT64, nil, nil},
		{"Ninf", etensor.FLOAT64, nil, nil},
		{"Tau", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "mAHP V Function Plot"
	plt.Params.XAxisCol = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Ninf", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Tau", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 1)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() {
	ss.Update()
	dt := ss.TimeTable

	mp := &ss.Mahp

	var n, tau float32
	mp.NinfTauFmV(ss.TimeVstart, &n, &tau)
	kna := float32(0)
	msdt := float32(0.001)
	v := ss.TimeVstart
	vinc := float32(2) * (ss.TimeVend - ss.TimeVstart) / float32(ss.TimeSteps)

	isi := int(1000 / ss.SpikeFreq)

	dt.SetNumRows(ss.TimeSteps)
	for ti := 1; ti <= ss.TimeSteps; ti++ {
		vnorm := chans.VFmBio(v)
		t := float32(ti) * msdt

		var ninf, tau float32
		mp.NinfTauFmV(v, &ninf, &tau)
		dn := mp.DNFmV(vnorm, n)
		g := mp.GmAHP(n)

		dt.SetCellFloat("Time", ti, float64(t))
		dt.SetCellFloat("V", ti, float64(v))
		dt.SetCellFloat("GmAHP", ti, float64(g))
		dt.SetCellFloat("N", ti, float64(n))
		dt.SetCellFloat("dN", ti, float64(dn))
		dt.SetCellFloat("Ninf", ti, float64(ninf))
		dt.SetCellFloat("Tau", ti, float64(tau))
		dt.SetCellFloat("Kna", ti, float64(kna))

		if ss.TimeSpike {
			si := ti % isi
			if si == 0 {
				v = ss.TimeVend
				kna += 0.05 * (1 - kna)
			} else {
				v = ss.TimeVstart + (float32(si)/float32(isi))*(ss.TimeVend-ss.TimeVstart)
				kna -= kna / 50
			}
		} else {
			v += vinc
			if v > ss.TimeVend {
				v = ss.TimeVend
			}
		}
		n += dn
	}
	ss.TimePlot.Update()
}

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "mAHPplotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"V", etensor.FLOAT64, nil, nil},
		{"GmAHP", etensor.FLOAT64, nil, nil},
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
	plt.SetColParams("V", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GmAHP", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
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

	gi.SetAppName("mahp_plot")
	gi.SetAppAbout(`This plots an equation. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("mahp_plot", "Plotting Equations", width, height)
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
			gi.OpenURL("https://github.com/emer/axon/blob/master/chans/mahp_plot/README.md")
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
