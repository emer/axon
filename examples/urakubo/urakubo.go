// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
urakubo: This simulation replicates the Urakubo et al, 2008 detailed model of spike-driven
learning, including intracellular Ca-driven signaling, involving CaMKII, CaN, PKA, PP1.
*/
package main

import (
	"fmt"
	"log"
	"strconv"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/netview"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Spine      Spine         `desc:"the spine state with Urakubo intracellular model"`
	Neuron     *axon.Neuron  `desc:"the neuron"`
	Stim       Stims         `desc:"what stimulation to drive with"`
	CaTarg     CaState       `desc:"target calcium level for CaTarg stim"`
	Msec       int           `inactive:"+" desc:"current cycle of updating"`
	Msec100Log *etable.Table `view:"no-inline" desc:"every 100 msec plot -- a point every 100 msec, shows full run"`
	Msec10Log  *etable.Table `view:"no-inline" desc:"every 10 msec plot -- a point every 10 msec, shows last 10 seconds"`
	MsecLog    *etable.Table `view:"no-inline" desc:"millisecond level log, shows last second"`
	Net        *axon.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`

	// internal state - view:"-"
	Win         *gi.Window       `view:"-" desc:"main GUI window"`
	NetView     *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar     *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	Msec100Plot *eplot.Plot2D    `view:"-" desc:"the plot"`
	Msec10Plot  *eplot.Plot2D    `view:"-" desc:"the plot"`
	MsecPlot    *eplot.Plot2D    `view:"-" desc:"the plot"`
	IsRunning   bool             `view:"-" desc:"true if sim is running"`
	StopNow     bool             `view:"-" desc:"flag to stop running"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Spine.Defaults()
	ss.Spine.Init()
	ss.Net = &axon.Network{}
	ss.MsecLog = &etable.Table{}
	ss.Msec10Log = &etable.Table{}
	ss.Msec100Log = &etable.Table{}
	ss.Defaults()
}

// Defaults sets default params
func (ss *Sim) Defaults() {
	ss.Spine.Defaults()
	ss.CaTarg.Cyt = 10
	ss.CaTarg.PSD = 10
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigNet(ss.Net)
	ss.ConfigLog(ss.MsecLog)
	ss.ConfigLog(ss.Msec10Log)
	ss.ConfigLog(ss.Msec100Log)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Neuron")
	ly := net.AddLayer2D("Neuron", 1, 1, emer.Hidden).(*axon.Layer)

	net.Defaults()
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
	ss.Neuron = &ly.Neurons[0]
}

// InitWts loads the saved weights
func (ss *Sim) InitWts(net *axon.Network) {
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Spine.Defaults()
	ss.Spine.Init()
	ss.Msec = 0
	ss.InitWts(ss.Net)
	ss.StopNow = false
	ss.UpdateView()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Msec:\t%d\t\t\t", ss.Msec)
}

func (ss *Sim) UpdateView() {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters())
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// RunStim runs current Stim selection
func (ss *Sim) RunStim() {
	fn, has := StimFuncs[ss.Stim]
	if !has {
		fmt.Printf("Stim function: %s not found!\n", ss.Stim)
		return
	}
	go fn()
}

// NeuronUpdt updates the neuron and spine for given msec
func (ss *Sim) NeuronUpdt(msec int) {
	ss.Msec = msec
	ly := ss.Net.LayerByName("Neuron").(axon.AxonLayer).AsAxon()
	nrn := ss.Neuron
	ly.Act.VmFmG(nrn)
	ly.Act.ActFmG(nrn)
	ss.Spine.StepTime(0.001)
}

// LogDefault does default logging for current Msec
func (ss *Sim) LogDefault() {
	msec := ss.Msec
	ss.Log(ss.MsecLog, msec%1000)
	if ss.Msec%10 == 0 {
		ss.Log(ss.Msec10Log, (msec/10)%1000)
		if ss.Msec%100 == 0 {
			ss.Log(ss.Msec100Log, (msec / 100))
			ss.MsecPlot.GoUpdate()
			ss.Msec10Plot.GoUpdate()
			ss.Msec100Plot.GoUpdate()
		}
	}
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

//////////////////////////////////////////////
//  Log

// Log adds data from current msec to the given table at given row
func (ss *Sim) Log(dt *etable.Table, row int) {
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}
	nrn := ss.Neuron

	dt.SetCellFloat("Time", row, float64(ss.Msec)*0.001)
	dt.SetCellFloat("Ge", row, float64(nrn.Ge))
	dt.SetCellFloat("Inet", row, float64(nrn.Inet))
	dt.SetCellFloat("Vm", row, float64(nrn.Vm))
	dt.SetCellFloat("Act", row, float64(nrn.Act))
	dt.SetCellFloat("Spike", row, float64(nrn.Spike))
	dt.SetCellFloat("Gk", row, float64(nrn.Gk))
	dt.SetCellFloat("ISI", row, float64(nrn.ISI))
	dt.SetCellFloat("AvgISI", row, float64(nrn.ISIAvg))

	ss.Spine.Log(dt, row)
}

func (ss *Sim) ConfigLog(dt *etable.Table) {
	dt.SetMetaData("name", "Data Log")
	dt.SetMetaData("desc", "Record of neuron / spine data")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"Ge", etensor.FLOAT64, nil, nil},
		{"Inet", etensor.FLOAT64, nil, nil},
		{"Vm", etensor.FLOAT64, nil, nil},
		{"Act", etensor.FLOAT64, nil, nil},
		{"Spike", etensor.FLOAT64, nil, nil},
		{"Gk", etensor.FLOAT64, nil, nil},
		{"ISI", etensor.FLOAT64, nil, nil},
		{"AvgISI", etensor.FLOAT64, nil, nil},
	}

	ss.Spine.ConfigLog(&sch)

	dt.SetFromSchema(sch, 1000)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Urakubo Data Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Ge", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Inet", eplot.Off, eplot.FixMin, -.2, eplot.FixMax, 1)
	plt.SetColParams("Vm", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Act", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Spike", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Gk", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("ISI", eplot.Off, eplot.FixMin, -2, eplot.FloatMax, 1)
	plt.SetColParams("AvgISI", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)

	for _, cn := range dt.ColNames {
		if cn != "Time" {
			plt.SetColParams(cn, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		}
	}

	plt.SetColParams("Cyt_Ca", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("PSD_Ca", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("Cyt_AC1act", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("PSD_AC1act", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("PSD_CaMKIIact", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("Trp_AMPAR", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)

	return plt
}

func (ss *Sim) ResetPlots() {
	ss.MsecLog.SetNumRows(0)
	ss.MsecPlot.Update()
	ss.Msec10Log.SetNumRows(0)
	ss.Msec10Plot.Update()
	ss.Msec100Log.SetNumRows(0)
	ss.Msec100Plot.Update()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("urakubo")
	gi.SetAppAbout(`This simulation replicates the Urakubo et al, 2008 biophysical model of LTP / LTD.
See <a href="https://github.com/emer/axon/blob/master/examples/urakubo/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("urakubo", "Urakubo", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMaxWidth()
	split.SetStretchMaxHeight()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv) // add labels etc

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "Msec100Plot").(*eplot.Plot2D)
	ss.Msec100Plot = ss.ConfigPlot(plt, ss.Msec100Log)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "Msec10Plot").(*eplot.Plot2D)
	ss.Msec10Plot = ss.ConfigPlot(plt, ss.Msec10Log)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "MsecPlot").(*eplot.Plot2D)
	ss.MsecPlot = ss.ConfigPlot(plt, ss.MsecLog)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Run", Icon: "step-fwd", Tooltip: "Runs current Stim.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			ss.RunStim() // does go
		}
	})

	tbar.AddSeparator("run-sep")

	tbar.AddAction(gi.ActOpts{Label: "Reset Plots", Icon: "update", Tooltip: "Reset Plots.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.ResetPlots()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Defaults", Icon: "update", Tooltip: "Restore initial default parameters.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Defaults()
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/neuron/README.md")
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

	/*
		inQuitPrompt := false
		gi.SetQuitReqFunc(func() {
			if inQuitPrompt {
				return
			}
			inQuitPrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
				Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inQuitPrompt = false
					}
				})
		})

		// gi.SetQuitCleanFunc(func() {
		// 	fmt.Printf("Doing final Quit cleanup here..\n")
		// })

		inClosePrompt := false
		win.SetCloseReqFunc(func(w *gi.Window) {
			if inClosePrompt {
				return
			}
			inClosePrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
				Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inClosePrompt = false
					}
				})
		})
	*/

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts",
				}},
			},
		}},
	},
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
