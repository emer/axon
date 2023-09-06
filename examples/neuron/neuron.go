// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
neuron: This simulation illustrates the basic properties of neural spiking and
rate-code activation, reflecting a balance of excitatory and inhibitory
influences (including leak and synaptic inhibition).
*/
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/econfig"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/netparams"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/minmax"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		gimain.Main(sim.RunGUI)
	} else {
		sim.RunNoGUI()
	}
}

// see config.go for Config

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: "Prjn", Desc: "no learning",
			Params: params.Params{
				"Prjn.Learn.Learn": "false",
			}},
		{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Inhib.Layer.On": "false",
				"Layer.Acts.Init.Vm":   "0.3",
			}},
	},
	"Testing": {
		{Sel: "Layer", Desc: "",
			Params: params.Params{
				"Layer.Acts.NMDA.Gbar":  "0.0",
				"Layer.Acts.GabaB.Gbar": "0.0",
			}},
	},
}

// Extra state for neuron
type NeuronEx struct {

	// input ISI countdown for spiking mode -- counts up
	InISI float32 `desc:"input ISI countdown for spiking mode -- counts up"`
}

func (nrn *NeuronEx) Init() {
	nrn.InISI = 0
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config `desc:"simulation configuration parameters -- set by .toml config file and / or args"`

	// [view: no-inline] the network -- click to view / edit parameters for layers, prjns, etc
	Net *axon.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`

	// [view: no-inline] extra neuron state for additional channels: VGCC, AK
	NeuronEx NeuronEx `view:"no-inline" desc:"extra neuron state for additional channels: VGCC, AK"`

	// axon timing parameters and state
	Context axon.Context `desc:"axon timing parameters and state"`

	// contains computed statistic values
	Stats estats.Stats `desc:"contains computed statistic values"`

	// [view: no-inline] logging
	Logs elog.Logs `view:"no-inline" desc:"logging"`

	// [view: inline] all parameter management
	Params emer.NetParams `view:"inline" desc:"all parameter management"`

	// current cycle of updating
	Cycle int `inactive:"+" desc:"current cycle of updating"`

	// [view: -] main GUI window
	Win *gi.Window `view:"-" desc:"main GUI window"`

	// [view: -] the network viewer
	NetView *netview.NetView `view:"-" desc:"the network viewer"`

	// [view: -] the master toolbar
	ToolBar *gi.ToolBar `view:"-" desc:"the master toolbar"`

	// [view: -] the test-trial plot
	TstCycPlot *eplot.Plot2D `view:"-" desc:"the test-trial plot"`

	// [view: -] map of values for detailed debugging / testing
	ValMap map[string]float32 `view:"-" desc:"map of values for detailed debugging / testing"`

	// [view: -] true if sim is running
	IsRunning bool `view:"-" desc:"true if sim is running"`

	// [view: -] flag to stop running
	StopNow bool `view:"-" desc:"flag to stop running"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	econfig.Config(&ss.Config, "config.toml")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.ValMap = make(map[string]float32)
}

func (ss *Sim) Defaults() {
	econfig.Config(&ss.Config, "config.toml")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context

	net.InitName(net, "Neuron")
	in := net.AddLayer2D("Input", 1, 1, axon.InputLayer)
	hid := net.AddLayer2D("Neuron", 1, 1, axon.SuperLayer)

	net.ConnectLayers(in, hid, prjn.NewFull(), axon.ForwardPrjn)

	err := net.Build(ctx)
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	ss.InitWts(net)
}

// InitWts loads the saved weights
func (ss *Sim) InitWts(net *axon.Network) {
	net.InitWts(&ss.Context)
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Context.Reset()
	ss.InitWts(ss.Net)
	ss.NeuronEx.Init()
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.UpdateView()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Cycle:\t%d\t\t\t", ss.Context.Cycle)
}

func (ss *Sim) UpdateView() {
	ss.TstCycPlot.UpdatePlot()
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(), int(ss.Context.Cycle))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// RunCycles updates neuron over specified number of cycles
func (ss *Sim) RunCycles() {
	ctx := &ss.Context
	ss.Init()
	ss.StopNow = false
	ss.Net.InitActs(ctx)
	ctx.NewState(etime.Train)
	ss.SetParams("", false)
	// ly := ss.Net.AxonLayerByName("Neuron")
	// nrn := &(ly.Neurons[0])
	inputOn := false
	for cyc := 0; cyc < ss.Config.NCycles; cyc++ {
		switch cyc {
		case ss.Config.OnCycle:
			inputOn = true
		case ss.Config.OffCycle:
			inputOn = false
		}
		ss.NeuronUpdt(ss.Net, inputOn)
		ctx.Cycle = int32(cyc)
		ss.Logs.LogRow(etime.Test, etime.Cycle, cyc)
		ss.RecordVals(cyc)
		if cyc%ss.Config.UpdtInterval == 0 {
			ss.UpdateView()
		}
		ss.Context.CycleInc()
		if ss.StopNow {
			break
		}
	}
	ss.UpdateView()
}

func (ss *Sim) RecordVals(cyc int) {
	var vals []float32
	ly := ss.Net.AxonLayerByName("Neuron")
	key := fmt.Sprintf("cyc: %03d", cyc)
	for _, vnm := range axon.NeuronVarNames {
		ly.UnitVals(&vals, vnm, 0)
		vkey := key + fmt.Sprintf("\t%s", vnm)
		ss.ValMap[vkey] = vals[0]
	}
}

// NeuronUpdt updates the neuron
// this just calls the relevant code directly, bypassing most other stuff.
func (ss *Sim) NeuronUpdt(nt *axon.Network, inputOn bool) {
	ctx := &ss.Context
	ly := ss.Net.AxonLayerByName("Neuron")
	ni := ly.NeurStIdx
	di := uint32(0)
	ac := &ly.Params.Acts
	nex := &ss.NeuronEx
	// nrn.Noise = float32(ly.Params.Act.Noise.Gen(-1))
	// nrn.Ge += nrn.Noise // GeNoise
	// nrn.Gi = 0
	if inputOn {
		if ss.Config.GeClamp {
			axon.SetNrnV(ctx, ni, di, axon.GeRaw, ss.Config.Ge)
			axon.SetNrnV(ctx, ni, di, axon.GeSyn, ac.Dt.GeSynFmRawSteady(axon.NrnV(ctx, ni, di, axon.GeRaw)))
		} else {
			nex.InISI += 1
			if nex.InISI > 1000/ss.Config.SpikeHz {
				axon.SetNrnV(ctx, ni, di, axon.GeRaw, ss.Config.Ge)
				nex.InISI = 0
			} else {
				axon.SetNrnV(ctx, ni, di, axon.GeRaw, 0)
			}
			axon.SetNrnV(ctx, ni, di, axon.GeSyn, ac.Dt.GeSynFmRaw(axon.NrnV(ctx, ni, di, axon.GeSyn), axon.NrnV(ctx, ni, di, axon.GeRaw)))
		}
	} else {
		axon.SetNrnV(ctx, ni, di, axon.GeRaw, 0)
		axon.SetNrnV(ctx, ni, di, axon.GeSyn, 0)
	}
	axon.SetNrnV(ctx, ni, di, axon.GiRaw, ss.Config.Gi)
	axon.SetNrnV(ctx, ni, di, axon.GiSyn, ac.Dt.GiSynFmRawSteady(axon.NrnV(ctx, ni, di, axon.GiRaw)))

	if ss.Net.GPU.On {
		ss.Net.GPU.SyncStateToGPU()
		ss.Net.GPU.RunPipelineWait("Cycle", 2)
		ss.Net.GPU.SyncStateFmGPU()
		ctx.CycleInc() // why is this not working!?
	} else {
		lpl := ly.Pool(0, di)
		ly.GInteg(ctx, ni, di, lpl, ly.LayerVals(0))
		ly.SpikeFmG(ctx, ni, di, lpl)
	}
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) {
	ss.Params.SetAll()
	ly := ss.Net.AxonLayerByName("Neuron")
	lyp := ly.Params
	lyp.Acts.Gbar.E = 1
	lyp.Acts.Gbar.L = 0.2
	lyp.Acts.Erev.E = float32(ss.Config.ErevE)
	lyp.Acts.Erev.I = float32(ss.Config.ErevI)
	// lyp.Acts.Noise.Var = float64(ss.Config.Noise)
	lyp.Acts.KNa.On.SetBool(ss.Config.KNaAdapt)
	lyp.Acts.Mahp.Gbar = ss.Config.MahpGbar
	lyp.Acts.NMDA.Gbar = ss.Config.NMDAGbar
	lyp.Acts.GabaB.Gbar = ss.Config.GABABGbar
	lyp.Acts.VGCC.Gbar = ss.Config.VGCCGbar
	lyp.Acts.AK.Gbar = ss.Config.AKGbar
	lyp.Acts.Update()
}

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()

	ss.Logs.PlotItems("Vm", "Spike")

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
}

func (ss *Sim) ConfigLogItems() {
	ly := ss.Net.AxonLayerByName("Neuron")
	// nex := &ss.NeuronEx
	lg := &ss.Logs

	lg.AddItem(&elog.Item{
		Name:   "Cycle",
		Type:   etensor.INT64,
		FixMax: false,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
				ctx.SetInt(int(ss.Context.Cycle))
			}}})

	vars := []string{"GeSyn", "Ge", "Gi", "Inet", "Vm", "Act", "Spike", "Gk", "ISI", "ISIAvg", "VmDend", "GnmdaSyn", "Gnmda", "GABAB", "GgabaB", "Gvgcc", "VgccM", "VgccH", "Gak", "MahpN", "GknaMed", "GknaSlow", "GiSyn"}

	for _, vnm := range vars {
		cvnm := vnm // closure
		lg.AddItem(&elog.Item{
			Name:   cvnm,
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					vl := ly.UnitVal(cvnm, []int{0, 0}, 0)
					ctx.SetFloat32(vl)
				}}})
	}

}

func (ss *Sim) ResetTstCycPlot() {
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
	ss.TstCycPlot.Update()
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

	gi.SetAppName("neuron")
	gi.SetAppAbout(`This simulation illustrates the basic properties of neural spiking and
rate-code activation, reflecting a balance of excitatory and inhibitory
influences (including leak and synaptic inhibition).
See <a href="https://github.com/emer/axon/blob/master/examples/neuron/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("neuron", "Neuron", width, height)
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

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	key := etime.Scope(etime.Test, etime.Cycle)
	plt.SetTable(ss.Logs.Table(etime.Test, etime.Cycle))
	egui.ConfigPlotFromLog("Neuron", plt, &ss.Logs, key)
	ss.TstCycPlot = plt

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

	tbar.AddAction(gi.ActOpts{Label: "Run Cycles", Icon: "step-fwd", Tooltip: "Runs neuron updating over NCycles.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.RunCycles()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddSeparator("run-sep")

	tbar.AddAction(gi.ActOpts{Label: "Reset Plot", Icon: "update", Tooltip: "Reset TstCycPlot.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.ResetTstCycPlot()
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

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUwithGUI(&ss.Context)
		gi.SetQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}

	win.MainMenuUpdated()
	return win
}

func (ss *Sim) RunGUI() {
	ss.Init()
	win := ss.ConfigGui()
	win.StartEventLoop()
}

func (ss *Sim) RunNoGUI() {
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWts {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name()

	// netdata := ss.Config.Log.NetData
	// if netdata {
	// 	mpi.Printf("Saving NetView data from testing\n")
	// 	ss.GUI.InitNetData(ss.Net, 200)
	// }

	ss.Init()

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	ss.RunCycles()

	if ss.Config.Log.Cycle {
		dt := ss.Logs.Table(etime.Test, etime.Cycle)
		fnm := ecmd.LogFileName("cyc", netName, runName)
		dt.SaveCSV(gi.FileName(fnm), etable.Tab, etable.Headers)
	}

	// if netdata {
	// 	ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	// }

	ss.Net.GPU.Destroy() // safe even if no GPU
}
