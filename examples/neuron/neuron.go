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

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/minmax"
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

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no learning",
				Params: params.Params{
					"Prjn.Learn.Enabled": "false",
				}},
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "false",
					"Layer.Act.Init.Vm":    "0.3",
				}},
		},
	}},
}

// Extra state for neuron
type NeuronEx struct {
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
	GeClamp      bool          `desc:"clamp constant Ge value -- otherwise drive discrete spiking input"`
	SpikeHz      float32       `desc:"frequency of input spiking for !GeClamp mode"`
	Ge           float32       `min:"0" step:"0.01" desc:"Raw synaptic excitatory conductance"`
	Gi           float32       `min:"0" step:"0.01" desc:"Inhibitory conductance "`
	ErevE        float32       `min:"0" max:"1" step:"0.01" def:"1" desc:"excitatory reversal (driving) potential -- determines where excitation pushes Vm up to"`
	ErevI        float32       `min:"0" max:"1" step:"0.01" def:"0.3" desc:"leak reversal (driving) potential -- determines where excitation pulls Vm down to"`
	Noise        float32       `min:"0" step:"0.01" desc:"the variance parameter for Gaussian noise added to unit activations on every cycle"`
	KNaAdapt     bool          `desc:"apply sodium-gated potassium adaptation mechanisms that cause the neuron to reduce spiking over time"`
	MAHPGbar     float32       `def:"0.05" desc:"strength of mAHP M-type channel -- used to be implemented by KNa but now using the more standard M-type channel mechanism"`
	NMDAGbar     float32       `def:"0,0.15" desc:"strength of NMDA current -- 0.15 default for posterior cortex"`
	GABABGbar    float32       `def:"0,0.2" desc:"strength of GABAB current -- 0.2 default for posterior cortex"`
	VGCCGbar     float32       `def:"0.02" desc:"strength of VGCC voltage gated calcium current -- only activated during spikes -- this is now an essential part of Ca-driven learning to reflect recv spiking in the Ca signal -- but if too strong leads to runaway excitatory bursting."`
	AKGbar       float32       `def:"0.1" desc:"strength of A-type potassium channel -- this is only active at high (depolarized) membrane potentials -- only during spikes -- useful to counteract VGCC's"`
	NCycles      int           `min:"10" def:"200" desc:"total number of cycles to run"`
	OnCycle      int           `min:"0" def:"10" desc:"when does excitatory input into neuron come on?"`
	OffCycle     int           `min:"0" def:"160" desc:"when does excitatory input into neuron go off?"`
	UpdtInterval int           `min:"1" def:"10"  desc:"how often to update display (in cycles)"`
	Net          *axon.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	NeuronEx     NeuronEx      `view:"no-inline" desc:"extra neuron state for additional channels: VGCC, AK"`
	Stats        estats.Stats  `desc:"contains computed statistic values"`
	Logs         elog.Logs     `view:"no-inline" desc:"logging"`
	Params       params.Sets   `view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"`

	Cycle int `inactive:"+" desc:"current cycle of updating"`

	// internal state - view:"-"
	Win        *gi.Window       `view:"-" desc:"main GUI window"`
	NetView    *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar    *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	TstCycPlot *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	IsRunning  bool             `view:"-" desc:"true if sim is running"`
	StopNow    bool             `view:"-" desc:"flag to stop running"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Params = ParamSets
	ss.Stats.Init()
	ss.Defaults()
}

// Defaults sets default params
func (ss *Sim) Defaults() {
	ss.UpdtInterval = 10
	ss.GeClamp = true
	ss.Cycle = 0
	ss.SpikeHz = 50
	ss.Ge = 0.5
	ss.Gi = 0.0
	ss.ErevE = 1
	ss.ErevI = 0.3
	ss.Noise = 0
	ss.KNaAdapt = true
	ss.MAHPGbar = 0.05
	ss.NMDAGbar = 0.15
	ss.GABABGbar = 0.2
	ss.VGCCGbar = 0.02
	ss.AKGbar = 0.1
	ss.NCycles = 200
	ss.OnCycle = 10
	ss.OffCycle = 160
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Neuron")
	net.AddLayer2D("Neuron", 1, 1, emer.Hidden)

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
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
	ss.Cycle = 0
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
	return fmt.Sprintf("Cycle:\t%d\t\t\t", ss.Cycle)
}

func (ss *Sim) UpdateView() {
	ss.TstCycPlot.UpdatePlot()
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(), ss.Cycle)
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// RunCycles updates neuron over specified number of cycles
func (ss *Sim) RunCycles() {
	ss.Init()
	ss.StopNow = false
	ss.Net.InitActs()
	ss.SetParams("", false)
	ly := ss.Net.LayerByName("Neuron").(axon.AxonLayer).AsAxon()
	nrn := &(ly.Neurons[0])
	inputOn := false
	for cyc := 0; cyc < ss.NCycles; cyc++ {
		ss.Cycle = cyc
		switch cyc {
		case ss.OnCycle:
			inputOn = true
		case ss.OffCycle:
			inputOn = false
		}
		// nrn.Noise = float32(ly.Act.Noise.Gen(-1))
		// nrn.Ge += nrn.Noise // GeNoise
		nrn.Gi = 0
		ss.NeuronUpdt(ss.Net, inputOn)
		ss.Logs.LogRow(etime.Test, etime.Cycle, ss.Cycle)
		if ss.Cycle%ss.UpdtInterval == 0 {
			ss.UpdateView()
		}
		if ss.StopNow {
			break
		}
	}
	ss.UpdateView()
}

// NeuronUpdt updates the neuron
// this just calls the relevant code directly, bypassing most other stuff.
func (ss *Sim) NeuronUpdt(nt *axon.Network, inputOn bool) {
	ly := ss.Net.LayerByName("Neuron").(axon.AxonLayer).AsAxon()
	ac := &ly.Act
	nrn := &(ly.Neurons[0])
	nex := &ss.NeuronEx
	if inputOn {
		if ss.GeClamp {
			nrn.GeRaw = ss.Ge
			nrn.GeSyn = nrn.GeRaw
		} else {
			nex.InISI += 1
			if nex.InISI > 1000/ss.SpikeHz {
				nrn.GeRaw = ss.Ge
				nex.InISI = 0
			} else {
				nrn.GeRaw = 0
			}
			ac.Dt.GeSynFmRaw(nrn.GeRaw, &nrn.GeSyn, ac.Init.Ge)
		}
	} else {
		nrn.GeRaw = 0
		nrn.GeSyn = 0
	}
	nrn.Ge = nrn.GeSyn
	nrn.Gi = ss.Gi

	ac.NMDAFmRaw(nrn, 0)
	ac.GvgccFmVm(nrn)

	nrn.GABAB, nrn.GABABx = ac.GABAB.GABAB(nrn.GABAB, nrn.GABABx, nrn.Gi)
	nrn.GgabaB = ac.GABAB.GgabaB(nrn.GABAB, nrn.VmDend)

	nrn.Gi += nrn.GgabaB

	ac.VmFmG(nrn)
	ac.ActFmG(nrn)
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
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	ly := ss.Net.LayerByName("Neuron").(axon.AxonLayer).AsAxon()
	ly.Act.Gbar.E = 1
	ly.Act.Gbar.L = 0.2
	ly.Act.Erev.E = float32(ss.ErevE)
	ly.Act.Erev.I = float32(ss.ErevI)
	// ly.Act.Noise.Var = float64(ss.Noise)
	ly.Act.KNa.On = ss.KNaAdapt
	ly.Act.MAHP.Gbar = ss.MAHPGbar
	ly.Act.NMDA.Gbar = ss.NMDAGbar
	ly.Act.GABAB.Gbar = ss.GABABGbar
	ly.Act.VGCC.Gbar = ss.VGCCGbar
	ly.Act.AK.Gbar = ss.AKGbar
	ly.Act.Update()
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()

	ss.Logs.PlotItems("Vm", "Spike")

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
}

func (ss *Sim) ConfigLogItems() {
	ly := ss.Net.LayerByName("Neuron").(axon.AxonLayer).AsAxon()
	nrn := &(ly.Neurons[0])
	// nex := &ss.NeuronEx
	lg := &ss.Logs

	lg.AddItem(&elog.Item{
		Name:   "Cycle",
		Type:   etensor.INT64,
		FixMax: false,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
				ctx.SetInt(ss.Cycle)
			}}})

	vars := []string{"GeSyn", "Ge", "Gi", "Inet", "Vm", "Act", "Spike", "Gk", "ISI", "ISIAvg", "VmDend", "GnmdaSyn", "Gnmda", "GABAB", "GgabaB", "Gvgcc", "VgccM", "VgccH", "Gak", "MahpN", "GknaMed", "GknaSlow"}

	for _, vnm := range vars {
		cvnm := vnm // closure
		lg.AddItem(&elog.Item{
			Name:   cvnm,
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					vl, _ := nrn.VarByName(cvnm)
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
