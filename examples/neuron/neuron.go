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
	"strconv"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/chans"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
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

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no learning",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
				}},
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "false",
					"Layer.Act.Init.Vm":    "0.3",
				}},
		},
	}},
}

// Extra state for neuron -- VGCC and AK
type NeuronEx struct {
	Gvgcc float32 `desc:"VGCC total conductance"`
	VGCCm float32 `desc:"VGCC M gate -- activates with increasing Vm"`
	VGCCh float32 `desc:"VGCC H gate -- deactivates with increasing Vm"`
	Gak   float32 `desc:"AK total conductance"`
	AKm   float32 `desc:"AK M gate -- activates with increasing Vm"`
	AKh   float32 `desc:"AK H gate -- deactivates with increasing Vm"`
	InISI float32 `desc:"input ISI countdown for spiking mode -- counts up"`
}

func (nrn *NeuronEx) Init() {
	nrn.Gvgcc = 0
	nrn.VGCCm = 0
	nrn.VGCCh = 1
	nrn.Gak = 0
	nrn.AKm = 0
	nrn.AKh = 1
	nrn.InISI = 0
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	GeClamp      bool             `desc:"clamp constant Ge value -- otherwise drive discrete spiking input"`
	SpikeHz      float32          `desc:"frequency of input spiking for !GeClamp mode"`
	Ge           float32          `min:"0" step:"0.01" desc:"Raw synaptic excitatory conductance"`
	Gi           float32          `min:"0" step:"0.01" desc:"Inhibitory conductance "`
	ErevE        float32          `min:"0" max:"1" step:"0.01" def:"1" desc:"excitatory reversal (driving) potential -- determines where excitation pushes Vm up to"`
	ErevI        float32          `min:"0" max:"1" step:"0.01" def:"0.3" desc:"leak reversal (driving) potential -- determines where excitation pulls Vm down to"`
	Noise        float32          `min:"0" step:"0.01" desc:"the variance parameter for Gaussian noise added to unit activations on every cycle"`
	KNaAdapt     bool             `desc:"apply sodium-gated potassium adaptation mechanisms that cause the neuron to reduce spiking over time"`
	NMDAGbar     float32          `def:"0,0.15" desc:"strength of NMDA current -- 0.15 default for posterior cortex"`
	GABABGbar    float32          `def:"0,0.2" desc:"strength of GABAB current -- 0.2 default for posterior cortex"`
	VGCC         chans.VGCCParams `desc:"VGCC parameters: set Gbar > 0 to include"`
	AK           chans.AKParams   `desc:"A-type potassium channel parameters: set Gbar > 0 to include"`
	NCycles      int              `min:"10" def:"200" desc:"total number of cycles to run"`
	OnCycle      int              `min:"0" def:"10" desc:"when does excitatory input into neuron come on?"`
	OffCycle     int              `min:"0" def:"160" desc:"when does excitatory input into neuron go off?"`
	UpdtInterval int              `min:"1" def:"10"  desc:"how often to update display (in cycles)"`
	Net          *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	NeuronEx     NeuronEx         `view:"no-inline" desc:"extra neuron state for additional channels: VGCC, AK"`
	TstCycLog    *etable.Table    `view:"no-inline" desc:"testing trial-level log data -- click to see record of network's response to each input"`
	Params       params.Sets      `view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"`

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
	ss.TstCycLog = &etable.Table{}
	ss.Params = ParamSets
	ss.Defaults()
}

// Defaults sets default params
func (ss *Sim) Defaults() {
	ss.UpdtInterval = 10
	ss.Cycle = 0
	ss.SpikeHz = 50
	ss.Ge = 1.0
	ss.Gi = 0.8
	ss.ErevE = 1
	ss.ErevI = 0.3
	ss.Noise = 0
	ss.KNaAdapt = true
	ss.NMDAGbar = 0
	ss.GABABGbar = 0
	ss.VGCC.Defaults()
	ss.VGCC.Gbar = 0
	ss.AK.Defaults()
	ss.AK.Gbar = 0
	ss.NCycles = 200
	ss.OnCycle = 10
	ss.OffCycle = 160
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigNet(ss.Net)
	ss.ConfigTstCycLog(ss.TstCycLog)
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
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters())
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
		ss.LogTstCyc(ss.TstCycLog, ss.Cycle)
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
			ly.Act.Dt.GeSynFmRaw(nrn.GeRaw, &nrn.GeSyn, ly.Act.Init.Ge)
		}
	} else {
		nrn.GeRaw = 0
		nrn.GeSyn = 0
	}
	nrn.Ge = nrn.GeSyn
	nrn.Gi = ss.Gi
	nrn.NMDA = ly.Act.NMDA.NMDA(nrn.NMDA, nrn.GeRaw, 0)
	nrn.Gnmda = ly.Act.NMDA.Gnmda(nrn.NMDA, nrn.VmDend)
	nrn.GABAB, nrn.GABABx = ly.Act.GABAB.GABAB(nrn.GABAB, nrn.GABABx, nrn.Gi)
	nrn.GgabaB = ly.Act.GABAB.GgabaB(nrn.GABAB, nrn.VmDend)

	nex.Gvgcc = ss.VGCC.Gvgcc(nrn.VmDend, nex.VGCCm, nex.VGCCh)
	dm, dh := ss.VGCC.DMHFmV(nrn.VmDend, nex.VGCCm, nex.VGCCh)
	nex.VGCCm += dm
	nex.VGCCh += dh

	nex.Gak = ss.AK.Gak(nex.AKm, nex.AKh)
	dm, dh = ss.AK.DMHFmV(nrn.VmDend, nex.AKm, nex.AKh)
	nex.AKm += dm
	nex.AKh += dh

	nrn.Gk += nex.Gak
	nrn.Ge += nex.Gvgcc + nrn.Gnmda
	nrn.Gi += nrn.GgabaB

	ly.Act.VmFmG(nrn)
	ly.Act.ActFmG(nrn)
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
	ly.Act.NMDA.Gbar = ss.NMDAGbar
	ly.Act.GABAB.Gbar = ss.GABABGbar
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

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current cycle to the TstCycLog table.
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}
	row := cyc

	ly := ss.Net.LayerByName("Neuron").(axon.AxonLayer).AsAxon()
	nrn := &(ly.Neurons[0])
	nex := &ss.NeuronEx

	dt.SetCellFloat("Cycle", row, float64(cyc))
	dt.SetCellFloat("GeSyn", row, float64(nrn.GeSyn))
	dt.SetCellFloat("Ge", row, float64(nrn.Ge))
	dt.SetCellFloat("Gi", row, float64(nrn.Gi))
	dt.SetCellFloat("Inet", row, float64(nrn.Inet))
	dt.SetCellFloat("Vm", row, float64(nrn.Vm))
	dt.SetCellFloat("Act", row, float64(nrn.Act))
	dt.SetCellFloat("Spike", row, float64(nrn.Spike))
	dt.SetCellFloat("Gk", row, float64(nrn.Gk+nex.Gak))
	dt.SetCellFloat("ISI", row, float64(nrn.ISI))
	dt.SetCellFloat("AvgISI", row, float64(nrn.ISIAvg))
	dt.SetCellFloat("VmDend", row, float64(nrn.VmDend))
	dt.SetCellFloat("NMDA", row, float64(nrn.NMDA))
	dt.SetCellFloat("Gnmda", row, float64(nrn.Gnmda))
	dt.SetCellFloat("GABAB", row, float64(nrn.GABAB))
	dt.SetCellFloat("GgabaB", row, float64(nrn.GgabaB))
	dt.SetCellFloat("Gvgcc", row, float64(nex.Gvgcc))
	dt.SetCellFloat("VGCCm", row, float64(nex.VGCCm))
	dt.SetCellFloat("VGCCh", row, float64(nex.VGCCh))
	dt.SetCellFloat("Gak", row, float64(nex.Gak))
	dt.SetCellFloat("AKm", row, float64(nex.AKm))
	dt.SetCellFloat("AKh", row, float64(nex.AKh))

	// note: essential to use Go version of update when called from another goroutine
	if cyc%ss.UpdtInterval == 0 {
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of testing per cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.NCycles // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
		{"GeSyn", etensor.FLOAT64, nil, nil},
		{"Ge", etensor.FLOAT64, nil, nil},
		{"Gi", etensor.FLOAT64, nil, nil},
		{"Inet", etensor.FLOAT64, nil, nil},
		{"Vm", etensor.FLOAT64, nil, nil},
		{"Act", etensor.FLOAT64, nil, nil},
		{"Spike", etensor.FLOAT64, nil, nil},
		{"Gk", etensor.FLOAT64, nil, nil},
		{"ISI", etensor.FLOAT64, nil, nil},
		{"AvgISI", etensor.FLOAT64, nil, nil},
		{"VmDend", etensor.FLOAT64, nil, nil},
		{"NMDA", etensor.FLOAT64, nil, nil},
		{"Gnmda", etensor.FLOAT64, nil, nil},
		{"GABAB", etensor.FLOAT64, nil, nil},
		{"GgabaB", etensor.FLOAT64, nil, nil},
		{"Gvgcc", etensor.FLOAT64, nil, nil},
		{"VGCCm", etensor.FLOAT64, nil, nil},
		{"VGCCh", etensor.FLOAT64, nil, nil},
		{"Gak", etensor.FLOAT64, nil, nil},
		{"AKm", etensor.FLOAT64, nil, nil},
		{"AKh", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Neuron Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GeSyn", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Ge", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Gi", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Inet", eplot.Off, eplot.FixMin, -.2, eplot.FixMax, 1)
	plt.SetColParams("Vm", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Act", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Spike", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Gk", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("ISI", eplot.Off, eplot.FixMin, -2, eplot.FloatMax, 1)
	plt.SetColParams("AvgISI", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("VmDend", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("NMDA", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("Gnmda", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("GABAB", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("GgabaB", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("Gvgcc", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("VGCCm", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("VGCCh", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("Gak", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("AKm", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("AKh", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	return plt
}

func (ss *Sim) ResetTstCycPlot() {
	ss.TstCycLog.SetNumRows(0)
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
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

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
