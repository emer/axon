// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

// kinaseq plots kinase learning simulation over time
package main

/*
import (
	"fmt"
	"math/rand"

	"cogentcore.org/core/icons"
	"cogentcore.org/core/ki"
	"cogentcore.org/core/math32"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/emer"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/plot/plotview"
	"cogentcore.org/core/tensor/table"
)

func main() {
	TheSim.Config()
	win := TheSim.ConfigGUI()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// the network -- click to view / edit parameters for layers, prjns, etc
	Net *axon.Network `view:"no-inline"`

	// the sending neuron
	SendNeur *axon.Neuron `view:"no-inline"`

	// the receiving neuron
	RecvNeur *axon.Neuron `view:"no-inline"`

	// prjn-level parameters -- for intializing synapse -- other params not used
	Prjn *axon.Prjn `view:"no-inline"`

	// extra neuron state
	NeuronEx NeuronEx `view:"no-inline"`

	// all parameter management
	Params emer.Params `view:"inline"`

	// multiplier on product factor to equate to SynC
	PGain float32

	// spike multiplier for display purposes
	SpikeDisp float32

	// use current Ge clamping for recv neuron -- otherwise spikes driven externally
	RGeClamp bool

	// gain multiplier for RGe clamp
	RGeGain float32

	// baseline recv Ge level
	RGeBase float32

	// baseline recv Gi level
	RGiBase float32

	// number of repetitions -- if > 1 then only final @ end of Dur shown
	NTrials int

	// number of msec in minus phase
	MinusMsec int

	// number of msec in plus phase
	PlusMsec int

	// quiet space between spiking
	ISIMsec int

	// total trial msec: minus, plus isi
	TrialMsec int `view:"-"`

	// minus phase firing frequency
	MinusHz int

	// plus phase firing frequency
	PlusHz int

	// additive difference in sending firing frequency relative to recv (recv has basic minus, plus)
	SendDiffHz int

	// synapse state values, NST_ in log
	SynNeurTheta axon.Synapse `view:"no-inline"`

	// synapse state values, SST_ in log
	SynSpkTheta axon.Synapse `view:"no-inline"`

	// synapse state values, SSC_ in log
	SynSpkCont axon.Synapse `view:"no-inline"`

	// synapse state values, SNC_ in log
	SynNMDACont axon.Synapse `view:"no-inline"`

	// axon time recording
	Context axon.Context

	// all logs
	Logs map[string]*table.Table `view:"no-inline"`

	// all plots
	Plots map[string]*plotview.PlotView `view:"-"`

	// main GUI window
	Win *core.Window `view:"-"`

	// the master toolbar
	ToolBar *core.ToolBar `view:"-"`

	// stop button
	StopNow bool `view:"-"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Context.Defaults()
	ss.PGain = 1
	ss.SpikeDisp = 0.1
	ss.RGeClamp = true
	ss.RGeGain = 0.2
	ss.RGeBase = 0
	ss.RGiBase = 0
	ss.NTrials = 1000
	ss.MinusMsec = 150
	ss.PlusMsec = 50
	ss.ISIMsec = 200
	ss.MinusHz = 50
	ss.PlusHz = 25
	ss.Update()
	ss.ConfigNet(ss.Net)
	ss.ConfigTable(ss.Log("TrialLog"))
	ss.ConfigTable(ss.Log("RunLog"))
	ss.ConfigTable(ss.Log("DWtLog"))
	ss.ConfigTable(ss.Log("DWtVarLog"))
	ss.Init()
}

// Update updates computed values
func (ss *Sim) Update() {
	ss.TrialMsec = ss.MinusMsec + ss.PlusMsec + ss.ISIMsec
}

// Init restarts the run and applies current parameters
func (ss *Sim) Init() {
	ss.Params.SetAll()
	ss.Context.Reset()
	ss.Net.InitWts()
	ss.NeuronEx.Init()
	ss.InitSyn(&ss.SynNeurTheta)
	ss.InitSyn(&ss.SynSpkTheta)
	ss.InitSyn(&ss.SynSpkCont)
	ss.InitSyn(&ss.SynNMDACont)
}

// Log returns / makes log table of given name
func (ss *Sim) Log(name string) *table.Table {
	if ss.Logs == nil {
		ss.Logs = make(map[string]*table.Table)
	}
	dt, ok := ss.Logs[name]
	if ok {
		return dt
	}
	dt = &table.Table{}
	ss.Logs[name] = dt
	return dt
}

func (ss *Sim) Plot(name string) *plotview.PlotView {
	return ss.Plots[name]
}

func (ss *Sim) AddPlot(name string, plt *plotview.PlotView) {
	if ss.Plots == nil {
		ss.Plots = make(map[string]*plotview.PlotView)
	}
	ss.Plots[name] = plt
}

// Sweep runs a sweep through minus-plus ranges
func (ss *Sim) Sweep() {
	ss.Update()
	dt := ss.Log("DWtLog")
	dvt := ss.Log("DWtVarLog")
	rdt := ss.Log("RunLog")

	hz := []int{25, 50, 100}
	nhz := len(hz)

	dt.SetNumRows(nhz * nhz)
	dvt.SetNumRows(nhz * nhz)

	row := 0
	for mi := 0; mi < nhz; mi++ {
		minusHz := hz[mi]
		for pi := 0; pi < nhz; pi++ {
			plusHz := hz[pi]

			ss.RunImpl(minusHz, plusHz, ss.NTrials)

			cond := fmt.Sprintf("%03d -> %03d", minusHz, plusHz)
			dwt := float64(plusHz-minusHz) / 100
			dt.SetFloat("ErrDWt", row, float64(dwt))
			dt.SetString("Cond", row, cond)
			dvt.SetFloat("ErrDWt", row, float64(dwt))
			dvt.SetString("Cond", row, cond)

			rix := table.NewIndexView(rdt)
			for ci := 2; ci < rdt.NumColumns(); ci++ {
				cnm := rdt.ColumnName(ci)
				mean := agg.Mean(rix, cnm)[0]
				sem := agg.Sem(rix, cnm)[0]
				dt.SetFloat(cnm, row, mean)
				dvt.SetFloat(cnm, row, sem)
			}
			row++
		}
	}
	ss.Plot("DWtPlot").Update()
	ss.Plot("DWtVarPlot").Update()
}

// Run runs for given parameters
func (ss *Sim) Run() {
	ss.Update()
	ss.RunImpl(ss.MinusHz, ss.PlusHz, ss.NTrials)
}

// RunImpl runs NTrials, recording to RunLog and TrialLog
func (ss *Sim) RunImpl(minusHz, plusHz, ntrials int) {
	dt := ss.Log("RunLog")
	dt.SetNumRows(ntrials)
	ss.Context.Reset()
	for nr := 0; nr < ntrials; nr++ {
		ss.TrialImpl(minusHz, plusHz)
		ss.LogState(dt, nr, nr, 0)
	}
	ss.Plot("RunPlot").Update()
}

func (ss *Sim) Trial() {
	ss.Update()
	ss.TrialImpl(ss.MinusHz, ss.PlusHz)
	ss.Plot("TrialPlot").Update()
}

// TrialImpl runs one trial for given parameters
func (ss *Sim) TrialImpl(minusHz, plusHz int) {
	dt := ss.Log("TrialLog")
	dt.SetNumRows(ss.TrialMsec)

	nex := &ss.NeuronEx
	gi := ss.RGiBase

	ss.InitWts()

	ss.Context.NewState(true)
	for phs := 0; phs < 3; phs++ {
		var maxms, rhz int
		switch phs {
		case 0:
			rhz = minusHz
			maxms = ss.MinusMsec
		case 1:
			rhz = plusHz
			maxms = ss.PlusMsec
		case 2:
			rhz = 0
			maxms = ss.ISIMsec
		}
		shz := rhz + ss.SendDiffHz
		if shz < 0 {
			shz = 0
		}

		ge := ss.RGeBase + ss.RGeGain*RGeStimForHz(float32(rhz))

		var Sint, Rint float32
		if rhz > 0 {
			Rint = math32.Exp(-1000.0 / float32(rhz))
		}
		if shz > 0 {
			Sint = math32.Exp(-1000.0 / float32(shz))
		}
		for t := 0; t < maxms; t++ {
			cyc := ss.Context.Cycle

			sSpk := false
			if Sint > 0 {
				nex.Sp *= rand.Float32()
				if nex.Sp <= Sint {
					sSpk = true
					nex.Sp = 1
				}
			}

			rSpk := false
			if Rint > 0 {
				nex.Rp *= rand.Float32()
				if nex.Rp <= Rint {
					rSpk = true
					nex.Rp = 1
				}
			}

			ss.NeuronUpdate(sSpk, rSpk, ge, gi)

			ss.LogState(dt, cyc, 0, cyc)
			ss.Context.CycleInc()
		}
	}
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Window {
	width := 1600
	height := 1200

	// core.WinEventTrace = true

	core.SetAppName("kinaseq")
	core.SetAppAbout(`Exploration of kinase equations. See <a href="https://github.com/emer/axon/blob/master/examples/kinaseq"> GitHub</a>.</p>`)

	win := core.NewMainWindow("kinaseq", "Kinase Equation Exploration", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := core.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := core.AddNewSplitView(mfr, "split")
	split.Dim = math32.X
	split.SetStretchMax()

	sv := views.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := core.AddNewTabView(split, "tv")

	plt := tv.AddNewTab(plotview.KiT_Plot2D, "RunPlot").(*plotview.PlotView)
	ss.AddPlot("RunPlot", ss.ConfigRunPlot(plt, ss.Log("RunLog")))

	plt = tv.AddNewTab(plotview.KiT_Plot2D, "TrialPlot").(*plotview.PlotView)
	ss.AddPlot("TrialPlot", ss.ConfigTrialPlot(plt, ss.Log("TrialLog")))

	plt = tv.AddNewTab(plotview.KiT_Plot2D, "DWtPlot").(*plotview.PlotView)
	ss.AddPlot("DWtPlot", ss.ConfigDWtPlot(plt, ss.Log("DWtLog")))

	plt = tv.AddNewTab(plotview.KiT_Plot2D, "DWtVarPlot").(*plotview.PlotView)
	ss.AddPlot("DWtVarPlot", ss.ConfigDWtPlot(plt, ss.Log("DWtVarLog")))

	split.SetSplits(.2, .8)

	tbar.AddAction(core.ActOpts{Label: "Init", Icon: icons.Update, Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "Trial", Icon: "step-fwd", Tooltip: "Run one trial of the equations and plot results in TrialPlot."}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Trial()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "Run", Icon: "play", Tooltip: "Run NTrials of the equations and plot results at end of each trial in RunPlot."}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Run()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "Sweep", Icon: "fast-fwd", Tooltip: "Sweep through minus-plus combinations and plot in DWtLogs."}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Sweep()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			core.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/kinaseq/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := core.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*core.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*core.Action)
	emen.Menu.AddCopyCutPaste(win)

	win.MainMenuUpdated()
	return win
}
*/
