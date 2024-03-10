// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

// kinaseq plots kinase learning simulation over time
package main

import (
	"fmt"
	"math/rand"

	"cogentcore.org/core/gi"
	"cogentcore.org/core/giv"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/ki"
	"cogentcore.org/core/mat32"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/etable/v2/agg"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
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
	Logs map[string]*etable.Table `view:"no-inline"`

	// all plots
	Plots map[string]*eplot.Plot2D `view:"-"`

	// main GUI window
	Win *gi.Window `view:"-"`

	// the master toolbar
	ToolBar *gi.ToolBar `view:"-"`

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
func (ss *Sim) Log(name string) *etable.Table {
	if ss.Logs == nil {
		ss.Logs = make(map[string]*etable.Table)
	}
	dt, ok := ss.Logs[name]
	if ok {
		return dt
	}
	dt = &etable.Table{}
	ss.Logs[name] = dt
	return dt
}

func (ss *Sim) Plot(name string) *eplot.Plot2D {
	return ss.Plots[name]
}

func (ss *Sim) AddPlot(name string, plt *eplot.Plot2D) {
	if ss.Plots == nil {
		ss.Plots = make(map[string]*eplot.Plot2D)
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
			dt.SetCellFloat("ErrDWt", row, float64(dwt))
			dt.SetCellString("Cond", row, cond)
			dvt.SetCellFloat("ErrDWt", row, float64(dwt))
			dvt.SetCellString("Cond", row, cond)

			rix := etable.NewIdxView(rdt)
			for ci := 2; ci < rdt.NumCols(); ci++ {
				cnm := rdt.ColName(ci)
				mean := agg.Mean(rix, cnm)[0]
				sem := agg.Sem(rix, cnm)[0]
				dt.SetCellFloat(cnm, row, mean)
				dvt.SetCellFloat(cnm, row, sem)
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
			Rint = mat32.Exp(-1000.0 / float32(rhz))
		}
		if shz > 0 {
			Sint = mat32.Exp(-1000.0 / float32(shz))
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

			ss.NeuronUpdt(sSpk, rSpk, ge, gi)

			ss.LogState(dt, cyc, 0, cyc)
			ss.Context.CycleInc()
		}
	}
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *gi.Window {
	width := 1600
	height := 1200

	// gi.WinEventTrace = true

	gi.SetAppName("kinaseq")
	gi.SetAppAbout(`Exploration of kinase equations. See <a href="https://github.com/emer/axon/blob/master/examples/kinaseq"> GitHub</a>.</p>`)

	win := gi.NewMainWindow("kinaseq", "Kinase Equation Exploration", width, height)
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

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.AddPlot("RunPlot", ss.ConfigRunPlot(plt, ss.Log("RunLog")))

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrialPlot").(*eplot.Plot2D)
	ss.AddPlot("TrialPlot", ss.ConfigTrialPlot(plt, ss.Log("TrialLog")))

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "DWtPlot").(*eplot.Plot2D)
	ss.AddPlot("DWtPlot", ss.ConfigDWtPlot(plt, ss.Log("DWtLog")))

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "DWtVarPlot").(*eplot.Plot2D)
	ss.AddPlot("DWtVarPlot", ss.ConfigDWtPlot(plt, ss.Log("DWtVarLog")))

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: icons.Update, Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Trial", Icon: "step-fwd", Tooltip: "Run one trial of the equations and plot results in TrialPlot."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Trial()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Run", Icon: "play", Tooltip: "Run NTrials of the equations and plot results at end of each trial in RunPlot."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Run()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Sweep", Icon: "fast-fwd", Tooltip: "Sweep through minus-plus combinations and plot in DWtLogs."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Sweep()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/kinaseq/README.md")
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
