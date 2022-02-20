// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// kinaseq plots kinase learning simulation over time
package main

import (
	"math/rand"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/kinase"
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
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
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {
	Net        *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	SendNeur   *axon.Neuron     `view:"no-inline" desc:"the sending neuron"`
	RecvNeur   *axon.Neuron     `view:"no-inline" desc:"the receiving neuron"`
	NeuronEx   NeuronEx         `view:"no-inline" desc:"extra neuron state"`
	Kinase     kinase.SynParams `desc:"kinase params"`
	Params     emer.Params      `view:"inline" desc:"all parameter management"`
	PGain      float32          `desc:"multiplier on product factor to equate to SynC"`
	SpikeDisp  float32          `desc:"spike multiplier for display purposes"`
	RGeClamp   bool             `desc:"use current Ge clamping for recv neuron -- otherwise spikes driven externally"`
	RGeBase    float32          `desc:"baseline recv Ge level"`
	RGiBase    float32          `desc:"baseline recv Gi level"`
	NReps      int              `desc:"number of repetitions -- if > 1 then only final @ end of Dur shown"`
	MinusMsec  int              `desc:"number of msec in minus phase"`
	PlusMsec   int              `desc:"number of msec in plus phase"`
	ISIMsec    int              `desc:"quiet space between spiking"`
	TrialMsec  int              `view:"-" desc:"total trial msec: minus, plus isi"`
	MinusHz    int              `desc:"minus phase firing frequency"`
	PlusHz     int              `desc:"plus phase firing frequency"`
	SendDiffHz int              `desc:"additive difference in sending firing frequency relative to recv (recv has basic minus, plus)"`
	SynNeur    axon.Synapse     `view:"no-inline" desc:"NeurSpkCa product synapse state values, P_ in log"`
	SynSpk     axon.Synapse     `view:"no-inline" desc:"SynSpkCa synapse state values, X_ in log"`
	SynNNMDA   axon.Synapse     `view:"no-inline" desc:"SynNMDACa synapse state values, N_ in log"`
	SynOpt     axon.Synapse     `view:"no-inline" desc:"optimized computation synapse state values, O_ in log"`
	Time       axon.Time        `desc:"axon time recording"`
	Table      *etable.Table    `view:"no-inline" desc:"table for plot"`
	Plot       *eplot.Plot2D    `view:"-" desc:"the plot"`
	Win        *gi.Window       `view:"-" desc:"main GUI window"`
	ToolBar    *gi.ToolBar      `view:"-" desc:"the master toolbar"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Kinase.Defaults()
	ss.Time.Defaults()
	ss.PGain = 10
	ss.SpikeDisp = 0.1
	ss.RGeBase = 0.5
	ss.RGiBase = 2
	ss.NReps = 1
	ss.MinusMsec = 150
	ss.PlusMsec = 50
	ss.ISIMsec = 0
	ss.MinusHz = 50
	ss.PlusHz = 25
	ss.Update()
	ss.ConfigNet(ss.Net)
	ss.Table = &etable.Table{}
	ss.ConfigTable(ss.Table)
}

// Update updates computed values
func (ss *Sim) Update() {
	ss.Kinase.Update()
	ss.TrialMsec = ss.MinusMsec + ss.PlusMsec + ss.ISIMsec
}

// Init restarts the run and applies current parameters
func (ss *Sim) Init() {
	ss.Params.SetAll()
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.NeuronEx.Init()
}

// Run runs the equation.
func (ss *Sim) Run() {
	ss.Update()
	dt := ss.Table

	if ss.NReps == 1 {
		dt.SetNumRows(ss.TrialMsec)
	} else {
		dt.SetNumRows(ss.NReps)
	}

	Sp := float32(1)
	Rp := float32(1)

	ss.Time.Reset()

	ge := ss.RGeBase
	gi := ss.RGiBase

	for nr := 0; nr < ss.NReps; nr++ {
		ss.Time.NewState()
		for phs := 0; phs < 3; phs++ {
			var maxms, rhz int
			switch phs {
			case 0:
				rhz = ss.MinusHz
				maxms = ss.MinusMsec
			case 1:
				rhz = ss.PlusHz
				maxms = ss.PlusMsec
			case 2:
				rhz = 0
				maxms = ss.ISIMsec
			}
			shz := rhz + ss.SendDiffHz
			if shz < 0 {
				shz = 0
			}

			Sint := mat32.Exp(-1000.0 / float32(shz))
			Rint := mat32.Exp(-1000.0 / float32(rhz))
			for t := 0; t < maxms; t++ {
				row := ss.Time.Cycle
				if ss.NReps == 1 {
					dt.SetCellFloat("Cycle", row, float64(row))
				} else {
					row = nr
					dt.SetCellFloat("Cycle", row, float64(row))
				}

				Sp *= rand.Float32()
				sSpk := false
				if Sp <= Sint {
					sSpk = true
					Sp = 1
				}

				Rp *= rand.Float32()
				rSpk := false
				if Rp <= Rint {
					rSpk = true
					Rp = 1
				}

				ss.NeuronUpdt(sSpk, rSpk, ge, gi)

				if ss.NReps == 1 {
					ss.Log(ss.Table, row, row)
				}
				ss.Time.CycleInc()
			}
			if ss.NReps > 1 {
				ss.Log(ss.Table, nr, nr)
			}
		}
	}
	ss.Plot.Update()
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
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

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "Plot").(*eplot.Plot2D)
	ss.Plot = ss.ConfigPlot(plt, ss.Table)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Run()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/kinaseq/README.md")
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
