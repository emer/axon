// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// kinaseq plots kinase learning simulation over time
package main

import (
	"math/rand"
	"strconv"
	"strings"

	"github.com/emer/axon/kinase"
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
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {
	Kinase    kinase.SynParams `desc:Kinase rate constants"`
	PGain     float32          `desc:"multiplier on product factor to equate to SynC"`
	SpikeDisp float32          `desc:"spike multiplier for display purposes"`
	NReps     int              `desc:"number of repetitions -- if > 1 then only final @ end of Dur shown"`
	DurMsec   int              `desc:"duration for activity window"`
	SendHz    float32          `desc:"sending firing frequency (used as minus phase for ThetaErr)"`
	RecvHz    float32          `desc:"receiving firing frequency (used as plus phase for ThetaErr)"`
	Table     *etable.Table    `view:"no-inline" desc:"table for plot"`
	Plot      *eplot.Plot2D    `view:"-" desc:"the plot"`
	Win       *gi.Window       `view:"-" desc:"main GUI window"`
	ToolBar   *gi.ToolBar      `view:"-" desc:"the master toolbar"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.Kinase.Defaults()
	ss.PGain = 10
	ss.SpikeDisp = 0.1
	ss.NReps = 100
	ss.DurMsec = 200
	ss.SendHz = 20
	ss.RecvHz = 20
	ss.Update()
	ss.Table = &etable.Table{}
	ss.ConfigTable(ss.Table)
}

// Update updates computed values
func (ss *Sim) Update() {
	ss.Kinase.Update()
}

// Run runs the equation.
func (ss *Sim) Run() {
	ss.Update()
	dt := ss.Table

	if ss.NReps == 1 {
		dt.SetNumRows(ss.DurMsec)
	} else {
		dt.SetNumRows(ss.NReps)
	}

	Sint := mat32.Exp(-1000.0 / float32(ss.SendHz))
	Rint := mat32.Exp(-1000.0 / float32(ss.RecvHz))
	Sp := float32(1)
	Rp := float32(1)

	kp := &ss.Kinase

	var rSpk, rSpkCaM, rSpkCaP, rSpkCaD float32                         // recv
	var sSpk, sSpkCaM, sSpkCaP, sSpkCaD float32                         // send
	var pSpkCaM, pSpkCaP, pSpkCaD, pDWt float32                         // product
	var cSpk, cSpkCaM, cSpkCaP, cSpkCaD, cDWt, cISI float32             // syn continuous
	var oSpk, oSpkCaM, oSpkCaP, oSpkCaD, oCaM, oCaP, oCaD, oDWt float32 // syn optimized compute

	for nr := 0; nr < ss.NReps; nr++ {
		cISI = -1
		for t := 0; t < ss.DurMsec; t++ {
			row := t
			if ss.NReps == 1 {
				dt.SetCellFloat("Time", row, float64(row))
			} else {
				row = nr
				dt.SetCellFloat("Time", row, float64(row))
			}

			Sp *= rand.Float32()
			if Sp <= Sint {
				sSpk = 1
				Sp = 1
			} else {
				sSpk = 0
			}

			Rp *= rand.Float32()
			if Rp <= Rint {
				rSpk = 1
				Rp = 1
			} else {
				rSpk = 0
			}

			kp.FmSpike(rSpk, &rSpkCaM, &rSpkCaP, &rSpkCaD)
			kp.FmSpike(sSpk, &sSpkCaM, &sSpkCaP, &sSpkCaD)

			// this is standard CHL form
			pSpkCaM = ss.PGain * rSpkCaM * sSpkCaM
			pSpkCaP = ss.PGain * rSpkCaP * sSpkCaP
			pSpkCaD = ss.PGain * rSpkCaD * sSpkCaD

			pDWt = kp.DWt(pSpkCaP, pSpkCaD)

			// either side drives up..
			cSpk = 0
			switch kp.Rule {
			case kinase.SynSpkCa:
				if sSpk > 0 || rSpk > 0 {
					cSpk = 1
				}
			}
			kp.FmSpike(cSpk, &cSpkCaM, &cSpkCaP, &cSpkCaD)
			cDWt = kp.DWt(cSpkCaP, cSpkCaD)

			// optimized
			if cSpk > 0 {
				isi := int(cISI)

				// get old before cam update, for previous isi
				if isi >= 0 {
					oSpkCaD = kp.DFmLastSpike(oSpkCaD, oSpkCaP, oSpkCaM, isi) // reverse order
					oSpkCaP = kp.PFmLastSpike(oSpkCaP, oSpkCaM, isi)
				}

				var mprv float32
				if isi >= 0 {
					mprv = oSpkCaM * mat32.FastExp(-cISI/(kp.MTau-0.5))
				}
				minc := kp.MDt * (kp.SpikeG*cSpk - mprv)
				oSpkCaM = mprv + minc

				oCaM = oSpkCaM
				oCaP = kp.PFmLastSpike(oSpkCaP, oSpkCaM, 0)
				oCaD = kp.DFmLastSpike(oSpkCaD, oSpkCaP, oSpkCaM, 0)
				// fmt.Printf("t: %d  isi: %d  ocam: %g  ocap: %g  ocad: %g  val: %g\n", t, isi, oSpkCaM, oSpkCaP, oSpkCaD, oCaD)
				cISI = 0
			} else if cISI >= 0 {
				cISI += 1
				isi := int(cISI)

				oCaM = oSpkCaM * mat32.FastExp(-cISI/(kp.MTau-0.5))
				oCaP = kp.PFmLastSpike(oSpkCaP, oSpkCaM, isi)
				oCaD = kp.DFmLastSpike(oSpkCaD, oSpkCaP, oSpkCaM, isi)
				// fmt.Printf("t: %d  isi: %d  ocam: %g  ocap: %g  ocad: %g  val: %g\n", t, isi, oSpkCaM, oSpkCaP, oSpkCaD, oCaD)
			}

			oDWt = kp.DWt(oCaP, oCaD)

			if ss.NReps == 1 || t == ss.DurMsec-1 {
				dt.SetCellFloat("RSpike", row, float64(ss.SpikeDisp*rSpk))
				dt.SetCellFloat("RSpkCaM", row, float64(rSpkCaM))
				dt.SetCellFloat("RSpkCaP", row, float64(rSpkCaP))
				dt.SetCellFloat("RSpkCaD", row, float64(rSpkCaD))
				dt.SetCellFloat("SSpike", row, float64(ss.SpikeDisp*sSpk))
				dt.SetCellFloat("SSpkCaM", row, float64(sSpkCaM))
				dt.SetCellFloat("SSpkCaP", row, float64(sSpkCaP))
				dt.SetCellFloat("SSpkCaD", row, float64(sSpkCaD))
				dt.SetCellFloat("SynPSpkCaM", row, float64(pSpkCaM))
				dt.SetCellFloat("SynPSpkCaP", row, float64(pSpkCaP))
				dt.SetCellFloat("SynPSpkCaD", row, float64(pSpkCaD))
				dt.SetCellFloat("SynPDWt", row, float64(pDWt))
				dt.SetCellFloat("SynCSpike", row, float64(ss.SpikeDisp*cSpk))
				dt.SetCellFloat("SynCSpkCaM", row, float64(cSpkCaM))
				dt.SetCellFloat("SynCSpkCaP", row, float64(cSpkCaP))
				dt.SetCellFloat("SynCSpkCaD", row, float64(cSpkCaD))
				dt.SetCellFloat("SynCDWt", row, float64(cDWt))
				dt.SetCellFloat("SynOSpike", row, float64(ss.SpikeDisp*oSpk))
				dt.SetCellFloat("SynOSpkCaM", row, float64(oCaM))
				dt.SetCellFloat("SynOSpkCaP", row, float64(oCaP))
				dt.SetCellFloat("SynOSpkCaD", row, float64(oCaD))
				dt.SetCellFloat("SynODWt", row, float64(oDWt))
			}
		}
	}
	ss.Plot.Update()
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "Kinase Opt Table")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"RSpike", etensor.FLOAT64, nil, nil},
		{"RSpkCaM", etensor.FLOAT64, nil, nil},
		{"RSpkCaP", etensor.FLOAT64, nil, nil},
		{"RSpkCaD", etensor.FLOAT64, nil, nil},
		{"SSpike", etensor.FLOAT64, nil, nil},
		{"SSpkCaM", etensor.FLOAT64, nil, nil},
		{"SSpkCaP", etensor.FLOAT64, nil, nil},
		{"SSpkCaD", etensor.FLOAT64, nil, nil},
		{"SynPSpkCaM", etensor.FLOAT64, nil, nil},
		{"SynPSpkCaP", etensor.FLOAT64, nil, nil},
		{"SynPSpkCaD", etensor.FLOAT64, nil, nil},
		{"SynPDWt", etensor.FLOAT64, nil, nil},
		{"SynCSpike", etensor.FLOAT64, nil, nil},
		{"SynCSpkCaM", etensor.FLOAT64, nil, nil},
		{"SynCSpkCaP", etensor.FLOAT64, nil, nil},
		{"SynCSpkCaD", etensor.FLOAT64, nil, nil},
		{"SynCDWt", etensor.FLOAT64, nil, nil},
		{"SynOSpkCaM", etensor.FLOAT64, nil, nil},
		{"SynOSpkCaP", etensor.FLOAT64, nil, nil},
		{"SynOSpkCaD", etensor.FLOAT64, nil, nil},
		{"SynODWt", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Kinase Learning Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// plt.Params.Points = true

	for _, cn := range dt.ColNames {
		if cn == "Time" {
			continue
		}
		if strings.Contains(cn, "DWt") {
			plt.SetColParams(cn, eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
		} else {
			plt.SetColParams(cn, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		}
	}
	// plt.SetColParams("SynCSpkCaM", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	// plt.SetColParams("SynOSpkCaM", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SynCSpkCaP", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SynOSpkCaP", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SynCSpkCaD", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SynOSpkCaD", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	return plt
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
