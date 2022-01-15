// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
geneplot plots genesis data from a directory
*/
package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/dirs"
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
const LogPrec = 6

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Log     *etable.Table `view:"no-inline" desc:"genesis data"`
	Plot    *eplot.Plot2D `view:"-" desc:"the plot"`
	Win     *gi.Window    `view:"-" desc:"main GUI window"`
	ToolBar *gi.ToolBar   `view:"-" desc:"the master toolbar"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Log = &etable.Table{}
}

//////////////////////////////////////////////
//  Log

// PlotDir plots directory
func (ss *Sim) PlotDir(dir gi.FileName) error {
	dt := ss.Log
	plt := ss.Plot

	pth := string(dir)
	st, err := os.Stat(pth)
	if err != nil {
		fmt.Println(err)
		return err
	}
	if !st.IsDir() {
		pth, _ = filepath.Split(pth)
		_, err := os.Stat(pth)
		if err != nil {
			fmt.Println(err)
			return err
		}
	}

	fls := dirs.ExtFileNames(pth, nil)
	nf := len(fls)
	if nf == 0 {
		err := fmt.Errorf("PlotDir -- no data in: %s", pth)
		log.Println(err)
		return err
	}
	sort.Strings(fls)

	dt.SetMetaData("name", "Data from: "+pth)
	dt.SetMetaData("desc", "Genesis data")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
	}
	for _, fl := range fls {
		sch = append(sch, etable.Column{fl, etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)

	time, err := ss.ReadFloats(filepath.Join(pth, fls[0]), 0)
	if err != nil {
		return err
	}

	dt.SetNumRows(len(time))
	copy(dt.Cols[0].(*etensor.Float64).Values, time)
	for i, fl := range fls {
		fp := filepath.Join(pth, fl)
		d, err := ss.ReadFloats(fp, 1)
		if err != nil {
			continue
		}
		copy(dt.Cols[i+1].(*etensor.Float64).Values, d)
	}
	plt.Update()
	for _, fl := range fls {
		if !(strings.HasPrefix(fl, "Vm") || strings.HasPrefix(fl, "membrane")) {
			plt.SetColParams(fl, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		}
	}
	plt.Update()
	return nil
}

// ReadFloats reads one file of data, from given column number
func (ss *Sim) ReadFloats(fn string, col int) ([]float64, error) {
	f, err := os.Open(fn)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}
	scan := bufio.NewScanner(f) // line at a time
	var vls []float64
	for scan.Scan() {
		ln := string(scan.Bytes())
		var v1, v2 float64
		_, err = fmt.Sscanf(ln, "%g %g", &v1, &v2)
		if err != nil {
			fmt.Println(err)
		}
		if col == 0 {
			vls = append(vls, v1)
		} else {
			vls = append(vls, v2)
		}
	}
	return vls, nil
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Geneplot Data Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)

	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("geneplot")
	gi.SetAppAbout(`This plots data in a genesis data directory.
See <a href="https://github.com/emer/axon/blob/master/examples/geneplot/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("geneplot", "Geneplot", width, height)
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

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "Plot").(*eplot.Plot2D)
	ss.Plot = ss.ConfigPlot(plt, ss.Log)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Plot", Icon: "step-fwd", Tooltip: "Plot data."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		giv.CallMethod(ss, "PlotDir", vp)
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
		{"PlotDir", ki.Props{
			"desc": "plot data in this dir",
			"icon": "file-open",
			"Args": ki.PropSlice{
				{"Directory Name", ki.Props{}},
			},
		}},
	},
}

func mainrun() {
	TheSim.New()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
