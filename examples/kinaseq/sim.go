// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
kinaseq: Explores calcium-based synaptic learning rules,
specifically at the synaptic level.
*/
package main

//go:generate core generate -add-types

import (
	"os"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tensor"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/kinase"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		sim.RunGUI()
	} else {
		sim.RunNoGUI()
	}
}

// see config.go for Config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config

	// Kinase NeurCa params
	NeurCa kinase.NeurCaParams

	// Kinase SynCa params
	SynCa kinase.SynCaParams

	// Kinase LinearSynCa params
	LinearSynCa kinase.SynCaLinear

	// Kinase state
	Kinase KinaseState

	// Training data for least squares solver
	TrainData tensor.Float64

	// axon timing parameters and state
	Context axon.Context

	// contains computed statistic values
	Stats estats.Stats

	// logging
	Logs elog.Logs `display:"no-inline"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// map of values for detailed debugging / testing
	ValMap map[string]float32 `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	econfig.Config(&ss.Config, "config.toml")
	ss.SynCa.Defaults()
	ss.NeurCa.Defaults()
	ss.LinearSynCa.Defaults()
	ss.Stats.Init()
	ss.ValMap = make(map[string]float32)
}

func (ss *Sim) Defaults() {
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigKinase()
	ss.ConfigLogs()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		os.Exit(0)
	}
}

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Context.Reset()
	ss.Kinase.Init()
	ss.ConfigKinase()
	ss.GUI.StopNow = false
}

func (ss *Sim) ConfigLogs() {
	ss.ConfigKinaseLogItems()
	ss.Logs.CreateTables()

	ss.Logs.PlotItems("Send.Spike", "Recv.Spike")

	ss.Logs.SetContext(&ss.Stats, nil)
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
}

func (ss *Sim) ResetTstCycPlot() {
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
	ss.GUI.UpdatePlot(etime.Test, etime.Cycle)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Kinase Eq"
	ss.GUI.MakeBody(ss, "kinaseq", title, `kinaseq: Explores calcium-based synaptic learning rules, specifically at the synaptic level. See <a href="https://github.com/emer/axon/blob/master/examples/kinaseq/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	ss.GUI.AddPlots(title, &ss.Logs)
	// key := etime.Scope(etime.Test, etime.Cycle)
	// plt := ss.GUI.NewPlot(key, ss.GUI.Tabs.NewTab("TstCycPlot"))
	// plt.SetTable(ss.Logs.Table(etime.Test, etime.Cycle))
	// egui.ConfigPlotFromLog("Neuron", plt, &ss.Logs, key)
	// ss.TstCycPlot = plt

	ss.GUI.Body.AddAppBar(func(p *core.Plan) {
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Stop", Icon: icons.Stop,
			Tooltip: "Stops running.",
			Active:  egui.ActiveRunning,
			Func: func() {
				// ss.Stop()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Sweep", Icon: icons.PlayArrow,
			Tooltip: "Runs Kinase sweep over set of minus / plus spiking levels.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					go func() {
						ss.GUI.IsRunning = true
						ss.Sweep()
						ss.GUI.IsRunning = false
						ss.GUI.UpdateWindow()
					}()
				}
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Run", Icon: icons.PlayArrow,
			Tooltip: "Runs NTrials of Kinase updating.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					go func() {
						ss.GUI.IsRunning = true
						ss.Run()
						ss.GUI.IsRunning = false
						ss.GUI.UpdateWindow()
					}()
				}
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Trial", Icon: icons.PlayArrow,
			Tooltip: "Runs one Trial of Kinase updating.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					go func() {
						ss.GUI.IsRunning = true
						ss.Trial()
						ss.GUI.IsRunning = false
						ss.GUI.UpdateWindow()
					}()
				}
			},
		})
		core.Add(p, func(w *core.Separator) {})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Plot", Icon: icons.Update,
			Tooltip: "Reset TstCycPlot.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.ResetTstCycPlot()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Defaults", Icon: icons.Update,
			Tooltip: "Restore initial default parameters.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Defaults()
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
			Icon:    "file-markdown",
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/neuron/README.md")
			},
		})
	})
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}

func (ss *Sim) RunNoGUI() {
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWts {
		mpi.Printf("Saving final weights per run\n")
	}

	ss.Init()
	// if ss.Config.Log.Cycle {
	// 	dt := ss.Logs.Table(etime.Test, etime.Cycle)
	// 	// fnm := ecmd.LogFilename("cyc", netName, runName)
	// 	dt.SaveCSV(core.Filename(fnm), table.Tab, table.Headers)
	// }
}
