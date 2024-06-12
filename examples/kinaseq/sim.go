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
	"reflect"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/kinase"
	"github.com/emer/emergent/v2/ecmd"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/netview"
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

	// Kinase SynCa params
	CaParams kinase.CaParams

	// Kinase state
	Kinase KinaseState

	// Training data for least squares solver
	TrainData tensor.Float64

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *axon.Network `view:"no-inline"`

	// extra neuron state for additional channels: VGCC, AK
	NeuronEx NeuronEx `view:"no-inline"`

	// axon timing parameters and state
	Context axon.Context

	// contains computed statistic values
	Stats estats.Stats

	// logging
	Logs elog.Logs `view:"no-inline"`

	// all parameter management
	Params emer.NetParams `view:"inline"`

	// current cycle of updating
	Cycle int `edit:"-"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `view:"inline"`

	// manages all the gui elements
	GUI egui.GUI `view:"-"`

	// map of values for detailed debugging / testing
	ValMap map[string]float32 `view:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	econfig.Config(&ss.Config, "config.toml")
	ss.Config.Params.Update()
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.CaParams.Defaults()
	ss.Stats.Init()
	ss.ValMap = make(map[string]float32)
}

func (ss *Sim) Defaults() {
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigNet(ss.Net)
	ss.ConfigKinase()
	ss.ConfigLogs()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Context.Reset()
	ss.InitWts(ss.Net)
	ss.Kinase.Init()
	ss.ConfigKinase()
	ss.NeuronEx.Init()
	ss.GUI.StopNow = false
	ss.SetParams("", false) // all sheets
}

func (ss *Sim) ConfigLogs() {
	if ss.Config.Run.Neuron {
		ss.ConfigNeuronLogItems()
	} else {
		ss.ConfigKinaseLogItems()
	}
	ss.Logs.CreateTables()

	if ss.Config.Run.Neuron {
		ss.Logs.PlotItems("Vm", "Spike")
	} else {
		ss.Logs.PlotItems("Send.Spike", "Recv.Spike")
	}

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
}

func (ss *Sim) ConfigNeuronLogItems() {
	ly := ss.Net.AxonLayerByName("Neuron")
	// nex := &ss.NeuronEx
	lg := &ss.Logs

	lg.AddItem(&elog.Item{
		Name:   "Cycle",
		Type:   reflect.Int,
		FixMax: false,
		Range:  minmax.F32{Max: 1},
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
				ctx.SetInt(int(ss.Context.Cycle))
			}}})

	vars := []string{"GeSyn", "Ge", "Gi", "Inet", "Vm", "Act", "Spike", "Gk", "ISI", "ISIAvg", "VmDend", "GnmdaSyn", "Gnmda", "GABAB", "GgabaB", "Gvgcc", "VgccM", "VgccH", "Gak", "MahpN", "GknaMed", "GknaSlow", "GiSyn", "CaSyn"}

	for _, vnm := range vars {
		lg.AddItem(&elog.Item{
			Name:   vnm,
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					vl := ly.UnitValue(vnm, []int{0, 0}, 0)
					ctx.SetFloat32(vl)
				}}})
	}

	pj := ly.RcvPaths[0]
	pvars := []string{"CaM", "CaP", "CaD", "CaUpT"}
	for _, vnm := range pvars {
		lg.AddItem(&elog.Item{
			Name:   "Syn." + vnm,
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					vl := pj.SynValue(vnm, 0, 0)
					ctx.SetFloat32(vl)
				}}})
	}
}

func (ss *Sim) ResetTstCycPlot() {
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
	ss.GUI.UpdatePlot(etime.Test, etime.Cycle)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Kinase Eq"
	ss.GUI.MakeBody(ss, "kinaseq", title, `kinaseq: Explores calcium-based synaptic learning rules, specifically at the synaptic level. See <a href="https://github.com/emer/axon/blob/master/examples/kinaseq/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.ConfigNetView(nv) // add labels etc
	ss.ViewUpdate.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)
	// key := etime.Scope(etime.Test, etime.Cycle)
	// plt := ss.GUI.NewPlot(key, ss.GUI.Tabs.NewTab("TstCycPlot"))
	// plt.SetTable(ss.Logs.Table(etime.Test, etime.Cycle))
	// egui.ConfigPlotFromLog("Neuron", plt, &ss.Logs, key)
	// ss.TstCycPlot = plt

	ss.GUI.Body.AddAppBar(func(tb *core.Toolbar) {
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Stop", Icon: icons.Stop,
			Tooltip: "Stops running.",
			Active:  egui.ActiveRunning,
			Func: func() {
				ss.Stop()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Sweep", Icon: icons.PlayArrow,
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
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Run", Icon: icons.PlayArrow,
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
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Trial", Icon: icons.PlayArrow,
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
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Train", Icon: icons.PlayArrow,
			Tooltip: "Train the Kinase approximation models.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					go func() {
						ss.GUI.IsRunning = true
						ss.Train()
						ss.GUI.IsRunning = false
						ss.GUI.UpdateWindow()
					}()
				}
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Run Neuron", Icon: icons.PlayArrow,
			Tooltip: "Runs neuron updating over NCycles.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					go func() {
						ss.GUI.IsRunning = true
						ss.RunCycles()
						ss.GUI.IsRunning = false
						ss.GUI.UpdateWindow()
					}()
				}
			},
		})
		core.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Reset Plot", Icon: icons.Update,
			Tooltip: "Reset TstCycPlot.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.ResetTstCycPlot()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Defaults", Icon: icons.Update,
			Tooltip: "Restore initial default parameters.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Defaults()
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "README",
			Icon:    "file-markdown",
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/neuron/README.md")
			},
		})
	})
	ss.GUI.FinalizeGUI(false)

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUwithGUI(&ss.Context)
		core.TheApp.AddQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
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
		fnm := ecmd.LogFilename("cyc", netName, runName)
		dt.SaveCSV(core.Filename(fnm), table.Tab, table.Headers)
	}

	// if netdata {
	// 	ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	// }

	ss.Net.GPU.Destroy() // safe even if no GPU
}
