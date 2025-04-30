// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// kinaseq: Explores calcium-based synaptic learning rules,
// specifically at the synaptic level.
package kinasesim

//go:generate core generate -add-types -add-funcs -gosl

import (
	"reflect"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/cli"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/kinase"
	"github.com/emer/emergent/v2/egui"
)

// Modes are the looping modes (Stacks) for running and statistics.
type Modes int32 //enums:enum
const (
	Test Modes = iota
)

// Levels are the looping levels for running and statistics.
type Levels int32 //enums:enum
const (
	Cycle Levels = iota
	Trial
	Condition
)

// StatsPhase is the phase of stats processing for given mode, level.
// Accumulated values are reset at Start, added each Step.
type StatsPhase int32 //enums:enum
const (
	Start StatsPhase = iota
	Step
)

// see config.go for Config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config *Config `new-window:"+"`

	// Kinase CaSpike params
	CaSpike kinase.CaSpikeParams `display:"no-inline" new-window:"+"`

	// SynCa20 determines whether to use 20 msec SynCa integration.
	SynCa20 bool

	// CaPWts are CaBin integration weights for CaP
	CaPWts []float32 `new-window:"+"`

	// CaDWts are CaBin integration weights for CaD
	CaDWts []float32 `new-window:"+"`

	// Kinase state
	Kinase KinaseState `new-window:"+"`

	// Training data for least squares solver
	TrainData tensor.Float64 `new-window:"+"`

	// Root is the root tensorfs directory, where all stats and other misc sim data goes.
	Root *tensorfs.Node `display:"-"`

	// Stats has the stats directory within Root.
	Stats *tensorfs.Node `display:"-"`

	// Current has the current stats values within Stats.
	Current *tensorfs.Node `display:"-"`

	// StatFuncs are statistics functions called at given mode and level,
	// to perform all stats computations. phase = Start does init at start of given level,
	// and all intialization / configuration (called during Init too).
	StatFuncs []func(mode Modes, level Levels, phase StatsPhase) `display:"-"`

	// GUI manages all the GUI elements
	GUI egui.GUI `display:"-"`

	// RandSeeds is a list of random seeds to use for each run.
	RandSeeds randx.Seeds `display:"-"`
}

// RunSim runs the simulation as a standalone app
// with given configuration.
func RunSim(cfg *Config) error {
	ss := &Sim{Config: cfg}
	ss.ConfigSim()
	if ss.Config.GUI {
		ss.RunGUI()
	} else {
		ss.RunNoGUI()
	}
	return nil
}

// EmbedSim runs the simulation with default configuration
// embedded within given body element.
func EmbedSim(b tree.Node) *Sim {
	cfg := NewConfig()
	cfg.GUI = true
	ss := &Sim{Config: cfg}
	ss.ConfigSim()
	ss.Init()
	ss.ConfigGUI(b)
	return ss
}

func (ss *Sim) Defaults() {
	ss.CaSpike.Defaults()
	ss.SynCa20 = false
	cli.SetFromDefaults(&ss.Config)
}

func (ss *Sim) ConfigSim() {
	ss.Defaults()
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.CaSpike.Defaults()
	ss.ConfigKinase()
	ss.ConfigStats()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		return
	}
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.SetRunName()
	ss.InitRandSeed(0)
	ss.GUI.StopNow = false
	ss.ConfigKinase()
	ss.StatsInit()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.GUI.StopNow = true
}

//////// Stats

// AddStat adds a stat compute function.
func (ss *Sim) AddStat(f func(mode Modes, level Levels, phase StatsPhase)) {
	ss.StatFuncs = append(ss.StatFuncs, f)
}

// StatsStart is called by Looper at the start of given level, for each iteration.
// It needs to call RunStats Start at the next level down.
// e.g., each Epoch is the start of the full set of Trial Steps.
func (ss *Sim) StatsStart(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level < Trial {
		return
	}
	ss.RunStats(mode, level-1, Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	ss.RunStats(mode, level, Step)
	tensorfs.DirTable(axon.StatsNode(ss.Stats, mode, level), nil).WriteToLog()
}

// RunStats runs the StatFuncs for given mode, level and phase.
func (ss *Sim) RunStats(mode Modes, level Levels, phase StatsPhase) {
	for _, sf := range ss.StatFuncs {
		sf(mode, level, phase)
	}
	if phase == Step && ss.GUI.Tabs != nil {
		nm := mode.String() + " " + level.String() + " Plot"
		ss.GUI.Tabs.AsLab().GoUpdatePlot(nm)
	}
}

// SetRunName sets the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) SetRunName() string {
	runName := "Run"
	ss.Current.StringValue("RunName", 1).SetString1D(runName, 0)
	return runName
}

// RunName returns the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) RunName() string {
	return ss.Current.StringValue("RunName", 1).String1D(0)
}

// StatsInit initializes all the stats by calling Start across all modes and levels.
func (ss *Sim) StatsInit() {
	ss.RunStats(Test, Cycle, Start)
	ss.RunStats(Test, Trial, Start)
	ss.RunStats(Test, Condition, Start)
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Cycle))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Condition))
		if idx < 0 {
			idx = 0
		}
		tbs.SelectTabIndex(idx)
	}
}

// ConfigStats handles configures functions to do all stats computation
// in the tensorfs system.
func (ss *Sim) ConfigStats() {
	ss.Stats = ss.Root.Dir("Stats")
	ss.Current = ss.Stats.Dir("Current")

	ss.SetRunName()

	vals := axon.StructValues(&ss.Kinase,
		func(parent reflect.Value, field reflect.StructField, value reflect.Value) bool {
			if field.Name == "CaBins" {
				return false
			}
			return true
		})
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		for _, sv := range vals {
			name := sv.Path
			kind := sv.Field.Type.Kind()
			isNumber := reflectx.KindIsNumber(kind)
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			tsr := tensorfs.ValueType(levelDir, name, kind)
			if phase == Start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0)
					switch level {
					case Cycle:
						switch name {
						case "Send.Spike", "Recv.Spike", "StdSyn.CaP", "StdSyn.CaD":
							s.On = true
						}
					case Trial:
						switch name {
						case "StdSyn.CaP", "StdSyn.CaD", "StdSyn.DWt", "LinearSyn.CaP", "LinearSyn.CaD", "LinearSyn.DWt":
							s.On = true
						case "Cycle":
							s.Group = "none"
						}
					case Condition:
						switch name {
						case "StdSyn.DWt", "LinearSyn.DWt", "ErrDWt":
							s.On = true
						case "Cycle", "Trial":
							s.Group = "none"
						}
					}
				})
				continue
			}
			switch level {
			case Cycle, Trial:
				if isNumber {
					stat := errors.Log1(reflectx.ToFloat(sv.Value.Interface()))
					tsr.AppendRowFloat(stat)
					tensorfs.ValueType(curModeDir, name, kind, 1).SetFloat1D(stat, 0)
				} else {
					stat := reflectx.ToString(sv.Value.Interface())
					tsr.AppendRowString(stat)
					curModeDir.StringValue(name, 1).SetString1D(stat, 0)
				}
			default:
				if isNumber {
					subDir := modeDir.Dir((level - 1).String())
					stat := stats.StatMean.Call(subDir.Value(name)).Float1D(0)
					tsr.AppendRowFloat(stat)
				}
			}
		}
	})
	// collect regression data
	// ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
	// 	if level != Trial {
	// 		return
	// 	}
	// 	regressDir := ss.Stats.Dir("Regress")
	// 	nbins := ss.Config.Run.NCaBins
	// 	vars := []string{"Trial", "Hz", "Bins", "SynCa", "PredCa", "ErrCa", "SSE"}
	// 	for vi, name := range vars {
	// 		ndim := 2
	// 		switch name {
	// 		case "Hz":
	// 			ndim = 4
	// 		case "Bins":
	// 			ndim = nbins
	// 		case "SSE":
	// 			ndim = 1
	// 		}
	// 		tsr := regressDir.Float64(name, 0, ndim)
	// 		if phase == Start {
	// 			tsr.SetNumRows(0)
	// 			continue
	// 		}
	// 	}
	// })
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.CycleUpdateInterval = 10

	ss.StatsInit()
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
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
			ss.Stop()
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
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Plot", Icon: icons.Update,
		Tooltip: "Reset TstCycPlot.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.StatsInit()
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
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL(ss.Config.URL)
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI(nil)
	ss.GUI.Body.RunMainWindow()
}

func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}

	// runName := ss.SetRunName()
	// netName := ss.Net.Name
	// axon.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{[]string{"Cycle"}})

	mpi.Printf("Running %d Cycles\n", ss.Config.Run.Cycles)
	ss.Sweep()
	// axon.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
}
