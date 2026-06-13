// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// consat: This simulation tests axon on constraint satisfaction
// using the travelling salesman problem.
package consat

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"os"
	"reflect"

	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/fsfffb"
	"github.com/emer/axon/v2/sims/consat/consatenv"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/paths"
)

// Modes are the looping modes (Stacks) for running and statistics.
type Modes int32 //enums:enum
const (
	Test  Modes = iota
	Train       // not used, but needed for some things
)

// Levels are the looping levels for running and statistics.
type Levels int32 //enums:enum
const (
	Cycle Levels = iota
	Trial
	Epoch
)

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// simulation configuration parameters -- set by .toml config file and / or args
	Config *Config `new-window:"+"`

	// Net is the network: click to view / edit parameters for layers, paths, etc.
	Net *axon.Network `new-window:"+" display:"no-inline"`

	// Params manages network parameter setting.
	Params axon.Params `display:"inline"`

	// Loops are the control loops for running the sim, in different Modes
	// across stacks of Levels.
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// Envs provides mode-string based storage of environments.
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// NetUpdate has netview update parameters.
	NetUpdate axon.NetViewUpdate `display:"inline"`

	// Root is the root tensorfs directory, where all stats and other misc sim data goes.
	Root *tensorfs.Node `display:"-"`

	// Stats has the stats directory within Root.
	Stats *tensorfs.Node `display:"-"`

	// Current has the current stats values within Stats.
	Current *tensorfs.Node `display:"-"`

	// StatFuncs are statistics functions called at given mode and level,
	// to perform all stats computations. phase = Start does init at start of given level,
	// and all intialization / configuration (called during Init too).
	StatFuncs []func(mode enums.Enum, level enums.Enum, start bool) `display:"-"`

	// GUI manages all the GUI elements
	GUI egui.GUI `display:"-"`

	// RandSeeds is a list of random seeds to use for each run.
	RandSeeds randx.Seeds `display:"-"`
}

func Embed(b tree.Node)               { egui.Embed[Sim, Config](b) }
func (ss *Sim) SetConfig(cfg *Config) { ss.Config = cfg }
func (ss *Sim) Body() *core.Body      { return ss.GUI.Body }

// func (ss *Sim) ShouldDisplay(field string) bool {
// 	switch field {
// 	case "Gi", "FB", "FSTau", "SS", "SSfTau", "SSiTau":
// 		return ss.FSFFFB
// 	case "InhibExcite", "InhibInhib":
// 		return !ss.FSFFFB
// 	default:
// 		return true
// 	}
// }

func (ss *Sim) Defaults() {
}

func (ss *Sim) ConfigSim() {
	ss.Defaults()
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.Net = axon.NewNetwork(ss.Config.Name)
	ss.Params.Config(LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag, reflect.ValueOf(ss))
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	if ss.Config.GPU {
		gpu.SelectAdapter = ss.Config.Run.GPUDevice
		axon.GPUInit()
		axon.UseGPU = true
	}
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLoops()
	ss.ConfigStats()
	// if ss.Config..GPU {
	// 	fmt.Println(axon.GPUSystem.Vars().StringDoc())
	// }
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var tst *consatenv.ConSatEnv
	if len(ss.Envs) == 0 {
		tst = &consatenv.ConSatEnv{}
	} else {
		tst = ss.Envs.ByMode(Test).(*consatenv.ConSatEnv)
	}

	// inputs := tensorfs.DirTable(ss.Root.Dir("Inputs/Test"), nil)

	tst.Name = Test.String()
	tst.Defaults()
	tst.Config(173)

	tst.Init(0)
	tst.Step() // have to run once!

	// note: names must be in place when adding
	ss.Envs.Add(tst)
}

func (ss *Sim) ReConfigNet() {
	ss.Net.DeleteAll()
	ss.ConfigNet(ss.Net)
	// ss.GUI.NetView.Config()
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(1)
	net.Context().ThetaCycles = int32(ss.Config.Run.Cycles)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByMode(Test).(*consatenv.ConSatEnv)
	n := ev.NStates
	pn := ev.NUnitsPer
	inlay := net.AddLayer4D("Input", axon.InputLayer, n, n, pn, pn)
	_ = inlay
	hlay := net.AddLayer4D("Cities", axon.SuperLayer, n, n, pn, pn)

	full := paths.NewFull()
	full.SelfCon = true
	one2one := paths.NewOneToOne()
	net.ConnectLayers(inlay, hlay, one2one, axon.ForwardPath)
	net.ConnectLayers(hlay, hlay, full, axon.LateralPath)
	net.ConnectLayers(hlay, hlay, full, axon.InhibPath)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.Script = ss.Config.Params.Script
	ss.Params.ApplyAll(ss.Net)
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.SetRunName()
	ss.InitRandSeed(0)
	ss.ApplyParams()
	ss.StatsInit()
	ss.NewRun()
	ss.NetUpdate.RecordSyns()
	ss.NetUpdate.Update(Test, Cycle)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run, ss.Net.Rand)
}

// NetViewUpdater returns the NetViewUpdate for given mode.
func (ss *Sim) NetViewUpdater(mode enums.Enum) *axon.NetViewUpdate {
	return &ss.NetUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	cycles := ss.Config.Run.Cycles

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, ss.Config.Run.Trials, 1).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, Cycle, Trial, Train,
		func(mode enums.Enum) { ss.Net.ClearInputs() },
		func(mode enums.Enum) { ss.ApplyInputs(mode.(Modes)) },
	)
	ls.Stacks[Test].OnInit.Add("Init", func() { ss.Init() })

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)
		ls.Stacks[Test].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment for given mode.
// Any other start-of-trial logic can also be put here.
func (ss *Sim) ApplyInputs(mode Modes) {
	// net := ss.Net
	curModeDir := ss.Current.Dir(mode.String())
	ev := ss.Envs.ByMode(mode)
	// lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	// net.InitExt()
	ev.Step()
	curModeDir.StringValue("TrialName", 1).SetString1D(ev.String(), 0)
	// for _, lnm := range lays {
	// 	ly := ss.Net.LayerByName(lnm)
	// 	st := ev.State("Input")
	// 	if st != nil {
	// 		ly.ApplyExt(uint32(0), st)
	// 	}
	// }
	// net.ApplyExts()
	ss.SetWeights()
}

func (ss *Sim) SetWeights() {
	net := ss.Net
	ctx := net.Context()
	ev := ss.Envs.ByMode(Test).(*consatenv.ConSatEnv)
	ly := net.LayerByName("Cities")
	n := ev.NStates
	np := ev.NUnitsPer
	gpn := np * np
	selfWt := ss.Config.Params.SelfWt
	inhibWt := ss.Config.Params.InhibWt
	pt := ly.RecvPaths[1] // excitatory
	pt.SetWeightsFunc(ctx, func(si, ri int, send, recv *tensor.Shape) float32 {
		sCity := (si / gpn) / n
		sPos := (si / gpn) % n
		rCity := (ri / gpn) / n
		rPos := (ri / gpn) % n
		dwt := ev.DistWeight(rCity, sCity)
		if rPos == sPos {
			if sCity == rCity {
				return selfWt
			}
			return dwt
		}
		return 0.0
	})
	pt = ly.RecvPaths[2] // inhibitory
	pt.SetWeightsFunc(ctx, func(si, ri int, send, recv *tensor.Shape) float32 {
		sCity := (si / gpn) / n
		sPos := (si / gpn) % n
		rCity := (ri / gpn) / n
		rPos := (ri / gpn) % n
		if rPos == sPos && sCity == rCity {
			return 0
		}
		if rPos == sPos || sCity == rCity {
			return inhibWt
		}
		return 0.0
	})
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	ss.InitRandSeed(0)
	ss.Envs.ByMode(Test).Init(0)
	ctx.Reset()
	ss.Net.InitWeights()
}

//////// Stats

// AddStatStd adds a standard stat compute function (defined in axon)
func (ss *Sim) AddStatStd(f func(mode enums.Enum, level enums.Enum, start bool)) {
	ss.StatFuncs = append(ss.StatFuncs, f)
}

// AddStat adds a custom stat compute function.
func (ss *Sim) AddStat(f func(mode Modes, level Levels, start bool)) {
	ss.AddStatStd(func(mode enums.Enum, level enums.Enum, start bool) {
		f(mode.(Modes), level.(Levels), start)
	})
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
	ss.RunStats(mode, level-1, axon.Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	ss.RunStats(mode, level, axon.Step)
	tensorfs.DirTable(axon.StatsNode(ss.Stats, mode, level), nil).WriteToLog()
}

// RunStats runs the StatFuncs for given mode, level and phase.
func (ss *Sim) RunStats(mode Modes, level Levels, start bool) {
	for _, sf := range ss.StatFuncs {
		sf(mode, level, start)
	}
	if !start && ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		nm := mode.String() + " " + level.String() + " Plot"
		tbs.GoUpdatePlot(nm)
		if level == Trial {
			ev := ss.Envs.ByMode(Test).(*consatenv.ConSatEnv)
			ev.UpdatePlot()
			plt := tbs.Plot("Optimal", ev.Plot)
			plt.Update()
		}
		tbs.SelectTabIndex(idx)
	}
}

// SetRunName sets the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) SetRunName() string {
	runName := ss.Params.RunName(0)
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
	for md, st := range ss.Loops.Stacks {
		mode := md.(Modes)
		for _, lev := range st.Order {
			level := lev.(Levels)
			ss.RunStats(mode, level, axon.Start)
		}
	}
	ev := ss.Envs.ByMode(Test).(*consatenv.ConSatEnv)
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Cycle))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
		plt := tbs.Plot("Optimal", ev.Plot)
		plt.Update()
		tbs.SelectTabIndex(idx)
	}
}

// ConfigStats handles configures functions to do all stats computation
// in the tensorfs system.
func (ss *Sim) ConfigStats() {
	net := ss.Net
	ss.Stats = ss.Root.Dir("Stats")
	ss.Current = ss.Stats.Dir("Current")

	ss.SetRunName()

	// last arg(s) are levels to exclude
	ss.AddStatStd(axon.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial))
	ss.AddStatStd(axon.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial))
	ss.AddStatStd(axon.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial))
	ss.AddStatStd(axon.StatPerTrialMSec(ss.Stats, Test, Trial))

	layers := []string{"Cities"}
	statNames := []string{"Spike", "Vm", "VmDend", "Ge", "Act", "Gi", "FFs", "FBs", "FSi", "SSi", "SSf", "FSGi", "SSGi"}
	ss.AddStat(func(mode Modes, level Levels, start bool) {
		for _, lnm := range layers {
			ly := ss.Net.LayerByName(lnm)
			pi := int(ly.Params.PoolIndex(0))
			di := 0
			for _, stnm := range statNames {
				name := lnm + "_" + stnm
				modeDir := ss.Stats.Dir(mode.String())
				curModeDir := ss.Current.Dir(mode.String())
				levelDir := modeDir.Dir(level.String())
				tsr := levelDir.Float64(name)
				ndata := 1
				if start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						// s.Range.SetMin(0).SetMax(1)
						s.On = false
						switch stnm {
						case "Act":
							s.On = true
						case "Vm", "VmDend":
							s.RightY = true
						}
					})
					continue
				}
				switch level {
				case Cycle:
					var stat float32
					switch stnm {
					case "Spike", "Vm", "VmDend":
						stat = ly.AvgMaxVarByPool(stnm, 0, di).Avg
					case "Ge":
						stat = axon.PoolAvgMax(axon.AMGeInt, axon.AMCycle, axon.Avg, uint32(pi), uint32(di))
					case "Act":
						stat = axon.PoolAvgMax(axon.AMAct, axon.AMCycle, axon.Avg, uint32(pi), uint32(di))
					case "Gi":
						stat = axon.Neurons.Value(int(ly.NeurStIndex), di, int(axon.Gi))
					default:
						var ivar fsfffb.InhibVars
						ivar.SetString(stnm)
						stat = axon.Pools.Value(pi, di, int(ivar))
					}
					curModeDir.Float64(name, ndata).SetFloat1D(float64(stat), di)
					tsr.AppendRowFloat(float64(stat))
				default:
					subDir := modeDir.Dir((level - 1).String())
					stat := stats.StatMean.Call(subDir.Value(name)).Float1D(0)
					tsr.AppendRowFloat(stat)
				}
			}
		}
	})
}

// StatCounters returns counters string to show at bottom of netview.
func (ss *Sim) StatCounters(mode, level enums.Enum) string {
	counters := ss.Loops.Stacks[mode].CountersString()
	vu := ss.NetViewUpdater(mode)
	if vu == nil || vu.View == nil {
		return counters
	}
	di := vu.View.Di
	counters += fmt.Sprintf(" Di: %d", di)
	curModeDir := ss.Current.Dir(mode.String())
	if curModeDir.Node("TrialName") == nil {
		return counters
	}
	counters += fmt.Sprintf(" TrialName: %s", curModeDir.StringValue("TrialName").String1D(di))
	if level == Cycle {
		return counters
	}
	return counters
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.StopLevel = Trial
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 2 * ss.Config.Run.Cycles
	nv.Options.Raster.Max = ss.Config.Run.Cycles
	nv.SetNet(ss.Net)
	ss.NetUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.5, 2.5)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ev := ss.Envs.ByMode(Test).(*consatenv.ConSatEnv)
	tbs := ss.GUI.Tabs.AsLab()
	_, idx := tbs.CurrentTab()
	plt := plot.New()
	tbs.Plot("Optimal", plt)
	ev.Plot = plt
	ev.MakePlot()
	tbs.SelectTabIndex(idx)

	ss.StatsInit()
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Defaults", Icon: icons.Update,
		Tooltip: "Restore initial default parameters.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Defaults()
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})
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

func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}

	runName := ss.SetRunName()
	netName := ss.Net.Name
	cfg := &ss.Config.Log
	axon.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Save})

	ss.Loops.Run(Test)

	axon.CloseLogFiles(ss.Loops, ss.Stats)
	axon.GPURelease()
}
