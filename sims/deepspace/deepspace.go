// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// deepspace simulates deep cerebellar nucleus and deep cortical layer predictive
// learning on spatial updating from the vestibular and visual systems.
package deepspace

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"image"
	"os"
	"reflect"

	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/sims/deepspace/emery"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/paths"
)

// Modes are the looping modes (Stacks) for running and statistics.
type Modes int32 //enums:enum
const (
	Train Modes = iota
	Test
)

// Levels are the looping levels for running and statistics.
type Levels int32 //enums:enum
const (
	Cycle Levels = iota
	Trial
	Epoch
	Run
	Expt
)

// StatsPhase is the phase of stats processing for given mode, level.
// Accumulated values are reset at Start, added each Step.
type StatsPhase int32 //enums:enum
const (
	Start StatsPhase = iota
	Step
)

// see params.go for params, config.go for Config

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

	// TrainUpdate has Train mode netview update parameters.
	TrainUpdate axon.NetViewUpdate `display:"inline"`

	// TestUpdate has Test mode netview update parameters.
	TestUpdate axon.NetViewUpdate `display:"inline"`

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

	// GUI for viewing env.
	EnvGUI *emery.GUI `display:"-"`

	// RandSeeds is a list of random seeds to use for each run.
	RandSeeds randx.Seeds `display:"-"`
}

func (ss *Sim) SetConfig(cfg *Config) { ss.Config = cfg }
func (ss *Sim) Body() *core.Body      { return ss.GUI.Body }

func (ss *Sim) ConfigSim() {
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.Net = axon.NewNetwork(ss.Config.Name)
	ss.Params.Config(LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag, reflect.ValueOf(ss))
	if ss.Config.Params.Hid2 {
		ss.Params.ExtraSheets = "Hid2"
	}
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
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	newEnv := (len(ss.Envs) == 0)

	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn, tst *emery.EmeryEnv
		if newEnv {
			trn = &emery.EmeryEnv{}
			tst = &emery.EmeryEnv{}
		} else {
			trn = ss.Envs.ByModeDi(Train, di).(*emery.EmeryEnv)
			tst = ss.Envs.ByModeDi(Test, di).(*emery.EmeryEnv)
		}

		// note: names must be standard here!
		trn.Defaults()
		trn.Name = env.ModeDi(Train, di)
		trn.RandSeed = 73 + int64(di)*73
		trn.UnitsPer = ss.Config.Env.UnitsPer
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
		}
		trn.Config()

		tst.Defaults()
		tst.Name = env.ModeDi(Test, di)
		tst.RandSeed = 181 + int64(di)*181
		trn.UnitsPer = ss.Config.Env.UnitsPer
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(tst, ss.Config.Env.Env)
		}
		tst.Config()

		trn.Init(0)
		tst.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn, tst)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.Context().ThetaCycles = int32(ss.Config.Run.Cycles)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByModeDi(Train, 0).(*emery.EmeryEnv)

	full := paths.NewFull()
	full.SelfCon = true // unclear if this makes a diff for self cons at all
	one2one := paths.NewOneToOne()
	_ = one2one

	// nPerAng := 5 // 30 total > 20 -- small improvement
	// nPerDepth := 2
	// rfDepth := 6
	// rfWidth := 3
	//
	// rect := paths.NewRect()
	// rect.Size.Set(rfWidth, rfDepth) // 6 > 8 > smaller
	// rect.Scale.Set(1.0/float32(nPerAng), 1.0/float32(nPerDepth))
	// _ = rect
	//
	// rectRecip := paths.NewRectRecip(rect)
	// _ = rectRecip

	space := float32(5)
	eyeSz := image.Point{ev.NextStates["EyeR"].DimSize(1), ev.NextStates["EyeR"].DimSize(0)}

	rotAct := net.AddLayer2D("ActRotate", axon.InputLayer, ev.UnitsPer, ev.LinearUnits)

	vvelIn, vvelInp := net.AddInputPulv2D("VNCAngVel", ev.UnitsPer, ev.LinearUnits, space)

	eyeLIn, eyeLInp := net.AddInputPulv2D("EyeLeft", eyeSz.Y, eyeSz.X, space)
	eyeRIn, eyeRInp := net.AddInputPulv2D("EyeRight", eyeSz.Y, eyeSz.X, space)
	eyeLIn.AddClass("VisIn")
	eyeLInp.AddClass("VisIn")
	eyeRIn.AddClass("VisIn")
	eyeRInp.AddClass("VisIn")

	vestHid, vestHidct := net.AddSuperCT2D("VestHidden", "", 10, 10, 2*space, one2one) // one2one learn > full
	// net.ConnectCTSelf(vestHidct, full, "") // self definitely doesn't make sense -- no need for 2-back ct
	// net.LateralConnectLayer(vestHidct, full).AddClass("CTSelfMaint") // no diff
	net.ConnectToPulv(vestHid, vestHidct, vvelInp, full, full, "")
	net.ConnectLayers(rotAct, vestHid, full, axon.ForwardPath)
	net.ConnectLayers(vvelIn, vestHid, full, axon.ForwardPath)

	visHid, visHidct := net.AddSuperCT2D("VisHidden", "", 10, 10, 2*space, one2one) // one2one learn > full
	// net.ConnectCTSelf(visHidct, full, "") // self definitely doesn't make sense -- no need for 2-back ct
	// net.LateralConnectLayer(visHidct, full).AddClass("CTSelfMaint") // no diff
	net.ConnectToPulv(visHid, visHidct, eyeLInp, full, full, "")
	net.ConnectToPulv(visHid, visHidct, eyeRInp, full, full, "")
	net.ConnectLayers(rotAct, visHid, full, axon.ForwardPath)
	net.ConnectLayers(vvelIn, visHid, full, axon.ForwardPath)

	// net.ConnectLayers(visHidct, visHid, full, BackPath)

	// var hid2, hid2ct *axon.Layer
	// if ss.Config.Params.Hid2 {
	// 	hid2, hid2ct = net.AddSuperCT2D("DepthHid2", "", 10, 20, 2*space, one2one) // one2one learn > full
	//
	// 	net.ConnectCTSelf(hid2ct, full, "")
	// 	net.ConnectToPulv(hid2, hid2ct, vvelInp, full, full, "")
	// 	net.ConnectLayers(rotAct, hid2, full, axon.ForwardPath)
	//
	// 	// net.ConnectLayers(hid, hid2, rect2, axon.ForwardPath)
	// 	// net.ConnectLayers(hid2, hid, rect2Recip, BackPath)
	//
	// 	net.BidirConnectLayers(hid, hid2, full)
	// 	net.ConnectLayers(hid2ct, hidct, full, axon.BackPath)
	// }

	// no benefit from these:
	// net.ConnectLayers(hdHid, hid, full, BackPath)
	// net.ConnectLayers(hdHidct, hidct, full, BackPath)

	vvelIn.PlaceBehind(rotAct, space)
	eyeLIn.PlaceRightOf(rotAct, space)
	eyeRIn.PlaceRightOf(eyeLIn, space)
	vestHid.PlaceAbove(rotAct)
	visHid.PlaceRightOf(vestHid, space)
	// if ss.Config.Params.Hid2 {
	// 	hid2.PlaceBehind(hdHidct, 2*space)
	// }

	net.Build()
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
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
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.ApplyParams()
	ss.StatsInit()
	ss.NewRun()
	ss.TrainUpdate.RecordSyns()
	ss.TrainUpdate.Update(Train, Trial)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// NetViewUpdater returns the NetViewUpdate for given mode.
func (ss *Sim) NetViewUpdater(mode enums.Enum) *axon.NetViewUpdate {
	if mode.Int64() == Train.Int64() {
		return &ss.TrainUpdate
	}
	return &ss.TestUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trials := int(math32.IntMultipleGE(float32(ss.Config.Run.Trials), float32(ss.Config.Run.NData)))
	cycles := ss.Config.Run.Cycles
	plusPhase := ss.Config.Run.PlusCycles

	ls.AddStack(Train, Trial).
		AddLevel(Expt, 1).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, cycles-plusPhase, cycles-1, Cycle, Trial, Train)

	ls.Stacks[Train].OnInit.Add("Init", ss.Init)

	ls.AddOnStartToLoop(Trial, "ApplyInputs", func(mode enums.Enum) {
		ss.ApplyInputs(mode.(Modes))
	})

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	trainEpoch := ls.Loop(Train, Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			ss.TestAll()
		}
	})

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.RunName())
	})

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
		ls.Stacks[Test].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
		ls.Loop(Train, Trial).OnEnd.Add("UpdateEnvGUI", func() {
			ss.UpdateEnvGUI(Train)
		})
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment for given mode.
// Any other start-of-trial logic can also be put here.
func (ss *Sim) ApplyInputs(mode Modes) {
	net := ss.Net
	ctx := ss.Net.Context()
	ndata := int(ctx.NData)
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	curModeDir := ss.Current.Dir(mode.String())

	net.InitExt()
	for di := uint32(0); di < ctx.NData; di++ {
		ev := ss.Envs.ByModeDi(mode, int(di)).(*emery.EmeryEnv)
		ev.Step()
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			pats := ev.State(lnm)
			if pats != nil {
				ly.ApplyExt(di, pats)
			}
		}
		curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), int(di))
	}
	ss.Net.ApplyExts()
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	ss.InitRandSeed(ss.Loops.Loop(Train, Run).Counter.Cur)
	for di := 0; di < int(ctx.NData); di++ {
		ss.Envs.ByModeDi(Train, di).Init(0)
		ss.Envs.ByModeDi(Test, di).Init(0)
	}
	ctx.Reset()
	ss.Net.InitWeights()
	if ss.Config.Run.StartWeights != "" { // this is just for testing -- not usually needed
		ss.Net.OpenWeightsJSON(core.Filename(ss.Config.Run.StartWeights))
		mpi.Printf("Starting with initial weights from: %s\n", ss.Config.Run.StartWeights)
	}
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ctx := ss.Net.Context()
	for di := 0; di < int(ctx.NData); di++ {
		ss.Envs.ByModeDi(Test, di).Init(0)
	}
	ss.Loops.ResetAndRun(Test)
	ss.Loops.Mode = Train // important because this is called from Train Run: go back.
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
	if level <= Trial {
		return
	}
	ss.RunStats(mode, level-1, Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level == Cycle {
		return
	}
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
	runName := ss.Params.RunName(ss.Config.Run.Run)
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
			if level == Cycle {
				continue
			}
			ss.RunStats(mode, level, Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
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
	counterFunc := axon.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		counterFunc(mode, level, phase == Start)
	})
	runNameFunc := axon.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		runNameFunc(mode, level, phase == Start)
	})
	trialNameFunc := axon.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trialNameFunc(mode, level, phase == Start)
	})
	perTrlFunc := axon.StatPerTrialMSec(ss.Stats, Train, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		perTrlFunc(mode, level, phase == Start)
	})

	plays := net.LayersByType(axon.PulvinarLayer)
	corSimFunc := axon.StatCorSim(ss.Stats, ss.Current, net, Trial, Run, plays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		corSimFunc(mode, level, phase == Start)
	})

	prevCorFunc := axon.StatPrevCorSim(ss.Stats, ss.Current, net, Trial, Run, plays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		prevCorFunc(mode, level, phase == Start)
	})

	lays := net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	actGeFunc := axon.StatLayerActGe(ss.Stats, net, Train, Trial, Run, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		actGeFunc(mode, level, phase == Start)
	})

	pcaFunc := axon.StatPCA(ss.Stats, ss.Current, net, ss.Config.Run.PCAInterval, Train, Trial, Run, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trnEpc := ss.Loops.Loop(Train, Epoch).Counter.Cur
		pcaFunc(mode, level, phase == Start, trnEpc)
	})

	// stateFunc := axon.StatLayerState(ss.Stats, net, Test, Trial, true, "ActM", "Depth", "DepthP", "HeadDir", "HeadDirP")
	// ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
	// 	stateFunc(mode, level, phase == Start)
	// })
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
	statNames := []string{"DepthP_CorSim", "HeadDirP_CorSim"}
	if level == Cycle || curModeDir.Node(statNames[0]) == nil {
		return counters
	}
	for _, name := range statNames {
		counters += fmt.Sprintf(" %s: %.4g", name, curModeDir.Float64(name).Float1D(di))
	}
	return counters
}

//////// GUI

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	// nv.ViewDefaults()
	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 2.1, 2.0)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.CycleUpdateInterval = 10
	ss.GUI.StopLevel = Trial
	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 2 * ss.Config.Run.Cycles
	nv.Options.Raster.Max = ss.Config.Run.Cycles
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.TestUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}
	ss.ConfigNetView(nv)

	evtab, _ := ss.GUI.Tabs.NewTab("Env")
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*emery.EmeryEnv)
	ss.EnvGUI = &emery.GUI{}
	ss.EnvGUI.ConfigGUI(ev, evtab)

	ss.StatsInit()

	ss.GUI.Tabs.SelectTabIndex(0)
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

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

func (ss *Sim) UpdateEnvGUI(mode Modes) {
	vu := ss.NetViewUpdater(mode)
	if vu == nil || vu.View == nil {
		return
	}
	ss.EnvGUI.Update()
}

func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}

	runName := ss.SetRunName()
	netName := ss.Net.Name
	cfg := &ss.Config.Log
	axon.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Train, cfg.Test})

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.Runs, ss.Config.Run.Run)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)

	ss.Loops.Run(Train)

	axon.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
	axon.GPURelease()
}
