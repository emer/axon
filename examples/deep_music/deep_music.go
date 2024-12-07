// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// deep_music runs a DeepAxon network on predicting the next note
// in a musical sequence of notes.
package main

//go:generate core generate -add-types -add-funcs

import (
	"fmt"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/cli"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/plot"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/tensorfs"
	"cogentcore.org/core/tree"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/paths"
)

func main() {
	// gpu.Debug = true
	cfg := &Config{}
	cli.SetFromDefaults(cfg)
	opts := cli.DefaultOptions(cfg.Name, cfg.Title)
	opts.DefaultFiles = append(opts.DefaultFiles, "config.toml")
	cli.Run(opts, cfg, RunSim)
}

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
	Params axon.Params

	// Loops are the the control loops for running the sim, in different Modes
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

	// RandSeeds is a list of random seeds to use for each run.
	RandSeeds randx.Seeds `display:"-"`
}

// RunSim runs the simulation with given configuration.
func RunSim(cfg *Config) error {
	sim := &Sim{}
	sim.Config = cfg
	sim.Run()
	return nil
}

func (ss *Sim) Run() {
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.Net = axon.NewNetwork(ss.Config.Name)
	ss.Params.Config(LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag)
	if ss.Config.Params.Hid2 {
		ss.Params.ExtraSheets = "Hid2"
	}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	if ss.Config.Run.GPU {
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
		return
	}
	if ss.Config.GUI {
		ss.RunGUI()
	} else {
		ss.RunNoGUI()
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *MusicEnv
	if len(ss.Envs) == 0 {
		trn = &MusicEnv{}
		tst = &MusicEnv{}
	} else {
		trn = ss.Envs.ByMode(Train).(*MusicEnv)
		tst = ss.Envs.ByMode(Test).(*MusicEnv)
	}

	song := "bach_goldberg.mid"
	// maxRows := 60 // 30 is good benchmark, 25 it almost fully solves
	// have to push it to 60 to get an effect of Tau=4 vs. 1
	maxRows := 32
	if ss.Config.Params.Hid2 {
		ss.Params.ExtraSheets = "Hid2 "
	} else {
		ss.Params.ExtraSheets = ""
	}
	if ss.Config.Env.FullSong {
		maxRows = 0 // full thing
		ss.Params.ExtraSheets += "FullSong"
	} else {
		ss.Params.ExtraSheets += "30Notes"
	}
	track := 0
	wrapNotes := false // does a bit better with false for short lengths (30)

	// note: names must be standard here!
	trn.Defaults()
	trn.WrapNotes = wrapNotes
	trn.Name = Train.String()
	trn.Debug = false
	if ss.Config.Env.Env != nil {
		reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
	}
	trn.Config(song, track, maxRows, ss.Config.Env.UnitsPer)
	trn.ConfigNData(ss.Config.Run.NData)

	fmt.Printf("song rows: %d\n", trn.Song.NumRows())

	tst.Defaults()
	tst.WrapNotes = wrapNotes
	tst.Name = Test.String()
	tst.Play = true // see notes in README for getting this to work
	if ss.Config.Env.Env != nil {
		reflectx.SetFieldsFromMap(tst, ss.Config.Env.Env)
	}
	tst.Config(song, track, maxRows, ss.Config.Env.UnitsPer)
	tst.ConfigNData(ss.Config.Run.NData)

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByMode(etime.Train).(*MusicEnv)
	nnotes := ev.NNotes

	full := paths.NewFull()
	full.SelfCon = true // unclear if this makes a diff for self cons at all
	one2one := paths.NewOneToOne()
	_ = one2one

	space := float32(5)

	nUnits := 10
	if ev.MaxSteps == 0 {
		nUnits = 20
	}

	in, inPulv := net.AddInputPulv4D("Input", 1, nnotes, ss.Config.Env.UnitsPer, 1, space)
	in.AddClass("InLay")
	inPulv.AddClass("InLay")

	var hidp, hid2, hid2ct *axon.Layer
	hid, hidct := net.AddSuperCT2D("Hidden", "", 10, nUnits, space, one2one) // one2one learn > full
	_ = hidp
	if ss.Config.Params.Hid2 {
		// hidp -> hid2 doesn't actually help at all..
		// hidp = net.AddPulvForSuper(hid, space)
	}
	net.ConnectCTSelf(hidct, full, "")
	net.ConnectToPulv(hid, hidct, inPulv, full, full, "")
	net.ConnectLayers(in, hid, full, axon.ForwardPath)
	// net.ConnectLayers(hidct, hid, full, BackPath) // not useful

	if ss.Config.Params.Hid2 {
		hid2, hid2ct = net.AddSuperCT2D("Hidden2", "", 20, nUnits, space, one2one) // one2one learn > full
		net.ConnectCTSelf(hid2ct, full, "")
		net.ConnectToPulv(hid2, hid2ct, inPulv, full, full, "") // shortcut top-down
		errors.Log1(inPulv.RecvPathBySendName(hid2ct.Name)).AsEmer().AddClass("CTToPulvHigher")
		// net.ConnectToPulv(hid2, hid2ct, hidp, full, full) // predict layer below -- not useful
	}

	if ss.Config.Params.Hid2 {
		net.BidirConnectLayers(hid, hid2, full)
		net.ConnectLayers(hid2ct, hidct, full, axon.BackPath)
		// net.ConnectLayers(hid2ct, hid, full, axon.BackPath)
	}

	hid.PlaceAbove(in)
	if ss.Config.Params.Hid2 {
		hid2.PlaceRightOf(hid, 2)
	}

	net.Build()
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWeights()
	fmt.Println(axon.GPUSystem.Vars().StringDoc())
}

func (ss *Sim) ApplyParams() {
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
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.InitStats()
	ss.NewRun()
	ss.TrainUpdate.RecordSyns()
	ss.TrainUpdate.Update(Train, Trial)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// CurrentMode returns the current Train / Test mode from Context.
func (ss *Sim) CurrentMode() Modes {
	ctx := ss.Net.Context()
	var md Modes
	md.SetInt64(int64(ctx.Mode))
	return md
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
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, 50, cycles-plusPhase, cycles-1, Cycle, Trial, Train)

	ls.Stacks[Train].OnInit.Add("Init", func() { ss.Init() })

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
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater, ss.StatCounters)

		ls.Stacks[Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
		ls.Stacks[Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
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
	ctx := net.Context()
	ndata := int(ctx.NData)
	lays := net.LayersByClass("InputLayer", "TargetLayer")
	curModeDir := ss.Current.RecycleDir(mode.String())
	ev := ss.Envs.ByMode(mode).(*MusicEnv)
	ev.Step()

	net.InitExt()
	for di := uint32(0); di < ctx.NData; di++ {
		ev.StepDi(int(di))
		if ctx.Mode == int32(Test) && !ss.Config.Env.TestClamp {
			// todo:
			lastnote := int(curModeDir.Float64("OutNote", int(ctx.NData)).Float1D(int(di))) + ev.NoteRange.Min
			ev.RenderNote(lastnote)
		}
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			pats := ev.State("Note")
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
	ss.Envs.ByMode(Train).Init(0)
	ss.Envs.ByMode(Test).Init(0)
	ctx.Reset()
	ss.Net.InitWeights()
	if ss.Config.Run.StartWeights != "" { // this is just for testing -- not usually needed
		ss.Net.OpenWeightsJSON(core.Filename(ss.Config.Run.StartWeights))
		mpi.Printf("Starting with initial weights from: %s\n", ss.Config.Run.StartWeights)
	}
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(Test).Init(0)
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
		nm := mode.String() + "/" + level.String() + " Plot"
		ss.GUI.Tabs.GoUpdatePlot(nm)
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

// InitStats initializes all the stats by calling Start across all modes and levels.
func (ss *Sim) InitStats() {
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
		_, idx := ss.GUI.Tabs.CurrentTab()
		ss.GUI.Tabs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		ss.GUI.Tabs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		ss.GUI.Tabs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
		ss.GUI.Tabs.SelectTabIndex(idx)
	}
}

// ConfigStats handles configures functions to do all stats computation
// in the tensorfs system.
func (ss *Sim) ConfigStats() {
	net := ss.Net
	ss.Stats, _ = ss.Root.Mkdir("Stats")
	ss.Current, _ = ss.Stats.Mkdir("Current")

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

	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		if level != Trial {
			return
		}
		name := "TrialName"
		modeDir := ss.Stats.RecycleDir(mode.String())
		curModeDir := ss.Current.RecycleDir(mode.String())
		levelDir := modeDir.RecycleDir(level.String())
		tsr := levelDir.StringValue(name)
		ndata := int(ss.Net.Context().NData)
		if phase == Start {
			tsr.SetNumRows(0)
			if ps := plot.GetStylersFrom(tsr); ps == nil {
				ps.Add(func(s *plot.Style) {
					s.On = false
				})
				plot.SetStylersTo(tsr, ps)
			}
			return
		}
		for di := range ndata {
			// saved in apply inputs
			trlNm := curModeDir.StringValue(name, ndata).String1D(di)
			tsr.AppendRowString(trlNm)
		}
	})

	// up to a point, it is good to use loops over stats in one function,
	// to reduce repetition of boilerplate.
	statNames := []string{"InputP_CorSim", "Err"}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		for _, name := range statNames {
			modeDir := ss.Stats.RecycleDir(mode.String())
			curModeDir := ss.Current.RecycleDir(mode.String())
			levelDir := modeDir.RecycleDir(level.String())
			subDir := modeDir.RecycleDir((level - 1).String()) // note: will fail for Cycle
			tsr := levelDir.Float64(name)
			ctx := ss.Net.Context()
			ndata := int(ctx.NData)
			var stat float64
			if phase == Start {
				tsr.SetNumRows(0)
				if ps := plot.GetStylersFrom(tsr); ps == nil {
					ps.Add(func(s *plot.Style) {
						s.Range.SetMin(0).SetMax(1)
						s.On = true
					})
					plot.SetStylersTo(tsr, ps)
				}
				continue
			}
			switch level {
			case Trial:
				inp := ss.Net.LayerByName("InputP")
				for di := range ndata {
					var stat float64
					switch name {
					case "InputP_CorSim":
						stat = 1.0 - float64(axon.LayerStates.Value(int(inp.Index), int(di), int(axon.LayerPhaseDiff)))
					case "Err":
						err, minusIndex, plusIndex := inp.LocalistErr4D(ctx)
						curModeDir.Float64("TargNote", ndata).SetFloat1D(float64(plusIndex[di]), di)
						curModeDir.Float64("OutNote", ndata).SetFloat1D(float64(minusIndex[di]), di)
						if err[di] {
							stat = 1
						} else {
							stat = 0
						}
						ev := ss.Envs.ByMode(ss.CurrentMode()).(*MusicEnv)
						if ev.Play {
							if ss.Config.Env.PlayTarg {
								ev.PlayNote(plusIndex[di])
							} else {
								ev.PlayNote(minusIndex[di])
							}
						}
					}
					curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
					tsr.AppendRowFloat(stat)
				}
			case Epoch:
				stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			case Run:
				stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			}
		}
	})

	perTrlFunc := axon.StatPerTrialMSec(ss.Stats, "InputP_CorSim", Train, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		perTrlFunc(mode, level, phase == Start)
	})

	prevCorFunc := axon.StatPrevCorSim(ss.Stats, ss.Current, net, Trial, "InputP")
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		prevCorFunc(mode, level, phase == Start)
	})

	lays := net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	actGeFunc := axon.StatLayerActGe(ss.Stats, net, Train, Trial, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		actGeFunc(mode, level, phase == Start)
	})

	pcaFunc := axon.StatPCA(ss.Stats, ss.Current, net, ss.Config.Run.PCAInterval, Train, Trial, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trnEpc := ss.Loops.Loop(Train, Epoch).Counter.Cur
		pcaFunc(mode, level, phase == Start, trnEpc)
	})

	stateFunc := axon.StatLayerState(ss.Stats, net, Test, Trial, true, "ActM", "Input", "InputP")
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		stateFunc(mode, level, phase == Start)
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
	curModeDir := ss.Current.RecycleDir(mode.String())
	if curModeDir.Node("TrialName") == nil {
		return counters
	}
	counters += fmt.Sprintf(" TrialName: %s", curModeDir.StringValue("TrialName").String1D(di))
	statNames := []string{"InputP_CorSim"}
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
func (ss *Sim) ConfigGUI() {
	ss.GUI.MakeBody(ss, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.FS = ss.Root
	ss.GUI.DataRoot = "Root"
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.TestUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}
	ss.ConfigNetView(nv)

	ss.GUI.UpdateFiles()
	ss.InitStats()
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

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
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
