// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// pcore_vs simulates the inhibitory dynamics in the STN and GPe
// leading to integration of Go vs. NoGo signal in the basal
// ganglia, for the Ventral Striatum (VS) global Go vs. No case.
package main

//go:generate core generate -add-types -add-funcs

import (
	"fmt"
	"math/rand/v2"

	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/cli"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
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
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/paths"
)

func main() {
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
	Theta
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
	// if ss.Config.Run.GPU {
	// 	fmt.Println(axon.GPUSystem.Vars().StringDoc())
	// }
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
	newEnv := (len(ss.Envs) == 0)

	var trn0 *VSPatchEnv
	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn, tst *VSPatchEnv
		if newEnv {
			trn = &VSPatchEnv{}
			tst = &VSPatchEnv{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*VSPatchEnv)
			tst = ss.Envs.ByModeDi(etime.Test, di).(*VSPatchEnv)
		}

		// note: names must be standard here!
		trn.Name = env.ModeDi(etime.Train, di)
		trn.Defaults()
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
		}
		trn.Config(etime.Train, di, 73+int64(di)*73)
		if di == 0 {
			trn.ConfigPats()
			trn0 = trn
		} else {
			trn.Pats = trn0.Pats
		}

		tst.Name = env.ModeDi(etime.Test, di)
		tst.Defaults()
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(tst, ss.Config.Env.Env)
		}
		tst.Config(etime.Test, di, 181+int64(di)*181)
		tst.Pats = trn0.Pats

		trn.Init(0)
		tst.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn, tst)
		if di == 0 {
			ss.ConfigRubicon(trn)
		}
	}
}

func (ss *Sim) ConfigRubicon(trn *VSPatchEnv) {
	pv := &ss.Net.Rubicon
	pv.SetNUSs(1, 1)
	pv.Defaults()
	pv.Urgency.U50 = 20 // 20 def
	pv.LHb.VSPatchGain = 3
	pv.LHb.VSPatchNonRewThr = 0.1
	pv.USs.PVposGain = 10
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.Context().ThetaCycles = int32(ss.Config.Run.Cycles)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByModeDi(Train, 0).(*VSPatchEnv)

	space := float32(2)
	full := paths.NewFull()

	// mtxRandPath := paths.NewPoolUniformRand()
	// mtxRandPath.PCon = 0.5
	// _ = mtxRandPath

	in := net.AddLayer2D("State", axon.InputLayer, ev.NUnitsY, ev.NUnitsX)
	vSpatchD1, vSpatchD2 := net.AddVSPatchLayers("", 1, 6, 6, space)

	net.ConnectToVSPatch(in, vSpatchD1, vSpatchD2, full)

	net.Build()
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWeights()
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
	ss.ConfigEnv() // always do -- otherwise env params not reset after run
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
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

	ev := ss.Envs.ByModeDi(Train, 0).(*VSPatchEnv)
	trials := int(math32.IntMultipleGE(float32(ss.Config.Run.Trials), float32(ss.Config.Run.NData)))
	cycles := ss.Config.Run.Cycles
	plusPhase := ss.Config.Run.PlusCycles

	ls.AddStack(Train, Trial).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Theta, ev.Thetas).
		AddLevel(Cycle, cycles)

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Theta, ev.Thetas).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, 50, cycles-plusPhase, cycles-1, Cycle, Theta, Train) // note: Theta

	ls.Stacks[Train].OnInit.Add("Init", func() { ss.Init() })

	ls.AddOnStartToLoop(Theta, "ApplyInputs", func(mode enums.Enum) {
		trial := ls.Stacks[mode].Loops[Trial].Counter.Cur
		theta := ls.Stacks[mode].Loops[Theta].Counter.Cur
		ss.ApplyInputs(mode.(Modes), trial, theta)
	})

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	trainEpoch := ls.Loop(Train, Epoch)
	trainEpoch.OnEnd.Add("NewConds", func() {
		trnEpc := trainEpoch.Counter.Cur
		if trnEpc > 1 && trnEpc%ss.Config.Run.CondEpochs == 0 {
			ord := rand.Perm(ev.NConds)
			for di := 0; di < ss.Config.Run.NData; di++ {
				ev := ss.Envs.ByModeDi(etime.Train, di).(*VSPatchEnv)
				ev.SetCondValuesPermute(ord)
			}
		}
	})

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.RunName())
	})

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Theta, ss.NetViewUpdater)

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
func (ss *Sim) ApplyInputs(mode Modes, trial, theta int) {
	net := ss.Net
	ndata := int(net.Context().NData)
	curModeDir := ss.Current.Dir(mode.String())
	lays := []string{"State"}
	net.InitExt()
	for di := range ndata {
		ev := ss.Envs.ByModeDi(mode, di).(*VSPatchEnv)
		ev.Step()
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			st := ev.State(ly.Name)
			if st != nil {
				ly.ApplyExt(uint32(di), st)
			}
		}
		curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), di)
		ss.ApplyRubicon(ev, mode, theta, uint32(di))
	}
	net.ApplyExts()
}

// ApplyRubicon applies Rubicon reward inputs
func (ss *Sim) ApplyRubicon(ev *VSPatchEnv, mode Modes, theta int, di uint32) {
	pv := &ss.Net.Rubicon
	pv.NewState(di, &ss.Net.Rand) // first before anything else is updated
	pv.EffortUrgencyUpdate(di, 1)
	// if mode == Test {
	pv.Urgency.Reset(di)
	// }
	if theta == ev.Thetas-1 {
		axon.GlobalScalars.Set(1, int(axon.GvACh), int(di))
		ss.ApplyRew(di, ev.Rew)
	} else {
		ss.ApplyRew(di, 0)
		axon.GlobalScalars.Set(0, int(axon.GvACh), int(di))
	}
}

// ApplyRew applies reward input based on gating action and input
func (ss *Sim) ApplyRew(di uint32, rew float32) {
	pv := &ss.Net.Rubicon
	if rew > 0 {
		pv.SetUS(di, axon.Positive, 0, rew)
	} else if rew < 0 {
		pv.SetUS(di, axon.Negative, 0, -rew)
	}
	pv.SetDrives(di, 1, 1)
	pv.Step(di, &ss.Net.Rand)
	// normally set by VTA layer, including CS:
	lhbDA := axon.GlobalScalars.Value(int(axon.GvLHbPVDA), int(di))
	axon.GlobalScalars.Set(lhbDA, int(axon.GvDA), int(di))
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
	if level <= Theta {
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
		ss.GUI.Tabs.AsLab().GoUpdatePlot(nm)
		ss.GUI.Tabs.AsLab().GoUpdatePlot("Train/TrialAll Plot")
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
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Trial))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Epoch))
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

	// up to a point, it is good to use loops over stats in one function,
	// to reduce repetition of boilerplate.
	statNames := []string{"Cond", "CondRew", "Rew", "RewPred", "DA", "RewPred_NR", "DA_NR"}
	numStats := len(statNames)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		for si, name := range statNames {
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			subDir := modeDir.Dir((level - 1).String()) // note: will fail for Cycle
			tsr := levelDir.Float64(name)
			ndata := int(ss.Net.Context().NData)
			var stat float64
			if phase == Start {
				tsr.SetNumRows(0)
				plot.SetFirstStylerTo(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					if si >= 2 && si <= 5 {
						s.On = true
					}
				})
				if si == numStats-1 && level == Epoch {
					all, _ := levelDir.Values()
					for _, ts := range all {
						ts.(tensor.Values).SetNumRows(0)
					}
				}
				continue
			}
			switch level {
			case Trial, Theta:
				for di := range ndata {
					ev := ss.Envs.ByModeDi(mode, di).(*VSPatchEnv)
					theta := ev.Theta.Cur
					var stat float32
					switch name {
					case "Cond":
						stat = float32(ev.Cond)
					case "CondRew":
						stat = ev.CondRew
					case "Rew":
						stat = axon.GlobalScalars.Value(int(axon.GvRew), di)
					case "RewPred":
						stat = axon.GlobalScalars.Value(int(axon.GvRewPred), di)
						ev.RewPred = stat
						if level == Theta && theta == ev.Thetas-1 {
							ev.RewPred_NR = stat
						}
					case "DA":
						stat = axon.GlobalScalars.Value(int(axon.GvDA), di)
						ev.DA = stat
						if level == Theta && theta == ev.Thetas-1 {
							ev.DA_NR = stat
						}
					case "RewPred_NR":
						stat = ev.RewPred_NR
					case "DA_NR":
						stat = ev.DA_NR
					}
					curModeDir.Float32(name, ndata).SetFloat1D(float64(stat), di)
					tsr.AppendRowFloat(float64(stat))
				}
			case Epoch:
				stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
				if si == 0 {
					stats.Groups(curModeDir, subDir.Value("Cond"))
					break
				} else {
					stats.GroupStats(curModeDir, stats.StatMean, subDir.Value(name))
				}
				// note: results go under Group name: Cond
				gp := curModeDir.Dir("Stats/Cond/" + name).Value("Mean")
				plot.SetFirstStylerTo(gp, func(s *plot.Style) {
					s.Range.SetMin(0)
					if si >= 2 && si <= 4 {
						s.On = true
					}
				})
				if si == numStats-1 {
					nrows := gp.DimSize(0)
					row := curModeDir.Dir("Stats").Int("Row", nrows)
					for i := range nrows {
						row.Set(i, i)
					}
					_, idx := ss.GUI.Tabs.AsLab().CurrentTab()
					gpst := curModeDir.Dir("Stats/Cond").Value("Cond")
					for j := range gpst.DimSize(0) {
						val := gpst.String1D(j)
						for si, name := range statNames {
							if si == 0 {
								continue
							}
							svals := curModeDir.Dir("Stats/Cond/" + name).Value("Mean")
							snm := "Cond_" + val + "_" + name
							tsr := levelDir.Float64(snm)
							tsr.AppendRowFloat(svals.Float1D(j))
						}
					}
					ss.GUI.Tabs.AsLab().PlotTensorFS(curModeDir.Dir("Stats"))
					ss.GUI.Tabs.AsLab().SelectTabIndex(idx)
				}
			case Run:
				stat = stats.StatFinal.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			}
		}
	})

	perTrlFunc := axon.StatPerTrialMSec(ss.Stats, Train, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		perTrlFunc(mode, level, phase == Start)
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
	statNames := []string{"Rew", "RewPred", "DA", "RewPred_NR", "DA_NR"}
	if level == Cycle || curModeDir.Node(statNames[0]) == nil {
		return counters
	}
	for _, name := range statNames {
		counters += fmt.Sprintf(" %s: %.4g", name, curModeDir.Value(name).Float1D(di))
	}
	return counters
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	ss.GUI.MakeBody(ss, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.FS = ss.Root
	ss.GUI.DataRoot = "Root"
	ss.GUI.CycleUpdateInterval = 10

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

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 2.15, 2.45)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.UpdateFiles()
	ss.StatsInit()
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "New seed",
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
