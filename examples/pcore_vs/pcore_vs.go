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
	"math"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/num"
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

	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn, tst *GoNoEnv
		if newEnv {
			trn = &GoNoEnv{}
			tst = &GoNoEnv{}
		} else {
			trn = ss.Envs.ByModeDi(Train, di).(*GoNoEnv)
			tst = ss.Envs.ByModeDi(Test, di).(*GoNoEnv)
		}

		// note: names must be standard here!
		trn.Name = env.ModeDi(Train, di)
		trn.Defaults()
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
		}
		trn.Config(Train, 73+int64(di)*73)

		tst.Name = env.ModeDi(Test, di)
		tst.Defaults()
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(tst, ss.Config.Env.Env)
		}
		tst.Config(Test, 181+int64(di)*181)

		trn.Init(0)
		tst.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn, tst)
		if di == 0 {
			ss.ConfigRubicon(trn)
		}
	}
}

func (ss *Sim) ConfigRubicon(trn *GoNoEnv) {
	pv := &ss.Net.Rubicon
	pv.SetNUSs(2, 1)
	pv.Urgency.U50 = 20 // 20 def
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByModeDi(Train, 0).(*GoNoEnv)

	np := 1
	nuY := ev.NUnitsY
	nuX := ev.NUnitsX
	space := float32(2)

	one2one := paths.NewOneToOne()
	full := paths.NewFull()
	_ = full
	mtxRandPath := paths.NewPoolUniformRand()
	mtxRandPath.PCon = 0.5
	_ = mtxRandPath

	mtxGo, mtxNo, gpePr, gpeAk, stn, gpi := net.AddVBG("", 1, np, nuY, nuX, nuY, nuX, space)
	_, _ = gpePr, gpeAk

	snc := net.AddLayer2D("SNc", axon.InputLayer, 1, 1)
	_ = snc

	urge := net.AddUrgencyLayer(5, 4)
	_ = urge

	accPos := net.AddLayer4D("ACCPos", axon.InputLayer, 1, np, nuY, nuX)
	accNeg := net.AddLayer4D("ACCNeg", axon.InputLayer, 1, np, nuY, nuX)
	accPos.AddClass("ACC")
	accNeg.AddClass("ACC")

	accPosPT, accPosVM := net.AddPTMaintThalForSuper(accPos, nil, "VM", "PFCPath", one2one, full, one2one, true, space)
	_ = accPosPT

	net.ConnectLayers(accPos, stn, full, axon.ForwardPath).AddClass("CortexToSTN")
	net.ConnectLayers(accNeg, stn, full, axon.ForwardPath).AddClass("CortexToSTN")

	net.ConnectLayers(gpi, accPosVM, full, axon.InhibPath).AddClass("BgFixed")

	mtxGo.SetBuildConfig("ThalLay1Name", accPosVM.Name)
	mtxNo.SetBuildConfig("ThalLay1Name", accPosVM.Name)

	net.ConnectToVSMatrix(accPos, mtxGo, full).AddClass("ACCToVMtx")
	net.ConnectToVSMatrix(accNeg, mtxNo, full).AddClass("ACCToVMtx")
	// cross connections:
	net.ConnectToVSMatrix(accPos, mtxNo, full).AddClass("ACCToVMtx")
	net.ConnectToVSMatrix(accNeg, mtxGo, full).AddClass("ACCToVMtx")

	net.ConnectToVSMatrix(urge, mtxGo, full)

	accPosVM.PlaceRightOf(gpi, space)
	snc.PlaceRightOf(accPosVM, space)
	urge.PlaceRightOf(snc, space)
	gpeAk.PlaceAbove(gpi)
	stn.PlaceRightOf(gpePr, space)
	mtxGo.PlaceAbove(gpeAk)
	accPos.PlaceAbove(mtxGo)
	accNeg.PlaceRightOf(accPos, space)

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

	ev := ss.Envs.ByModeDi(Test, 0).(*GoNoEnv)
	trials := int(math32.IntMultipleGE(float32(ss.Config.Run.Trials), float32(ss.Config.Run.NData)))
	cycles := ss.Config.Run.Cycles
	plusPhase := ss.Config.Run.PlusCycles

	ls.AddStack(Train, Trial).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Theta, 3).
		AddLevel(Cycle, cycles)

	nTestInc := int(1.0/ev.TestInc) + 1
	totTstTrls := ev.TestReps * nTestInc * nTestInc
	testTrials := int(math32.IntMultipleGE(float32(totTstTrls), float32(ss.Config.Run.NData)))

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, testTrials, ss.Config.Run.NData).
		AddLevel(Theta, 3).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, 50, cycles-plusPhase, cycles-1, Cycle, Theta, Train) // note: Theta

	ls.Stacks[Train].OnInit.Add("Init", func() { ss.Init() })

	ls.AddOnStartToLoop(Theta, "ApplyInputs", func(mode enums.Enum) {
		trial := ls.Stacks[mode].Loops[Trial].Counter.Cur
		theta := ls.Stacks[mode].Loops[Theta].Counter.Cur
		ss.ApplyInputs(mode.(Modes), trial, theta)
	})

	ls.AddOnEndToLoop(Theta, "GatedAction", func(mode enums.Enum) {
		theta := ls.Stacks[mode].Loops[Theta].Counter.Cur
		if theta == 1 {
			ss.GatedAction(mode.(Modes))
		}
	})

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.RunName())
	})

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Theta, ss.NetViewUpdater, ss.StatCounters)

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
	curModeDir := ss.Current.RecycleDir(mode.String())
	lays := []string{"ACCPos", "ACCNeg"}
	net.InitExt()
	for di := range ndata {
		idx := trial + di
		ev := ss.Envs.ByModeDi(mode, di).(*GoNoEnv)
		ev.Trial.Set(idx)
		if theta == 0 {
			ev.Step()
		} else {
			for _, lnm := range lays {
				ly := ss.Net.LayerByName(lnm)
				st := ev.State(ly.Name)
				if st != nil {
					ly.ApplyExt(uint32(di), st)
				}
			}
		}
		curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), di)
		ss.ApplyRubicon(ev, mode, theta, uint32(di))
	}
	net.ApplyExts()
}

// ApplyRubicon applies Rubicon reward inputs
func (ss *Sim) ApplyRubicon(ev *GoNoEnv, mode Modes, trial int, di uint32) {
	pv := &ss.Net.Rubicon
	pv.EffortUrgencyUpdate(di, 1)
	if mode == Test {
		pv.Urgency.Reset(di)
	}

	switch trial {
	case 0:
		axon.GlobalSetRew(di, 0, false) // no rew
		axon.GlobalScalars.Set(0, int(axon.GvACh), int(di))
	case 1:
		axon.GlobalSetRew(di, 0, false) // no rew
		axon.GlobalScalars.Set(1, int(axon.GvACh), int(di))
	case 2:
		axon.GlobalScalars.Set(1, int(axon.GvACh), int(di))
		ss.GatedRew(ev, di)
	}
}

// GatedRew applies reward input based on gating action and input
func (ss *Sim) GatedRew(ev *GoNoEnv, di uint32) {
	// note: not using RPE here at this point
	rew := ev.Rew
	ss.SetRew(rew, di)
}

func (ss *Sim) SetRew(rew float32, di uint32) {
	pv := &ss.Net.Rubicon
	axon.GlobalSetRew(di, rew, true)
	axon.GlobalScalars.Set(rew, int(axon.GvDA), int(di)) // no reward prediction error
	if rew > 0 {
		pv.SetUS(di, axon.Positive, 0, 1)
	} else if rew < 0 {
		pv.SetUS(di, axon.Negative, 0, 1)
	}
}

// GatedAction records gating action and generates reward
// this happens at the end of Trial == 1 (2nd trial)
// so that the reward is present during the final trial when learning occurs.
func (ss *Sim) GatedAction(mode Modes) {
	ctx := ss.Net.Context()
	curModeDir := ss.Current.RecycleDir(mode.String())
	mtxly := ss.Net.LayerByName("VMtxGo")
	vmly := ss.Net.LayerByName("ACCPosVM")
	vmlpi := vmly.Params.PoolIndex(0)
	mtxlpi := mtxly.Params.PoolIndex(0)
	nan := math.NaN()
	ndata := int(ctx.NData)
	for di := 0; di < ndata; di++ {
		ev := ss.Envs.ByModeDi(mode, di).(*GoNoEnv)
		didGate := mtxly.Params.AnyGated(uint32(di))
		action := "Gated"
		if !didGate {
			action = "NoGate"
		}
		ev.Action(action, nil)
		rt := axon.LayerStates.Value(vmly.Index, di, int(axon.LayerRT))
		if rt > 0 {
			curModeDir.Float32("ACCPosVM_RT", ndata).SetFloat1D(float64(rt/200), di)
		} else {
			curModeDir.Float32("ACCPosVM_RT", ndata).SetFloat1D(nan, di)
		}
		cycavg := float64(axon.PoolAvgMax(axon.AMCaPMax, axon.AMCycle, axon.Avg, vmlpi, uint32(di)))
		curModeDir.Float32("ACCPosVM_ActAvg", ndata).SetFloat1D(cycavg, di)
		cycavg = float64(axon.PoolAvgMax(axon.AMCaPMax, axon.AMCycle, axon.Avg, mtxlpi, uint32(di)))
		curModeDir.Float32("VMtxGo_ActAvg", ndata).SetFloat1D(cycavg, di)
	}
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
		ss.GUI.Tabs.GoUpdatePlot("Train/TrialAll Plot")
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
		_, idx := ss.GUI.Tabs.CurrentTab()
		ss.GUI.Tabs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Trial))
		ss.GUI.Tabs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		ss.GUI.Tabs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		ss.GUI.Tabs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
		ss.GUI.Tabs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Epoch))
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
	statNames := []string{"ACCPos", "ACCNeg", "Gated", "Should", "Match", "Rew", "ACCPosVM_RT", "ACCPosVM_ActAvg", "VMtxGo_ActAvg"}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		for si, name := range statNames {
			modeDir := ss.Stats.RecycleDir(mode.String())
			curModeDir := ss.Current.RecycleDir(mode.String())
			levelDir := modeDir.RecycleDir(level.String())
			subDir := modeDir.RecycleDir((level - 1).String()) // note: will fail for Cycle
			tsr := levelDir.Float64(name)
			ndata := int(ss.Net.Context().NData)
			var stat float64
			if phase == Start {
				tsr.SetNumRows(0)
				if ps := plot.GetStylersFrom(tsr); ps == nil {
					ps.Add(func(s *plot.Style) {
						s.Range.SetMin(0).SetMax(1)
						if si < 3 {
							s.On = true
						}
					})
					plot.SetStylersTo(tsr, ps)
				}
				continue
			}
			switch level {
			case Trial:
				for di := range ndata {
					ev := ss.Envs.ByModeDi(mode, di).(*GoNoEnv)
					var stat float32
					switch name {
					case "ACCPos":
						stat = ev.ACCPos
					case "ACCNeg":
						stat = ev.ACCNeg
					case "Gated":
						stat = num.FromBool[float32](ev.Gated)
					case "Should":
						stat = num.FromBool[float32](ev.Should)
					case "Match":
						stat = num.FromBool[float32](ev.Match)
					case "Rew":
						stat = ev.Rew
					case "ACCPosVM_RT", "ACCPosVM_ActAvg", "VMtxGo_ActAvg":
						stat = float32(curModeDir.Float32(name, ndata).Float1D(di))
					}
					curModeDir.Float32(name, ndata).SetFloat1D(float64(stat), di)
					tsr.AppendRowFloat(float64(stat))
				}
			case Epoch:
				if mode == Train {
					stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
					tsr.AppendRowFloat(stat)
					break
				}
				switch name {
				case "TrialName":
					stats.Groups(curModeDir, subDir.Value(name))
				default:
					stats.GroupStats(curModeDir, stats.StatMean, subDir.Value(name))
				}
			case Run:
				stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			}
		}
	})

	perTrlFunc := axon.StatPerTrialMSec(ss.Stats, "Gated", Train, Trial)
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
	curModeDir := ss.Current.RecycleDir(mode.String())
	if curModeDir.Node("TrialName") == nil {
		return counters
	}
	counters += fmt.Sprintf(" TrialName: %s", curModeDir.StringValue("TrialName").String1D(di))
	statNames := []string{"Gated"}
	if level == Cycle || curModeDir.Node(statNames[0]) == nil {
		return counters
	}
	for _, name := range statNames {
		counters += fmt.Sprintf(" %s: %.4g", name, curModeDir.Float32(name).Float1D(di))
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
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	// nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	// nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

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
