// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// consat: This simulation tests axon on constraint satisfaction
// using the traveling salesperson problem.
package consat

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"os"
	"reflect"

	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/patterns"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/sims/consat/consatenv"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/looper"
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

// see params.go for params, config.go for config

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
	StatFuncs []func(mode enums.Enum, level enums.Enum, start bool) `display:"-"`

	// GUI manages all the GUI elements
	GUI egui.GUI `display:"-"`

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
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	if ss.Config.GPU {
		// gpu.DebugAdapter = true
		gpu.SelectAdapter = ss.Config.Run.GPUDevice
		axon.GPUInit()
		axon.UseGPU = true
	}
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLoops()
	ss.ConfigStats()
	// if ss.Config.GPU {
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
	newEnv := (len(ss.Envs) == 0)

	ndata := ss.Config.Run.NData

	for di := range ndata {
		var trn, tst *consatenv.ConSatEnv
		if newEnv {
			trn = &consatenv.ConSatEnv{}
			tst = &consatenv.ConSatEnv{}
		} else {
			trn = ss.Envs.ByModeDi(Train, di).(*consatenv.ConSatEnv)
			tst = ss.Envs.ByModeDi(Test, di).(*consatenv.ConSatEnv)
		}

		// note: names must be standard here!
		trn.Name = env.ModeDi(Train, di)
		trn.Defaults()
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
		}
		trn.Config(ndata, di, 73+int64(di)*73)

		tst.Name = env.ModeDi(Test, di)
		tst.Defaults()
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(tst, ss.Config.Env.Env)
		}
		tst.Config(ndata, ndata+di, 181+int64(di)*181) // unique items

		trn.Init(0)
		tst.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn, tst)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.Context().SetISICycles(int32(ss.Config.Run.ISICycles)).
		SetMinusCycles(int32(ss.Config.Run.MinusCycles)).
		SetPlusCycles(int32(ss.Config.Run.PlusCycles)).Update()
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByModeDi(Train, 0).(*consatenv.ConSatEnv)
	n := ev.NVars
	nu := ev.NUnitsPer
	np := ev.PopCodeUnits
	nc := ev.NConstraints + 1
	nHidUnits := 20 // 20 == 30 > 10
	// nHid1Units := 20

	inp := net.AddLayer4D("Input", axon.InputLayer, n, 1, 1, np)
	// hid1 := net.AddLayer2D("Hidden1", axon.SuperLayer, nHidUnits, nHidUnits)
	hid1 := net.AddLayer2D("Hidden1", axon.SuperLayer, nHidUnits, nHidUnits)
	// hid1.SetSampleShape(emer.CenterPoolIndexes(hid1, 2), emer.CenterPoolShape(hid1, 2))
	// hid2 := net.AddLayer2D("Hidden2", axon.SuperLayer, nHidUnits, nHidUnits)
	// no hid2 better!

	out := net.AddLayer4D("Output", axon.TargetLayer, 1, 1, nu, nu*nc)

	// inp.PlaceBehind(pos, 2)
	// hid1.PlaceAbove(pos)

	full := paths.NewFull()

	// topo := paths.NewPoolTile()
	// topo.Size.Set(1, 3)
	// topo.Skip.Set(1, 0)
	// topo.Start.Set(0, 0)
	// _ = topo

	net.ConnectLayers(inp, hid1, full, axon.ForwardPath)
	// net.BidirConnectLayers(hid1, hid2, full)
	// net.BidirConnectLayers(hid2, out, full)
	net.BidirConnectLayers(hid1, out, full) // shortcut

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
	ss.RandSeeds.Set(run, ss.Net.Rand)
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
	cycles := ss.Config.Run.Cycles()

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

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, Cycle, Trial, Train,
		func(mode enums.Enum) { ss.Net.ClearInputs() },
		func(mode enums.Enum) { ss.ApplyInputs(mode.(Modes)) },
	)
	ls.Stacks[Train].OnInit.Add("Init", ss.Init)
	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	trainEpoch := ls.Loop(Train, Epoch)
	// trainEpoch.IsDone.AddBool("NZeroStop", func() bool {
	// 	stopNz := ss.Config.Run.NZero
	// 	if stopNz <= 0 {
	// 		return false
	// 	}
	// 	curModeDir := ss.Current.Dir(Train.String())
	// 	curNZero := int(curModeDir.Value("NZero").Float1D(-1))
	// 	stop := curNZero >= stopNz
	// 	return stop
	// 	return false
	// })

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
	ndata := int(net.Context().NData)
	curModeDir := ss.Current.Dir(mode.String())
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	net.InitExt()
	for di := range ndata {
		ev := ss.Envs.ByModeDi(mode, di)
		ev.Step()
		curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), di)
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			st := ev.State(ly.Name)
			if st != nil {
				ly.ApplyExt(uint32(di), st)
			}
		}
	}
	net.ApplyExts()
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	run := ss.Loops.Loop(Train, Run).Counter.Cur
	ss.InitRandSeed(run)
	for di := 0; di < int(ctx.NData); di++ {
		ss.Envs.ByModeDi(Train, di).Init(run)
		ss.Envs.ByModeDi(Test, di).Init(run)
	}
	ctx.Reset()
	ss.Net.InitWeights()
	if ss.Config.Run.StartWeights != "" {
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

////////  Inputs

func (ss *Sim) ConfigInputs() {
	dt := table.New()
	metadata.SetName(dt, "Train")
	metadata.SetDoc(dt, "Training inputs")
	dt.AddStringColumn("Name")
	dt.AddFloat32Column("Input", 5, 5)
	dt.AddFloat32Column("Output", 5, 5)
	dt.SetNumRows(25)

	patterns.PermutedBinaryMinDiff(dt.Columns.Values[1], 6, 1, 0, 3)
	patterns.PermutedBinaryMinDiff(dt.Columns.Values[2], 6, 1, 0, 3)
	dt.SaveCSV("random_5x5_25_gen.tsv", tensor.Tab, table.Headers)

	tensorfs.DirFromTable(ss.Root.Dir("Inputs/Train"), dt)
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
	if level <= Trial {
		return
	}
	ss.RunStats(mode, level-1, axon.Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level == Cycle {
		return
	}
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
		switch level {
		// case Trial:
		// 	ev := ss.Envs.ByModeDi(Train, 0).(*consatenv.ConSatEnv)
		// 	ev.UpdatePlot()
		// 	fr := tbs.TabByName("Optimal")
		// 	fr.Update()
		case Run:
			tbs.GoUpdatePlot("Train RunAll Plot")
		}
		tbs.SelectTabIndex(idx)
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
			ss.RunStats(mode, level, axon.Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Epoch))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
		tbs.PlotTensorFS(ss.Stats.Dir("Train/RunAll"))
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
	ss.AddStatStd(axon.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle))
	ss.AddStatStd(axon.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle))
	ss.AddStatStd(axon.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial))
	ss.AddStatStd(axon.StatPerTrialMSec(ss.Stats, Train, Trial))

	// up to a point, it is good to use loops over stats in one function,
	// to reduce repetition of boilerplate.
	statNames := []string{"CorSim", "UnitErr", "Err"}
	statDocs := map[string]string{
		"CorSim":  "The correlation-based similarity of the neural activity patterns between the minus and plus phase (1 = patterns are effectively identical). For target layers, this is good continuous, normalized measure of learning performance, which can be more sensitive than thresholded SSE measures.",
		"UnitErr": "Normalized proportion of neurons with activities on the wrong side of 0.5 relative to the target values. This is a good normalized error measure.",
		"Err":     "At the trial level this indicates the presence of an error (i.e., UnitErr > 0), and at higher levels, it is the proportion of errors across the epoch. Thus, when this is zero, the network is performing perfectly (with respect to target outputs).",
	}
	ss.AddStat(func(mode Modes, level Levels, start bool) {
		for _, name := range statNames {
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			subDir := modeDir.Dir((level - 1).String()) // note: will fail for Cycle
			tsr := levelDir.Float64(name)
			ndata := int(ss.Net.Context().NData)
			var stat float64
			if start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					if name != "UnitErr" {
						s.On = true
					}
				})
				metadata.SetDoc(tsr, statDocs[name])
				continue
			}
			switch level {
			case Trial:
				out := ss.Net.LayerByName("Output")
				ltsr := curModeDir.Float64(out.Name+"_ActM", out.Shape.Sizes...)
				for di := range ndata {
					var stat float64
					switch name {
					case "CorSim":
						stat = 1.0 - float64(axon.LayerStates.Value(int(out.Index), int(di), int(axon.LayerPhaseDiff)))
					case "UnitErr":
						stat = out.PctUnitErr(ss.Net.Context())[di]
					case "Err":
						ev := ss.Envs.ByModeDi(mode, di).(*consatenv.ConSatEnv)
						out.UnitValuesSampleTensor(ltsr, "ActM", di)
						stat = ev.OutErr(ltsr)
					}
					curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
					tsr.AppendRowFloat(stat)
				}
			case Run:
				stat = stats.StatFinal.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			default:
				stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			}
		}
	})

	lays := net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.InputLayer, axon.TargetLayer)
	ss.AddStatStd(axon.StatLayerActGe(ss.Stats, net, Train, Trial, Run, lays...))
	ss.AddStatStd(axon.StatLayerGiMult(ss.Stats, net, Train, Epoch, Run, lays...))

	superLays := net.LayersByType(axon.SuperLayer, axon.CTLayer)
	ss.AddStatStd(axon.StatLearnTiming(ss.Stats, ss.Current, net, Trial, Run, superLays...))

	noTarglays := net.LayersByType(axon.SuperLayer, axon.CTLayer)
	pcaFunc := axon.StatPCA(ss.Stats, ss.Current, net, ss.Config.Run.PCAInterval, Train, Trial, Run, noTarglays...)
	ss.AddStat(func(mode Modes, level Levels, start bool) {
		trnEpc := ss.Loops.Loop(Train, Epoch).Counter.Cur
		pcaFunc(mode, level, start, trnEpc)
	})

	// ss.AddStatStd(axon.StatLayerState(ss.Stats, net, Test, Trial, true, "ActM", "Input", "Output"))

	ss.AddStatStd(axon.StatLevelAll(ss.Stats, Train, Run, func(s *plot.Style, cl tensor.Values) {
		name := metadata.Name(cl)
		switch name {
		case "FirstZero", "LastZero":
			s.On = true
			s.Range.SetMin(0)
		}
	}))
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
	statNames := []string{"CorSim", "UnitErr", "Err"}
	if level == Cycle || curModeDir.Node(statNames[0]) == nil {
		return counters
	}
	for _, name := range statNames {
		counters += fmt.Sprintf(" %s: %.4g", name, curModeDir.Float64(name).Float1D(di))
	}
	return counters
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.StopLevel = Trial
	nv := ss.GUI.AddNetView("Network")
	nv.Settings.MaxRecs = 2 * ss.Config.Run.Cycles()
	nv.Settings.Raster.Max = ss.Config.Run.Cycles()
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.TestUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	// ev := ss.Envs.ByModeDi(Train, 0).(*consatenv.ConSatEnv)
	// tbs := ss.GUI.Tabs.AsLab()
	// _, idx := tbs.CurrentTab()
	// plt := plot.New()
	// tbs.Plot("Optimal", plt)
	// ev.Plot = plt
	// ev.MakePlot()
	// tbs.SelectTabIndex(idx)

	ss.StatsInit()
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
