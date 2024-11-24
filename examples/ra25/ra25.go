// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ra25 runs a simple random-associator four-layer axon network
// that uses the standard supervised learning paradigm to learn
// mappings between 25 random input / output patterns
// defined over 5x5 input / output layers (i.e., 25 units)
package main

//go:generate core generate -add-types -add-funcs

import (
	"fmt"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/cli"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/vecint"
	"cogentcore.org/core/plot"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tensor/tensorfs"
	"cogentcore.org/core/tree"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
)

func main() {
	opts := cli.DefaultOptions("ra25", "Random associator.")
	opts.DefaultFiles = append(opts.DefaultFiles, "config.toml")
	cfg := &Config{}
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

// see params.go for params

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// Hidden1Size is the size of hidden 1 layer.
	Hidden1Size vecint.Vector2i `default:"{'X':10,'Y':10}" nest:"+"`

	// Hidden2Size is the size of hidden 2 layer.
	Hidden2Size vecint.Vector2i `default:"{'X':10,'Y':10}" nest:"+"`

	// Sheet is the extra params sheet name(s) to use (space separated
	// if multiple). Must be valid name as listed in compiled-in params
	// or loaded params.
	Sheet string

	// Tag is an extra tag to add to file names and logs saved from this run.
	Tag string

	// Note is additional info to describe the run params etc,
	// like a git commit message for the run.
	Note string

	// SaveAll will save a snapshot of all current param and config settings
	// in a directory named params_<datestamp> (or _good if Good is true),
	// then quit. Useful for comparing to later changes and seeing multiple
	// views of current params.
	SaveAll bool `nest:"+"`

	// Good is for SaveAll, save to params_good for a known good params state.
	// This can be done prior to making a new release after all tests are passing.
	// Add results to git to provide a full diff record of all params over level.
	Good bool `nest:"+"`
}

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// GPU uses the GPU for computation, generally faster than CPU even for
	// small models if NData ~16.
	GPU bool `default:"true"`

	// NData is the number of data-parallel items to process in parallel per trial.
	// Is significantly faster for both CPU and GPU.  Results in an effective
	// mini-batch of learning.
	NData int `default:"16" min:"1"`

	// NThreads is the number of parallel threads for CPU computation;
	// 0 = use default.
	NThreads int `default:"0"`

	// Run is the _starting_ run number, which determines the random seed.
	// NRuns counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, NRuns = 1.
	Run int `default:"0"`

	// NRuns is the total number of runs to do when running Train,
	// starting from Run.
	NRuns int `default:"5" min:"1"`

	// NEpochs is the total number of epochs per run.
	NEpochs int `default:"100"`

	// NZero is how many perfect, zero-error epochs before stopping a Run.
	NZero int `default:"2"`

	// NTrials is the total number of trials per epoch.
	// Should be an even multiple of NData.
	NTrials int `default:"32"`

	// TestInterval is how often (in epochs) to run through all the test patterns,
	// in terms of training epochs. Can use 0 or -1 for no testing.
	TestInterval int `default:"5"`

	// PCAInterval is how often (in epochs) to compute PCA on hidden
	// representations to measure variance.
	PCAInterval int `default:"5"`

	// StartWts is the name of weights file to load at start of first run.
	StartWts string
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// SaveWeights will save final weights after each run.
	SaveWeights bool

	// Train has the list of Train mode levels to save log files for.
	Train []string `default:"['Run', 'Epoch']" nest:"+"`

	// Test has the list of Test mode levels to save log files for.
	Test []string `nest:"+"`
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {

	// Includes has a list of additional config files to include.
	// After configuration, it contains list of include files added.
	Includes []string

	// GUI means open the GUI. Otherwise it runs automatically and quits,
	// saving results to log files.
	GUI bool `default:"true"`

	// Debug reports debugging information.
	Debug bool

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }

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
	ss.Net = axon.NewNetwork("RA25")
	ss.Params.Config(LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag)
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	if ss.Config.Run.GPU {
		axon.GPUInit()
		axon.UseGPU = true
	}
	// ss.ConfigPats()
	ss.OpenPats()
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
	var trn, tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(Train).(*env.FixedTable)
		tst = ss.Envs.ByMode(Test).(*env.FixedTable)
	}

	pats := tensorfs.DirTable(ss.Root.RecycleDir("Pats"), nil)

	// this logic can be used to create train-test splits of a set of patterns:
	// n := pats.NumRows()
	// order := rand.Perm(n)
	// ntrn := int(0.85 * float64(n))
	// trnEnv := table.NewView(pats)
	// tstEnv := table.NewView(pats)
	// trnEnv.Indexes = order[:ntrn]
	// tstEnv.Indexes = order[ntrn:]

	// note: names must be standard here!
	trn.Name = Train.String()
	trn.Config(table.NewView(pats))
	trn.Validate()

	tst.Name = Test.String()
	tst.Config(table.NewView(pats))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	inp := net.AddLayer2D("Input", axon.InputLayer, 5, 5)
	hid1 := net.AddLayer2D("Hidden1", axon.SuperLayer, ss.Config.Params.Hidden1Size.Y, ss.Config.Params.Hidden1Size.X)
	hid2 := net.AddLayer2D("Hidden2", axon.SuperLayer, ss.Config.Params.Hidden2Size.Y, ss.Config.Params.Hidden2Size.X)
	out := net.AddLayer2D("Output", axon.TargetLayer, 5, 5)

	// use this to position layers relative to each other
	// hid2.PlaceRightOf(hid1, 2)

	// note: see emergent/path module for all the options on how to connect
	// NewFull returns a new paths.Full connectivity pattern
	full := paths.NewFull()

	net.ConnectLayers(inp, hid1, full, axon.ForwardPath)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	// net.LateralConnectLayerPath(hid1, full, &axon.HebbPath{}).SetType(InhibPath)

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// out.Type = axon.CompareLayer
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

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

	trials := int(math32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))
	cycles := 200
	plusPhase := 50

	ls.AddStack(Train, Trial).
		AddLevel(Run, ss.Config.Run.NRuns).
		AddLevel(Epoch, ss.Config.Run.NEpochs).
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
	trainEpoch.IsDone.AddBool("NZeroStop", func() bool {
		stopNz := ss.Config.Run.NZero
		if stopNz <= 0 {
			return false
		}
		curModeDir := ss.Current.RecycleDir(Train.String())
		curNZero := int(curModeDir.Value("NZero").Float1D(-1))
		stop := curNZero >= stopNz
		return stop
		return false
	})

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
	ndata := int(net.Context().NData)
	curModeDir := ss.Current.RecycleDir(mode.String())
	ev := ss.Envs.ByMode(mode)
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	net.InitExt()
	for di := range ndata {
		ev.Step()
		tensorfs.Value[string](curModeDir, "TrialName", ndata).SetString1D(ev.String(), di)
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
	ss.InitRandSeed(ss.Loops.Loop(Train, Run).Counter.Cur)
	ss.Envs.ByMode(Train).Init(0)
	ss.Envs.ByMode(Test).Init(0)
	ctx.Reset()
	ss.Net.InitWeights()
	if ss.Config.Run.StartWts != "" { // this is just for testing -- not usually needed
		ss.Net.OpenWeightsJSON(core.Filename(ss.Config.Run.StartWts))
		mpi.Printf("Starting with initial weights from: %s\n", ss.Config.Run.StartWts)
	}
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(Test).Init(0)
	ss.Loops.ResetAndRun(Test)
	ss.Loops.Mode = Train // important because this is called from Train Run: go back.
}

////////  Patterns

func (ss *Sim) ConfigPats() {
	dt := table.New()
	metadata.SetName(dt, "TrainPats")
	metadata.SetDoc(dt, "Training patterns")
	dt.AddStringColumn("Name")
	dt.AddFloat32Column("Input", 5, 5)
	dt.AddFloat32Column("Output", 5, 5)
	dt.SetNumRows(25)

	patgen.PermutedBinaryMinDiff(dt.ColumnByIndex(1).Tensor.(*tensor.Float32), 6, 1, 0, 3)
	patgen.PermutedBinaryMinDiff(dt.ColumnByIndex(2).Tensor.(*tensor.Float32), 6, 1, 0, 3)
	dt.SaveCSV("random_5x5_25_gen.tsv", tensor.Tab, table.Headers)

	tensorfs.DirFromTable(ss.Root.RecycleDir("Pats"), dt)
}

func (ss *Sim) OpenPats() {
	dt := table.New()
	metadata.SetName(dt, "TrainPats")
	metadata.SetDoc(dt, "Training patterns")
	errors.Log(dt.OpenCSV("random_5x5_25.tsv", tensor.Tab))
	tensorfs.DirFromTable(ss.Root.RecycleDir("Pats"), dt)
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
	tensorfs.Scalar[string](ss.Current, "RunName").SetString1D(runName, 0)
	return runName
}

// RunName returns the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) RunName() string {
	return tensorfs.Scalar[string](ss.Current, "RunName").String1D(0)
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
		tsr := tensorfs.Value[string](levelDir, name)
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
			trlNm := tensorfs.Value[string](curModeDir, name, ndata).String1D(di)
			tsr.AppendRowString(trlNm)
		}
	})

	// up to a point, it is good to use loops over stats in one function,
	// to reduce repetition of boilerplate.
	statNames := []string{"CorSim", "UnitErr", "Err", "NZero", "FirstZero", "LastZero"}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		for _, name := range statNames {
			if name == "NZero" && (mode != Train || level == Trial) {
				return
			}
			modeDir := ss.Stats.RecycleDir(mode.String())
			curModeDir := ss.Current.RecycleDir(mode.String())
			levelDir := modeDir.RecycleDir(level.String())
			subDir := modeDir.RecycleDir((level - 1).String()) // note: will fail for Cycle
			tsr := tensorfs.Value[float64](levelDir, name)
			ndata := int(ss.Net.Context().NData)
			var stat float64
			if phase == Start {
				tsr.SetNumRows(0)
				if ps := plot.GetStylersFrom(tsr); ps == nil {
					ps.Add(func(s *plot.Style) {
						s.Range.SetMin(0).SetMax(1)
						s.On = true
						switch name {
						case "NZero":
							s.On = false
						case "FirstZero", "LastZero":
							if level < Run {
								s.On = false
							}
						}
					})
					plot.SetStylersTo(tsr, ps)
				}
				switch name {
				case "NZero":
					if level == Epoch {
						tensorfs.Scalar[float64](curModeDir, name).SetFloat1D(0, 0)
					}
				case "FirstZero", "LastZero":
					if level == Epoch {
						tensorfs.Scalar[float64](curModeDir, name).SetFloat1D(-1, 0)
					}
				}
				continue
			}
			switch level {
			case Trial:
				out := ss.Net.LayerByName("Output")
				for di := range ndata {
					var stat float64
					switch name {
					case "CorSim":
						stat = 1.0 - float64(axon.LayerStates.Value(int(out.Index), int(di), int(axon.LayerPhaseDiff)))
					case "UnitErr":
						stat = out.PctUnitErr(ss.Net.Context())[di]
					case "Err":
						uniterr := tensorfs.Value[float64](curModeDir, "UnitErr", ndata).Float1D(di)
						stat = 1.0
						if uniterr == 0 {
							stat = 0
						}
					}
					tensorfs.Value[float64](curModeDir, name, ndata).SetFloat1D(stat, di)
					tsr.AppendRowFloat(stat)
				}
			case Epoch:
				nz := tensorfs.Scalar[float64](curModeDir, "NZero").Float1D(0)
				switch name {
				case "NZero":
					err := stats.StatSum.Call(subDir.Value("Err")).Float1D(0)
					stat = tensorfs.Scalar[float64](curModeDir, name).Float1D(0)
					if err == 0 {
						stat++
					} else {
						stat = 0
					}
					tensorfs.Scalar[float64](curModeDir, name).SetFloat1D(stat, 0)
				case "FirstZero":
					stat = tensorfs.Scalar[float64](curModeDir, name).Float1D(0)
					if stat < 0 && nz == 1 {
						stat = tensorfs.Scalar[float64](curModeDir, "Epoch").Float1D(0)
					}
					tensorfs.Scalar[float64](curModeDir, name).SetFloat1D(stat, 0)
				case "LastZero":
					stat = tensorfs.Scalar[float64](curModeDir, name).Float1D(0)
					if stat < 0 && nz >= float64(ss.Config.Run.NZero) {
						stat = tensorfs.Scalar[float64](curModeDir, "Epoch").Float1D(0)
					}
					tensorfs.Scalar[float64](curModeDir, name).SetFloat1D(stat, 0)
				default:
					stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				}
				tsr.AppendRowFloat(stat)
			case Run:
				switch name {
				case "NZero", "FirstZero", "LastZero":
					stat = subDir.Value(name).Float1D(-1)
				default:
					stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				}
				tsr.AppendRowFloat(stat)
			}
		}
	})

	perTrlFunc := axon.StatPerTrialMSec(ss.Stats, "Err", Train, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		perTrlFunc(mode, level, phase == Start)
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

	stateFunc := axon.StatLayerState(ss.Stats, net, Test, Trial, true, "ActM", "Input", "Output")
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
	counters += fmt.Sprintf(" TrialName: %s", tensorfs.Value[string](curModeDir, "TrialName").String1D(di))
	statNames := []string{"CorSim", "UnitErr", "Err"}
	if level == Cycle || curModeDir.Node(statNames[0]) == nil {
		return counters
	}
	for _, name := range statNames {
		counters += fmt.Sprintf(" %s: %.4g", name, tensorfs.Value[float64](curModeDir, name).Float1D(di))
	}
	return counters
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Axon Random Associator"
	ss.GUI.MakeBody(ss, "ra25", title, `This demonstrates a basic Axon model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.FS = ss.Root
	ss.GUI.DataRoot = "Root"
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Phase, ss.StatCounters)
	ss.TestUpdate.Config(nv, axon.Phase, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level) // todo: carry this all the way through
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.UpdateFiles()
	ss.InitStats()
	ss.GUI.FinalizeGUI(false)
}

// todo: persistent run log
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
			core.TheApp.OpenURL("https://github.com/emer/axon/blob/main/examples/ra25/README.md")
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

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.NRuns, ss.Config.Run.Run)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)

	ss.Loops.Run(Train)

	axon.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
	axon.GPURelease()
}
