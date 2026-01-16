// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// deepspace simulates deep cerebellar nucleus and deep cortical layer predictive
// learning on spatial updating from the vestibular and visual systems.
package deepspace

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
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
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

	// GUI for viewing env. `display:"-"`
	EnvGUI *emery.GUI

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
	ndata := ss.Config.Run.NData
	var trn *emery.EmeryEnv
	if len(ss.Envs) == 0 {
		trn = &emery.EmeryEnv{}
	} else {
		trn = ss.Envs.ByMode(Train).(*emery.EmeryEnv)
	}

	// note: names must be standard here!
	trn.Defaults()
	trn.Name = Train.String()
	trn.Params.UnitsPer = ss.Config.Env.UnitsPer
	// if ss.Config.Env.Env != nil {
	// 	reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
	// }
	trn.Config(ndata, ss.Config.Run.Cycles(), ss.Root.Dir("Env"), axon.ComputeGPU)
	trn.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.Context().SetISICycles(int32(ss.Config.Run.ISICycles)).
		SetMinusCycles(int32(ss.Config.Run.MinusCycles)).
		SetPlusCycles(int32(ss.Config.Run.PlusCycles)).Update()
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0
	cycles := ss.Config.Run.Cycles()

	ev := ss.Envs.ByMode(Train).(*emery.EmeryEnv)

	full := paths.NewFull()
	full.SelfCon = true // unclear if this makes a diff for self cons at all
	one2one := paths.NewOneToOne()
	_ = one2one
	p1to1 := paths.NewPoolOneToOne()
	_ = p1to1

	space := float32(2)
	// eyeSz := image.Point{2, 1}

	addInput := func(nm string, doc string) (in, mf, thal *axon.Layer) {
		in = net.AddLayer4D(nm, axon.InputLayer, 1, 2, ev.Params.UnitsPer, 1)
		in.AddClass("RateIn")
		in.Doc = "Rate code version. " + doc

		mf = net.AddLayer4D(nm+"MF", axon.InputLayer, ev.Params.TimeBins, 1, 1, ev.Params.PopCodeUnits)
		mf.AddClass("MFIn")
		mf.Doc = "MF mossy fiber input, transient population code. " + doc
		mf.PlaceBehind(in, space)

		thal = net.AddLayer4D(nm+"Thal", axon.InputLayer, ev.Params.TimeBins, 1, 1, ev.Params.PopCodeUnits)
		thal.AddClass("ThalIn")
		thal.Doc = "Thalamic input, integrated population code. " + doc
		thal.PlaceBehind(mf, space)
		return
	}

	rotAct, rotActMF, rotActThal := addInput("Rotate", "Full body horizontal rotation action, population coded left to right with gaussian tuning curves for a range of degrees for each unit (X axis) and redundant units for population code in the Y axis.")

	// rotActPrev, rotActPrevPop := addInput("ActRotatePrev", "Previous trial's version of ActRotate. This should be implicitly maintained but currently is not.")
	// _ = rotActPrevPop

	addInputPulv := func(nm string, doc string) (in, thal, thalP *axon.Layer) {
		in = net.AddLayer4D(nm, axon.InputLayer, 1, 2, ev.Params.UnitsPer, 1)
		in.AddClass("RateIn")
		in.Doc = "Rate code version. " + doc

		thal, thalP = net.AddInputPulv4D(nm+"Thal", ev.Params.TimeBins, 1, 1, ev.Params.PopCodeUnits, space)
		thal.AddClass("ThalIn")
		thalP.AddClass("ThalIn")
		thal.Doc = "Thalamic input, integrated population code. " + doc
		thal.PlaceBehind(in, space)
		return
	}

	vsRotVel, vsRotVelThal, vsRotVelThalP := addInputPulv("VSRotHVel", "Vestibular horizontal rotation velocity, computed from the physics model over time. Population coded left to right with gaussian tuning curves for a range of degrees for each unit (X axis) and redundant units for population code in the Y axis.")
	_ = vsRotVelThalP

	vmRotVel, vmRotVelThal, vmRotVelThalP := addInputPulv("VMRotHVel", "Full-field visual motion computed from the eye using retinal motion filter (see Env tab for visual environment). Population coded left to right with gaussian tuning curves for a range of velocities for each unit (X axis) and redundant units for population code in the Y axis.")

	s1, s1ct := net.AddSuperCT2D("S1", "", 10, 10, space, one2one) // one2one learn > full
	s1.Doc = "Neocortical integrated vestibular and full-field visual motion processing. Does predictive learning on both input signals, more like S2 (secondary), but just using one for simplicity."
	// net.ConnectCTSelf(s1ct, full, "") // self definitely doesn't make sense -- no need for 2-back ct
	// net.LateralConnectLayer(s1ct, full).AddClass("CTSelfMaint") // no diff
	net.ConnectToPulv(s1, s1ct, vsRotVelThalP, full, full, "")
	net.ConnectLayers(rotActThal, s1, full, axon.ForwardPath).AddClass("FFToHid", "FromAct")
	net.ConnectLayers(vsRotVelThal, s1, full, axon.ForwardPath).AddClass("FFToHid")

	// visHid, visHidct := net.AddSuperCT2D("VisHid", "", 10, 10, space, one2one) // one2one learn > full

	// net.ConnectToPulv(visHid, visHidct, vmRotVelp, full, full, "")
	// net.ConnectLayers(rotAct, visHid, full, axon.ForwardPath).AddClass("FFToHid", "FromAct")
	// net.ConnectLayers(vmRotVel, visHid, full, axon.ForwardPath).AddClass("FFToHid")

	net.ConnectToPulv(s1, s1ct, vmRotVelThalP, full, full, "")
	net.ConnectLayers(vmRotVelThal, s1, full, axon.ForwardPath).AddClass("FFToHid")

	// net.ConnectLayers(vsRotVelThal, visHid, full, axon.ForwardPath).AddClass("FFToHid")

	if ev.Params.LeftEye {
		// net.ConnectToPulv(visHidThal, visHidct, eyeLInp, full, full, "")
		// net.ConnectLayers(eyeLInThal, visHid, full, axon.ForwardPath).AddClass("FFToHid")
	}

	// cerebellum:
	// cycles-20 is sufficient to allow time for motor to engage
	ioUp, cniIOUp, cniUp, cneUp := net.AddNuclearCNUp(vsRotVel, rotAct, cycles-20, space)
	_, _ = ioUp, cneUp

	pt := net.ConnectLayers(vsRotVel, cneUp, p1to1, axon.ForwardPath).AddClass("SenseToCNeUp")
	pt.AddDefaultParams(func(pt *axon.PathParams) { pt.SetFixedWts() })

	// net.ConnectLayers(rotActPrev, cniIOUp, p1to1, axon.CNIOPath).AddClass("MFUp", "MFToCNiIOUp")
	// net.ConnectLayers(s1ct, cniIOUp, p1to1, axon.CNIOPath).AddClass("MFUp", "MFToCNiIOUp")
	net.ConnectLayers(rotActMF, cniIOUp, full, axon.CNIOPath).AddClass("MFUp", "MFToCNiIOUp")

	// net.ConnectLayers(rotActPrev, cniUp, p1to1, axon.CNIOPath).AddClass("MFUp", "MFToCNiUp")
	// net.ConnectLayers(s1ct, cniUp, p1to1, axon.CNIOPath).AddClass("MFUp", "MFToCNiUp")
	net.ConnectLayers(rotActMF, cniUp, full, axon.CNIOPath).AddClass("MFUp", "MFToCNiUp")

	// position

	// rotActPrev.PlaceBehind(rotActThal, space)
	vsRotVel.PlaceRightOf(rotAct, float32(ev.Params.PopCodeUnits))
	vmRotVel.PlaceRightOf(vsRotVel, float32(ev.Params.PopCodeUnits))
	// if ev.LeftEye {
	// 	eyeLIn.PlaceRightOf(vmRotVel, space)
	// }
	s1.PlaceAbove(rotAct)

	cniIOUp.PlaceRightOf(s1, space*3)

	// visHid.PlaceRightOf(s1, space)
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
	ss.UpdateEnvGUI(Train)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// NetViewUpdater returns the NetViewUpdate for given mode.
func (ss *Sim) NetViewUpdater(mode enums.Enum) *axon.NetViewUpdate {
	return &ss.TrainUpdate
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

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, Cycle, Trial, Train,
		func(mode enums.Enum) { ss.Net.ClearInputs() },
		func(mode enums.Enum) { ss.TakeNextActions(mode.(Modes)) },
	)
	ls.Stacks[Train].OnInit.Add("Init", ss.Init)
	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	for mode, st := range ls.Stacks {
		st.Loops[Cycle].OnStart.Add("ApplyInputs", func() { ss.ApplyInputs(mode.(Modes)) })
		plusPhase := st.Loops[Cycle].EventByName("MinusPhase:End")
		plusPhase.OnEvent.InsertBefore("PlusPhase:Start", "NextAction", func() bool {
			// note: critical to have this happen *after* MinusPhase:End and *before* PlusPhase:Start
			// because minus phase end has gated info, and plus phase start applies action input
			ss.NextAction(mode.(Modes))
			return false
		})
	}

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.RunName())
	})

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
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
	ev := ss.Envs.ByMode(mode).(*emery.EmeryEnv)
	cyc := ss.Loops.Loop(mode, Cycle).Counter.Cur
	render := cyc%ev.Params.TimeBinCycles == 0
	ev.RenderStates = render
	ev.Step()
	if !render {
		return
	}
	net.InitExt()
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(lnm)
		if !reflectx.IsNil(reflect.ValueOf(pats)) {
			ly.ApplyExtAll(ctx, pats)
		} else {
			// fmt.Println("nil pats:", lnm)
		}
	}
	ss.Net.ApplyExts()
	if cyc == 0 {
		for di := uint32(0); di < ctx.NData; di++ {
			curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), int(di))
		}
	}
}

// NextAction sets next actions.
// Called at end of minus phase.
func (ss *Sim) NextAction(mode Modes) {
	net := ss.Net
	ctx := net.Context()
	ndata := int(ctx.NData)
	ev := ss.Envs.ByMode(mode).(*emery.EmeryEnv)
	for di := 0; di < ndata; di++ {
		ang := 2.0 * (ev.Rand.Float32() - 0.5) * ev.Params.MaxRotate
		ev.NextAction(di, emery.Rotate, ang)
	}
}

// TakeNextActions starts executing actions specified in NextAction.
// This is called at start of trial.
func (ss *Sim) TakeNextActions(mode Modes) {
	ev := ss.Envs.ByMode(mode).(*emery.EmeryEnv)
	ev.TakeNextActions()
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	run := ss.Loops.Loop(Train, Run).Counter.Cur
	ss.InitRandSeed(run)
	ss.Envs.ByMode(Train).Init(run)
	ctx.Reset()
	ss.Net.InitWeights()
	if ss.Config.Run.StartWeights != "" { // this is just for testing -- not usually needed
		ss.Net.OpenWeightsJSON(core.Filename(ss.Config.Run.StartWeights))
		mpi.Printf("Starting with initial weights from: %s\n", ss.Config.Run.StartWeights)
	}
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
	if level < Trial { // < not <=
		return
	}
	ss.RunStats(mode, level-1, Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	// if level == Cycle {
	// 	return
	// }
	ss.RunStats(mode, level, Step)
	tensorfs.DirTable(axon.StatsNode(ss.Stats, mode, level), nil).WriteToLog()
}

// RunStats runs the StatFuncs for given mode, level and phase.
func (ss *Sim) RunStats(mode Modes, level Levels, phase StatsPhase) {
	for _, sf := range ss.StatFuncs {
		sf(mode, level, phase)
	}
	if level > Cycle {
		if phase == Step && ss.GUI.Tabs != nil {
			nm := mode.String() + " " + level.String() + " Plot"
			ss.GUI.Tabs.AsLab().GoUpdatePlot(nm)
			if level == Trial {
				ss.GUI.Tabs.AsLab().GoUpdatePlot("Train Cycle Plot")
			}
		}
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
			// if level == Cycle {
			// 	continue
			// }
			ss.RunStats(mode, level, Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Trial))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Cycle))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
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
	counterFunc := axon.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial)
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

	ss.ConfigStatAdaptFilt()
	ss.ConfigStatVis()
	ss.ConfigStatNuclear()

	lays := net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer, axon.InputLayer, axon.PulvinarLayer)
	actGeFunc := axon.StatLayerActGe(ss.Stats, net, Train, Trial, Run, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		actGeFunc(mode, level, phase == Start)
	})

	pcaFunc := axon.StatPCA(ss.Stats, ss.Current, net, ss.Config.Run.PCAInterval, Train, Trial, Run, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trnEpc := ss.Loops.Loop(Train, Epoch).Counter.Cur
		pcaFunc(mode, level, phase == Start, trnEpc)
	})

	stateFunc := axon.StatLayerState(ss.Stats, net, Test, Trial, true, "ActM", "Depth", "DepthP", "HeadDir", "HeadDirP")
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		stateFunc(mode, level, phase == Start)
	})
}

func (ss *Sim) ConfigStatVis() {
	statNames := []string{"VisVestibCor", "EmeryAng"}
	statDescs := map[string]string{
		"VisVestibCor": "Correlation between the visual motion and vestibular rotation velocity signals, indicating quality of visual motion filters",
		"EmeryAng":     "Emery's current body angle",
	}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		if level < Trial {
			return
		}
		for _, name := range statNames {
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			subDir := modeDir.Dir((level - 1).String()) // note: will fail for Cycle
			tsr := levelDir.Float64(name)
			ndata := int(ss.Net.Context().NData)
			if phase == Start {
				tsr.SetNumRows(0)
				// plot.SetFirstStyler(tsr, func(s *plot.Style) {
				// 	s.On = true
				// })
				metadata.SetDoc(tsr, statDescs[name])
				continue
			}
			switch level {
			case Trial:
				ev := ss.Envs.ByMode(mode).(*emery.EmeryEnv)
				for di := range ndata {
					var stat float64
					switch name {
					case "VisVestibCor":
						stat = ev.VisVestibCorrelCycle(di)
					case "EmeryAng":
						stat = float64(ev.SenseValue(di, emery.VSRotHDir, false)) // current
					}
					curModeDir.Float64(name, ndata).SetFloat1D(float64(stat), di)
					tsr.AppendRowFloat(float64(stat))
				}
			case Run:
				stat := stats.StatFinal.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			default:
				stat := stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			}
		}
	})
}

func (ss *Sim) ConfigStatNuclear() {
	net := ss.Net
	prefix := "VSRotHVel"
	pool := 1 // 0 = layer pool, get first pool
	layerNames := []string{"IO", "CNiIO", "CNiUp", "CNeUp"}
	layers := make([]*axon.Layer, len(layerNames))
	pools := make([]uint32, len(layerNames))
	for li, lnm := range layerNames {
		layers[li] = net.LayerByName(prefix + lnm)
		pools[li] = layers[li].Params.PoolIndex(1) // 4D
	}
	statNames := []string{"IOenv", "IOe", "IOi", "IOioff", "IOerr", "IOspike", "CNiIO", "CNiUp", "CNeUp", "CNeUpGe", "CNeUpGi"}
	statDescs := map[string]string{
		"IOenv":   "IO envelope initiated by action input to IO neurons",
		"IOe":     "Integrated excitatory input to IO",
		"IOi":     "Integrated inhibitory input to IO at the current time",
		"IOioff":  "Integrated inhibitory input to IO offset from TimeOff, which is compared against IOe",
		"IOerr":   "IOe - IOi (positive only): the error signal that drives IO spiking, if above threshold",
		"IOspike": "IO spike, either from IOerr or at end of the IOenv for the baseline spiking",
		"CNiIO":   "integrated activity (CaP) of CNiIO predictive inhibitory input to IO, generates IOi at a temporal offset 'in the future'",
		"CNiUp":   "inhibitory interneuron that projects to CNeUp, learns to inhibit CNeUp just prior to its activation",
		"CNeUp":   "excitatory output, driven directly by excitatory sensory input, which should be cancelled by CNiUp inputs",
		"CNeUpGe": "excitatory conductance into CNeUp, from sensory input",
		"CNeUpGi": "inhibitory conductance into CNeUp, from CNiUp",
	}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		if level != Cycle {
			return
		}
		di := 0
		diu := uint32(di)
		for _, name := range statNames {
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			tsr := levelDir.Float64(name)
			ndata := 1
			if phase == Start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					if name != "CNeUpGe" && name != "CNeUpGi" {
						s.On = true
					}
				})
				metadata.SetDoc(tsr, statDescs[name])
				continue
			}
			var stat float32
			switch name {
			case "IOenv":
				stat = layers[0].AvgMaxVarByPool("TimeCycle", pool, di).Avg
				if stat > 0 {
					stat = 1
				}
			case "IOe":
				stat = layers[0].AvgMaxVarByPool("GaP", pool, di).Avg
			case "IOi":
				stat = layers[0].AvgMaxVarByPool("GaM", pool, di).Avg
			case "IOioff":
				stat = layers[0].AvgMaxVarByPool("GaD", pool, di).Avg
			case "IOerr":
				stat = layers[0].AvgMaxVarByPool("TimeDiff", pool, di).Avg
			case "IOspike":
				stat = layers[0].AvgMaxVarByPool("Spike", pool, di).Avg
			case "CNiIO":
				stat = axon.PoolAvgMax(axon.AMCaP, axon.AMCycle, axon.Avg, pools[1], diu)
			case "CNiUp":
				stat = axon.PoolAvgMax(axon.AMCaP, axon.AMCycle, axon.Avg, pools[2], diu)
			case "CNeUp":
				stat = axon.PoolAvgMax(axon.AMCaP, axon.AMCycle, axon.Avg, pools[3], diu)
			case "CNeUpGe":
				stat = layers[3].AvgMaxVarByPool("Ge", pool, di).Avg
			case "CNeUpGi":
				stat = layers[3].AvgMaxVarByPool("Gi", pool, di).Avg
			}
			curModeDir.Float64(name, ndata).SetFloat1D(float64(stat), di)
			tsr.AppendRowFloat(float64(stat))
		}
	})
}

func (ss *Sim) ConfigStatAdaptFilt() {
	net := ss.Net
	prefix := "VSRotHVel"
	cnely := net.LayerByName(prefix + "CNeUp")
	cnepi := cnely.Params.PoolIndex(0)
	ioly := net.LayerByName(prefix + "IO")
	// iopi := ioly.Params.PoolIndex(0)
	statNames := []string{"CNeUpMax", "IOErrs"}
	statDescs := map[string]string{
		"CNeUpMax": "Maximum activity across the trial for CNeUp Adaptive Filtering layer. Should be around .5 in general",
		"IOErrs":   "Average number of IO error spikes across trials",
	}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		if level < Trial {
			return
		}
		for _, name := range statNames {
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			subDir := modeDir.Dir((level - 1).String()) // note: will fail for Cycle
			tsr := levelDir.Float64(name)
			ndata := int(ss.Net.Context().NData)
			if phase == Start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					s.On = true
				})
				metadata.SetDoc(tsr, statDescs[name])
				continue
			}
			switch level {
			case Trial:
				for di := range ndata {
					var stat float32
					switch name {
					case "CNeUpMax":
						stat = axon.PoolAvgMax(axon.AMCaPMax, axon.AMCycle, axon.Max, cnepi, uint32(di))
					case "IOErrs":
						stat = ioly.AvgMaxVarByPool("TimePeak", 0, di).Avg
					}
					curModeDir.Float64(name, ndata).SetFloat1D(float64(stat), di)
					tsr.AppendRowFloat(float64(stat))
				}
			case Run:
				stat := stats.StatFinal.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			default:
				stat := stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
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
	nv.Options.MaxRecs = 2 * ss.Config.Run.Cycles()
	nv.Options.Raster.Max = ss.Config.Run.Cycles()
	nv.Options.LayerNameSize = 0.03
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Cycle, ss.StatCounters) // Theta
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}
	ss.ConfigNetView(nv)

	evtab, _ := ss.GUI.Tabs.NewTab("Env")
	ev := ss.Envs.ByMode(etime.Train).(*emery.EmeryEnv)
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
	if vu == nil || vu.View == nil || ss.EnvGUI == nil {
		return
	}
	ss.EnvGUI.Update()
}

func (ss *Sim) RunNoGUI() {
	// profile.Profiling = true
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

	// profile.Report(time.Millisecond)

	axon.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
	axon.GPURelease()
}
