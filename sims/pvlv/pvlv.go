// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// pvlv simulates the primary value, learned value model of classical
// conditioning and phasic dopamine the amygdala, ventral striatum
// and associated areas.
package pvlv

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"os"
	"reflect"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/num"
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
	"github.com/emer/axon/v2/sims/pvlv/cond"
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
	Sequence
	Block
	Condition
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
	// if ss.Config.Run.GPU {
	// 	fmt.Println(axon.GPUSystem.Vars().StringDoc())
	// }
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigEnv() {
	var trn *cond.CondEnv
	if len(ss.Envs) == 0 {
		trn = &cond.CondEnv{}
	} else {
		trn = ss.Envs.ByMode(Train).(*cond.CondEnv)
	}
	trn.Name = Train.String()
	if ss.Config.Env.Env != nil {
		reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
	}
	trn.Config(ss.Config.Run.Runs, ss.Config.Env.RunName)
	trn.Init(0)

	ss.ConfigRubicon()
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigRubicon() {
	rp := &ss.Net.Rubicon
	rp.SetNUSs(cond.NUSs, 1) // 1=neg
	rp.Defaults()
	rp.USs.PVposGain = 2
	rp.USs.PVnegGain = 1
	rp.LHb.VSPatchGain = 4        // 4 def -- needs more for shorter trial count here
	rp.LHb.VSPatchNonRewThr = 0.1 // 0.1 def
	rp.USs.USnegGains[0] = 2      // big salient input!
	// note: costs weights are very low by default..

	rp.Urgency.U50 = 50 // no pressure during regular trials
	if ss.Config.Params.Rubicon != nil {
		reflectx.SetFieldsFromMap(rp, ss.Config.Params.Rubicon)
	}
	rp.Update()
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(1)
	net.Context().ThetaCycles = int32(ss.Config.Run.Cycles)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByMode(Train).(*cond.CondEnv)
	ny := ev.NYReps

	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	popY := 4
	popX := 4
	space := float32(2)

	pone2one := paths.NewPoolOneToOne()
	one2one := paths.NewOneToOne()
	_ = one2one
	full := paths.NewFull()
	_ = pone2one

	stim := ev.CurStates["CS"]
	ctxt := ev.CurStates["ContextIn"]

	vSgpi, vSmtxGo, vSmtxNo, vSpatchD1, vSpatchD2, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPos, ofcPosCT, ofcPosPTp, ofcPosPT, ilPos, ilPosCT, ilPosPT, ilPosPTp, ilPosMD, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, accCost, accCostCT, accCostPT, accCostPTp, accCostMD, ilNeg, ilNegCT, ilNegPT, ilNegPTp, ilNegMD, sc := net.AddRubiconOFCus(ny, popY, popX, nuBgY, nuBgX, nuCtxY, nuCtxX, space)
	// note: list all above so can copy / paste and validate correct return values
	_, _, _, _, _, _ = vSgpi, vSmtxGo, vSmtxNo, vSpatchD1, vSpatchD2, urgency
	_, _, _, _, _, _ = usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP
	_, _, _, _ = ilPos, ilPosCT, ilPosPTp, ilPosMD
	_, _, _ = ofcNeg, ofcNegCT, ofcNegPTp
	_, _, _, _ = ilNeg, ilNegCT, ilNegPTp, ilNegMD
	_, _, _, _ = accCost, accCostCT, accCostPTp, accCostMD
	_, _, _, _, _ = ofcPosPT, ofcNegPT, ilPosPT, ilNegPT, accCostPT
	// todo: connect more of above

	time, timeP := net.AddInputPulv4D("Time", 1, cond.MaxTime, ny, 1, space)

	cs, csP := net.AddInputPulv4D("CS", stim.DimSize(0), stim.DimSize(1), stim.DimSize(2), stim.DimSize(3), space)

	ctxIn := net.AddLayer4D("ContextIn", axon.InputLayer, ctxt.DimSize(0), ctxt.DimSize(1), ctxt.DimSize(2), ctxt.DimSize(3))

	//////// CS -> BLA, OFC

	net.ConnectToSC1to1(cs, sc)

	net.ConnectCSToBLApos(cs, blaPosAcq, blaNov)
	net.ConnectToBLAAcq(cs, blaNegAcq, full)
	net.ConnectLayers(cs, vSpatchD1, full, axon.ForwardPath) // these are critical for discriminating A vs. B
	net.ConnectLayers(cs, vSpatchD2, full, axon.ForwardPath)

	// note: context is hippocampus -- key thing is that it comes on with stim
	// most of ctxIn is same as CS / CS in this case, but a few key things for extinction
	// ptpred input is important for learning to make conditional on actual engagement
	net.ConnectToBLAExt(ctxIn, blaPosExt, full)
	net.ConnectToBLAExt(ctxIn, blaNegExt, full)

	// OFCus predicts cs
	net.ConnectToPFCBack(cs, csP, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, full, "CSToPFC")
	net.ConnectToPFCBack(cs, csP, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, full, "CSToPFC")

	//////// OFC predicts time, effort, urgency

	// todo: a more dynamic US rep is needed to drive predictions in OFC

	net.ConnectToPFCBack(time, timeP, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, full, "TimeToPFC")
	net.ConnectToPFCBack(time, timeP, ilPos, ilPosCT, ilPosPT, ilPosPTp, full, "TimeToPFC")

	net.ConnectToPFCBack(time, timeP, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, full, "TimeToPFC")
	net.ConnectToPFCBack(time, timeP, accCost, accCostCT, accCostPT, accCostPTp, full, "TimeToPFC")
	net.ConnectToPFCBack(time, timeP, ilNeg, ilNegCT, ilNegPT, ilNegPTp, full, "TimeToPFC")

	//////// position

	time.PlaceRightOf(pvPos, space*2)
	cs.PlaceRightOf(time, space)
	ctxIn.PlaceRightOf(cs, space)

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
	// ss.ConfigEnv() // always do -- otherwise env params not reset after run
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
	return &ss.TrainUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	cycles := ss.Config.Run.Cycles
	plusPhase := ss.Config.Run.PlusCycles

	// Note: actual max counters set by env
	ls.AddStack(Train, Trial).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Condition, 1).
		AddLevel(Block, 50).
		AddLevel(Sequence, 8).
		AddLevel(Trial, 5).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, cycles-plusPhase, Cycle, Trial, Train)
	ls.Stacks[Train].OnInit.Add("Init", ss.Init)
	ls.AddOnStartToLoop(Trial, "ApplyInputs", func(mode enums.Enum) {
		ss.ApplyInputs(mode.(Modes))
	})

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// UpdateLoopMax gets the latest loop counter Max values from env
func (ss *Sim) UpdateLoopMax() {
	ev := ss.Envs.ByMode(Train).(*cond.CondEnv)
	trn := ss.Loops.Stacks[Train]
	trn.Loops[Condition].Counter.Max = ev.Condition.Max
	trn.Loops[Block].Counter.Max = ev.Block.Max
	trn.Loops[Sequence].Counter.Max = ev.Sequence.Max
	trn.Loops[Trial].Counter.Max = ev.Tick.Max

	if ss.Config.Env.SetNBlocks {
		trn.Loops[Block].Counter.Max = ss.Config.Env.NBlocks
	}
}

// ApplyInputs applies input patterns from given environment for given mode.
// Any other start-of-trial logic can also be put here.
func (ss *Sim) ApplyInputs(mode Modes) {
	net := ss.Net
	ss.Net.InitExt()
	curModeDir := ss.Current.Dir(mode.String())
	ev := ss.Envs.ByMode(mode).(*cond.CondEnv)
	ev.Step()
	ss.UpdateLoopMax()
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(0, pats)
		}
		switch lnm {
		case "CS":
			lpi := ly.Params.PoolIndex(0)
			axon.PoolsInt.Set(num.FromBool[int32](ev.CurTick.CSOn), int(lpi), 0, int(axon.Clamped))
		}
	}
	curModeDir.StringValue("TrialName", 1).SetString1D(ev.String(), 0)
	curModeDir.StringValue("SeqType", 1).SetString1D(ev.SequenceType, 0)
	curModeDir.StringValue("Cond", 1).SetString1D(ev.CondName, 0)
	curModeDir.StringValue("TickType", 1).SetString1D(fmt.Sprintf("%02d_%s", ev.Tick.Prev, ev.CurTick.Type.String()), 0)

	ss.ApplyRubicon(ev, mode, &ev.CurTick)
	net.ApplyExts()
}

// ApplyRubicon applies Rubicon reward inputs.
func (ss *Sim) ApplyRubicon(ev *cond.CondEnv, mode Modes, seq *cond.Sequence) {
	rp := &ss.Net.Rubicon
	di := uint32(0)               // not doing NData here -- otherwise loop over
	rp.NewState(di, &ss.Net.Rand) // first before anything else is updated
	rp.SetGoalMaintFromLayer(di, ss.Net, "ILposPT", 0.3)
	rp.DecodePVEsts(di, ss.Net)
	dist := math32.Abs(float32(3 - ev.Tick.Cur))
	rp.SetGoalDistEst(di, dist)
	rp.EffortUrgencyUpdate(di, 1)
	if seq.USOn {
		if seq.Valence == cond.Pos {
			rp.SetUS(di, axon.Positive, seq.US, seq.USMag)
		} else {
			rp.SetUS(di, axon.Negative, seq.US, seq.USMag) // adds to neg us
		}
	}
	drvs := make([]float32, cond.NUSs)
	drvs[seq.US] = 1
	rp.SetDrives(di, 1, drvs...)
	rp.Step(di, &ss.Net.Rand)
}

// InitEnvRun intializes a new environment run, as when the RunName is changed
// or at NewRun()
func (ss *Sim) InitEnvRun() {
	ev := ss.Envs.ByMode(Train).(*cond.CondEnv)
	ev.RunName = ss.Config.Env.RunName
	ev.Init(0)
	ss.LoadCondWeights(ev.CurRun.Weights) // only if nonempty
	// todo:
	// ss.Loops.ResetCountersBelow(Train, Sequence)
	// ss.Logs.ResetLog(Train, Trial)
	// ss.Logs.ResetLog(Train, Sequence)
}

// LoadRunWeights loads weights specified in current run, if any
func (ss *Sim) LoadRunWeights() {
	ev := ss.Envs.ByMode(Train).(*cond.CondEnv)
	ss.LoadCondWeights(ev.CurRun.Weights) // only if nonempty
}

// LoadCondWeights loads weights saved after named condition, in wts/cond.wts.gz
func (ss *Sim) LoadCondWeights(cond string) {
	if cond == "" {
		return
	}
	wfn := "wts/" + cond + ".wts.gz"
	errors.Log(ss.Net.OpenWeightsJSON(core.Filename(wfn)))
}

// SaveCondWeights saves weights based on current condition, in wts/cond.wts.gz
func (ss *Sim) SaveCondWeights() {
	ev := ss.Envs.ByMode(Train).(*cond.CondEnv)
	cnm, _ := ev.CurRun.Cond(ev.Condition.Cur)
	if cnm == "" {
		return
	}
	wfn := "wts/" + cnm + ".wts.gz"
	err := errors.Log(ss.Net.SaveWeightsJSON(core.Filename(wfn)))
	if err == nil {
		fmt.Printf("Saved weights to: %s\n", wfn)
	}
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	ss.InitRandSeed(ss.Loops.Loop(Train, Run).Counter.Cur)
	ss.InitEnvRun()
	ctx.Reset()
	ss.Net.InitWeights()
	ss.LoadRunWeights()
	ss.UpdateLoopMax()
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
	if level < Trial {
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
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Sequence))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Block))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Trial))
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

	// note: Trial level is not recorded, only the sequence

	// last arg(s) are levels to exclude
	counterFunc := axon.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		counterFunc(mode, level, phase == Start)
	})
	// todo: add Cond
	runNameFunc := axon.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		runNameFunc(mode, level, phase == Start)
	})
	// todo: add SeqType, TickType
	trialNameFunc := axon.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trialNameFunc(mode, level, phase == Start)
	})
	perTrlFunc := axon.StatPerTrialMSec(ss.Stats, Train, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		perTrlFunc(mode, level, phase == Start)
	})

	// trialStats := []string{"Action", "Target", "Correct"}
	// ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
	// 	if level != Trial {
	// 		return
	// 	}
	// 	for si, name := range trialStats {
	// 		modeDir := ss.Stats.Dir(mode.String())
	// 		curModeDir := ss.Current.Dir(mode.String())
	// 		levelDir := modeDir.Dir(level.String())
	// 		di := 0 //
	// 		tsr := levelDir.Float64(name)
	// 		if phase == Start {
	// 			tsr.SetNumRows(0)
	// 			plot.SetFirstStyler(tsr, func(s *plot.Style) {
	// 				s.Range.SetMin(0).SetMax(1)
	// 				if si >= 2 && si <= 5 {
	// 					s.On = true
	// 				}
	// 			})
	// 			continue
	// 		}
	// 		ev := ss.Envs.ByModeDi(mode, di).(*MotorSeqEnv)
	// 		var stat float32
	// 		switch name {
	// 		case "Action":
	// 			stat = float32(ev.CurAction)
	// 		case "Target":
	// 			stat = float32(ev.Target)
	// 		case "Correct":
	// 			stat = num.FromBool[float32](ev.Correct)
	// 		}
	// 		curModeDir.Float32(name, 1).SetFloat1D(float64(stat), di)
	// 		tsr.AppendRowFloat(float64(stat))
	// 	}
	// })
	//
	// seqStats := []string{"NCorrect", "Rew", "RewPred", "RPE", "RewEpc"}
	// ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
	// 	if level <= Trial {
	// 		return
	// 	}
	// 	for _, name := range seqStats {
	// 		modeDir := ss.Stats.Dir(mode.String())
	// 		curModeDir := ss.Current.Dir(mode.String())
	// 		levelDir := modeDir.Dir(level.String())
	// 		subDir := modeDir.Dir((level - 1).String()) // note: will fail for Cycle
	// 		tsr := levelDir.Float64(name)
	// 		ndata := int(ss.Net.Context().NData)
	// 		var stat float64
	// 		if phase == Start {
	// 			tsr.SetNumRows(0)
	// 			plot.SetFirstStyler(tsr, func(s *plot.Style) {
	// 				s.Range.SetMin(0).SetMax(1)
	// 				s.On = true
	// 			})
	// 			continue
	// 		}
	// 		switch level {
	// 		case Trial:
	// 			curModeDir.Float32(name, ndata).SetFloat1D(float64(stat), di)
	// 			tsr.AppendRowFloat(float64(stat))
	// 		default:
	// 			stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
	// 			tsr.AppendRowFloat(stat)
	// 		}
	// 	}
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
	strNames := []string{"Cond", "TrialName", "SeqType", "TickType"}
	for _, name := range strNames {
		counters += fmt.Sprintf(" %s: %s", name, curModeDir.StringValue(name).String1D(di))
	}
	statNames := []string{"DA", "RewPred"}
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
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.StopLevel = Trial
	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 2 * ss.Config.Run.Cycles
	nv.Options.Raster.Max = ss.Config.Run.Cycles
	nv.Options.LayerNameSize = 0.02
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.4, 2.6)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

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
	axon.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Train})

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.Runs, ss.Config.Run.Run)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)

	ss.Loops.Run(Train)

	axon.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
	axon.GPURelease()
}
