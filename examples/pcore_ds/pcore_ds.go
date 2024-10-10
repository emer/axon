// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
pcore_ds: This project simulates the inhibitory dynamics in the STN and GPe leading to integration of Go vs. NoGo signal in the basal ganglia, for the Dorsal Striatum (DS) motor action case.
*/
package main

//go:generate core generate -add-types

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"strconv"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/num"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/plot/plotcore"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/split"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tree"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/ecmd"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		sim.RunGUI()
	} else if sim.Config.Params.Tweak {
		sim.RunParamTweak()
	} else {
		sim.RunNoGUI()
	}
}

// see params.go for network params, config.go for Config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config `new-window:"+"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *axon.Network `new-window:"+" display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Manager `new-window:"+" display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats `new-window:"+"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `new-window:"+"`

	// Environments
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// axon timing parameters and state
	Context axon.Context `new-window:"+"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"add-fields"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = axon.NewNetwork("PCore")
	econfig.Config(&ss.Config, "config.toml")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
	ss.Context.ThetaCycles = int32(ss.Config.Run.ThetaCycles)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	newEnv := (len(ss.Envs) == 0)

	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn, tst *MotorSeqEnv
		if newEnv {
			trn = &MotorSeqEnv{}
			tst = &MotorSeqEnv{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*MotorSeqEnv)
			tst = ss.Envs.ByModeDi(etime.Test, di).(*MotorSeqEnv)
		}

		// note: names must be standard here!
		trn.Name = env.ModeDi(etime.Train, di)
		trn.Defaults()
		trn.NActions = ss.Config.Env.NActions
		trn.SeqLen = ss.Config.Env.SeqLen
		if ss.Config.Env.Env != nil {
			params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
		}
		trn.Config(etime.Train, 73+int64(di)*73)

		tst.Name = env.ModeDi(etime.Test, di)
		tst.Defaults()
		tst.NActions = ss.Config.Env.NActions
		tst.SeqLen = ss.Config.Env.SeqLen
		if ss.Config.Env.Env != nil {
			params.ApplyMap(tst, ss.Config.Env.Env, ss.Config.Debug)
		}
		tst.Config(etime.Test, 181+int64(di)*181)

		trn.Init(0)
		tst.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn, tst)
		if di == 0 {
			ss.ConfigRubicon(trn)
		}
	}
}

func (ss *Sim) ConfigRubicon(trn *MotorSeqEnv) {
	pv := &ss.Net.Rubicon
	pv.SetNUSs(&ss.Context, 2, 1)
	pv.Urgency.U50 = 20 // 20 def
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*MotorSeqEnv)

	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	np := 1
	nuPer := ev.NUnitsPer
	nAct := ev.NActions
	nuX := 6
	nuY := 6
	nuCtxY := 6
	nuCtxX := 6
	space := float32(2)

	p1to1 := paths.NewPoolOneToOne()
	one2one := paths.NewOneToOne()
	_ = one2one
	full := paths.NewFull()
	_ = full
	mtxRandPath := paths.NewUniformRand()
	mtxRandPath.PCon = 0.5
	_ = mtxRandPath

	mtxGo, mtxNo, gpePr, gpeAk, stn, gpi, pf := net.AddDBG("", 1, nAct, nuY, nuX, nuY, nuX, space)
	_, _ = gpePr, gpeAk

	snc := net.AddLayer2D("SNc", axon.InputLayer, 1, 1)
	_ = snc

	state := net.AddLayer4D("State", axon.InputLayer, 1, np, nuPer, nAct)
	s1 := net.AddLayer4D("S1", axon.InputLayer, 1, np, nuPer, nAct+1)

	targ := net.AddLayer2D("Target", axon.InputLayer, nuPer, nAct) // Target: just for vis

	motor := net.AddLayer4D("MotorBS", axon.TargetLayer, 1, nAct, nuPer, 1)
	pf.Shape.CopyShape(&motor.Shape)

	vl := net.AddPulvLayer4D("VL", 1, nAct, nuPer, 1) // VL predicts brainstem Action
	vl.SetBuildConfig("DriveLayName", motor.Name)

	// bool before space is selfmaint or not: selfcons much better (false)
	m1, m1CT, m1PT, m1PTp, m1VM := net.AddPFC2D("M1", "VM", nuCtxY, nuCtxX, false, false, space)
	_ = m1PT
	// todo: M1PTp should be VL interconnected, prior to PT, not after it.

	// vl is a predictive thalamus but we don't have direct access to its source
	net.ConnectToPFC(nil, vl, m1, m1CT, m1PT, m1PTp, full, "VLM1") // m1 predicts vl

	// these pathways are *essential* -- must get current state here
	net.ConnectLayers(m1, vl, full, axon.ForwardPath).AddClass("ToVL")

	net.ConnectLayers(gpi, motor, p1to1, axon.InhibPath)
	net.ConnectLayers(m1PT, motor, full, axon.ForwardPath).AddClass("M1ToMotorBS")
	// net.ConnectLayers(m1PTp, motor, full, axon.ForwardPath).AddClass("M1ToMotorBS")
	net.ConnectLayers(m1, motor, full, axon.ForwardPath).AddClass("M1ToMotorBS")

	net.ConnectLayers(motor, pf, one2one, axon.ForwardPath)

	net.ConnectLayers(state, stn, full, axon.ForwardPath).AddClass("ToDSTN")
	net.ConnectLayers(state, m1, full, axon.ForwardPath).AddClass("ToM1")
	net.ConnectLayers(s1, stn, full, axon.ForwardPath).AddClass("ToDSTN")
	net.ConnectLayers(s1, m1, full, axon.ForwardPath).AddClass("ToM1")

	net.ConnectLayers(gpi, m1VM, full, axon.InhibPath).AddClass("DBGInhib")

	mtxGo.SetBuildConfig("ThalLay1Name", m1VM.Name)
	mtxNo.SetBuildConfig("ThalLay1Name", m1VM.Name)

	toMtx := full
	// toMtx := mtxRandPath // works, but not as reliably
	net.ConnectToDSMatrix(state, mtxGo, toMtx).AddClass("StateToMtx")
	net.ConnectToDSMatrix(state, mtxNo, toMtx).AddClass("StateToMtx")
	net.ConnectToDSMatrix(s1, mtxNo, toMtx).AddClass("S1ToMtx")
	net.ConnectToDSMatrix(s1, mtxGo, toMtx).AddClass("S1ToMtx")

	net.ConnectToDSMatrix(m1, mtxGo, toMtx).AddClass("M1ToMtx")
	net.ConnectToDSMatrix(m1, mtxNo, toMtx).AddClass("M1ToMtx")

	// note: just using direct pathways here -- theoretically through CL
	// not working! -- need to make these modulatory in the right way.
	// net.ConnectToDSMatrix(motor, mtxGo, p1to1).AddClass("CLToMtx")
	// net.ConnectToDSMatrix(motor, mtxNo, p1to1).AddClass("CLToMtx")

	pf.PlaceRightOf(gpi, space)
	snc.PlaceBehind(stn, space)
	m1VM.PlaceRightOf(pf, space)
	vl.PlaceRightOf(m1VM, space)
	motor.PlaceBehind(vl, space)
	targ.PlaceBehind(motor, space)

	gpeAk.PlaceBehind(gpePr, space)
	stn.PlaceRightOf(gpePr, space)
	mtxGo.PlaceAbove(gpi)
	mtxNo.PlaceBehind(mtxGo, space)
	state.PlaceAbove(mtxGo)
	s1.PlaceRightOf(state, space)

	m1.PlaceRightOf(s1, space)
	m1PT.PlaceRightOf(m1, space)

	net.Build(ctx)
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWeights(ctx)
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll() // first hard-coded defaults
	if ss.Config.Params.Network != nil {
		ss.Params.SetNetworkMap(ss.Net, ss.Config.Params.Network)
	}
	// compensate for expected activity levels based on max seq len
	lnms := []string{"State", "S1", "MotorBS", "VL"}
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*MotorSeqEnv)
	for _, lnm := range lnms {
		ly := ss.Net.LayerByName(lnm)
		// fmt.Println(ly.Params.Inhib.ActAvg.Nominal)
		ly.Params.Inhib.ActAvg.Nominal = 0.5 / float32(ev.NActions)
	}
}

func (ss *Sim) TurnOffTheNoise() {
	return // not doing this now -- not better
	mtxGo := ss.Net.LayerByName("MtxGo")
	if mtxGo.Params.Acts.Noise.On.IsFalse() {
		return
	}
	ss.Params.SetAllSheet("NoiseOff")
	ss.Net.GPU.SyncParamsToGPU()
	fmt.Println("Turned noise off")
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if ss.Config.GUI {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
	ss.ConfigEnv() // note: critical -- must reset rewpred etc.
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	ev := ss.Envs.ByModeDi(etime.Train, 0).(*MotorSeqEnv)
	trls := int(math32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	nSeqTrials := ev.SeqLen + 1 // 1 reward at end

	nCycles := ss.Config.Run.ThetaCycles
	plusCycles := 50

	man.AddStack(etime.Train).
		AddTime(etime.Expt, 1).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Sequence, trls, ss.Config.Run.NData).
		AddTime(etime.Trial, nSeqTrials).
		AddTime(etime.Cycle, nCycles)

	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTimeIncr(etime.Sequence, trls, ss.Config.Run.NData).
		AddTime(etime.Trial, nSeqTrials).
		AddTime(etime.Cycle, nCycles)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, nCycles-plusCycles, nCycles-1) // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate)         // std algo code

	for m, _ := range man.Stacks {
		stack := man.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			seq := man.Stacks[m].Loops[etime.Sequence].Counter.Cur
			trial := man.Stacks[m].Loops[etime.Trial].Counter.Cur
			ss.ApplyInputs(m, seq, trial)
		})
	}
	// note: auto applies to all
	plusPhase := man.Stacks[etime.Train].Loops[etime.Cycle].EventByName("PlusPhase")
	plusPhase.OnEvent.InsertBefore("PlusPhase:Start", "TakeAction", func() {
		// note: critical to have this happen *after* MinusPhase:End and *before* PlusPhase:Start
		// because minus phase end has gated info, and plus phase start applies action input
		ss.TakeAction(ss.Net)
	})

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	man.GetLoop(etime.Train, etime.Epoch).IsDone["StopCrit"] = func() bool {
		rew := ss.Stats.Float("RewEpc")
		stop := rew >= 0.98
		return stop
	}

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs, etime.Sequence)

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintValues([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.Stats.String("RunName"))
	})

	// man.GetLoop(etime.Train, etime.Run).Main.Add("TestAll", func() {
	// 	ss.Loops.Run(etime.Test)
	// })

	////////////////////////////////////////////
	// GUI

	if !ss.Config.GUI {
		if ss.Config.Log.NetData {
			man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
				ss.GUI.NetDataRecord(ss.ViewUpdate.Text)
			})
		}
	} else {
		axon.LooperUpdateNetView(man, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
		axon.LooperUpdatePlots(man, &ss.GUI)
	}

	if ss.Config.Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(mode etime.Modes, seq, trial int) {
	ctx := &ss.Context
	net := ss.Net
	ss.Net.InitExt(ctx)

	lays := []string{"State", "S1", "Target", "SNc"}
	states := []string{"State", "PrevAction", "Target", "SNc"}

	for di := 0; di < ss.Config.Run.NData; di++ {
		ev := ss.Envs.ByModeDi(mode, di).(*MotorSeqEnv)
		inRew := ev.IsRewTrial()
		ev.Step()
		for li, lnm := range lays {
			snm := states[li]
			ly := net.LayerByName(lnm)
			itsr := ev.State(snm)
			ly.ApplyExt(ctx, uint32(di), itsr)
		}
		ss.ApplyRubicon(ev, uint32(di), inRew)
	}
	net.ApplyExts(ctx) // now required for GPU mode
}

// ApplyRubicon applies Rubicon reward inputs
func (ss *Sim) ApplyRubicon(ev *MotorSeqEnv, di uint32, inRew bool) {
	ctx := &ss.Context
	pv := &ss.Net.Rubicon
	pv.EffortUrgencyUpdate(ctx, di, 1)
	pv.Urgency.Reset(ctx, di)

	if inRew {
		axon.GlobalScalars[axon.GvACh, di] = 1
		ss.SetRew(ev.RPE, di)
	} else {
		axon.GlobalSetRew(ctx, di, 0, false) // no rew
		axon.GlobalScalars[axon.GvACh, di] = 0
	}
}

func (ss *Sim) SetRew(rew float32, di uint32) {
	ctx := &ss.Context
	pv := &ss.Net.Rubicon
	axon.GlobalSetRew(ctx, di, rew, true)
	axon.GlobalScalars[axon.GvDA, di] = rew // reward prediction error
	if rew > 0 {
		pv.SetUS(ctx, di, axon.Positive, 0, 1)
	} else if rew < 0 {
		pv.SetUS(ctx, di, axon.Negative, 0, 1)
	}
}

// TakeAction takes action for this step, using decoded cortical action.
// Called at end of minus phase.
func (ss *Sim) TakeAction(net *axon.Network) {
	ctx := &ss.Context
	for di := 0; di < ss.Config.Run.NData; di++ {
		ev := ss.Envs.ByModeDi(ctx.Mode, di).(*MotorSeqEnv)
		if !ev.IsRewTrialPostStep() {
			netAct := ss.DecodeAct(ev, di)
			ev.Action(fmt.Sprintf("%d", netAct), nil)
			ss.ApplyAction(di)
		}
	}
	ss.Net.ApplyExts(ctx) // required!
	ss.Net.GPU.SyncPoolsToGPU()
}

// DecodeAct decodes the VL ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ev *MotorSeqEnv, di int) int {
	vt := ss.Stats.SetLayerTensor(ss.Net, "MotorBS", "CaSpkPM", di)
	return ss.SoftMaxChoose4D(vt)
	// return ss.HardChoose4D(vt)
}

// SoftMaxChoose2D probabalistically selects column with most activity in layer,
// using a softmax with Config.Env.ActSoftMaxGain gain factor
func (ss *Sim) SoftMaxChoose2D(vt *tensor.Float32) int {
	dy := vt.DimSize(0)
	nact := vt.DimSize(1)
	var tot float32
	probs := make([]float32, nact)
	for i := range probs {
		var sum float32
		for j := 0; j < dy; j++ {
			sum += vt.Value([]int{j, i})
		}
		p := math32.FastExp(ss.Config.Env.ActSoftMaxGain * sum)
		probs[i] = p
		tot += p
	}
	for i, p := range probs {
		probs[i] = p / tot
	}
	chs := randx.PChoose32(probs)
	return chs
}

// SoftMaxChoose4D probabalistically selects column with most activity in layer,
// using a softmax with Config.Env.ActSoftMaxGain gain factor
func (ss *Sim) SoftMaxChoose4D(vt *tensor.Float32) int {
	nact := vt.DimSize(1)
	nuY := vt.DimSize(2)
	nuX := vt.DimSize(3)
	var tot float32
	probs := make([]float32, nact)
	for i := range probs {
		var sum float32
		for j := 0; j < nuY; j++ {
			for k := 0; k < nuX; k++ {
				sum += vt.Value([]int{0, i, j, k})
			}
		}
		p := math32.FastExp(ss.Config.Env.ActSoftMaxGain * sum)
		probs[i] = p
		tot += p
	}
	for i, p := range probs {
		probs[i] = p / tot
		// fmt.Println(i, p, probs[i])
	}
	chs := randx.PChoose32(probs)
	return chs
}

// HardChoose2D deterministically selects column with most activity in layer,
func (ss *Sim) HardChoose2D(vt *tensor.Float32) int {
	nact := vt.DimSize(1)
	nuY := vt.DimSize(2)
	nuX := vt.DimSize(3)
	var mx float32
	var mxi int
	for i := 0; i < nact; i++ {
		var sum float32
		for j := 0; j < nuY; j++ {
			for k := 0; k < nuX; k++ {
				sum += vt.Value([]int{0, i, j, k})
			}
		}
		if sum > mx {
			mx = sum
			mxi = i
		}
	}
	return mxi
}

// HardChoose4D deterministically selects column with most activity in layer,
func (ss *Sim) HardChoose4D(vt *tensor.Float32) int {
	nact := vt.DimSize(1)
	nuY := vt.DimSize(2)
	nuX := vt.DimSize(3)
	var mx float32
	var mxi int
	for i := 0; i < nact; i++ {
		var sum float32
		for j := 0; j < nuY; j++ {
			for k := 0; k < nuX; k++ {
				sum += vt.Value([]int{0, i, j, k})
			}
		}
		if sum > mx {
			mx = sum
			mxi = i
		}
	}
	return mxi
}

func (ss *Sim) ApplyAction(di int) {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByModeDi(ss.Context.Mode, di).(*MotorSeqEnv)
	ap := ev.State("Action")
	ly := net.LayerByName("MotorBS")
	ly.ApplyExt(ctx, uint32(di), ap)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	for di := 0; di < int(ctx.NData); di++ {
		ss.Envs.ByModeDi(etime.Train, di).Init(0)
		ss.Envs.ByModeDi(etime.Test, di).Init(0)
	}
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights(ctx)
	ss.InitStats()
	ss.StatCounters(0)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("Action", 0)
	ss.Stats.SetFloat("Target", 0)
	ss.Stats.SetFloat("Correct", 0)
	ss.Stats.SetFloat("NCorrect", 0)
	ss.Stats.SetFloat("Rew", 0)
	ss.Stats.SetFloat("RewPred", 0)
	ss.Stats.SetFloat("RPE", 0)
	ss.Stats.SetFloat("RewEpc", 0)
	ss.Stats.SetFloat("EpochsToCrit", math.NaN())
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters(di int) {
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl+di)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ss.Stats.SetString("TrialName", "")
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	di := ss.ViewUpdate.View.Di
	if tm == etime.Trial {
		ss.TrialStats(di) // get trial stats for current di
	}
	ss.StatCounters(di)
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Sequence", "Trial", "Di", "TrialName", "Cycle", "Action", "Target", "Correct", "Rew", "RewPred", "RPE"})
}

// TrialStats records the trial-level statistics
func (ss *Sim) TrialStats(di int) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*MotorSeqEnv)
	ss.Stats.SetFloat32("Action", float32(ev.CurAction))
	ss.Stats.SetFloat32("Target", float32(ev.Target))
	ss.Stats.SetFloat32("Correct", num.FromBool[float32](ev.Correct))
	ss.Stats.SetFloat32("NCorrect", float32(ev.PrevNCorrect))
	ss.Stats.SetFloat32("RewPred", ev.RewPred)
	ss.Stats.SetFloat32("Rew", ev.Rew)
	ss.Stats.SetFloat32("RPE", ev.RPE)
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Expt, etime.Run, etime.Epoch, etime.Sequence, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Sequence, "TrialName")
	ss.Logs.AddStatStringItem(etime.Test, etime.Sequence, "TrialName")

	ss.Logs.AddStatAggItem("NCorrect", etime.Expt, etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("RewPred", etime.Expt, etime.Run, etime.Epoch, etime.Sequence)
	li := ss.Logs.AddStatAggItem("Rew", etime.Expt, etime.Run, etime.Epoch, etime.Sequence)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("RPE", etime.Expt, etime.Run, etime.Epoch, etime.Sequence)
	li.FixMin = false
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Sequence)

	ss.Logs.AddItem(&elog.Item{
		Name:  "EpochsToCrit",
		Type:  reflect.Float64,
		Range: minmax.F32{Min: -1},
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
				elg := ss.Logs.Table(ctx.Mode, etime.Epoch)
				epc := int(elg.Float("Epoch", elg.Rows-1))
				etc := math.NaN()
				if epc < ss.Config.Run.NEpochs-1 {
					etc = float64(epc)
				}
				ss.Stats.SetInt("EpochsToCrit", ss.Stats.Int("Epoch")) // only set if makes criterion
				ctx.SetFloat64(etc)
			},
			etime.Scope(etime.Train, etime.Expt): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Run, stats.Mean)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:  "NToCrit",
		Type:  reflect.Float64,
		Range: minmax.F32{Min: -1},
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Expt): func(ctx *elog.Context) {
				ix := ss.Logs.IndexView(etime.Train, etime.Run)
				ix.Filter(func(et *table.Table, row int) bool {
					return !math.IsNaN(et.Float("EpochsToCrit", row))
				})
				ctx.SetInt(len(ix.Indexes))
			}}})

	// axon.LogAddDiagnosticItems(&ss.Logs, ss.Net, etime.Epoch, etime.Trial)

	ss.Logs.PlotItems("NCorrect", "RewPred", "RPE", "Rew")

	ss.Logs.CreateTables()

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	// ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	ss.Logs.NoPlot(etime.Test, etime.Expt)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	// ss.Logs.SetMeta(etime.Test, etime.Cycle, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode != etime.Analyze {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
	}

	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
		// row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		return // skip
	case time == etime.Sequence:
		for di := 0; di < ss.Config.Run.NData; di++ {
			ss.TrialStats(di)
			ss.StatCounters(di)
			ss.Logs.LogRowDi(mode, time, row, di)
		}
		// ss.TurnOffTheNoise()
		return // don't do reg
	case time == etime.Epoch && mode == etime.Test:
		ss.TestStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc

	if time == etime.Epoch && mode == etime.Train {
		rew := dt.Float("Rew", dt.Rows-1)
		ss.Stats.SetFloat("RewEpc", rew) // used for stopping criterion
	}
}

func (ss *Sim) TestStats() {
	tststnm := "TestTrialStats"
	ix := ss.Logs.IndexView(etime.Test, etime.Sequence)
	spl := split.GroupBy(ix, "TrialName")
	for _, ts := range ix.Table.ColumnNames {
		if ts == "TrialName" {
			continue
		}
		split.AggColumn(spl, ts, stats.Mean)
	}
	tstst := spl.AggsToTable(table.ColumnNameOnly)
	tstst.SetMetaData("precision", strconv.Itoa(elog.LogPrec))
	ss.Logs.MiscTables[tststnm] = tstst

	if ss.Config.GUI {
		plt := ss.GUI.Plots[etime.ScopeKey(tststnm)]
		plt.SetTable(tstst)
		plt.Options.XAxis = "Sequence"
		plt.SetColumnOptions("NCorrect", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
		plt.SetColumnOptions("Rew", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
		plt.SetColumnOptions("RPE", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
		plt.GoUpdatePlot()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "PCore DS Test"
	ss.GUI.MakeBody(ss, "pcore", title, `This project simulates the Dorsal Basal Ganglia, starting with the Dorsal Striatum, centered on the Pallidum Core (GPe) areas that drive disinhibitory motor plan selection in descending motor pathways. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 400
	nv.Options.Raster.Max = ss.Config.Run.ThetaCycles
	nv.Options.LayerNameSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.0, 2.5)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, -0.03, 0.02), math32.Vec3(0, 1, 0))

	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	tststnm := "TestTrialStats"
	tstst := ss.Logs.MiscTable(tststnm)
	ttp, _ := ss.GUI.Tabs.NewTab(tststnm + " Plot")
	plt := plotcore.NewSubPlot(ttp)
	ss.GUI.Plots[etime.ScopeKey(tststnm)] = plt
	plt.Options.Title = tststnm
	plt.Options.XAxis = "Trial"
	plt.SetTable(tstst)

	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		// vgpu.Debug = ss.Config.Debug
		ss.Net.ConfigGPUnoGUI(&ss.Context) // must happen after gui or no gui
		core.TheApp.AddQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
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

	ss.GUI.AddLooperCtrl(p, ss.Loops, []etime.Modes{etime.Train, etime.Test})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "TestInit", Icon: icons.Update,
		Tooltip: "reinitialize the testing control so it re-runs.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Loops.ResetCountersByMode(etime.Test)
			ss.GUI.UpdateWindow()
		},
	})

	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    icons.Reset,
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/emer/axon/blob/main/examples/pcore/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}

func (ss *Sim) RunNoGUI() {
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Train, etime.Trial, "trl", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Expt, etime.Train, etime.Expt, "expt", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.TestTrial, etime.Test, etime.Trial, "tst_trl", netName, runName)

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	ss.Init()

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.NRuns, ss.Config.Run.Run)
	ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	if ss.Config.Log.TestEpoch {
		dt := ss.Logs.MiscTable("TestTrialStats")
		fnm := ecmd.LogFilename("tst_epc", netName, runName)
		dt.SaveCSV(core.Filename(fnm), table.Tab, table.Headers)
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
