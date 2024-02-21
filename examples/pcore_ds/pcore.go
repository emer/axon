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
	"strconv"

	"cogentcore.org/core/gi"
	"cogentcore.org/core/glop/num"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/mat32"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/ecmd"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/empi/v2/mpi"
	"github.com/emer/etable/v2/agg"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/etable/v2/minmax"
	"github.com/emer/etable/v2/split"
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
	Config Config

	// the network -- click to view / edit parameters for layers, prjns, etc
	Net *axon.Network `view:"no-inline"`

	// all parameter management
	Params emer.NetParams `view:"inline"`

	// contains looper control loops for running sim
	Loops *looper.Manager `view:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats

	// Contains all the logs and information about the logs.'
	Logs elog.Logs

	// Environments
	Envs env.Envs `view:"no-inline"`

	// axon timing parameters and state
	Context axon.Context

	// netview update parameters
	ViewUpdt netview.ViewUpdt `view:"inline"`

	// manages all the gui elements
	GUI egui.GUI `view:"-"`

	// a list of random seeds to use for each run
	RndSeeds erand.Seeds `view:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	econfig.Config(&ss.Config, "config.toml")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.InitRndSeed(0)
	ss.Context.Defaults()
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
		trn.Nm = env.ModeDi(etime.Train, di)
		trn.Defaults()
		if ss.Config.Env.Env != nil {
			params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
		}
		trn.Config(etime.Train, 73+int64(di)*73)
		trn.Validate()

		tst.Nm = env.ModeDi(etime.Test, di)
		tst.Defaults()
		if ss.Config.Env.Env != nil {
			params.ApplyMap(tst, ss.Config.Env.Env, ss.Config.Debug)
		}
		tst.Config(etime.Test, 181+int64(di)*181)
		tst.Validate()

		trn.Init(0)
		tst.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn, tst)
		if di == 0 {
			ss.ConfigPVLV(trn)
		}
	}
}

func (ss *Sim) ConfigPVLV(trn *MotorSeqEnv) {
	pv := &ss.Net.PVLV
	pv.SetNUSs(&ss.Context, 2, 1)
	pv.Urgency.U50 = 20 // 20 def
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*MotorSeqEnv)

	net.InitName(net, "PCore")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	np := 1
	nuPer := ev.NUnitsPer
	nAct := ev.MaxSeqLen
	nuX := 5
	nuY := 5
	nuCtxY := 5
	nuCtxX := 5
	space := float32(2)

	p1to1 := prjn.NewPoolOneToOne()
	one2one := prjn.NewOneToOne()
	_ = one2one
	full := prjn.NewFull()
	_ = full
	mtxRndPrjn := prjn.NewPoolUnifRnd()
	mtxRndPrjn.PCon = 0.5
	_ = mtxRndPrjn

	mtxGo, mtxNo, gpePr, gpeAk, stn, gpi := net.AddBG4D("", 1, nAct, nuY, nuX, nuY, nuX, space)
	_, _ = gpePr, gpeAk

	snc := net.AddLayer2D("SNc", 1, 1, axon.InputLayer)
	_ = snc

	state := net.AddLayer4D("State", 1, np, nuPer, nAct, axon.InputLayer)
	s1 := net.AddLayer4D("S1", 1, np, nuPer, nAct, axon.InputLayer)

	targ := net.AddLayer2D("Target", nuPer, nAct, axon.InputLayer) // Target: just for vis

	motor := net.AddLayer4D("MotorBS", 1, nAct, nuPer, 1, axon.TargetLayer)
	pf := net.AddLayer4D("PF", 1, nAct, nuPer, 1, axon.SuperLayer)
	net.ConnectLayers(motor, pf, one2one, axon.ForwardPrjn)

	vl := net.AddPulvLayer4D("VL", 1, nAct, nuPer, 1) // VL predicts brainstem Action
	vl.SetBuildConfig("DriveLayName", motor.Name())

	m1, m1CT, m1PT, m1PTp, m1VM := net.AddPFC2D("M1", "VM", nuCtxY, nuCtxX, false, space)
	_ = m1PT
	// todo: M1PTp should be VL interconnected, prior to PT, not after it.

	// vl is a predictive thalamus but we don't have direct access to its source
	net.ConnectToPFC(nil, vl, m1, m1CT, m1PTp, full) // m1 predicts vl

	// these projections are *essential* -- must get current state here
	net.ConnectLayers(m1, vl, full, axon.ForwardPrjn).SetClass("ToVL")

	net.ConnectLayers(gpi, motor, p1to1, axon.InhibPrjn)
	net.ConnectLayers(m1PT, motor, full, axon.ForwardPrjn).SetClass("M1ToMotorBS")
	net.ConnectLayers(m1, motor, full, axon.ForwardPrjn).SetClass("M1ToMotorBS")

	net.ConnectLayers(motor, pf, p1to1, axon.ForwardPrjn)

	net.ConnectLayers(state, stn, full, axon.ForwardPrjn).SetClass("ToSTN")
	net.ConnectLayers(s1, stn, full, axon.ForwardPrjn).SetClass("ToSTN")

	net.ConnectLayers(state, m1, full, axon.ForwardPrjn)
	net.ConnectLayers(s1, m1, full, axon.ForwardPrjn)

	net.ConnectLayers(gpi, m1VM, full, axon.InhibPrjn).SetClass("BgFixed")
	net.ConnectLayers(gpi, pf, full, axon.InhibPrjn).SetClass("BgFixed")

	mtxGo.SetBuildConfig("ThalLay1Name", m1VM.Name())
	mtxNo.SetBuildConfig("ThalLay1Name", m1VM.Name())

	net.ConnectToDSMatrix(state, mtxGo, full).SetClass("StateToMtx")
	net.ConnectToDSMatrix(state, mtxNo, full).SetClass("StateToMtx")
	net.ConnectToDSMatrix(s1, mtxNo, full).SetClass("StateToMtx")
	net.ConnectToDSMatrix(s1, mtxGo, full).SetClass("StateToMtx")

	net.ConnectToDSMatrix(m1, mtxGo, full).SetClass("M1ToMtx")
	net.ConnectToDSMatrix(m1, mtxNo, full).SetClass("M1ToMtx")

	net.ConnectLayers(pf, mtxGo, p1to1, axon.ForwardPrjn).SetClass("PFToMtx")
	net.ConnectLayers(pf, mtxNo, p1to1, axon.ForwardPrjn).SetClass("PFToMtx")

	m1VM.PlaceRightOf(gpi, space)
	pf.PlaceRightOf(m1VM, space)
	snc.PlaceRightOf(pf, space)
	gpeAk.PlaceAbove(gpi)
	stn.PlaceRightOf(gpePr, space)
	mtxGo.PlaceAbove(gpeAk)
	state.PlaceAbove(mtxGo)
	s1.PlaceRightOf(state, space)
	vl.PlaceRightOf(mtxNo, space)
	motor.PlaceBehind(vl, space)
	targ.PlaceBehind(motor, space)

	net.Build(ctx)
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWts(ctx)
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll() // first hard-coded defaults
	if ss.Config.Params.Network != nil {
		ss.Params.SetNetworkMap(ss.Net, ss.Config.Params.Network)
	}
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
	ss.InitRndSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdt.Update()
	ss.ViewUpdt.RecordSyns()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed(run int) {
	ss.RndSeeds.Set(run)
	ss.RndSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	ev := ss.Envs.ByModeDi(etime.Train, 0).(*MotorSeqEnv)
	trls := int(mat32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	nSeqTrials := ev.SeqLen + 1 // 1 reward at end

	man.AddStack(etime.Train).
		AddTime(etime.Expt, 1).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Sequence, trls, ss.Config.Run.NData).
		AddTime(etime.Trial, nSeqTrials).
		AddTime(etime.Cycle, 300)

	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTimeIncr(etime.Sequence, trls, ss.Config.Run.NData).
		AddTime(etime.Trial, nSeqTrials).
		AddTime(etime.Cycle, 300)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 250, 299)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			seq := man.Stacks[mode].Loops[etime.Sequence].Counter.Cur
			trial := man.Stacks[mode].Loops[etime.Trial].Counter.Cur
			ss.ApplyInputs(mode, seq, trial)
		})
	}
	// note: auto applies to all
	plusPhase, _ := man.Stacks[etime.Train].Loops[etime.Cycle].EventByName("PlusPhase")
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
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWts, ctrString, ss.Stats.String("RunName"))
	})

	// man.GetLoop(etime.Train, etime.Run).Main.Add("TestAll", func() {
	// 	ss.Loops.Run(etime.Test)
	// })

	////////////////////////////////////////////
	// GUI

	if !ss.Config.GUI {
		if ss.Config.Log.NetData {
			man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
				ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
			})
		}
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net, ss.NetViewCounters)
		axon.LooperUpdtPlots(man, &ss.GUI)
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

	lays := []string{"State", "S1", "Target"}
	states := []string{"State", "PrevAction", "Target"}

	for di := 0; di < ss.Config.Run.NData; di++ {
		ev := ss.Envs.ByModeDi(mode, di).(*MotorSeqEnv)
		inRew := ev.IsRewTrial()
		ev.Step()
		for li, lnm := range lays {
			snm := states[li]
			ly := net.AxonLayerByName(lnm)
			itsr := ev.State(snm)
			ly.ApplyExt(ctx, uint32(di), itsr)
		}
		ss.ApplyPVLV(ev, uint32(di), inRew)
	}
	net.ApplyExts(ctx) // now required for GPU mode
}

// ApplyPVLV applies PVLV reward inputs
func (ss *Sim) ApplyPVLV(ev *MotorSeqEnv, di uint32, inRew bool) {
	ctx := &ss.Context
	pv := &ss.Net.PVLV
	pv.EffortUrgencyUpdt(ctx, di, 1)
	pv.Urgency.Reset(ctx, di)

	if inRew {
		axon.SetGlbV(ctx, di, axon.GvACh, 1)
		ss.SetRew(ev.RPE, di)
	} else {
		axon.GlobalSetRew(ctx, di, 0, false) // no rew
		axon.SetGlbV(ctx, di, axon.GvACh, 0)
	}
}

func (ss *Sim) SetRew(rew float32, di uint32) {
	ctx := &ss.Context
	pv := &ss.Net.PVLV
	axon.GlobalSetRew(ctx, di, rew, true)
	axon.SetGlbV(ctx, di, axon.GvDA, rew) // reward prediction error
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
}

// DecodeAct decodes the VL ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ev *MotorSeqEnv, di int) int {
	vt := ss.Stats.SetLayerTensor(ss.Net, "MotorBS", "CaSpkPM", di)
	return ss.SoftMaxChoose4D(vt)
	// return ss.HardChoose4D(vt)
}

// SoftMaxChoose2D probabalistically selects column with most activity in layer,
// using a softmax with Config.Env.ActSoftMaxGain gain factor
func (ss *Sim) SoftMaxChoose2D(vt *etensor.Float32) int {
	dy := vt.Dim(0)
	nact := vt.Dim(1)
	var tot float32
	probs := make([]float32, nact)
	for i := range probs {
		var sum float32
		for j := 0; j < dy; j++ {
			sum += vt.Value([]int{j, i})
		}
		p := mat32.FastExp(ss.Config.Env.ActSoftMaxGain * sum)
		probs[i] = p
		tot += p
	}
	for i, p := range probs {
		probs[i] = p / tot
	}
	chs := erand.PChoose32(probs, -1)
	return chs
}

// SoftMaxChoose4D probabalistically selects column with most activity in layer,
// using a softmax with Config.Env.ActSoftMaxGain gain factor
func (ss *Sim) SoftMaxChoose4D(vt *etensor.Float32) int {
	nact := vt.Dim(1)
	nuY := vt.Dim(2)
	nuX := vt.Dim(3)
	var tot float32
	probs := make([]float32, nact)
	for i := range probs {
		var sum float32
		for j := 0; j < nuY; j++ {
			for k := 0; k < nuX; k++ {
				sum += vt.Value([]int{0, i, j, k})
			}
		}
		p := mat32.FastExp(ss.Config.Env.ActSoftMaxGain * sum)
		probs[i] = p
		tot += p
	}
	for i, p := range probs {
		probs[i] = p / tot
		// fmt.Println(i, p, probs[i])
	}
	chs := erand.PChoose32(probs, -1)
	return chs
}

// HardChoose2D deterministically selects column with most activity in layer,
func (ss *Sim) HardChoose2D(vt *etensor.Float32) int {
	nact := vt.Dim(1)
	nuY := vt.Dim(2)
	nuX := vt.Dim(3)
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
func (ss *Sim) HardChoose4D(vt *etensor.Float32) int {
	nact := vt.Dim(1)
	nuY := vt.Dim(2)
	nuX := vt.Dim(3)
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
	ly := net.AxonLayerByName("MotorBS")
	ly.ApplyExt(ctx, uint32(di), ap)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRndSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	for di := 0; di < int(ctx.NetIdxs.NData); di++ {
		ss.Envs.ByModeDi(etime.Train, di).Init(0)
		ss.Envs.ByModeDi(etime.Test, di).Init(0)
	}
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWts(ctx)
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
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters(di int) {
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
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
	if ss.ViewUpdt.View == nil {
		return
	}
	di := ss.ViewUpdt.View.Di
	if tm == etime.Trial {
		ss.TrialStats(di) // get trial stats for current di
	}
	ss.StatCounters(di)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Sequence", "Trial", "Di", "TrialName", "Cycle", "Action", "Target", "Correct", "Rew", "RewPred", "RPE"})
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
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Expt, etime.Run, etime.Epoch, etime.Sequence)

	ss.Logs.AddItem(&elog.Item{
		Name:  "EpochsToCrit",
		Type:  etensor.FLOAT64,
		Range: minmax.F64{Min: -1},
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
				elg := ss.Logs.Table(ctx.Mode, etime.Epoch)
				epc := int(elg.CellFloat("Epoch", elg.Rows-1))
				etc := math.NaN()
				if epc < ss.Config.Run.NEpochs-1 {
					etc = float64(epc)
				}
				ss.Stats.SetInt("EpochsToCrit", ss.Stats.Int("Epoch")) // only set if makes criterion
				ctx.SetFloat64(etc)
			},
			etime.Scope(etime.Train, etime.Expt): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Run, agg.AggMean)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:  "NToCrit",
		Type:  etensor.FLOAT64,
		Range: minmax.F64{Min: -1},
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Expt): func(ctx *elog.Context) {
				ix := ss.Logs.IdxView(etime.Train, etime.Run)
				ix.Filter(func(et *etable.Table, row int) bool {
					return !math.IsNaN(et.CellFloat("EpochsToCrit", row))
				})
				ctx.SetInt(len(ix.Idxs))
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
		return // don't do reg
	case time == etime.Epoch && mode == etime.Test:
		ss.TestStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc

	if time == etime.Epoch && mode == etime.Train {
		rew := dt.CellFloat("Rew", dt.Rows-1)
		ss.Stats.SetFloat("RewEpc", rew) // used for stopping criterion
	}
}

func (ss *Sim) TestStats() {
	tststnm := "TestTrialStats"
	ix := ss.Logs.IdxView(etime.Test, etime.Sequence)
	spl := split.GroupBy(ix, []string{"TrialName"})
	for _, ts := range ix.Table.ColNames {
		if ts == "TrialName" {
			continue
		}
		split.Agg(spl, ts, agg.AggMean)
	}
	tstst := spl.AggsToTable(etable.ColNameOnly)
	tstst.SetMetaData("precision", strconv.Itoa(elog.LogPrec))
	ss.Logs.MiscTables[tststnm] = tstst

	if ss.Config.GUI {
		plt := ss.GUI.Plots[etime.ScopeKey(tststnm)]
		plt.SetTable(tstst)
		plt.Params.XAxisCol = "Sequence"
		plt.SetColParams("NCorrect", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams("Rew", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams("RPE", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.GoUpdatePlot()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "PCore Test"
	ss.GUI.MakeBody(ss, "pcore", title, `This project simulates the inhibitory dynamics in the STN and GPe leading to integration of Go vs. NoGo signal in the basal ganglia. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 400
	nv.Params.Raster.Max = 300
	nv.Params.LayNmSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.Phase, etime.Phase)

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.3, 2.4)
	nv.SceneXYZ().Camera.LookAt(mat32.V3(0, -0.03, 0.02), mat32.V3(0, 1, 0))

	ss.GUI.ViewUpdt = &ss.ViewUpdt

	ss.GUI.AddPlots(title, &ss.Logs)

	tststnm := "TestTrialStats"
	tstst := ss.Logs.MiscTable(tststnm)
	plt := eplot.NewSubPlot(ss.GUI.Tabs.NewTab(tststnm + " Plot"))
	ss.GUI.Plots[etime.ScopeKey(tststnm)] = plt
	plt.Params.Title = tststnm
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(tstst)

	ss.GUI.Body.AddAppBar(func(tb *gi.Toolbar) {
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddLooperCtrl(tb, ss.Loops, []etime.Modes{etime.Train, etime.Test})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "TestInit", Icon: icons.Update,
			Tooltip: "reinitialize the testing control so it re-runs.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Loops.ResetCountersByMode(etime.Test)
				ss.GUI.UpdateWindow()
			},
		})

		////////////////////////////////////////////////
		gi.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Reset RunLog",
			Icon:    icons.Reset,
			Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.Logs.ResetLog(etime.Train, etime.Run)
				ss.GUI.UpdatePlot(etime.Train, etime.Run)
			},
		})
		////////////////////////////////////////////////
		gi.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "New Seed",
			Icon:    icons.Add,
			Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.RndSeeds.NewSeeds()
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "README",
			Icon:    "file-markdown",
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				gi.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/pcore/README.md")
			},
		})
	})
	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		// vgpu.Debug = ss.Config.Debug
		ss.Net.ConfigGPUwithGUI(&ss.Context) // must happen after gui or no gui
		gi.TheApp.AddQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.NewWindow().Run().Wait()
}

func (ss *Sim) RunNoGUI() {
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWts {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name()

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Train, etime.Trial, "trl", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)
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
		dt.SaveCSV(gi.Filename(fnm), etable.Tab, etable.Headers)
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
