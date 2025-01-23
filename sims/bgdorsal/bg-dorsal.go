// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// bgdorsal simulates the dorsal Basal Ganglia, starting with the
// Dorsal Striatum, centered on the Pallidum Core (GPe) areas that
// drive Go vs. No selection of motor actions.
package main

//go:generate core generate -add-types -add-funcs

import (
	"fmt"
	"reflect"

	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/base/num"
	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/cli"
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
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
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
	Trial
	Sequence
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
	ss.Params.Config(LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag, reflect.ValueOf(ss))
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	if ss.Config.Run.GPU {
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
		var trn, tst *MotorSeqEnv
		if newEnv {
			trn = &MotorSeqEnv{}
			tst = &MotorSeqEnv{}
		} else {
			trn = ss.Envs.ByModeDi(Train, di).(*MotorSeqEnv)
			tst = ss.Envs.ByModeDi(Test, di).(*MotorSeqEnv)
		}

		// note: names must be standard here!
		trn.Name = env.ModeDi(Train, di)
		trn.Defaults()
		trn.NActions = ss.Config.Env.NActions
		trn.SeqLen = ss.Config.Env.SeqLen
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
		}
		trn.Config(Train, 73+int64(di)*73)

		tst.Name = env.ModeDi(Test, di)
		tst.Defaults()
		tst.NActions = ss.Config.Env.NActions
		tst.SeqLen = ss.Config.Env.SeqLen
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

func (ss *Sim) ConfigRubicon(trn *MotorSeqEnv) {
	rp := &ss.Net.Rubicon
	rp.SetNUSs(2, 1)
	rp.Urgency.U50 = 20 // 20 def
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.Context().SetThetaCycles(int32(ss.Config.Run.Cycles)).
		SetPlusCycles(int32(ss.Config.Run.PlusCycles))
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByModeDi(Train, 0).(*MotorSeqEnv)

	np := 1
	nu := ss.Config.Params.NUnits
	nuPer := ev.NUnitsPer
	nAct := ev.NActions
	nSeq := ev.SeqLen
	maxSeqAct := max(nAct, nSeq) // layer size

	nuX := nu
	nuY := nu
	nuCtxY := nu
	nuCtxX := nu
	space := float32(2)

	p1to1 := paths.NewPoolOneToOne()
	one2one := paths.NewOneToOne()
	_ = one2one
	full := paths.NewFull()
	_ = full
	mtxRandPath := paths.NewUniformRand()
	mtxRandPath.PCon = 0.5
	_ = mtxRandPath

	mtxGo, mtxNo, gpePr, gpeAk, stn, gpi, pf := net.AddDorsalBG("", 1, nAct, nuY, nuX, nuY, nuX, space)
	_, _ = gpePr, gpeAk

	snc := net.AddLayer2D("SNc", axon.InputLayer, 1, 1)
	_ = snc

	state := net.AddLayer4D("State", axon.InputLayer, 1, np, nuPer, maxSeqAct)
	s1 := net.AddLayer4D("S1", axon.InputLayer, 1, np, nuPer, nAct+1)

	targ := net.AddLayer2D("Target", axon.InputLayer, nuPer, nAct) // Target: just for vis

	motor := net.AddLayer4D("MotorBS", axon.TargetLayer, 1, nAct, nuPer, 1)
	pf.Shape.CopyFrom(&motor.Shape)

	vl := net.AddPulvLayer4D("VL", 1, nAct, nuPer, 1) // VL predicts brainstem Action
	vl.SetBuildConfig("DriveLayName", motor.Name)

	// bool before space is selfmaint or not: selfcons much better (false)
	m1, m1CT, m1PT, m1PTp, m1VM := net.AddPFC2D("M1", "VM", nuCtxY, nuCtxX, false, false, space)
	_ = m1PT
	m1.SetBuildConfig("GateLayName", m1VM.Name)
	m1CT.SetBuildConfig("GateLayName", m1VM.Name)
	m1PT.SetBuildConfig("GateLayName", m1VM.Name)
	m1PTp.SetBuildConfig("GateLayName", m1VM.Name)
	m1VM.SetBuildConfig("GateLayName", m1VM.Name)
	motor.SetBuildConfig("GateLayName", m1VM.Name)

	// todo: M1PTp should be VL interconnected, prior to PT, not after it.

	// vl is a predictive thalamus but we don't have direct access to its source
	net.ConnectToPFC(nil, vl, m1, m1CT, m1PT, m1PTp, full, "VLM1") // m1 predicts vl

	// these pathways are *essential* -- must get current state here
	net.ConnectLayers(m1, vl, full, axon.ForwardPath).AddClass("VLM1")

	net.ConnectLayers(gpi, motor, p1to1, axon.InhibPath).AddClass("FmGPI")
	net.ConnectLayers(m1PT, motor, full, axon.ForwardPath).AddClass("M1ToMotorBS ToMotor")
	// net.ConnectLayers(m1PTp, motor, full, axon.ForwardPath).AddClass("M1ToMotorBS")
	net.ConnectLayers(m1, motor, full, axon.ForwardPath).AddClass("M1ToMotorBS ToMotor")

	net.ConnectLayers(motor, pf, one2one, axon.ForwardPath)

	net.ConnectLayers(state, stn, full, axon.ForwardPath).AddClass("ToDSTN FmState")
	net.ConnectLayers(state, m1, full, axon.ForwardPath).AddClass("ToM1 FmState")
	net.ConnectLayers(s1, stn, full, axon.ForwardPath).AddClass("ToDSTN")
	net.ConnectLayers(s1, m1, full, axon.ForwardPath).AddClass("ToM1")

	net.ConnectLayers(gpi, m1VM, full, axon.InhibPath).AddClass("DBGInhib")

	mtxGo.SetBuildConfig("ThalLay1Name", m1VM.Name)
	mtxNo.SetBuildConfig("ThalLay1Name", m1VM.Name)

	toMtx := full
	// toMtx := mtxRandPath // works, but not as reliably
	net.ConnectToDSMatrix(state, mtxGo, toMtx).AddClass("StateToMtx FmState")
	net.ConnectToDSMatrix(state, mtxNo, toMtx).AddClass("StateToMtx FmState")
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

	net.Build()
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.Script = ss.Config.Params.Script
	ss.Params.ApplyAll(ss.Net)

	// compensate for expected activity levels based on max seq len
	lnms := []string{"State", "S1", "MotorBS", "VL"}
	ev := ss.Envs.ByModeDi(Train, 0).(*MotorSeqEnv)
	for _, lnm := range lnms {
		ly := ss.Net.LayerByName(lnm)
		// fmt.Println(ly.Params.Inhib.ActAvg.Nominal)
		if lnm == "State" {
			ly.Params.Inhib.ActAvg.Nominal = 0.5 / float32(max(ev.SeqLen, ev.NActions))
		} else {
			ly.Params.Inhib.ActAvg.Nominal = 0.5 / float32(ev.NActions)
		}
	}
}

func (ss *Sim) TurnOffTheNoise() {
	return // not doing this now -- not better
	mtxGo := ss.Net.LayerByName("MtxGo")
	if mtxGo.Params.Acts.Noise.On.IsFalse() {
		return
	}
	ss.Params.ApplySheet(ss.Net, "NoiseOff")
	fmt.Println("Turned noise off")
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

	ev := ss.Envs.ByModeDi(Train, 0).(*MotorSeqEnv)
	seqs := int(math32.IntMultipleGE(float32(ss.Config.Run.Sequences), float32(ss.Config.Run.NData)))
	cycles := ss.Config.Run.Cycles
	plusPhase := ss.Config.Run.PlusCycles
	seqLen := ev.SeqLen + 1 // 1 reward at end

	ls.AddStack(Train, Trial).
		AddLevel(Expt, 1).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Sequence, seqs, ss.Config.Run.NData).
		AddLevel(Trial, seqLen).
		AddLevel(Cycle, cycles)

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Sequence, seqs, ss.Config.Run.NData).
		AddLevel(Trial, seqLen).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, cycles-plusPhase, cycles-1, Cycle, Trial, Train)

	ls.Stacks[Train].OnInit.Add("Init", func() { ss.Init() })

	ls.AddOnStartToLoop(Trial, "ApplyInputs", func(mode enums.Enum) {
		ss.ApplyInputs(mode.(Modes))
	})

	for mode, st := range ls.Stacks {
		plusPhase := st.Loops[Cycle].EventByName("MinusPhase:End")
		plusPhase.OnEvent.InsertBefore("PlusPhase:Start", "TakeAction", func() bool {
			// note: critical to have this happen *after* MinusPhase:End and *before* PlusPhase:Start
			// because minus phase end has gated info, and plus phase start applies action input
			ss.TakeAction(ss.Net, mode.(Modes))
			return false
		})
		// plusPhase := st.Loops[Cycle].EventByName("MinusPhase:End")
		// st.Loops[Trial].OnEnd.Prepend("TakeAction", func() bool {
		// 	ss.TakeAction(ss.Net, mode.(Modes))
		// 	return false
		// })
	}

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	ls.Loop(Train, Epoch).IsDone.AddBool("StopCrit", func() bool {
		epcDir := ss.Stats.Dir(Train.String()).Dir(Epoch.String())
		rew := epcDir.Value("Rew").Float1D(-1)
		stop := rew >= 0.98
		return stop
	})

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.RunName())
	})

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

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
	ss.Net.InitExt()

	ndata := int(net.Context().NData)
	curModeDir := ss.Current.Dir(mode.String())
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
			ly.ApplyExt(uint32(di), itsr)
		}
		curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), di)
		ss.ApplyRubicon(ev, mode, inRew, uint32(di))
	}
	net.ApplyExts()
}

// ApplyRubicon applies Rubicon reward inputs
func (ss *Sim) ApplyRubicon(ev *MotorSeqEnv, mode Modes, inRew bool, di uint32) {
	rp := &ss.Net.Rubicon
	rp.EffortUrgencyUpdate(di, 1)
	rp.Urgency.Reset(di)

	if inRew {
		axon.GlobalScalars.Set(1, int(axon.GvACh), int(di))
		ss.SetRew(ev.RPE, di)
	} else {
		axon.GlobalSetRew(di, 0, false) // no rew
		axon.GlobalScalars.Set(0, int(axon.GvACh), int(di))
	}
}

func (ss *Sim) SetRew(rew float32, di uint32) {
	rp := &ss.Net.Rubicon
	axon.GlobalSetRew(di, rew, true)
	axon.GlobalScalars.Set(rew, int(axon.GvDA), int(di)) // no reward prediction error
	if rew > 0 {
		rp.SetUS(di, axon.Positive, 0, 1)
	} else if rew < 0 {
		rp.SetUS(di, axon.Negative, 0, 1)
	}
}

// TakeAction takes action for this step, using decoded cortical action.
// Called at end of minus phase.
func (ss *Sim) TakeAction(net *axon.Network, mode Modes) {
	for di := 0; di < ss.Config.Run.NData; di++ {
		ev := ss.Envs.ByModeDi(mode, di).(*MotorSeqEnv)
		if !ev.IsRewTrialPostStep() {
			netAct := ss.DecodeAct(ev, mode, di)
			ev.Action(fmt.Sprintf("%d", netAct), nil)
			ss.ApplyAction(mode, di)
		}
	}
	ss.Net.ApplyExts() // required!
}

// DecodeAct decodes the MotorBS ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ev *MotorSeqEnv, mode Modes, di int) int {
	tsr := axon.StatsLayerValues(ss.Net, ss.Current, mode, di, "MotorBS", "ActM")
	return ss.SoftMaxChoose4D(tsr, mode)
	// return ss.HardChoose4D(tsr, mode)
}

// SoftMaxChoose2D probabalistically selects column with most activity in layer,
// using a softmax with Config.Env.ActSoftMaxGain gain factor
func (ss *Sim) SoftMaxChoose2D(vt *tensor.Float32, mode Modes) int {
	dy := vt.DimSize(0)
	nact := vt.DimSize(1)
	var tot float32
	probs := make([]float32, nact)
	for i := range probs {
		var sum float32
		for j := 0; j < dy; j++ {
			sum += vt.Value(j, i)
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
func (ss *Sim) SoftMaxChoose4D(vt *tensor.Float32, mode Modes) int {
	nact := vt.DimSize(1)
	nuY := vt.DimSize(2)
	nuX := vt.DimSize(3)
	var tot float32
	probs := make([]float32, nact)
	for i := range probs {
		var sum float32
		for j := 0; j < nuY; j++ {
			for k := 0; k < nuX; k++ {
				sum += vt.Value(0, i, j, k)
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
func (ss *Sim) HardChoose2D(vt *tensor.Float32, mode Modes) int {
	nact := vt.DimSize(1)
	nuY := vt.DimSize(2)
	nuX := vt.DimSize(3)
	var mx float32
	var mxi int
	for i := 0; i < nact; i++ {
		var sum float32
		for j := 0; j < nuY; j++ {
			for k := 0; k < nuX; k++ {
				sum += vt.Value(0, i, j, k)
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
func (ss *Sim) HardChoose4D(vt *tensor.Float32, mode Modes) int {
	nact := vt.DimSize(1)
	nuY := vt.DimSize(2)
	nuX := vt.DimSize(3)
	var mx float32
	var mxi int
	for i := 0; i < nact; i++ {
		var sum float32
		for j := 0; j < nuY; j++ {
			for k := 0; k < nuX; k++ {
				sum += vt.Value(0, i, j, k)
			}
		}
		if sum > mx {
			mx = sum
			mxi = i
		}
	}
	return mxi
}

func (ss *Sim) ApplyAction(mode Modes, di int) {
	net := ss.Net
	ev := ss.Envs.ByModeDi(mode, di).(*MotorSeqEnv)
	ap := ev.State("Action")
	ly := net.LayerByName("MotorBS")
	ly.ApplyExt(uint32(di), ap)
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
		nm := mode.String() + "/" + level.String() + " Plot"
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
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Trial))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Sequence))
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

	// note: Trial level is not recorded, only the sequence

	// last arg(s) are levels to exclude
	counterFunc := axon.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Sequence, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		counterFunc(mode, level, phase == Start)
	})
	runNameFunc := axon.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Sequence, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		runNameFunc(mode, level, phase == Start)
	})
	trialNameFunc := axon.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trialNameFunc(mode, level, phase == Start)
	})
	perTrlFunc := axon.StatPerTrialMSec(ss.Stats, Train, Sequence)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		perTrlFunc(mode, level, phase == Start)
	})

	vmly := ss.Net.LayerByName("M1VM")
	trialStats := []string{"Di", "Action", "Target", "Correct", "RT"}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		if level != Trial {
			return
		}
		for si, name := range trialStats {
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			ndata := int(ss.Net.Context().NData)
			for di := range ndata {
				tsr := levelDir.Float64(name)
				if phase == Start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0).SetMax(1)
						if si >= 3 && si <= 4 {
							s.On = true
						}
					})
					continue
				}
				ev := ss.Envs.ByModeDi(mode, di).(*MotorSeqEnv)
				var stat float32
				switch name {
				case "Di":
					stat = float32(di)
				case "Action":
					stat = float32(ev.CurAction)
				case "Target":
					stat = float32(ev.Target)
				case "Correct":
					stat = num.FromBool[float32](ev.Correct)
				case "RT":
					stat = axon.LayerStates.Value(vmly.Index, di, int(axon.GatedRT))
					if stat < 0 {
						stat = math32.NaN()
					}
				}
				curModeDir.Float32(name, ndata).SetFloat1D(float64(stat), di)
				tsr.AppendRowFloat(float64(stat))
			}
		}
	})

	seqStats := []string{"NCorrect", "Rew", "RewPred", "RPE", "RT"}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		if level <= Trial {
			return
		}
		for _, name := range seqStats {
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			subDir := modeDir.Dir((level - 1).String()) // note: will fail for Cycle
			tsr := levelDir.Float64(name)
			ndata := int(ss.Net.Context().NData)
			var stat float64
			if phase == Start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					if name != "RT" {
						s.On = true
					}
				})
				continue
			}
			switch level {
			case Sequence:
				for di := range ndata {
					ev := ss.Envs.ByModeDi(mode, di).(*MotorSeqEnv)
					var stat float32
					switch name {
					case "NCorrect":
						stat = float32(ev.PrevNCorrect)
					case "Rew":
						stat = ev.Rew
					case "RewPred":
						stat = ev.RewPred
					case "RPE":
						stat = ev.RPE
					case "RT":
						stat = float32(stats.StatMean.Call(subDir.Value(name)).Float1D(0))
					}
					curModeDir.Float32(name, ndata).SetFloat1D(float64(stat), di)
					tsr.AppendRowFloat(float64(stat))
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
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		if level <= Epoch {
			return
		}
		name := "EpochsToCrit"
		modeDir := ss.Stats.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		subDir := modeDir.Dir((level - 1).String()) // note: will fail for Cycle
		tsr := levelDir.Float64(name)
		if phase == Start {
			tsr.SetNumRows(0)
			plot.SetFirstStyler(tsr, func(s *plot.Style) {
				s.Range.SetMin(0)
				s.On = true
			})
			return
		}
		var stat float64
		switch level {
		case Run:
			stat = float64(ss.Loops.Loop(mode, (level - 1)).Counter.Cur)
			tsr.AppendRowFloat(stat)
		default: // expt
			stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
			tsr.AppendRowFloat(stat)
		}
	})
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		if level < Expt {
			return
		}
		name := "NFail"
		modeDir := ss.Stats.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		subDir := modeDir.Dir((level - 1).String())
		tsr := levelDir.Float64(name)
		if phase == Start {
			tsr.SetNumRows(0)
			plot.SetFirstStyler(tsr, func(s *plot.Style) {
				s.Range.SetMin(0)
				s.On = true
			})
			return
		}
		run := subDir.Value("EpochsToCrit")
		nfail := 0
		for i := range run.Len() {
			epc := run.Float1D(i)
			if int(epc) == ss.Config.Run.Epochs {
				nfail++
			}
		}
		tsr.AppendRowFloat(float64(nfail))
	})
	runAllFunc := axon.StatLevelAll(ss.Stats, Train, Run, func(s *plot.Style, cl tensor.Values) {
		name := metadata.Name(cl)
		switch name {
		case "EpochsToCrit", "NCorrect":
			s.On = true
			s.Range.SetMin(0)
		}
	})
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		runAllFunc(mode, level, phase == Start)
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
	statNames := []string{"Action", "Target", "Correct", "RT"}
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
	nv.Options.LayerNameSize = 0.03
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.TestUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.0, 2.5)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, -0.1, 0.02), math32.Vec3(0, 1, 0))

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
	gpu.DebugAdapter = true
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
