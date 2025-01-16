// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// choose: This project tests the Rubicon framework
// making cost-benefit based choices.
package main

//go:generate core generate -add-types -add-funcs

import (
	"fmt"
	"os"
	"reflect"

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
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/sims/choose/armaze"
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

	// StopOnSeq stops running at end of a sequence (for NetView Di data parallel index).
	StopOnSeq bool

	// StopOnErr stops running when an error programmed into the code occurs.
	StopOnErr bool

	// Params manages network parameter setting.
	Params axon.Params `display:"inline"`

	// Loops are the the control loops for running the sim, in different Modes
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

	// GUI for viewing env.
	EnvGUI *armaze.GUI `display:"-"`

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

	if ss.Config.Env.Config != "" {
		fmt.Println("Env Config:", ss.Config.Env.Config)
	}

	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn *armaze.Env
		if newEnv {
			trn = &armaze.Env{}
		} else {
			trn = ss.Envs.ByModeDi(Train, di).(*armaze.Env)
		}

		// note: names must be standard here!
		trn.Name = env.ModeDi(Train, di)
		trn.Defaults()
		trn.RandSeed = 73
		if !ss.Config.Env.SameSeed {
			trn.RandSeed += int64(di) * 73
		}
		trn.Config.NDrives = ss.Config.Env.NDrives
		if ss.Config.Env.Config != "" {
			args := os.Args
			os.Args = args[:1]
			// todo: need config
			// _, err := cli.Config(&trn.Config, ss.Config.Env.Config)
			// if err != nil {
			// 	slog.Error(err.Error())
			// }
		}
		trn.ConfigEnv(di)
		trn.Validate()
		trn.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn)
		if di == 0 {
			ss.ConfigRubicon(trn)
		}
	}
}

func (ss *Sim) ConfigRubicon(trn *armaze.Env) {
	rp := &ss.Net.Rubicon
	rp.SetNUSs(trn.Config.NDrives, 1)
	rp.Defaults()
	rp.USs.PVposGain = 2 // higher = more pos reward (saturating logistic func)
	rp.USs.PVnegGain = 1 // global scaling of RP neg level -- was 1
	rp.LHb.VSPatchGain = 4
	rp.LHb.VSPatchNonRewThr = 0.15

	rp.USs.USnegGains[0] = 2 // big salient input!

	rp.Drive.DriveMin = 0.5 // 0.5 -- should be
	rp.Urgency.U50 = 10
	if ss.Config.Params.Rubicon != nil {
		reflectx.SetFieldsFromMap(rp, ss.Config.Params.Rubicon)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.Context().ThetaCycles = int32(ss.Config.Run.Cycles)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByModeDi(Train, 0).(*armaze.Env)

	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	nAct := int(armaze.ActionsN)
	popY := 4
	popX := 4
	space := float32(2)

	pone2one := paths.NewPoolOneToOne()
	one2one := paths.NewOneToOne()
	full := paths.NewFull()
	mtxRandPath := paths.NewPoolUniformRand()
	mtxRandPath.PCon = 0.75
	_ = mtxRandPath
	_ = pone2one
	pathClass := "PFCPath"

	ny := ev.Config.Params.NYReps
	narm := ev.Config.NArms

	vSgpi, vSmtxGo, vSmtxNo, urgency, pvPos, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPosUS, ofcPosUSCT, ofcPosUSPT, ofcPosUSPTp, ilPos, ilPosCT, ilPosPT, ilPosPTp, ofcNegUS, ofcNegUSCT, ofcNegUSPT, ofcNegUSPTp, ilNeg, ilNegCT, ilNegPT, ilNegPTp, accCost, plUtil, sc := net.AddRubicon(ny, popY, popX, nuBgY, nuBgX, nuCtxY, nuCtxX, space)
	_, _ = plUtil, urgency
	_, _ = ofcNegUSCT, ofcNegUSPTp
	_, _ = vSmtxGo, vSmtxNo

	plUtilPTp := net.LayerByName("PLutilPTp")

	cs, csP := net.AddInputPulv2D("CS", ny, narm, space)
	dist, distP := net.AddInputPulv2D("Dist", ny, ev.MaxLength+1, space)

	//////// M1, VL, ALM

	act := net.AddLayer2D("Act", axon.InputLayer, ny, nAct) // Action: what is actually done
	vl := net.AddPulvLayer2D("VL", ny, nAct)                // VL predicts brainstem Action
	vl.SetBuildConfig("DriveLayName", act.Name)

	m1, m1CT := net.AddSuperCT2D("M1", "PFCPath", nuCtxY, nuCtxX, space, one2one)
	m1P := net.AddPulvForSuper(m1, space)

	alm, almCT, almPT, almPTp, almMD := net.AddPFC2D("ALM", "MD", nuCtxY, nuCtxX, true, true, space)
	_ = almPT

	net.ConnectLayers(vSgpi, almMD, full, axon.InhibPath)
	// net.ConnectToMatrix(alm, vSmtxGo, full) // todo: explore
	// net.ConnectToMatrix(alm, vSmtxNo, full)

	net.ConnectToPFCBidir(m1, m1P, alm, almCT, almPT, almPTp, full, "M1ALM") // alm predicts m1

	// vl is a predictive thalamus but we don't have direct access to its source
	net.ConnectToPulv(m1, m1CT, vl, full, full, pathClass)
	net.ConnectToPFC(nil, vl, alm, almCT, almPT, almPTp, full, "VLALM") // alm predicts m1

	// sensory inputs guiding action
	// note: alm gets effort, dist via predictive coding below

	net.ConnectLayers(dist, m1, full, axon.ForwardPath).AddClass("ToM1")
	net.ConnectLayers(ofcNegUS, m1, full, axon.ForwardPath).AddClass("ToM1")

	// shortcut: not needed
	// net.ConnectLayers(dist, vl, full, axon.ForwardPath).AddClass("ToVL")

	// these pathways are *essential* -- must get current state here
	net.ConnectLayers(m1, vl, full, axon.ForwardPath).AddClass("ToVL")
	net.ConnectLayers(alm, vl, full, axon.ForwardPath).AddClass("ToVL")

	net.ConnectLayers(m1, accCost, full, axon.ForwardPath).AddClass("MToACC")
	net.ConnectLayers(alm, accCost, full, axon.ForwardPath).AddClass("MToACC")

	// key point: cs does not project directly to alm -- no simple S -> R mappings!?

	//////// CS -> BLA, OFC

	net.ConnectToSC1to1(cs, sc)

	net.ConnectCSToBLApos(cs, blaPosAcq, blaNov)
	net.ConnectToBLAExt(cs, blaPosExt, full)

	net.ConnectToBLAAcq(cs, blaNegAcq, full)
	net.ConnectToBLAExt(cs, blaNegExt, full)

	// for some reason this really makes things worse:
	// net.ConnectToVSMatrix(cs, vSmtxGo, full)
	// net.ConnectToVSMatrix(cs, vSmtxNo, full)

	// OFCus predicts cs
	net.ConnectToPFCBack(cs, csP, ofcPosUS, ofcPosUSCT, ofcPosUSPT, ofcPosUSPTp, full, "CSToPFC")
	net.ConnectToPFCBack(cs, csP, ofcNegUS, ofcNegUSCT, ofcPosUSPT, ofcNegUSPTp, full, "CSToPFC")

	//////// OFC, ACC, ALM predicts dist

	// todo: a more dynamic US rep is needed to drive predictions in OFC
	// using distance and effort here in the meantime
	net.ConnectToPFCBack(dist, distP, ofcPosUS, ofcPosUSCT, ofcPosUSPT, ofcPosUSPTp, full, "DistToPFC")
	net.ConnectToPFCBack(dist, distP, ilPos, ilPosCT, ilPosPT, ilPosPTp, full, "PosToPFC")

	net.ConnectToPFC(dist, distP, ofcNegUS, ofcNegUSCT, ofcNegUSPT, ofcNegUSPTp, full, "DistToPFC")
	net.ConnectToPFC(dist, distP, ilNeg, ilNegCT, ilNegPT, ilNegPTp, full, "DistToPFC")

	//	alm predicts all effort, cost, sensory state vars
	net.ConnectToPFC(dist, distP, alm, almCT, almPT, almPTp, full, "DistToPFC")

	//////// ALM, M1 <-> OFC, ACC

	// action needs to know if maintaining a goal or not
	// using plUtil as main summary "driver" input to action system
	// PTp provides good notmaint signal for action.
	net.ConnectLayers(plUtilPTp, alm, full, axon.ForwardPath).AddClass("ToALM")
	net.ConnectLayers(plUtilPTp, m1, full, axon.ForwardPath).AddClass("ToM1")

	// note: in Obelisk this helps with the Consume action
	// but here in this example it produces some instability
	// at later time points -- todo: investigate later.
	// net.ConnectLayers(notMaint, vl, full, axon.ForwardPath).AddClass("ToVL")

	//////// position

	cs.PlaceRightOf(pvPos, space*2)
	dist.PlaceRightOf(cs, space)

	m1.PlaceRightOf(dist, space)
	alm.PlaceRightOf(m1, space)
	vl.PlaceBehind(m1P, space)
	act.PlaceBehind(vl, space)

	net.Build()
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.Script = ss.Config.Params.Script
	net := ss.Net
	ss.Params.ApplyAll(net)

	// params that vary as number of CSs
	ev := ss.Envs.ByModeDi(Train, 0).(*armaze.Env)

	nCSTot := ev.Config.NArms

	cs := net.LayerByName("CS")
	cs.Params.Inhib.ActAvg.Nominal = 0.32 / float32(nCSTot)
	csp := net.LayerByName("CSP")
	csp.Params.Inhib.ActAvg.Nominal = 0.32 / float32(nCSTot)
	bla := net.LayerByName("BLAposAcqD1")
	pji, _ := bla.RecvPathBySendName("BLANovelCS")
	pj := pji.(*axon.Path)

	// this is very sensitive param to get right
	// too little and the hamster does not try CSs at the beginning,
	// too high and it gets stuck trying the same location over and over
	pj.Params.PathScale.Abs = float32(math32.Min(2.3+(float32(nCSTot)/10.0), 3.0))
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.SetRunName()
	ss.InitRandSeed(0)
	// ss.ConfigEnv() // always do -- otherwise env params not reset after run
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
	return &ss.TrainUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	cycles := ss.Config.Run.Cycles
	plusPhase := ss.Config.Run.PlusCycles
	trials := int(math32.IntMultipleGE(float32(ss.Config.Run.Trials), float32(ss.Config.Run.NData)))

	// Note: actual max counters set by env
	ls.AddStack(Train, Trial).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
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
	}

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
// Called at end of minus phase. However, it can still gate sometimes
// after this point, so that is dealt with at end of plus phase.
func (ss *Sim) TakeAction(net *axon.Network, mode Modes) {
	rp := &net.Rubicon
	ctx := net.Context()
	curModeDir := ss.Current.Dir(mode.String())
	mtxLy := net.LayerByName("VMtxGo")
	vlly := net.LayerByName("VL")
	threshold := float32(0.1)
	ndata := int(ctx.NData)

	for di := 0; di < ndata; di++ {
		diu := uint32(di)
		ev := ss.Envs.ByModeDi(mode, di).(*armaze.Env)
		justGated := mtxLy.Params.AnyGated(diu) // not updated until plus phase: rp.VSMatrix.JustGated.IsTrue()
		hasGated := axon.GlobalScalars.Value(int(axon.GvVSMatrixHasGated), di) > 0
		ev.InstinctAct(justGated, hasGated)
		csGated := (justGated && !rp.HasPosUS(diu))
		ach := axon.GlobalScalars.Value(int(axon.GvACh), di)
		mtxLpi := mtxLy.Params.PoolIndex(0)
		mtxCaPMax := axon.PoolAvgMax(axon.AMCaPMax, axon.AMCycle, axon.Max, mtxLpi, diu)
		deciding := !csGated && !hasGated && (ach > threshold && mtxCaPMax > threshold) // give it time
		wasDeciding := num.ToBool(curModeDir.Float32("Deciding", ndata).Float1D(di))
		if wasDeciding {
			deciding = false // can't keep deciding!
		}
		curModeDir.Float32("Deciding", ndata).SetFloat1D(num.FromBool[float64](deciding), di)

		trSt := armaze.TrSearching
		if hasGated {
			trSt = armaze.TrApproaching
		}

		if csGated || deciding {
			act := "CSGated"
			trSt = armaze.TrJustEngaged
			if !csGated {
				act = "Deciding"
				trSt = armaze.TrDeciding
			}
			// ss.Stats.SetStringDi("Debug", di, act)
			ev.Action("None", nil)
			ss.ApplyAction(mode, di)
			curModeDir.StringValue("ActAction", ndata).SetString1D("None", di)
			curModeDir.StringValue("Instinct", ndata).SetString1D("None", di)
			curModeDir.StringValue("NetAction", ndata).SetString1D(act, di)
			curModeDir.Float64("ActMatch", ndata).SetFloat1D(1, di)
			lpi := vlly.Params.PoolIndex(0)
			axon.PoolsInt.Set(0, int(lpi), 0, int(axon.Clamped)) // not clamped this trial
		} else {
			// ss.Stats.SetStringDi("Debug", di, "acting")
			netAct := ss.DecodeAct(ev, mode, di)
			genAct := ev.InstinctAct(justGated, hasGated)
			curModeDir.StringValue("NetAction", ndata).SetString1D(netAct.String(), di)
			curModeDir.StringValue("Instinct", ndata).SetString1D(genAct.String(), di)
			if netAct == genAct {
				curModeDir.Float64("ActMatch", ndata).SetFloat1D(1, di)
			} else {
				curModeDir.Float64("ActMatch", ndata).SetFloat1D(0, di)
			}

			actAct := genAct
			if curModeDir.Float64("CortexDriving", ndata).Float1D(di) > 0 {
				actAct = netAct
			}
			curModeDir.StringValue("ActAction", ndata).SetString1D(actAct.String(), di)

			ev.Action(actAct.String(), nil)
			ss.ApplyAction(mode, di)

			switch {
			case rp.HasPosUS(diu):
				trSt = armaze.TrRewarded
			case actAct == armaze.Consume:
				trSt = armaze.TrConsuming
			}
		}
		if axon.GlobalScalars.Value(int(axon.GvGiveUp), di) > 0 {
			trSt = armaze.TrGiveUp
		}
		curModeDir.Int("TraceStateInt", ndata).SetInt1D(int(trSt), di)
		curModeDir.StringValue("TraceState", ndata).SetString1D(trSt.String(), di)
	}
	net.ApplyExts()
}

// DecodeAct decodes the VL ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ev *armaze.Env, mode Modes, di int) armaze.Actions {
	tsr := axon.StatsLayerValues(ss.Net, ss.Current, mode, di, "VL", "CaP")
	return ev.DecodeAct(tsr)
}

func (ss *Sim) ApplyAction(mode Modes, di int) {
	net := ss.Net
	ev := ss.Envs.ByModeDi(mode, di).(*armaze.Env)
	ap := ev.State("Action")
	ly := net.LayerByName("Act")
	ly.ApplyExt(uint32(di), ap)
}

// ApplyInputs applies input patterns from given environment for given mode.
// Any other start-of-trial logic can also be put here.
func (ss *Sim) ApplyInputs(mode Modes) {
	net := ss.Net
	ss.Net.InitExt()
	curModeDir := ss.Current.Dir(mode.String())
	lays := []string{"Dist", "CS"}
	ndata := int(net.Context().NData)

	for di := 0; di < ndata; di++ {
		ev := ss.Envs.ByModeDi(mode, di).(*armaze.Env)
		giveUp := axon.GlobalScalars.Value(int(axon.GvGiveUp), di) > 0
		if giveUp {
			ev.JustConsumed = true // triggers a new start -- we just consumed the giving up feeling :)
		}
		ev.Step()
		if ev.Tick == 0 {
			driving := num.FromBool[float64](randx.BoolP32(ss.Config.Env.PctCortex))
			curModeDir.Float64("CortexDriving", ndata).SetFloat1D(driving, di)
		}
		for _, lnm := range lays {
			ly := net.LayerByName(lnm)
			itsr := ev.State(lnm)
			ly.ApplyExt(uint32(di), itsr)
		}
		curModeDir.StringValue("TrialName", 1).SetString1D(ev.String(), 0)
		ss.ApplyRubicon(ev, mode, uint32(di))
	}

	net.ApplyExts()
}

// ApplyRubicon applies Rubicon reward inputs.
func (ss *Sim) ApplyRubicon(ev *armaze.Env, mode Modes, di uint32) {
	rp := &ss.Net.Rubicon
	rp.NewState(di, &ss.Net.Rand) // first before anything else is updated
	rp.SetGoalMaintFromLayer(di, ss.Net, "PLutilPT", 0.2)
	rp.DecodePVEsts(di, ss.Net)
	rp.SetGoalDistEst(di, float32(ev.Dist))
	rp.EffortUrgencyUpdate(di, ev.Effort)
	if ev.USConsumed >= 0 {
		rp.SetUS(di, axon.Positive, ev.USConsumed, ev.USValue)
	}
	rp.SetDrives(di, 0.5, ev.Drives...)
	rp.Step(di, &ss.Net.Rand)
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	ss.InitRandSeed(ss.Loops.Loop(Train, Run).Counter.Cur)
	for di := 0; di < int(ctx.NData); di++ {
		ss.Envs.ByModeDi(Train, di).Init(0)
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
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
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
	// 			plot.SetFirstStyle(tsr, func(s *plot.Style) {
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
func (ss *Sim) ConfigGUI() {
	ss.GUI.MakeBody(ss, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.FS = ss.Root
	ss.GUI.DataRoot = "Root"
	ss.GUI.CycleUpdateInterval = 10

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
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{}, math32.Vec3(0, 1, 0))

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
	axon.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Train})

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.Runs, ss.Config.Run.Run)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)

	ss.Loops.Run(Train)

	axon.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
	axon.GPURelease()
}
