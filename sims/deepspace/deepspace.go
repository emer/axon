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
	// to perform all stats computations. start does init at start of given level,
	// and all intialization / configuration (called during Init too).
	StatFuncs []func(mode enums.Enum, level enums.Enum, start bool) `display:"-"`

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
	var trn, tst *emery.EmeryEnv
	if len(ss.Envs) == 0 {
		trn = &emery.EmeryEnv{}
		tst = &emery.EmeryEnv{}
	} else {
		trn = ss.Envs.ByMode(Train).(*emery.EmeryEnv)
		tst = ss.Envs.ByMode(Test).(*emery.EmeryEnv)
	}

	// note: names must be standard here!
	trn.Defaults()
	trn.Name = Train.String()
	trn.Params.UnitsPer = ss.Config.Env.UnitsPer
	if ss.Config.Env.Env != nil {
		reflectx.SetFieldsFromMap(&trn.Params, ss.Config.Env.Env)
	}
	trn.Config(ndata, ss.Config.Run.Cycles(), ss.Root.Dir("Env"), axon.ComputeGPU)
	trn.Init(0)

	tst.Defaults()
	tst.Name = Test.String()
	tst.Params.UnitsPer = ss.Config.Env.UnitsPer
	if ss.Config.Env.Env != nil {
		reflectx.SetFieldsFromMap(&tst.Params, ss.Config.Env.Env)
	}
	tst.Config(ndata, ss.Config.Run.Cycles(), ss.Root.Dir("Env"), axon.ComputeGPU)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn)
	ss.Envs.Add(tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.Context().SetISICycles(int32(ss.Config.Run.ISICycles)).
		SetMinusCycles(int32(ss.Config.Run.MinusCycles)).
		SetPlusCycles(int32(ss.Config.Run.PlusCycles)).Update()
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0
	cycles := ss.Config.Run.Cycles()
	ev := ss.Envs.ByMode(Train).(*emery.EmeryEnv)
	unitsPer := ev.Params.UnitsPer

	full := paths.NewFull()
	full.SelfCon = true // unclear if this makes a diff for self cons at all
	one2one := paths.NewOneToOne()
	_ = one2one
	p1to1 := paths.NewPoolOneToOne()
	_ = p1to1
	firstPool := paths.NewPoolRect()
	firstPool.Size.Set(1, unitsPer)
	firstPool.Scale.Set(0, 0)
	firstPool.Wrap = false
	secondPool := paths.NewPoolRect()
	secondPool.Size.Set(1, unitsPer)
	secondPool.Scale.Set(0, 0)
	secondPool.Start.Set(1, 0)
	secondPool.Wrap = false

	space := float32(2)
	// eyeSz := image.Point{2, 1}

	addInput := func(nm string, doc string) (in, mf, thal *axon.Layer) {
		in = net.AddLayer4D(nm, axon.InputLayer, 1, 2, unitsPer, 1)
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

	eyeH, _, _ := addInput("EyeH", "VOR eye position control layer: relative balance in L - R activity drives changes in eye position set point, driven by anticipated vestibular signals from efferent copy of motor actions")
	eyeH.Type = axon.SuperLayer
	eyeH.Class = ""
	eyeH.AddClass("MotorOut")

	vorCtrl := net.AddLayer4D("VORCtrl", axon.InputLayer, 1, 2, unitsPer, 1)
	vorCtrl.AddClass("RateIn")
	vorCtrl.Doc = "VOR (vestibulo-ocular reflex) control input -- if left units active then cerebellar anticipation of vestibular signals drives compensatory eye movements, if right units active, it does not (inhibited)"

	fixedCon := func(send, recv *axon.Layer, path paths.Pattern, typ axon.PathTypes, cls string) *axon.Path {
		pt := net.ConnectLayers(send, recv, path, typ).AddClass(cls)
		pt.AddDefaultParams(func(pt *axon.PathParams) { pt.SetFixedWts() })
		return pt
	}
	fixedConMod := func(send, recv *axon.Layer, path paths.Pattern, typ axon.PathTypes, cls string) *axon.Path {
		pt := net.ConnectLayers(send, recv, path, typ).AddClass(cls)
		pt.AddDefaultParams(func(pt *axon.PathParams) {
			pt.SetFixedWts()
			pt.Com.GType = axon.ModulatoryG
		})
		return pt
	}

	fixedCon(vorCtrl, eyeH, secondPool, axon.InhibPath, "MotorInhib")

	// rotActPrev, rotActPrevPop := addInput("ActRotatePrev", "Previous trial's version of ActRotate. This should be implicitly maintained but currently is not.")
	// _ = rotActPrevPop

	addInputPulv := func(nm string, doc string) (in, thal, thalP, mf *axon.Layer) {
		in = net.AddLayer4D(nm, axon.InputLayer, 1, 2, unitsPer, 1)
		in.AddClass("RateIn")
		in.Doc = "Rate code version. " + doc

		mf = net.AddLayer4D(nm+"MF", axon.InputLayer, ev.Params.TimeBins, 1, 1, ev.Params.PopCodeUnits)
		mf.AddClass("MFIn")
		mf.Doc = "MF mossy fiber input, transient population code. " + doc
		mf.PlaceBehind(in, space)

		thal, thalP = net.AddInputPulv4D(nm+"Thal", ev.Params.TimeBins, 1, 1, ev.Params.PopCodeUnits, space)
		thal.AddClass("ThalIn")
		thalP.AddClass("ThalIn")
		thal.Doc = "Thalamic input, integrated population code. " + doc
		thal.PlaceBehind(mf, space)
		return
	}

	vsHV, vsHVThal, vsHVThalP, vsMF := addInputPulv("VShv", "Vestibular (VS) horizontal rotation velocity, computed from the physics model over time. Population coded left to right with gaussian tuning curves for a range of degrees for each unit (X axis) and redundant units for population code in the Y axis.")
	_, _ = vsHVThalP, vsMF

	vmHV, vmHVThal, vmHVThalP, vmMF := addInputPulv("VMhv", "Visual motion (VM) computed from the full-field of eye using retinal motion filter (see Env tab for visual environment). Population coded left to right with gaussian tuning curves for a range of velocities for each unit (X axis) and redundant units for population code in the Y axis.")
	_ = vmMF

	s1, s1ct := net.AddSuperCT2D("S1", "", 10, 10, space, one2one) // one2one learn > full
	s1.Doc = "Neocortical integrated vestibular and full-field visual motion processing. Does predictive learning on both input signals, more like S2 (secondary), but just using one for simplicity."
	// net.ConnectCTSelf(s1ct, full, "") // self definitely doesn't make sense -- no need for 2-back ct
	// net.LateralConnectLayer(s1ct, full).AddClass("CTSelfMaint") // no diff
	net.ConnectToPulv(s1, s1ct, vsHVThalP, full, full, "")
	net.ConnectLayers(rotActThal, s1, full, axon.ForwardPath).AddClass("FFToHid", "FromAct")
	net.ConnectLayers(vsHVThal, s1, full, axon.ForwardPath).AddClass("FFToHid")

	// visHid, visHidct := net.AddSuperCT2D("VisHid", "", 10, 10, space, one2one) // one2one learn > full

	// net.ConnectToPulv(visHid, visHidct, vmHVp, full, full, "")
	// net.ConnectLayers(rotAct, visHid, full, axon.ForwardPath).AddClass("FFToHid", "FromAct")
	// net.ConnectLayers(vmHV, visHid, full, axon.ForwardPath).AddClass("FFToHid")

	net.ConnectToPulv(s1, s1ct, vmHVThalP, full, full, "")
	net.ConnectLayers(vmHVThal, s1, full, axon.ForwardPath).AddClass("FFToHid")

	// net.ConnectLayers(vsHVThal, visHid, full, axon.ForwardPath).AddClass("FFToHid")

	if ev.Params.LeftEye {
		// net.ConnectToPulv(visHidThal, visHidct, eyeLInp, full, full, "")
		// net.ConnectLayers(eyeLInThal, visHid, full, axon.ForwardPath).AddClass("FFToHid")
	}

	//////// cerebellum:

	// cycles-20 is sufficient to allow time for motor to engage
	actionEnv := cycles - 20
	vsIOUp, vsCNiIOUp, vsCNiUp, vsCNeUp := net.AddNuclearCNUp(vsHV, rotAct, "", "VS", "Vestibular horizontal velocity (VShv), Upbound (adaptive filter), cancels vestib.", actionEnv, space)
	_, _ = vsIOUp, vsCNeUp

	vsIODn, vsCNiIODn, vsCNeDn := net.AddNuclearCNDn(vsHV, rotAct, "", "VS", "Vestibular horizontal velocity (VShv), Downbound (forward model), drives VOR.", actionEnv, space)
	_, _ = vsIODn, vsCNeDn

	vmrIOUp, vmrCNiIOUp, vmrCNiUp, vmrCNeUp := net.AddNuclearCNUp(vmHV, rotAct, "VMhvr", "VM", "Visual Motion horizontal velocity (VMhv), Upbound (adaptive filter), cancels VM, VOR Reflexive case (active with VOR reflex)", actionEnv, space)
	_, _ = vmrIOUp, vmrCNeUp

	vmiIOUp, vmiCNiIOUp, vmiCNiUp, vmiCNeUp := net.AddNuclearCNUp(vmHV, rotAct, "VMhvi", "VM", "Visual Motion horizontal velocity (VMhv), Upbound (adaptive filter), cancels VM, VOR Inhibited case (active when VOR inhibted)", actionEnv, space)
	_, _ = vmiIOUp, vmiCNeUp

	// upbound adaptive filter model
	fixedCon(vsHV, vsCNeUp, p1to1, axon.ForwardPath, "SenseToCNeUp")

	// motor efferent
	net.ConnectLayers(rotActMF, vsCNiIOUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiIO")
	net.ConnectLayers(rotActMF, vsCNiUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiUp")

	net.ConnectLayers(rotActMF, vsCNiIODn, full, axon.CNIOPath).AddClass("MF", "MFToCNiIO")
	net.ConnectLayers(rotActMF, vsCNeDn, full, axon.CNIOPath).AddClass("MF", "MFToCNeDn")

	fixedCon(vmHV, vmrCNeUp, p1to1, axon.ForwardPath, "SenseToCNeUp")

	net.ConnectLayers(rotActMF, vmrCNiIOUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiIO")
	net.ConnectLayers(rotActMF, vmrCNiUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiUp")

	fixedCon(vmHV, vmiCNeUp, p1to1, axon.ForwardPath, "SenseToCNeUp")

	net.ConnectLayers(rotActMF, vmiCNiIOUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiIO")
	net.ConnectLayers(rotActMF, vmiCNiUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiUp")

	// cross-modality input: vmr -> vs and vs -> vmr
	// vmr->vs beneficial even if vmr happens later.. not clear why
	net.ConnectLayers(vmMF, vsCNiIOUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiIO")
	net.ConnectLayers(vmMF, vsCNiUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiUp")
	// but this interferes with VOR function -- bad!!
	// net.ConnectLayers(vmrMF, vsCNiIODn, full, axon.CNIOPath).AddClass("MF", "MFToCNiIO")
	// net.ConnectLayers(vmrMF, vsCNeDn, full, axon.CNIOPath).AddClass("MF", "MFToCNiUp")

	net.ConnectLayers(vsMF, vmrCNiIOUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiIO")
	net.ConnectLayers(vsMF, vmrCNiUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiUp")

	net.ConnectLayers(vsMF, vmiCNiIOUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiIO")
	net.ConnectLayers(vsMF, vmiCNiUp, full, axon.CNIOPath).AddClass("MF", "MFToCNiUp")

	// vor reflex (not-inhib) is firstPool, inhibits vmi
	fixedCon(vorCtrl, vmiCNiIOUp, firstPool, axon.InhibPath, "VORCtrlToCN")
	fixedCon(vorCtrl, vmiCNeUp, firstPool, axon.InhibPath, "VORCtrlToCN")
	fixedConMod(vorCtrl, vmiIOUp, firstPool, axon.ForwardPath, "VORCtrlToCNIO")

	// vor inhib is secondPool, inhibits vmr
	fixedCon(vorCtrl, vmrCNiIOUp, secondPool, axon.InhibPath, "VORCtrlToCN")
	fixedCon(vorCtrl, vmrCNeUp, secondPool, axon.InhibPath, "VORCtrlToCN")
	fixedConMod(vorCtrl, vmrIOUp, secondPool, axon.ForwardPath, "VORCtrlToCNIO")

	// s1
	// net.ConnectLayers(s1ct, vsCNiIOUp, p1to1, axon.CNIOPath).AddClass("MF", "MFToCNiIOUp")
	// net.ConnectLayers(s1ct, vsCNiUp, p1to1, axon.CNIOPath).AddClass("MF", "MFToCNiUp")
	// net.ConnectLayers(s1ct, vsCNiIODn, p1to1, axon.CNIOPath).AddClass("MF", "MFToCNiIODn")

	// note: it is critical that this is from vs, because vm will be zeroed!
	// and indeed, the VORCtrl inhibits this when VOR is active.
	fixedCon(vsCNeDn, eyeH, p1to1, axon.ForwardPath, "Reflex")

	// position

	eyeH.PlaceRightOf(rotAct, float32(ev.Params.PopCodeUnits))
	vorCtrl.PlaceRightOf(eyeH, space)
	// rotActPrev.PlaceBehind(rotActThal, space)
	vsHV.PlaceRightOf(eyeH, float32(ev.Params.PopCodeUnits))
	vmHV.PlaceRightOf(vsHV, float32(ev.Params.PopCodeUnits))
	// if ev.LeftEye {
	// 	eyeLIn.PlaceRightOf(vmHV, space)
	// }
	s1.PlaceAbove(rotAct)

	vsCNiIOUp.PlaceRightOf(s1, space*3)
	vsCNiIODn.PlaceRightOf(vsCNiIOUp, space*3)
	vmrCNiIOUp.PlaceRightOf(vsCNiIODn, space*3)
	vmiCNiIOUp.PlaceRightOf(vmrCNiIOUp, space*3)

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
	ss.RandSeeds.Set(run, ss.Net.Rand)
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
	net := ss.Net

	ls.AddStack(Train, Trial).
		AddLevel(Expt, 1).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	nTests := len(emery.Tests)
	testTrials := int(math32.IntMultipleGE(float32(nTests), float32(ss.Config.Run.NData)))

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, testTrials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, net, ss.NetViewUpdater, Cycle, Trial, Train,
		func(mode enums.Enum) { net.ClearInputs() },
		func(mode enums.Enum) { ss.TakeNextActions(mode.(Modes)) },
	)
	ls.Stacks[Train].OnInit.Add("Init", ss.Init)
	ls.Stacks[Test].OnInit.Add("Init", ss.TestInit)
	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	for mode, st := range ls.Stacks {
		cl := st.Loops[Cycle]
		cl.OnStart.Replace("Cycle", func() bool {
			getNeurons := axon.LooperCycleGetNeurons(ls, net, ss.NetViewUpdater, Cycle, Train)
			cyc := cl.Counter.Cur
			if cyc%ss.Config.Env.ReadNetInterval == 0 {
				getNeurons = true
			}
			net.Cycle(getNeurons)
			if axon.UseGPU && !getNeurons {
				net.Context().CycleInc() // keep synced
			}
			return false
		})
		cl.OnStart.Add("ApplyInputs", func() { ss.ApplyInputs(mode.(Modes)) })
		plusPhase := cl.EventByName("MinusPhase:End")
		plusPhase.OnEvent.InsertBefore("PlusPhase:Start", "NextAction", func() bool {
			// note: critical to have this happen *after* MinusPhase:End and *before* PlusPhase:Start
			// because minus phase end has gated info, and plus phase start applies action input
			ss.NextAction(mode.(Modes))
			return false
		})
	}

	trainEpoch := ls.Loop(Train, Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			ss.TestAll()
		}
	})

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		axon.SaveWeightsIfConfigSet(net, ss.Config.Log.SaveWeights, ctrString, ss.RunName())
	})

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
		ls.Stacks[Test].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
		ls.Loop(Train, Trial).OnEnd.Add("UpdateEnvGUI", func() {
			ss.UpdateEnvGUI(Train)
		})
		ls.Loop(Test, Cycle).OnEnd.Add("UpdateEnvGUI", func() {
			ss.UpdateEnvGUI(Test)
		})
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns for given mode.
// This is called every Cycle, not trial.
func (ss *Sim) ApplyInputs(mode Modes) {
	net := ss.Net
	ctx := net.Context()
	ndata := int(ctx.NData)

	ss.ReadNetState(mode)

	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	curModeDir := ss.Current.Dir(mode.String())
	ev := ss.Envs.ByMode(mode).(*emery.EmeryEnv)
	cyc := int(ctx.Cycle)
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
			fmt.Println("nil pats:", lnm)
		}
	}
	ss.Net.ApplyExts()
	if cyc == 0 {
		for di := uint32(0); di < ctx.NData; di++ {
			curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), int(di))
		}
	}
}

// ReadNetState reads state back from the model, including motor control layer
// activity, which drives corresponding motor actions.
func (ss *Sim) ReadNetState(mode Modes) {
	net := ss.Net
	ctx := net.Context()
	ndata := int(ctx.NData)
	lays := []string{"EyeH"}
	curModeDir := ss.Current.Dir(mode.String())
	ev := ss.Envs.ByMode(mode).(*emery.EmeryEnv)
	cyc := int(ctx.Cycle) - 1
	interval := ss.Config.Env.ReadNetInterval
	read := cyc%interval == 0
	cycMax := int(ctx.ThetaCycles) / interval
	cycIndex := cyc / interval

	readIdx := cycIndex
	readMax := cycMax
	if !axon.UseGPU { // use per-cycle in non-GPU mode -- free
		readIdx = cyc
		readMax = int(ctx.ThetaCycles)
	}

	if !axon.UseGPU || read {
		axon.NuclearReadIO("VShv", "Up", ss.Current, net, mode, interval, readIdx, readMax)
		axon.NuclearReadIO("VMhvr", "Up", ss.Current, net, mode, interval, readIdx, readMax)
		axon.NuclearReadIO("VMhvi", "Up", ss.Current, net, mode, interval, readIdx, readMax)
		axon.NuclearReadUp("VShv", ss.Current, net, mode, interval, readIdx, readMax)
		axon.NuclearReadDn("VShv", ss.Current, net, mode, interval, readIdx, readMax)
		axon.NuclearReadUp("VMhvr", ss.Current, net, mode, interval, readIdx, readMax)
		axon.NuclearReadUp("VMhvi", ss.Current, net, mode, interval, readIdx, readMax)
	}
	if !read {
		return
	}

	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		for di := range ndata {
			l := ly.AvgMaxVarByPool("CaP", 1, di).Avg
			r := ly.AvgMaxVarByPool("CaP", 2, di).Avg
			lr := l - r
			curModeDir.Float64(lnm, ndata, cycMax).Set(float64(lr), di, cycIndex)
			ev.TakeAction(di, emery.EyeH, lr) // subject to usual delay
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
	if mode == Test {
		ev.NextTest()
		return
	}
	for di := 0; di < ndata; di++ {
		ang := 2.0 * (ev.Rand.Float32() - 0.5) * ev.Params.MaxRotate
		ev.NextAction(di, emery.Rotate, ang)
		ev.NextAction(di, emery.VORCtrl, 0.5) // uses its own probability
	}
}

// TakeNextActions starts executing actions specified in NextAction.
// This is called at start of trial. Renders efferent copy of actions.
func (ss *Sim) TakeNextActions(mode Modes) {
	ev := ss.Envs.ByMode(mode).(*emery.EmeryEnv)
	ev.TakeNextActions()
	ndata := int(ss.Net.Context().NData)
	curModeDir := ss.Current.Dir(mode.String())
	di := 0 // todo: get current view
	curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), di)
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

// TestInit initializes the test process
func (ss *Sim) TestInit() {
	run := ss.Loops.Loop(Train, Run).Counter.Cur
	ss.Envs.ByMode(Test).Init(run)
	ss.Net.InitActs()
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(Test).Init(0)
	ss.Loops.ResetAndRun(Test)
	ss.Loops.Mode = Train // important because this is called from Train Run: go back.
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
	if level < Trial { // < not <=
		return
	}
	ss.RunStats(mode, level-1, axon.Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	// if level == Cycle {
	// 	return
	// }
	ss.RunStats(mode, level, axon.Step)
	tensorfs.DirTable(axon.StatsNode(ss.Stats, mode, level), nil).WriteToLog()
}

// RunStats runs the StatFuncs for given mode, level and phase.
func (ss *Sim) RunStats(mode Modes, level Levels, start bool) {
	for _, sf := range ss.StatFuncs {
		sf(mode, level, start)
	}
	if level > Cycle {
		if !start && ss.GUI.Tabs != nil {
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
			ss.RunStats(mode, level, axon.Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Trial))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Cycle))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Cycle))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
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
	ss.AddStatStd(axon.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial))
	ss.AddStatStd(axon.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle))
	ss.AddStatStd(axon.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial))
	ss.AddStatStd(axon.StatPerTrialMSec(ss.Stats, Train, Trial))

	ss.ConfigStatVOR()
	ss.ConfigStatVis()

	pool := 0
	interval := ss.Config.Env.ReadNetInterval
	ss.AddStatStd(axon.StatNuclearCycleIO("VShv", "Up", interval, pool, ss.Stats, ss.Current, net, Cycle))
	ss.AddStatStd(axon.StatNuclearCycleIO("VMhvr", "Up", interval, pool, ss.Stats, ss.Current, net, Cycle))
	ss.AddStatStd(axon.StatNuclearCycleIO("VMhvi", "Up", interval, pool, ss.Stats, ss.Current, net, Cycle))
	ss.AddStatStd(axon.StatNuclearCycleUp("VShv", interval, pool, ss.Stats, ss.Current, net, Cycle))
	ss.AddStatStd(axon.StatNuclearCycleDn("VShv", interval, pool, ss.Stats, ss.Current, net, Cycle))
	ss.AddStatStd(axon.StatNuclearCycleUp("VMhvr", interval, pool, ss.Stats, ss.Current, net, Cycle))
	ss.AddStatStd(axon.StatNuclearCycleUp("VMhvi", interval, pool, ss.Stats, ss.Current, net, Cycle))

	ss.AddStatStd(axon.StatNuclearTrialUp("VShv", pool, ss.Stats, ss.Current, net, Trial, Run))
	ss.AddStatStd(axon.StatNuclearTrialUp("VMhvr", pool, ss.Stats, ss.Current, net, Trial, Run))
	ss.AddStatStd(axon.StatNuclearTrialUp("VMhvi", pool, ss.Stats, ss.Current, net, Trial, Run))

	plays := net.LayersByType(axon.PulvinarLayer)
	ss.AddStatStd(axon.StatCorSim(ss.Stats, ss.Current, net, Trial, Run, plays...))
	ss.AddStatStd(axon.StatPrevCorSim(ss.Stats, ss.Current, net, Trial, Run, plays...))

	lays := net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer, axon.InputLayer, axon.PulvinarLayer, axon.CNiIOLayer, axon.CNiUpLayer, axon.CNeDnLayer)
	ss.AddStatStd(axon.StatLayerActGe(ss.Stats, net, Train, Trial, Run, lays...))

	pcaFunc := axon.StatPCA(ss.Stats, ss.Current, net, ss.Config.Run.PCAInterval, Train, Trial, Run, lays...)
	ss.AddStat(func(mode Modes, level Levels, start bool) {
		trnEpc := ss.Loops.Loop(Train, Epoch).Counter.Cur
		pcaFunc(mode, level, start, trnEpc)
	})

	// ss.AddStatStd(axon.StatLayerState(ss.Stats, net, Test, Trial, true, "ActM", "Depth", "DepthP", "HeadDir", "HeadDirP"))
}

func (ss *Sim) ConfigStatVis() {
	statNames := []string{"VisVestibCor", "EmeryAng"}
	statDescs := map[string]string{
		"VisVestibCor": "Correlation between the visual motion and vestibular rotation velocity signals, indicating quality of visual motion filters",
		"EmeryAng":     "Emery's current body angle",
	}
	ss.AddStat(func(mode Modes, level Levels, start bool) {
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
			if start {
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
						stat = float64(ev.SenseValue(di, emery.VShd, false)) // current
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

func (ss *Sim) ConfigStatVOR() {
	statNames := []string{"VORCtrl", "VORSlip", "VORiSlip"}
	statDescs := map[string]string{
		"VORCtrl":  "whether VOR was inhibited (1) or not (0)",
		"VORSlip":  "Max visual slip on VOR engaged trials.",
		"VORiSlip": "Max visual slip on VOR inhibited trials -- for reference.",
	}
	ss.AddStat(func(mode Modes, level Levels, start bool) {
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
			ev := ss.Envs.ByMode(mode).(*emery.EmeryEnv)
			if start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					if name != "VORCtrl" {
						s.On = true
						s.RightY = true
					}
				})
				metadata.SetDoc(tsr, statDescs[name])
				continue
			}
			switch level {
			case Trial:
				for di := range ndata {
					es := ev.EmeryState(di)
					var stat float32
					switch name {
					case "VORCtrl":
						stat = es.CurActions[emery.VORCtrl]
					case "VORSlip":
						vi := es.CurActions[emery.VORCtrl]
						if vi > 0 {
							stat = math32.NaN()
						} else {
							stat = math32.Abs(es.SenseNormed[emery.VMhp] - es.SenseStart[emery.VMhp])
						}
					case "VORiSlip":
						vi := es.CurActions[emery.VORCtrl]
						if vi > 0 {
							stat = math32.Abs(es.SenseNormed[emery.VMhp] - es.SenseStart[emery.VMhp])
						} else {
							stat = math32.NaN()
						}
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
	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 2.0, 2.1)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.CycleUpdateInterval = 10
	ss.GUI.StopLevel = Trial
	nv := ss.GUI.AddNetView("Network")
	nv.Settings.Paths = false
	nv.Settings.MaxRecs = 2 * ss.Config.Run.Cycles()
	nv.Settings.Raster.Max = ss.Config.Run.Cycles()
	nv.Settings.LayerNameSize = 0.02
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Theta, ss.StatCounters) // Theta
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
