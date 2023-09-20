// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
boa: This project tests BG, OFC & ACC learning in a CS-driven approach task.
*/
package main

import (
	"fmt"
	"log"
	"math"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/examples/boa/armaze"
	"github.com/emer/emergent/econfig"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/timer"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/ki/bools"
	"github.com/goki/mat32"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		gimain.Main(sim.RunGUI)
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
	Config Config `desc:"simulation configuration parameters -- set by .toml config file and / or args"`

	// [view: no-inline] the network -- click to view / edit parameters for layers, prjns, etc
	Net *axon.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`

	// if true, stop running at end of a sequence (for NetView Di data parallel index)
	StopOnSeq bool `desc:"if true, stop running at end of a sequence (for NetView Di data parallel index)"`

	// if true, stop running when an error programmed into the code occurs
	StopOnErr bool `desc:"if true, stop running when an error programmed into the code occurs"`

	// [view: inline] network parameter management
	Params emer.NetParams `view:"inline" desc:"network parameter management"`

	// [view: no-inline] contains looper control loops for running sim
	Loops *looper.Manager `view:"no-inline" desc:"contains looper control loops for running sim"`

	// contains computed statistic values
	Stats estats.Stats `desc:"contains computed statistic values"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `desc:"Contains all the logs and information about the logs.'"`

	// [view: no-inline] Environments
	Envs env.Envs `view:"no-inline" desc:"Environments"`

	// axon timing parameters and state
	Context axon.Context `desc:"axon timing parameters and state"`

	// [view: inline] netview update parameters
	ViewUpdt netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	// [view: -] manages all the gui elements
	GUI egui.GUI `view:"-" desc:"manages all the gui elements"`

	// [view: -] gui for viewing env
	EnvGUI *armaze.GUI `view:"-" desc:"gui for viewing env"`

	// [view: -] a list of random seeds to use for each run
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`

	// [view: -] testing data, from -test arg
	TestData map[string]float32 `view:"-" desc:"testing data, from -test arg"`
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
		var trn *armaze.Env
		if newEnv {
			trn = &armaze.Env{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*armaze.Env)
		}

		// note: names must be standard here!
		trn.Nm = env.ModeDi(etime.Train, di)
		trn.Defaults()
		trn.RndSeed = 73
		if !ss.Config.Env.SameSeed {
			trn.RndSeed += int64(di) * 73
		}
		if ss.Config.Env.Env != nil {
			params.ApplyMap(&trn.Config, ss.Config.Env.Env, ss.Config.Debug)
		}
		trn.Config.NDrives = ss.Config.Env.NDrives
		trn.ConfigEnv(di)
		trn.Validate()

		trn.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn)
		if di == 0 {
			ss.ConfigPVLV(trn)
		}
	}
}

func (ss *Sim) ConfigPVLV(trn *armaze.Env) {
	pv := &ss.Net.PVLV
	pv.SetNUSs(&ss.Context, trn.Config.NDrives, 1)
	pv.Defaults()
	pv.USs.PVposGain = 2  // higher = more pos reward (saturating logistic func)
	pv.USs.PVnegGain = .1 // global scaling of PV neg level -- was 1

	pv.USs.USnegGains[0] = 0.1 // time: if USneg pool is saturating, reduce
	pv.USs.USnegGains[1] = 0.1 // effort: if USneg pool is saturating, reduce
	pv.USs.USnegGains[2] = 2   // big salient input!

	pv.USs.PVnegWts[0] = 0.02 // time: controls overall PVneg -- if too high, not enough reward..
	pv.USs.PVnegWts[1] = 0.02 // effort: controls overall PVneg -- if too high, not enough reward..
	pv.USs.PVnegWts[2] = 1

	pv.Drive.DriveMin = 0.5 // 0.5 -- should be
	pv.Urgency.U50 = 10
	if ss.Config.Params.PVLV != nil {
		params.ApplyMap(pv, ss.Config.Params.PVLV, ss.Config.Debug)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)
	net.InitName(net, "Boa")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	nAct := int(armaze.ActionsN)
	popY := 4
	popX := 4
	space := float32(2)

	pone2one := prjn.NewPoolOneToOne()
	one2one := prjn.NewOneToOne()
	full := prjn.NewFull()
	mtxRndPrjn := prjn.NewPoolUnifRnd()
	mtxRndPrjn.PCon = 0.75
	_ = mtxRndPrjn
	_ = pone2one
	prjnClass := "PFCPrjn"

	ny := ev.Config.Params.NYReps
	narm := ev.Config.NArms

	vSgpi, urgency, pvPos, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPosUS, ofcPosUSCT, ofcPosUSPTp, ofcPosVal, ofcPosValCT, ofcPosValPTp, ofcNegUS, ofcNegUSCT, ofcNegUSPTp, accNegVal, accNegValCT, accNegValPTp, accUtil, sc, notMaint := net.AddBOA(ctx, ny, popY, popX, nuBgY, nuBgX, nuCtxY, nuCtxX, space)
	_, _ = accUtil, urgency
	_, _ = ofcNegUSCT, ofcNegUSPTp

	accUtilPTp := net.AxonLayerByName("ACCutilPTp")

	cs, csP := net.AddInputPulv2D("CS", ny, ev.Config.NCSs, space)
	pos, posP := net.AddInputPulv2D("Pos", ny, ev.MaxLength+1, space)
	arm := net.AddLayer2D("Arm", ny, narm, axon.InputLayer) // irrelevant here

	///////////////////////////////////////////
	// M1, VL, ALM

	act := net.AddLayer2D("Act", ny, nAct, axon.InputLayer) // Action: what is actually done
	vl := net.AddPulvLayer2D("VL", ny, nAct)                // VL predicts brainstem Action
	vl.SetBuildConfig("DriveLayName", act.Name())

	m1, m1CT := net.AddSuperCT2D("M1", "PFCPrjn", nuCtxY, nuCtxX, space, one2one)
	m1P := net.AddPulvForSuper(m1, space)

	alm, almCT, almPT, almPTp, almMD := net.AddPFC2D("ALM", "MD", nuCtxY, nuCtxX, true, space)
	_ = almPT

	net.ConnectLayers(vSgpi, almMD, full, axon.InhibPrjn)
	// net.ConnectToMatrix(alm, vSmtxGo, full) // todo: explore
	// net.ConnectToMatrix(alm, vSmtxNo, full)

	net.ConnectToPFCBidir(m1, m1P, alm, almCT, almPTp, full) // alm predicts m1

	// vl is a predictive thalamus but we don't have direct access to its source
	net.ConnectToPulv(m1, m1CT, vl, full, full, prjnClass)
	net.ConnectToPFC(nil, vl, alm, almCT, almPTp, full) // alm predicts m1

	// sensory inputs guiding action
	// note: alm gets effort, pos via predictive coding below

	net.ConnectLayers(pos, m1, full, axon.ForwardPrjn).SetClass("ToM1")
	net.ConnectLayers(ofcNegUS, m1, full, axon.ForwardPrjn).SetClass("ToM1")

	// shortcut: not needed
	// net.ConnectLayers(pos, vl, full, axon.ForwardPrjn).SetClass("ToVL")

	// these projections are *essential* -- must get current state here
	net.ConnectLayers(m1, vl, full, axon.ForwardPrjn).SetClass("ToVL")
	net.ConnectLayers(alm, vl, full, axon.ForwardPrjn).SetClass("ToVL")

	// key point: cs does not project directly to alm -- no simple S -> R mappings!?

	///////////////////////////////////////////
	// CS -> BLA, OFC

	net.ConnectToSC1to1(cs, sc)

	net.ConnectCSToBLAPos(cs, blaPosAcq, blaNov)
	net.ConnectToBLAExt(cs, blaPosExt, full)

	net.ConnectToBLAAcq(cs, blaNegAcq, full)
	net.ConnectToBLAExt(cs, blaNegExt, full)

	// OFCus predicts cs
	net.ConnectToPFCBack(cs, csP, ofcPosUS, ofcPosUSCT, ofcPosUSPTp, full)

	///////////////////////////////////////////
	// OFC, ACC, ALM predicts pos

	// todo: a more dynamic US rep is needed to drive predictions in OFC
	// using distance and effort here in the meantime
	net.ConnectToPFCBack(pos, posP, ofcPosUS, ofcPosUSCT, ofcPosUSPTp, full)
	net.ConnectToPFCBack(pos, posP, ofcPosVal, ofcPosValCT, ofcPosValPTp, full)

	net.ConnectToPFC(pos, posP, ofcNegUS, ofcNegUSCT, ofcNegUSPTp, full)
	net.ConnectToPFC(pos, posP, accNegVal, accNegValCT, accNegValPTp, full)

	//	alm predicts all effort, cost, sensory state vars
	net.ConnectToPFC(pos, posP, alm, almCT, almPTp, full)

	///////////////////////////////////////////
	// ALM, M1 <-> OFC, ACC

	// action needs to know if maintaining a goal or not
	// using accUtil as main summary "driver" input to action system
	// PTp provides good notmaint signal for action.
	net.ConnectLayers(accUtilPTp, alm, full, axon.ForwardPrjn).SetClass("ToALM")
	net.ConnectLayers(accUtilPTp, m1, full, axon.ForwardPrjn).SetClass("ToM1")

	// note: in Obelisk this helps with the Consume action
	// but here in this example it produces some instability
	// at later time points -- todo: investigate later.
	// net.ConnectLayers(notMaint, vl, full, axon.ForwardPrjn).SetClass("ToVL")

	////////////////////////////////////////////////
	// position

	cs.PlaceRightOf(pvPos, space)
	pos.PlaceRightOf(cs, space)
	arm.PlaceRightOf(pos, space)

	m1.PlaceRightOf(arm, space)
	alm.PlaceRightOf(m1, space)
	vl.PlaceBehind(m1P, space)
	act.PlaceBehind(vl, space)

	notMaint.PlaceRightOf(alm, space)

	net.Build(ctx)
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	ss.Net.InitWts(ctx)
}

func (ss *Sim) ApplyParams() {
	net := ss.Net
	ss.Params.SetAll() // first hard-coded defaults

	// params that vary as number of CSs
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)

	nCSTot := ev.Config.NCSs

	cs := net.AxonLayerByName("CS")
	cs.Params.Inhib.ActAvg.Nominal = 0.32 / float32(nCSTot)
	csp := net.AxonLayerByName("CSP")
	csp.Params.Inhib.ActAvg.Nominal = 0.32 / float32(nCSTot)
	bla := net.AxonLayerByName("BLAPosAcqD1")
	pji, _ := bla.SendNameTry("BLANovelCS")
	pj := pji.(*axon.Prjn)

	// this is very sensitive param to get right
	// too little and the hamster does not try CSs at the beginning,
	// too high and it gets stuck trying the same location over and over
	pj.Params.PrjnScale.Abs = float32(math.Min(float64(2.3+(float32(nCSTot)/10.0)), 3.0))

	// then apply config-set params.
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
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
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
	// ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)

	// note: sequence stepping does not work in NData > 1 mode -- just going back to raw trials
	trls := int(mat32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

	for m := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	// note: phase is shared between all stacks!
	plusPhase, _ := man.Stacks[etime.Train].Loops[etime.Cycle].EventByName("PlusPhase")
	plusPhase.OnEvent.InsertBefore("PlusPhase:Start", "TakeAction", func() {
		// note: critical to have this happen *after* MinusPhase:End and *before* PlusPhase:Start
		// because minus phase end has gated info, and plus phase start applies action input
		ss.TakeAction(ss.Net)
	})

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			axon.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)
	if ss.Config.GUI {
		man.GetLoop(etime.Train, etime.Trial).OnStart.Add("ResetDebugTrial", func() {
			di := uint32(ss.ViewUpdt.View.Di)
			hadRew := axon.GlbV(&ss.Context, di, axon.GvHadRew) > 0
			if hadRew {
				ss.Logs.ResetLog(etime.Debug, etime.Trial)
			}
		})
	}

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	if ss.Config.Log.Testing {
		man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("RecordTestData", func() {
			ss.RecordTestData()
		})
	}

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWts, ctrString, ss.Stats.String("RunName"))
	})

	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PctCortex", func() {
		trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		ss.Config.Env.CurPctCortex(trnEpc)
	})

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

		man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("UpdateWorldGui", func() {
			ss.UpdateEnvGUI(etime.Train)
		})
	}

	if ss.Config.Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
// Called at end of minus phase. However, it can still gate sometimes
// after this point, so that is dealt with at end of plus phase.
func (ss *Sim) TakeAction(net *axon.Network) {
	ctx := &ss.Context
	pv := &ss.Net.PVLV
	mtxLy := ss.Net.AxonLayerByName("VsMtxGo")
	vlly := ss.Net.AxonLayerByName("VL")
	threshold := float32(0.1)
	for di := 0; di < int(ctx.NetIdxs.NData); di++ {
		diu := uint32(di)
		ev := ss.Envs.ByModeDi(ctx.Mode, di).(*armaze.Env)
		justGated := mtxLy.AnyGated(diu) // not updated until plus phase: pv.VSMatrix.JustGated.IsTrue()
		hasGated := axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0
		ev.InstinctAct(justGated, hasGated)
		csGated := (justGated && !pv.HasPosUS(ctx, diu))
		deciding := !csGated && !hasGated && (axon.GlbV(ctx, diu, axon.GvACh) > threshold && mtxLy.Pool(0, diu).AvgMax.SpkMax.Cycle.Max > threshold) // give it time
		wasDeciding := bools.FromFloat32(ss.Stats.Float32Di("Deciding", di))
		if wasDeciding {
			deciding = false // can't keep deciding!
		}
		ss.Stats.SetFloat32Di("Deciding", di, bools.ToFloat32(deciding))

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
			ss.Stats.SetStringDi("Debug", di, act)
			ev.Action("None", nil)
			ss.ApplyAction(di)
			ss.Stats.SetStringDi("ActAction", di, "None")
			ss.Stats.SetStringDi("Instinct", di, "None")
			ss.Stats.SetStringDi("NetAction", di, act)
			ss.Stats.SetFloatDi("ActMatch", di, 1)                // whatever it is, it is ok
			vlly.Pool(0, uint32(di)).Inhib.Clamped.SetBool(false) // not clamped this trial
		} else {
			ss.Stats.SetStringDi("Debug", di, "acting")
			netAct := ss.DecodeAct(ev, di)
			genAct := ev.InstinctAct(justGated, hasGated)
			ss.Stats.SetStringDi("NetAction", di, netAct.String())
			ss.Stats.SetStringDi("Instinct", di, genAct.String())
			if netAct == genAct {
				ss.Stats.SetFloatDi("ActMatch", di, 1)
			} else {
				ss.Stats.SetFloatDi("ActMatch", di, 0)
			}

			actAct := genAct
			if ss.Stats.FloatDi("CortexDriving", di) > 0 {
				actAct = netAct
			}
			ss.Stats.SetStringDi("ActAction", di, actAct.String())

			ev.Action(actAct.String(), nil)
			ss.ApplyAction(di)

			switch {
			case pv.HasPosUS(ctx, diu):
				trSt = armaze.TrRewarded
			case actAct == armaze.Consume:
				trSt = armaze.TrConsuming
			}
		}
		if axon.GlbV(ctx, diu, axon.GvGiveUp) > 0 {
			trSt = armaze.TrGiveUp
		}
		ss.Stats.SetIntDi("TraceStateInt", di, int(trSt))
		ss.Stats.SetStringDi("TraceState", di, trSt.String())

	}
	ss.Net.ApplyExts(ctx)
	ss.Net.GPU.SyncPoolsToGPU()
}

// DecodeAct decodes the VL ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ev *armaze.Env, di int) armaze.Actions {
	vt := ss.Stats.SetLayerTensor(ss.Net, "VL", "CaSpkP", di) // was "Act"
	return ev.DecodeAct(vt)
}

func (ss *Sim) ApplyAction(di int) {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByModeDi(ss.Context.Mode, di).(*armaze.Env)
	ap := ev.State("Action")
	ly := net.AxonLayerByName("Act")
	ly.ApplyExt(ctx, uint32(di), ap)
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	ss.Stats.SetString("Debug", "") // start clear
	net := ss.Net
	lays := []string{"Pos", "Arm", "CS"}

	ss.Net.InitExt(ctx)
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		ev := ss.Envs.ByModeDi(ctx.Mode, int(di)).(*armaze.Env)
		giveUp := axon.GlbV(ctx, di, axon.GvGiveUp) > 0
		if giveUp {
			ev.JustConsumed = true // triggers a new start -- we just consumed the giving up feeling :)
		}
		ev.Step()
		if ev.Tick == 0 {
			ss.Stats.SetFloat32Di("CortexDriving", int(di), bools.ToFloat32(erand.BoolP32(ss.Config.Env.PctCortex, -1)))
			ev.ExValueUtil(&ss.Net.PVLV, ctx)
		}
		for _, lnm := range lays {
			ly := net.AxonLayerByName(lnm)
			itsr := ev.State(lnm)
			ly.ApplyExt(ctx, di, itsr)
		}
		ss.ApplyPVLV(ctx, ev, di)
	}
	ss.Net.ApplyExts(ctx)
}

// ApplyPVLV applies current PVLV values to Context.PVLV,
// from given trial data.
func (ss *Sim) ApplyPVLV(ctx *axon.Context, ev *armaze.Env, di uint32) {
	pv := &ss.Net.PVLV
	pv.NewState(ctx, di, &ss.Net.Rand) // first before anything else is updated
	pv.EffortUrgencyUpdt(ctx, di, 1)   // note: effort can vary with terrain!
	if ev.USConsumed >= 0 {
		pv.SetUS(ctx, di, axon.Positive, ev.USConsumed, ev.USValue)
	}
	pv.SetDrives(ctx, di, 0.5, ev.Drives...)
	pv.Step(ctx, di, &ss.Net.Rand)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRndSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	for di := 0; di < int(ctx.NetIdxs.NData); di++ {
		ss.Envs.ByModeDi(etime.Train, di).Init(0)
	}
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Config.Env.PctCortex = 0
	ss.Net.InitWts(ctx)
	ss.InitStats()
	ss.StatCounters(0)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	if ss.Config.OpenWts != "" {
		ss.Net.OpenWtsJSON(gi.FileName(ss.Config.OpenWts))
		log.Println("Opened weights:", ss.Config.OpenWts)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetInt("Di", 0)
	ss.Stats.SetFloat("PctCortex", 0)
	ss.Stats.SetFloat("Pos", 0)
	ss.Stats.SetFloat("Drive", 0)
	ss.Stats.SetFloat("CS", 0)
	ss.Stats.SetFloat("US", 0)
	ss.Stats.SetFloat("HasRew", 0)
	ss.Stats.SetString("NetAction", "")
	ss.Stats.SetString("Instinct", "")
	ss.Stats.SetString("ActAction", "")
	ss.Stats.SetString("TraceState", "")
	ss.Stats.SetInt("TraceStateInt", 0)
	ss.Stats.SetFloat("ActMatch", 0)
	ss.Stats.SetFloat("AllGood", 0)

	ss.Stats.SetFloat("JustGated", 0)
	ss.Stats.SetFloat("Should", 0)
	ss.Stats.SetFloat("GateUS", 0)
	ss.Stats.SetFloat("GateCS", 0)
	ss.Stats.SetFloat("Deciding", 0)
	ss.Stats.SetFloat("GatedEarly", 0)
	ss.Stats.SetFloat("MaintEarly", 0)
	ss.Stats.SetFloat("GatedAgain", 0)
	ss.Stats.SetFloat("WrongCSGate", 0)
	ss.Stats.SetFloat("AChShould", 0)
	ss.Stats.SetFloat("AChShouldnt", 0)
	ss.Stats.SetFloat("Rew", 0)
	ss.Stats.SetFloat("DA", 0)
	ss.Stats.SetFloat("RewPred", 0)
	ss.Stats.SetFloat("DA_NR", 0)
	ss.Stats.SetFloat("RewPred_NR", 0)
	ss.Stats.SetFloat("DA_GiveUp", 0)
	ss.Stats.SetFloat("VSPatchThr", 0)

	ss.Stats.SetFloat("Time", 0)
	ss.Stats.SetFloat("Effort", 0)
	ss.Stats.SetFloat("Urgency", 0)

	ss.Stats.SetFloat("NegUSOutcome", 0)
	ss.Stats.SetFloat("PVpos", 0)
	ss.Stats.SetFloat("PVneg", 0)

	ss.Stats.SetFloat("PVposEst", 0)
	ss.Stats.SetFloat("PVposEstDisc", 0)
	ss.Stats.SetFloat("GiveUpDiff", 0)
	ss.Stats.SetFloat("GiveUpProb", 0)
	ss.Stats.SetFloat("GiveUp", 0)

	ss.Stats.SetFloat("LHbDip", 0)
	ss.Stats.SetFloat("LHbBurst", 0)
	ss.Stats.SetFloat("LHbDA", 0)

	ss.Stats.SetFloat("CeMpos", 0)
	ss.Stats.SetFloat("CeMneg", 0)
	ss.Stats.SetFloat("SC", 0)

	lays := ss.Net.LayersByType(axon.PTMaintLayer)
	for _, lnm := range lays {
		ss.Stats.SetFloat("Maint"+lnm, 0)
		ss.Stats.SetFloat("MaintFail"+lnm, 0)
		ss.Stats.SetFloat("PreAct"+lnm, 0)
	}
	ss.Stats.SetString("Debug", "") // special debug notes per trial
}

// StatCounters saves current counters to Stats, so they are available for logging etc
func (ss *Sim) StatCounters(di int) {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.ActionStatsDi(di)
	ev := ss.Envs.ByModeDi(mode, di).(*armaze.Env)
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl+di)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	ss.Stats.SetFloat32("PctCortex", ss.Config.Env.PctCortex)
	ss.Stats.SetFloat32("Pos", float32(ev.Pos))
	ss.Stats.SetFloat32("Arm", float32(ev.Arm))
	// ss.Stats.SetFloat32("Drive", float32(ev.Drive))
	ss.Stats.SetFloat32("CS", float32(ev.CurCS()))
	ss.Stats.SetFloat32("US", float32(ev.USConsumed))
	ss.Stats.SetFloat32("HasRew", axon.GlbV(ctx, uint32(di), axon.GvHasRew))
	ss.Stats.SetString("TrialName", "trl") // todo: could have dist, US etc
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
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di", "Cycle", "NetAction", "Instinct", "ActAction", "ActMatch", "JustGated", "Should", "Rew"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
	ss.GatedStats(di)
	ss.MaintStats(di)

	diu := uint32(di)
	ctx := &ss.Context
	pv := &ss.Net.PVLV
	nan := math.NaN()
	ss.Stats.SetFloat("DA", nan)
	ss.Stats.SetFloat("RewPred", nan)
	ss.Stats.SetFloat("Rew", nan)
	ss.Stats.SetFloat("HasRew", nan)
	ss.Stats.SetFloat("DA_NR", nan)
	ss.Stats.SetFloat("RewPred_NR", nan)
	ss.Stats.SetFloat("DA_GiveUp", nan)
	if pv.HasPosUS(ctx, diu) {
		ss.Stats.SetFloat32("DA", axon.GlbV(ctx, diu, axon.GvDA))
		ss.Stats.SetFloat32("RewPred", axon.GlbV(ctx, diu, axon.GvRewPred)) // gets from VSPatch or RWPred etc
		ss.Stats.SetFloat32("Rew", axon.GlbV(ctx, diu, axon.GvRew))
		ss.Stats.SetFloat("HasRew", 1)
	} else {
		if axon.GlbV(ctx, diu, axon.GvGiveUp) > 0 || axon.GlbV(ctx, diu, axon.GvNegUSOutcome) > 0 {
			ss.Stats.SetFloat32("DA_GiveUp", axon.GlbV(ctx, diu, axon.GvDA))
		} else {
			ss.Stats.SetFloat32("DA_NR", axon.GlbV(ctx, diu, axon.GvDA))
			ss.Stats.SetFloat32("RewPred_NR", axon.GlbV(ctx, diu, axon.GvRewPred))
			ss.Stats.SetFloat("HasRew", 0)
		}
	}

	vsLy := ss.Net.AxonLayerByName("VsPatch")
	ss.Stats.SetFloat32("VSPatchThr", vsLy.Vals[0].ActAvg.AdaptThr)

	ss.Stats.SetFloat32("Time", axon.GlbV(ctx, diu, axon.GvTime))
	ss.Stats.SetFloat32("Effort", axon.GlbV(ctx, diu, axon.GvEffort))
	ss.Stats.SetFloat32("Urgency", axon.GlbV(ctx, diu, axon.GvUrgency))

	ss.Stats.SetFloat32("NegUSOutcome", axon.GlbV(ctx, diu, axon.GvNegUSOutcome))
	ss.Stats.SetFloat32("PVpos", axon.GlbV(ctx, diu, axon.GvPVpos))
	ss.Stats.SetFloat32("PVneg", axon.GlbV(ctx, diu, axon.GvPVneg))

	ss.Stats.SetFloat32("PVposEst", axon.GlbV(ctx, diu, axon.GvPVposEst))
	ss.Stats.SetFloat32("PVposEstDisc", axon.GlbV(ctx, diu, axon.GvPVposEstDisc))
	ss.Stats.SetFloat32("GiveUpDiff", axon.GlbV(ctx, diu, axon.GvGiveUpDiff))
	ss.Stats.SetFloat32("GiveUpProb", axon.GlbV(ctx, diu, axon.GvGiveUpProb))
	ss.Stats.SetFloat32("GiveUp", axon.GlbV(ctx, diu, axon.GvGiveUp))

	ss.Stats.SetFloat32("LHbDip", axon.GlbV(ctx, diu, axon.GvLHbDip))
	ss.Stats.SetFloat32("LHbBurst", axon.GlbV(ctx, diu, axon.GvLHbBurst))
	ss.Stats.SetFloat32("LHbDA", axon.GlbV(ctx, diu, axon.GvLHbPVDA))

	ss.Stats.SetFloat32("CeMpos", axon.GlbV(ctx, diu, axon.GvCeMpos))
	ss.Stats.SetFloat32("CeMneg", axon.GlbV(ctx, diu, axon.GvCeMneg))

	ss.Stats.SetFloat32("SC", ss.Net.AxonLayerByName("SC").Pool(0, 0).AvgMax.CaSpkD.Cycle.Max)

	ss.Stats.SetFloat32("ACh", axon.GlbV(ctx, diu, axon.GvACh))
	ss.Stats.SetFloat32("AChRaw", axon.GlbV(ctx, diu, axon.GvAChRaw))

	var allGood float64
	agN := 0
	if fv := ss.Stats.Float("GateUS"); !math.IsNaN(fv) {
		allGood += fv
		agN++
	}
	if fv := ss.Stats.Float("GateCS"); !math.IsNaN(fv) {
		allGood += fv
		agN++
	}
	if fv := ss.Stats.Float("ActMatch"); !math.IsNaN(fv) {
		allGood += fv
		agN++
	}
	if fv := ss.Stats.Float("GatedEarly"); !math.IsNaN(fv) {
		allGood += 1 - fv
		agN++
	}
	if fv := ss.Stats.Float("GatedAgain"); !math.IsNaN(fv) {
		allGood += 1 - fv
		agN++
	}
	if fv := ss.Stats.Float("WrongCSGate"); !math.IsNaN(fv) {
		allGood += 1 - fv
		agN++
	}
	if agN > 0 {
		allGood /= float64(agN)
	}
	ss.Stats.SetFloat("AllGood", allGood)
}

// ActionStatsDi copies the action info from given data parallel index
// into the global action stats
func (ss *Sim) ActionStatsDi(di int) {
	if _, has := ss.Stats.Strings[estats.DiName("NetAction", di)]; !has {
		return
	}
	ss.Stats.SetString("NetAction", ss.Stats.StringDi("NetAction", di))
	ss.Stats.SetString("Instinct", ss.Stats.StringDi("Instinct", di))
	ss.Stats.SetFloat("ActMatch", ss.Stats.FloatDi("ActMatch", di))
	ss.Stats.SetString("ActAction", ss.Stats.StringDi("ActAction", di))
	ss.Stats.SetString("TraceState", ss.Stats.StringDi("TraceState", di))
	ss.Stats.SetInt("TraceStateInt", ss.Stats.IntDi("TraceStateInt", di))
}

// GatedStats updates the gated states
func (ss *Sim) GatedStats(di int) {
	ctx := &ss.Context
	pv := &ss.Net.PVLV
	diu := uint32(di)
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*armaze.Env)
	justGated := axon.GlbV(ctx, diu, axon.GvVSMatrixJustGated) > 0
	hasGated := axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0
	nan := mat32.NaN()
	ss.Stats.SetString("Debug", ss.Stats.StringDi("Debug", di))
	ss.ActionStatsDi(di)

	ss.Stats.SetFloat32("JustGated", bools.ToFloat32(justGated))
	ss.Stats.SetFloat32("Should", bools.ToFloat32(ev.ShouldGate))
	ss.Stats.SetFloat32("HasGated", bools.ToFloat32(hasGated))
	ss.Stats.SetFloat32("GateUS", nan)
	ss.Stats.SetFloat32("GateCS", nan)
	ss.Stats.SetFloat32("GatedEarly", nan)
	ss.Stats.SetFloat32("MaintEarly", nan)
	ss.Stats.SetFloat32("GatedAgain", nan)
	ss.Stats.SetFloat32("WrongCSGate", nan)
	ss.Stats.SetFloat32("AChShould", nan)
	ss.Stats.SetFloat32("AChShouldnt", nan)
	hasPos := pv.HasPosUS(ctx, diu)
	if justGated {
		ss.Stats.SetFloat32("WrongCSGate", bools.ToFloat32(!ev.ArmIsMaxUtil(ev.Arm)))
	}
	if ev.ShouldGate {
		if hasPos {
			ss.Stats.SetFloat32("GateUS", bools.ToFloat32(justGated))
		} else {
			ss.Stats.SetFloat32("GateCS", bools.ToFloat32(justGated))
		}
	} else {
		if hasGated {
			ss.Stats.SetFloat32("GatedAgain", bools.ToFloat32(justGated))
		} else { // !should gate means early..
			ss.Stats.SetFloat32("GatedEarly", bools.ToFloat32(justGated))
		}
	}
	// We get get ACh when new CS or Rew
	if hasPos || ev.LastCS != ev.CurCS() {
		ss.Stats.SetFloat32("AChShould", axon.GlbV(ctx, diu, axon.GvACh))
	} else {
		ss.Stats.SetFloat32("AChShouldnt", axon.GlbV(ctx, diu, axon.GvACh))
	}
}

// MaintStats updates the PFC maint stats
func (ss *Sim) MaintStats(di int) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*armaze.Env)
	// should be maintaining while going forward
	isFwd := ev.LastAct == armaze.Forward
	isCons := ev.LastAct == armaze.Consume
	actThr := float32(0.05) // 0.1 too high
	net := ss.Net
	lays := net.LayersByType(axon.PTMaintLayer)
	hasMaint := false
	for _, lnm := range lays {
		mnm := "Maint" + lnm
		fnm := "MaintFail" + lnm
		pnm := "PreAct" + lnm
		ptly := net.AxonLayerByName(lnm)
		var mact float32
		if ptly.Is4D() {
			for pi := uint32(1); pi < ptly.NPools; pi++ {
				avg := ptly.Pool(pi, uint32(di)).AvgMax.Act.Plus.Avg
				if avg > mact {
					mact = avg
				}
			}
		} else {
			mact = ptly.Pool(0, uint32(di)).AvgMax.Act.Plus.Avg
		}
		overThr := mact > actThr
		if overThr {
			hasMaint = true
		}
		ss.Stats.SetFloat32(pnm, mat32.NaN())
		ss.Stats.SetFloat32(mnm, mat32.NaN())
		ss.Stats.SetFloat32(fnm, mat32.NaN())
		if isFwd {
			ss.Stats.SetFloat32(mnm, mact)
			ss.Stats.SetFloat32(fnm, bools.ToFloat32(!overThr))
		} else if !isCons {
			ss.Stats.SetFloat32(pnm, bools.ToFloat32(overThr))
		}
	}
	if hasMaint {
		ss.Stats.SetFloat32("MaintEarly", bools.ToFloat32(!ev.ArmIsMaxUtil(ev.Arm)))
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	// ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "PctCortex")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.Trial, "Drive", "CS", "Pos", "US", "HasRew")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "NetAction", "Instinct", "ActAction", "TraceState")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	axon.LogAddPulvCorSimItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	// ss.ConfigActRFs()

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer, axon.CeMLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	// todo: PCA items should apply to CT layers too -- pass a type here.
	axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.PlotItems("ActMatch", "GateCS", "Deciding", "GateUS", "WrongCSGate", "Rew", "RewPred", "RewPred_NR", "MaintEarly")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	// ss.Logs.SetMeta(etime.Test, etime.Cycle, "LegendCol", "RunName")

	axon.LayerActsLogConfig(ss.Net, &ss.Logs)
}

func (ss *Sim) ConfigLogItems() {
	ss.Logs.AddStatAggItem("AllGood", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ActMatch", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("JustGated", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Should", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GateUS", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GateCS", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Deciding", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GatedEarly", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("MaintEarly", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GatedAgain", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("WrongCSGate", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AChShould", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AChShouldnt", etime.Run, etime.Epoch, etime.Trial)

	li := ss.Logs.AddStatAggItem("Rew", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("ACh", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("AChRaw", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("RewPred", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA_NR", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("RewPred_NR", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA_GiveUp", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	ss.Logs.AddStatAggItem("VSPatchThr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("Time", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Effort", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Urgency", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("NegUSOutcome", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("PVpos", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("PVneg", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("PVposEst", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("PVposEstDisc", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GiveUpDiff", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GiveUpProb", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GiveUp", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("LHbDip", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("LHbBurst", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("LHbDA", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("CeMpos", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("CeMneg", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("SC", etime.Run, etime.Epoch, etime.Trial)

	// Add a special debug message -- use of etime.Debug triggers
	// inclusion
	if ss.Config.GUI {
		ss.Logs.AddStatStringItem(etime.Debug, etime.Trial, "Debug")
	}

	ss.Logs.AddItem(&elog.Item{
		Name:      "ActCor",
		Type:      etensor.FLOAT64,
		CellShape: []int{int(armaze.ActionsN)},
		DimNames:  []string{"Acts"},
		// Plot:      true,
		Range:     minmax.F64{Min: 0},
		TensorIdx: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IdxView(ctx.Mode, etime.Trial)
				spl := split.GroupBy(ix, []string{"Instinct"})
				split.AggTry(spl, "ActMatch", agg.AggMean)
				ags := spl.AggsToTable(etable.ColNameOnly)
				ss.Logs.MiscTables["ActCor"] = ags
				ctx.SetTensor(ags.Cols[0]) // cors
			}}})
	for act := armaze.Actions(0); act < armaze.ActionsN; act++ { // per-action % correct
		anm := act.String()
		ss.Logs.AddItem(&elog.Item{
			Name: anm + "Cor",
			Type: etensor.FLOAT64,
			// Plot:  true,
			Range: minmax.F64{Min: 0},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ags := ss.Logs.MiscTables["ActCor"]
					rw := ags.RowsByString("Instinct", anm, etable.Equals, etable.UseCase)
					if len(rw) > 0 {
						ctx.SetFloat64(ags.CellFloat("ActMatch", rw[0]))
					}
				}}})
	}

	lays := ss.Net.LayersByType(axon.PTMaintLayer)
	for _, lnm := range lays {
		nm := "Maint" + lnm
		ss.Logs.AddStatAggItem(nm, etime.Run, etime.Epoch, etime.Trial)
		nm = "MaintFail" + lnm
		ss.Logs.AddStatAggItem(nm, etime.Run, etime.Epoch, etime.Trial)
		nm = "PreAct" + lnm
		ss.Logs.AddStatAggItem(nm, etime.Run, etime.Epoch, etime.Trial)
	}
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	pv := &ss.Net.PVLV
	if mode != etime.Analyze && mode != etime.Debug {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}

	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return /// not doing cycle-level logging -- too slow for gpu in general
		// row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		if mode == etime.Train {
			for di := 0; di < int(ctx.NetIdxs.NData); di++ {
				diu := uint32(di)
				ss.TrialStats(di)
				ss.StatCounters(di)
				ss.Logs.LogRowDi(mode, time, row, di)
				if !pv.HasPosUS(ctx, diu) && axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0 { // maint
					axon.LayerActsLog(ss.Net, &ss.Logs, di, &ss.GUI)
				}
				if ss.ViewUpdt.View != nil && di == ss.ViewUpdt.View.Di {
					drow := ss.Logs.Table(etime.Debug, time).Rows
					ss.Logs.LogRow(etime.Debug, time, drow)
					if ss.StopOnSeq {
						hasRew := axon.GlbV(ctx, uint32(di), axon.GvHasRew) > 0
						if hasRew {
							ss.Loops.Stop(etime.Trial)
						}
					}
					ss.GUI.UpdateTableView(etime.Debug, etime.Trial)
				}
				// if ss.Stats.Float("GatedEarly") > 0 {
				// 	fmt.Printf("STOPPED due to gated early: %d  %g\n", ev.US, ev.Rew)
				// 	ss.Loops.Stop(etime.Trial)
				// }
				// ev := ss.Envs.ByModeDi(etime.Train, di).(*armaze.Env)
				// if ss.StopOnErr && trnEpc > 5 && ss.Stats.Float("MaintEarly") > 0 {
				// 	fmt.Printf("STOPPED due to early maint for US: %d\n", ev.US)
				// 	ss.Loops.Stop(etime.Trial)
				// }
			}
			return // don't do reg
		}
	case mode == etime.Train && time == etime.Epoch:
		axon.LayerActsLogAvg(ss.Net, &ss.Logs, &ss.GUI, true) // reset recs
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		GUI

func (ss *Sim) UpdateEnvGUI(mode etime.Modes) {
	di := ss.GUI.ViewUpdt.View.Di
	diu := uint32(di)
	ev := ss.Envs.ByModeDi(mode, di).(*armaze.Env)
	ctx := &ss.Context
	net := ss.Net
	pv := &net.PVLV
	dp := ss.EnvGUI.USposData
	ofcPosUS := net.AxonLayerByName("OFCposUSPT")
	ofcmul := float32(1)
	np := pv.NPosUSs
	for i := uint32(0); i < np; i++ {
		drv := axon.GlbUSposV(ctx, diu, axon.GvDrives, i)
		us := axon.GlbUSposV(ctx, diu, axon.GvUSpos, i)
		ofcP := ofcPosUS.Pool(i+1, diu)
		ofc := ofcP.AvgMax.CaSpkD.Plus.Avg * ofcmul
		dp.SetCellFloat("Drive", int(i), float64(drv))
		dp.SetCellFloat("USin", int(i), float64(us))
		dp.SetCellFloat("OFC", int(i), float64(ofc))
	}
	dn := ss.EnvGUI.USnegData
	ofcNegUS := net.AxonLayerByName("OFCnegUSPT")
	nn := pv.NNegUSs
	for i := uint32(0); i < nn; i++ {
		us := axon.GlbUSneg(ctx, diu, axon.GvUSneg, i)
		ofcP := ofcNegUS.Pool(i+1, diu)
		ofc := ofcP.AvgMax.CaSpkD.Plus.Avg * ofcmul
		dn.SetCellFloat("USin", int(i), float64(us))
		dn.SetCellFloat("OFC", int(i), float64(ofc))
	}
	ss.EnvGUI.USposPlot.Update()
	ss.EnvGUI.UpdateWorld(ctx, ev, net, armaze.TraceStates(ss.Stats.IntDi("TraceStateInt", di)))
}

// ConfigGUI configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGUI() *gi.Window {
	title := "BOA = BG, OFC ACC"
	ss.GUI.MakeWindow(ss, "boa", title, `This project tests learning in the BG, OFC & ACC for basic approach learning to a CS associated with a US. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.02
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.Phase, etime.Phase)

	nv.Scene().Camera.Pose.Pos.Set(0, 1.4, 2.6)
	nv.Scene().Camera.LookAt(mat32.Vec3{X: 0, Y: 0, Z: 0}, mat32.Vec3{X: 0, Y: 1, Z: 0})

	ss.GUI.ViewUpdt = &ss.ViewUpdt

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Debug, etime.Trial)

	axon.LayerActsLogConfigGUI(&ss.Logs, &ss.GUI)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all NRuns, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("misc")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "New Seed",
		Icon:    "new",
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RndSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/boa/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUwithGUI(&ss.Context)
		gi.SetQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
	return ss.GUI.Win
}

func (ss *Sim) RunGUI() {
	ss.Init()
	win := ss.ConfigGUI()
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)
	ss.EnvGUI = &armaze.GUI{}
	fwin := ss.EnvGUI.ConfigWorldGUI(ev)
	fwin.GoStartEventLoop()
	win.StartEventLoop()
}

// RecordTestData returns key testing data from the network
func (ss *Sim) RecordTestData() {
	net := ss.Net
	lays := net.LayersByType(axon.PTMaintLayer, axon.PTNotMaintLayer, axon.MatrixLayer, axon.STNLayer, axon.BLALayer, axon.CeMLayer, axon.VSPatchLayer, axon.LHbLayer, axon.LDTLayer, axon.VTALayer)

	key := ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di"})

	net.AllGlobalVals(key, ss.TestData)
	for _, lnm := range lays {
		ly := net.AxonLayerByName(lnm)
		ly.TestVals(key, ss.TestData)
	}
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

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	if ss.Config.Log.Testing {
		ss.TestData = make(map[string]float32)
	}

	ss.Init()

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.NRuns, ss.Config.Run.Run)
	ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context) // must happen after gui or no gui
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	tmr := timer.Time{}
	tmr.Start()

	ss.Loops.Run(etime.Train)

	tmr.Stop()
	fmt.Printf("Total Time: %6.3g\n", tmr.TotalSecs())
	ss.Net.TimerReport()

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
