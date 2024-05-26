// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
choose: This project tests the Rubicon framework making cost-benefit based choices
*/
package main

//go:generate core generate -add-types

import (
	"fmt"
	"log"
	"log/slog"
	"math"
	"os"
	"reflect"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/num"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/plot/plotview"
	"cogentcore.org/core/tensor/stats/split"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/examples/choose/armaze"
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

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *axon.Network `view:"no-inline"`

	// if true, stop running at end of a sequence (for NetView Di data parallel index)
	StopOnSeq bool

	// if true, stop running when an error programmed into the code occurs
	StopOnErr bool

	// network parameter management
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
	ViewUpdate netview.ViewUpdate `view:"inline"`

	// manages all the gui elements
	GUI egui.GUI `view:"-"`

	// gui for viewing env
	EnvGUI *armaze.GUI `view:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `view:"-"`

	// testing data, from -test arg
	TestData map[string]float32 `view:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	_, err := econfig.Config(&ss.Config, "config.toml")
	if err != nil {
		slog.Error(err.Error())
	}
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
	ss.Context.ThetaCycles = int32(ss.Config.Run.NCycles)
}

////////////////////////////////////////////////////////////////////////////////
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

	if ss.Config.Env.Config != "" {
		fmt.Println("Env Config:", ss.Config.Env.Config)
	}

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
		trn.RandSeed = 73
		if !ss.Config.Env.SameSeed {
			trn.RandSeed += int64(di) * 73
		}
		trn.Config.NDrives = ss.Config.Env.NDrives
		if ss.Config.Env.Config != "" {
			args := os.Args
			os.Args = args[:1]
			_, err := econfig.Config(&trn.Config, ss.Config.Env.Config)
			if err != nil {
				slog.Error(err.Error())
			}
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
	rp.SetNUSs(&ss.Context, trn.Config.NDrives, 1)
	rp.Defaults()
	rp.USs.PVposGain = 2 // higher = more pos reward (saturating logistic func)
	rp.USs.PVnegGain = 1 // global scaling of RP neg level -- was 1
	rp.LHb.VSPatchGain = 5
	rp.LHb.VSPatchNonRewThr = 0.15

	rp.USs.USnegGains[0] = 2 // big salient input!

	rp.Drive.DriveMin = 0.5 // 0.5 -- should be
	rp.Urgency.U50 = 10
	if ss.Config.Params.Rubicon != nil {
		params.ApplyMap(rp, ss.Config.Params.Rubicon, ss.Config.Debug)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)
	net.InitName(net, "Choose")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

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

	vSgpi, vSmtxGo, vSmtxNo, urgency, pvPos, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPosUS, ofcPosUSCT, ofcPosUSPT, ofcPosUSPTp, ilPos, ilPosCT, ilPosPT, ilPosPTp, ofcNegUS, ofcNegUSCT, ofcNegUSPT, ofcNegUSPTp, ilNeg, ilNegCT, ilNegPT, ilNegPTp, accCost, plUtil, sc := net.AddRubicon(ctx, ny, popY, popX, nuBgY, nuBgX, nuCtxY, nuCtxX, space)
	_, _ = plUtil, urgency
	_, _ = ofcNegUSCT, ofcNegUSPTp
	_, _ = vSmtxGo, vSmtxNo

	plUtilPTp := net.AxonLayerByName("PLutilPTp")

	cs, csP := net.AddInputPulv2D("CS", ny, narm, space)
	dist, distP := net.AddInputPulv2D("Dist", ny, ev.MaxLength+1, space)

	///////////////////////////////////////////
	// M1, VL, ALM

	act := net.AddLayer2D("Act", ny, nAct, axon.InputLayer) // Action: what is actually done
	vl := net.AddPulvLayer2D("VL", ny, nAct)                // VL predicts brainstem Action
	vl.SetBuildConfig("DriveLayName", act.Name())

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

	///////////////////////////////////////////
	// CS -> BLA, OFC

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

	///////////////////////////////////////////
	// OFC, ACC, ALM predicts dist

	// todo: a more dynamic US rep is needed to drive predictions in OFC
	// using distance and effort here in the meantime
	net.ConnectToPFCBack(dist, distP, ofcPosUS, ofcPosUSCT, ofcPosUSPT, ofcPosUSPTp, full, "DistToPFC")
	net.ConnectToPFCBack(dist, distP, ilPos, ilPosCT, ilPosPT, ilPosPTp, full, "PosToPFC")

	net.ConnectToPFC(dist, distP, ofcNegUS, ofcNegUSCT, ofcNegUSPT, ofcNegUSPTp, full, "DistToPFC")
	net.ConnectToPFC(dist, distP, ilNeg, ilNegCT, ilNegPT, ilNegPTp, full, "DistToPFC")

	//	alm predicts all effort, cost, sensory state vars
	net.ConnectToPFC(dist, distP, alm, almCT, almPT, almPTp, full, "DistToPFC")

	///////////////////////////////////////////
	// ALM, M1 <-> OFC, ACC

	// action needs to know if maintaining a goal or not
	// using plUtil as main summary "driver" input to action system
	// PTp provides good notmaint signal for action.
	net.ConnectLayers(plUtilPTp, alm, full, axon.ForwardPath).AddClass("ToALM")
	net.ConnectLayers(plUtilPTp, m1, full, axon.ForwardPath).AddClass("ToM1")

	// note: in Obelisk this helps with the Consume action
	// but here in this example it produces some instability
	// at later time points -- todo: investigate later.
	// net.ConnectLayers(notMaint, vl, full, axon.ForwardPath).AddClass("ToVL")

	////////////////////////////////////////////////
	// position

	cs.PlaceRightOf(pvPos, space*2)
	dist.PlaceRightOf(cs, space)

	m1.PlaceRightOf(dist, space)
	alm.PlaceRightOf(m1, space)
	vl.PlaceBehind(m1P, space)
	act.PlaceBehind(vl, space)

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

	nCSTot := ev.Config.NArms

	cs := net.AxonLayerByName("CS")
	cs.Params.Inhib.ActAvg.Nominal = 0.32 / float32(nCSTot)
	csp := net.AxonLayerByName("CSP")
	csp.Params.Inhib.ActAvg.Nominal = 0.32 / float32(nCSTot)
	bla := net.AxonLayerByName("BLAposAcqD1")
	pji, _ := bla.SendNameTry("BLANovelCS")
	pj := pji.(*axon.Path)

	// this is very sensitive param to get right
	// too little and the hamster does not try CSs at the beginning,
	// too high and it gets stuck trying the same location over and over
	pj.Params.PathScale.Abs = float32(math32.Min(2.3+(float32(nCSTot)/10.0), 3.0))

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
	ss.Logs.ResetLog(etime.Debug, etime.Trial)
	ss.InitRandSeed(0)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdate.Update()
	ss.ViewUpdate.RecordSyns()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()
	// ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)

	ncyc := ss.Config.Run.NCycles

	// note: sequence stepping does not work in NData > 1 mode -- just going back to raw trials
	trls := int(math32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, ncyc)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, ncyc-50, ncyc-1)       // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

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

	// man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
	// 	trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	// 	if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
	// 		axon.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
	// 		ss.Logs.ResetLog(etime.Analyze, etime.Trial)
	// 	}
	// })

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)
	if ss.Config.GUI {
		man.GetLoop(etime.Train, etime.Trial).OnStart.Add("ResetDebugTrial", func() {
			di := uint32(ss.ViewUpdate.View.Di)
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
		ctrString := ss.Stats.PrintValues([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
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
				ss.GUI.NetDataRecord(ss.ViewUpdate.Text)
			})
		}
	} else {
		axon.LooperUpdateNetView(man, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
		axon.LooperUpdatePlots(man, &ss.GUI)

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
	rp := &ss.Net.Rubicon
	mtxLy := ss.Net.AxonLayerByName("VMtxGo")
	vlly := ss.Net.AxonLayerByName("VL")
	threshold := float32(0.1)
	for di := 0; di < int(ctx.NetIndexes.NData); di++ {
		diu := uint32(di)
		ev := ss.Envs.ByModeDi(ctx.Mode, di).(*armaze.Env)
		justGated := mtxLy.AnyGated(diu) // not updated until plus phase: rp.VSMatrix.JustGated.IsTrue()
		hasGated := axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0
		ev.InstinctAct(justGated, hasGated)
		csGated := (justGated && !rp.HasPosUS(ctx, diu))
		deciding := !csGated && !hasGated && (axon.GlbV(ctx, diu, axon.GvACh) > threshold && mtxLy.Pool(0, diu).AvgMax.SpkMax.Cycle.Max > threshold) // give it time
		wasDeciding := num.ToBool(ss.Stats.Float32Di("Deciding", di))
		if wasDeciding {
			deciding = false // can't keep deciding!
		}
		ss.Stats.SetFloat32Di("Deciding", di, num.FromBool[float32](deciding))

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
			case rp.HasPosUS(ctx, diu):
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
	lays := []string{"Dist", "CS"}

	ss.Net.InitExt(ctx)
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		ev := ss.Envs.ByModeDi(ctx.Mode, int(di)).(*armaze.Env)
		giveUp := axon.GlbV(ctx, di, axon.GvGiveUp) > 0
		if giveUp {
			ev.JustConsumed = true // triggers a new start -- we just consumed the giving up feeling :)
		}
		ev.Step()
		if ev.Tick == 0 {
			ss.Stats.SetFloat32Di("CortexDriving", int(di), num.FromBool[float32](randx.BoolP32(ss.Config.Env.PctCortex)))
		}
		for _, lnm := range lays {
			ly := net.AxonLayerByName(lnm)
			itsr := ev.State(lnm)
			ly.ApplyExt(ctx, di, itsr)
		}
		ss.ApplyRubicon(ctx, ev, di)
	}
	ss.Net.ApplyExts(ctx)
}

// ApplyRubicon applies current Rubicon values to Context.Rubicon,
// from given trial data.
func (ss *Sim) ApplyRubicon(ctx *axon.Context, ev *armaze.Env, di uint32) {
	rp := &ss.Net.Rubicon
	rp.NewState(ctx, di, &ss.Net.Rand) // first before anything else is updated
	rp.SetGoalMaintFromLayer(ctx, di, ss.Net, "PLutilPT", 0.2)
	rp.DecodePVEsts(ctx, di, ss.Net)
	rp.SetGoalDistEst(ctx, di, float32(ev.Dist))
	rp.EffortUrgencyUpdate(ctx, di, ev.Effort)
	if ev.USConsumed >= 0 {
		rp.SetUS(ctx, di, axon.Positive, ev.USConsumed, ev.USValue)
	}
	rp.SetDrives(ctx, di, 0.5, ev.Drives...)
	rp.Step(ctx, di, &ss.Net.Rand)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	for di := 0; di < int(ctx.NetIndexes.NData); di++ {
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
		ss.Net.OpenWtsJSON(core.Filename(ss.Config.OpenWts))
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
	ss.Stats.SetFloat("Dist", 0)
	ss.Stats.SetFloat("Drive", 0)
	ss.Stats.SetFloat("CS", 0)
	ss.Stats.SetFloat("US", 0)
	ss.Stats.SetString("NetAction", "")
	ss.Stats.SetString("Instinct", "")
	ss.Stats.SetString("ActAction", "")
	ss.Stats.SetString("TraceState", "")
	ss.Stats.SetInt("TraceStateInt", 0)
	ss.Stats.SetFloat("ActMatch", 0)
	ss.Stats.SetFloat("AllGood", 0)

	ss.Stats.SetFloat("Should", 0)
	ss.Stats.SetFloat("GateUS", 0)
	ss.Stats.SetFloat("GateCS", 0)
	ss.Stats.SetFloat("Deciding", 0)
	ss.Stats.SetFloat("GatedEarly", 0)
	ss.Stats.SetFloat("MaintEarly", 0)
	ss.Stats.SetFloat("MaintIncon", 0)
	ss.Stats.SetFloat("GatedAgain", 0)
	ss.Stats.SetFloat("WrongCSGate", 0)
	ss.Stats.SetFloat("AChShould", 0)
	ss.Stats.SetFloat("AChShouldnt", 0)

	ss.Stats.SetFloat("DA", 0)
	ss.Stats.SetFloat("DA_NR", 0)
	ss.Stats.SetFloat("RewPred_NR", 0)
	ss.Stats.SetFloat("DA_GiveUp", 0)

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
	ss.Stats.SetFloat32("Dist", float32(ev.Dist))
	ss.Stats.SetFloat32("Arm", float32(ev.Arm))
	// ss.Stats.SetFloat32("Drive", float32(ev.Drive))
	ss.Stats.SetFloat32("CS", float32(ev.CurCS()))
	ss.Stats.SetFloat32("US", float32(ev.USConsumed))
	ss.Stats.SetString("TrialName", "trl") // todo: could have dist, US etc
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di", "Cycle", "NetAction", "Instinct", "ActAction", "ActMatch", "JustGated", "Should"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
	ss.GatedStats(di)
	ss.MaintStats(di)

	diu := uint32(di)
	ctx := &ss.Context
	rp := &ss.Net.Rubicon
	rp.DecodePVEsts(ctx, diu, ss.Net) // get this for current trial!
	hasRew := axon.GlbV(ctx, diu, axon.GvHasRew) > 0
	if hasRew { // exclude data for logging -- will be re-computed at start of next trial
		// this allows the WrongStats to only record estimates, not actuals
		nan := math32.NaN()
		axon.SetGlbV(ctx, diu, axon.GvPVposEst, nan)
		axon.SetGlbV(ctx, diu, axon.GvPVposVar, nan)
		axon.SetGlbV(ctx, diu, axon.GvPVnegEst, nan)
		axon.SetGlbV(ctx, diu, axon.GvPVnegVar, nan)
	}

	ss.Stats.SetFloat32("SC", ss.Net.AxonLayerByName("SC").Pool(0, 0).AvgMax.CaSpkD.Cycle.Max)

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
	rp := &ss.Net.Rubicon
	diu := uint32(di)
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*armaze.Env)
	justGated := axon.GlbV(ctx, diu, axon.GvVSMatrixJustGated) > 0
	justGatedF := num.FromBool[float32](justGated)
	hasGated := axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0
	nan := math32.NaN()
	ss.Stats.SetString("Debug", ss.Stats.StringDi("Debug", di))
	ss.ActionStatsDi(di)

	ss.Stats.SetFloat32("JustGated", justGatedF)
	ss.Stats.SetFloat32("Should", num.FromBool[float32](ev.ShouldGate))
	ss.Stats.SetFloat32("HasGated", num.FromBool[float32](hasGated))
	ss.Stats.SetFloat32("GateUS", nan)
	ss.Stats.SetFloat32("GateCS", nan)
	ss.Stats.SetFloat32("GatedEarly", nan)
	ss.Stats.SetFloat32("MaintEarly", nan)
	ss.Stats.SetFloat32("GatedAgain", nan)
	ss.Stats.SetFloat32("WrongCSGate", nan)
	ss.Stats.SetFloat32("AChShould", nan)
	ss.Stats.SetFloat32("AChShouldnt", nan)
	hasPos := rp.HasPosUS(ctx, diu)
	if justGated {
		ss.Stats.SetFloat32("WrongCSGate", num.FromBool[float32](!ev.ArmIsBest(ev.Arm)))
	}
	if ev.ShouldGate {
		if hasPos {
			ss.Stats.SetFloat32("GateUS", justGatedF)
		} else {
			ss.Stats.SetFloat32("GateCS", justGatedF)
		}
	} else {
		if hasGated {
			ss.Stats.SetFloat32("GatedAgain", justGatedF)
		} else { // !should gate means early..
			ss.Stats.SetFloat32("GatedEarly", justGatedF)
		}
	}
	// We get get ACh when new CS or Rew
	ach := axon.GlbV(ctx, diu, axon.GvACh)
	if hasPos || ev.LastCS != ev.CurCS() {
		ss.Stats.SetFloat32("AChShould", ach)
	} else {
		ss.Stats.SetFloat32("AChShouldnt", ach)
	}
}

// MaintStats updates the PFC maint stats
func (ss *Sim) MaintStats(di int) {
	ctx := &ss.Context
	diu := uint32(di)
	nan := math32.NaN()
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*armaze.Env)
	// should be maintaining while going forward
	isFwd := ev.LastAct == armaze.Forward
	isCons := ev.LastAct == armaze.Consume
	actThr := float32(0.05) // 0.1 too high
	net := ss.Net
	hasGated := axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0
	goalMaint := axon.GlbV(ctx, diu, axon.GvGoalMaint)
	ss.Stats.SetFloat32("GoalMaint", goalMaint)
	hasGoalMaint := goalMaint > actThr
	lays := net.LayersByType(axon.PTMaintLayer)
	otherMaint := false
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
			otherMaint = true
		}
		ss.Stats.SetFloat32(pnm, math32.NaN())
		ss.Stats.SetFloat32(mnm, math32.NaN())
		ss.Stats.SetFloat32(fnm, math32.NaN())
		if isFwd {
			ss.Stats.SetFloat32(mnm, mact)
			ss.Stats.SetFloat32(fnm, num.FromBool[float32](!overThr))
		} else if !isCons {
			ss.Stats.SetFloat32(pnm, num.FromBool[float32](overThr))
		}
	}
	ss.Stats.SetFloat32("MaintIncon", num.FromBool[float32](otherMaint != hasGoalMaint))
	if hasGoalMaint && !hasGated {
		ss.Stats.SetFloat32("MaintEarly", 1)
	} else if !hasGated {
		ss.Stats.SetFloat32("MaintEarly", 0)
	} else {
		ss.Stats.SetFloat32("MaintEarly", nan)
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
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.Trial, "Drive", "CS", "Pos", "Dist", "US")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "NetAction", "Instinct", "ActAction", "TraceState")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	axon.LogAddGlobals(&ss.Logs, &ss.Context, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	// axon.LogAddPulvCorSimItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	// ss.ConfigActRFs()

	// layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer, axon.CeMLayer)
	// axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	// todo: PCA items should apply to CT layers too -- pass a type here.
	// axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	// ss.Logs.PlotItems("GateCS", "GateUS", "WrongCSGate", "Rew_R", "RewPred_R", "DA_R", "MaintEarly")
	ss.Logs.PlotItems("WrongCSGate", "Wrong0_RewPred_R", "Wrong0_DA_R", "Wrong0_PVposEst", "Wrong0_PVnegEst", "Wrong1_RewPred_R", "Wrong1_DA_R", "Wrong1_PVposEst", "Wrong1_PVnegEst")

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
	ss.Logs.AddStatAggItem("MaintIncon", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GatedAgain", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("WrongCSGate", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AChShould", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AChShouldnt", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("SC", etime.Run, etime.Epoch, etime.Trial)

	// Add a special debug message -- use of etime.Debug triggers
	// inclusion
	if ss.Config.GUI {
		ss.Logs.AddStatStringItem(etime.Debug, etime.Trial, "Debug")
	}

	for wrong := 0; wrong < 2; wrong++ {
		wrong := wrong
		for _, st := range ss.Config.Log.AggStats {
			st := st
			itmName := fmt.Sprintf("Wrong%d_%s", wrong, st)
			ss.Logs.AddItem(&elog.Item{
				Name: itmName,
				Type: reflect.Float64,
				// FixMin: true,
				// FixMax: true,
				Range: minmax.F32{Max: 1},
				Write: elog.WriteMap{
					etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
						ctx.SetFloat64(ctx.Stats.Float(itmName))
					}, etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
						ctx.SetAgg(ctx.Mode, etime.Epoch, stats.Mean)
					}}})
		}
	}

	ss.Logs.AddItem(&elog.Item{
		Name:      "ActCor",
		Type:      reflect.Float64,
		CellShape: []int{int(armaze.ActionsN)},
		DimNames:  []string{"Acts"},
		// Plot:      true,
		Range:       minmax.F32{Min: 0},
		TensorIndex: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IndexView(ctx.Mode, etime.Trial)
				spl := split.GroupBy(ix, []string{"Instinct"})
				split.AggColumnTry(spl, "ActMatch", stats.Mean)
				ags := spl.AggsToTable(table.ColumnNameOnly)
				ss.Logs.MiscTables["ActCor"] = ags
				ctx.SetTensor(ags.Columns[0]) // cors
			}}})
	for act := armaze.Actions(0); act < armaze.ActionsN; act++ { // per-action % correct
		anm := act.String()
		ss.Logs.AddItem(&elog.Item{
			Name: anm + "Cor",
			Type: reflect.Float64,
			// Plot:  true,
			Range: minmax.F32{Min: 0},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ags := ss.Logs.MiscTables["ActCor"]
					rw := ags.RowsByString("Instinct", anm, table.Equals, table.UseCase)
					if len(rw) > 0 {
						ctx.SetFloat64(ags.Float("ActMatch", rw[0]))
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

// EpochWrongStats aggregates stats separately for WrongCSGate = 0 vs. 1
// i.e., for trials when it selects the "wrong" option (not the best) = (Wrong1)
// vs. when it does select the best option (Wrong0)
func (ss *Sim) EpochWrongStats() {
	lgnm := "EpochWrongStats"

	ix := ss.Logs.IndexView(etime.Train, etime.Trial)
	ix.Filter(func(et *table.Table, row int) bool {
		return !math.IsNaN(et.Float("WrongCSGate", row)) // && (et.StringValue("ActAction", row) == "Consume")
	})
	spl := split.GroupBy(ix, []string{"WrongCSGate"})
	for _, ts := range ix.Table.ColumnNames {
		col := ix.Table.ColumnByName(ts)
		if col.DataType() == reflect.String || ts == "WrongCSGate" {
			continue
		}
		split.AggColumn(spl, ts, stats.Mean)
	}
	dt := spl.AggsToTable(table.ColumnNameOnly)
	dt.SetMetaData("Rew_R:On", "+")
	dt.SetMetaData("DA_R:On", "+")
	dt.SetMetaData("RewPred_R:On", "+")
	dt.SetMetaData("VtaDA:On", "+")
	dt.SetMetaData("PVposEst:On", "+")
	dt.SetMetaData("PVnegEst:On", "+")
	dt.SetMetaData("DA_R:FixMin", "+")
	dt.SetMetaData("DA_R:Min", "-1")
	dt.SetMetaData("DA_R:FixMax", "-")
	dt.SetMetaData("DA_R:Max", "1")
	// dt.SetMetaData("XAxisRot", "45")
	dt.SetMetaData("Type", "Bar")
	ss.Logs.MiscTables[lgnm] = dt

	// grab selected stats at CS and US for higher level aggregation,
	nrows := dt.Rows
	for ri := 0; ri < nrows; ri++ {
		wrong := dt.Float("WrongCSGate", ri)
		for _, st := range ss.Config.Log.AggStats {
			ss.Stats.SetFloat(fmt.Sprintf("Wrong%d_%s", int(wrong), st), dt.Float(st, ri))
		}
	}

	if ss.Config.GUI {
		plt := ss.GUI.Plots[etime.ScopeKey(lgnm)]
		plt.SetTable(dt)
		plt.GoUpdatePlot()
	}

}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	rp := &ss.Net.Rubicon
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
			for di := 0; di < int(ctx.NetIndexes.NData); di++ {
				diu := uint32(di)
				ss.TrialStats(di)
				ss.StatCounters(di)
				ss.Logs.LogRowDi(mode, time, row, di)
				if !rp.HasPosUS(ctx, diu) && axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0 { // maint
					axon.LayerActsLog(ss.Net, &ss.Logs, di, &ss.GUI)
				}
				if ss.ViewUpdate.View != nil && di == ss.ViewUpdate.View.Di {
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
		ss.EpochWrongStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		GUI

func (ss *Sim) UpdateEnvGUI(mode etime.Modes) {
	di := ss.GUI.ViewUpdate.View.Di
	diu := uint32(di)
	ev := ss.Envs.ByModeDi(mode, di).(*armaze.Env)
	ctx := &ss.Context
	net := ss.Net
	rp := &net.Rubicon
	dp := ss.EnvGUI.USposData
	ofcPosUS := net.AxonLayerByName("OFCposPT")
	ofcmul := float32(1)
	np := rp.NPosUSs
	for i := uint32(0); i < np; i++ {
		drv := axon.GlbUSposV(ctx, diu, axon.GvDrives, i)
		us := axon.GlbUSposV(ctx, diu, axon.GvUSpos, i)
		ofcP := ofcPosUS.Pool(i+1, diu)
		ofc := ofcP.AvgMax.CaSpkD.Plus.Avg * ofcmul
		dp.SetFloat("Drive", int(i), float64(drv))
		dp.SetFloat("USin", int(i), float64(us))
		dp.SetFloat("OFC", int(i), float64(ofc))
	}
	dn := ss.EnvGUI.USnegData
	ofcNegUS := net.AxonLayerByName("OFCnegPT")
	nn := rp.NNegUSs
	for i := uint32(0); i < nn; i++ {
		us := axon.GlbUSnegV(ctx, diu, axon.GvUSneg, i)
		ofcP := ofcNegUS.Pool(i+1, diu)
		ofc := ofcP.AvgMax.CaSpkD.Plus.Avg * ofcmul
		dn.SetFloat("USin", int(i), float64(us))
		dn.SetFloat("OFC", int(i), float64(ofc))
	}
	ss.EnvGUI.USposPlot.GoUpdatePlot()
	ss.EnvGUI.USnegPlot.GoUpdatePlot()
	ss.EnvGUI.UpdateWorld(ctx, ev, net, armaze.TraceStates(ss.Stats.IntDi("TraceStateInt", di)))
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Choose: Rubicon"
	ss.GUI.MakeBody(ss, "choose", title, `This project tests the Rubicon framework in simple cost-benefit choice scenarios, using an N-arm bandit maze task. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = ss.Config.Run.NCycles * 2
	nv.Params.LayNmSize = 0.02
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.4, 2.6)
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{}, math32.Vec3(0, 1, 0))

	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Debug, etime.Trial)

	axon.LayerActsLogConfigGUI(&ss.Logs, &ss.GUI)

	lgnm := "EpochWrongStats"
	dt := ss.Logs.MiscTable(lgnm)
	plt := plotview.NewSubPlot(ss.GUI.Tabs.NewTab(lgnm + " Plot"))
	ss.GUI.Plots[etime.ScopeKey(lgnm)] = plt
	plt.Params.Title = lgnm
	plt.Params.XAxisColumn = "WrongCSGate"
	plt.SetTable(dt)

	ss.GUI.Body.AddAppBar(func(tb *core.Toolbar) {
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddLooperCtrl(tb, ss.Loops, []etime.Modes{etime.Train})

		////////////////////////////////////////////////
		core.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Reset RunLog",
			Icon:    icons.Reset,
			Tooltip: "Reset the accumulated log of all NRuns, which are tagged with the ParamSet used",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.Logs.ResetLog(etime.Train, etime.Run)
				ss.GUI.UpdatePlot(etime.Train, etime.Run)
			},
		})
		////////////////////////////////////////////////
		core.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "New Seed",
			Icon:    icons.Add,
			Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.RandSeeds.NewSeeds()
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "README",
			Icon:    "file-markdown",
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/axon/blob/main/examples/choose/README.md")
			},
		})
	})
	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUwithGUI(&ss.Context)
		core.TheApp.AddQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)
	ss.EnvGUI = &armaze.GUI{}
	eb := ss.EnvGUI.ConfigWorldGUI(ev)
	eb.RunWindow()
	ss.GUI.Body.RunMainWindow()
}

// RecordTestData returns key testing data from the network
func (ss *Sim) RecordTestData() {
	net := ss.Net
	lays := net.LayersByType(axon.PTMaintLayer, axon.MatrixLayer, axon.STNLayer, axon.BLALayer, axon.CeMLayer, axon.VSPatchLayer, axon.LHbLayer, axon.LDTLayer, axon.VTALayer)

	key := ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di"})

	net.AllGlobalValues(key, ss.TestData)
	for _, lnm := range lays {
		ly := net.AxonLayerByName(lnm)
		ly.TestValues(key, ss.TestData)
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
	fmt.Printf("Total Time: %v\n", tmr.Total)
	ss.Net.TimerReport()

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
