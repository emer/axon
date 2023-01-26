// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
boa: This project tests BG, OFC & ACC learning in a CS-driven approach task.
*/
package main

import (
	"log"
	"math"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
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

// Debug triggers various messages etc
var Debug = false

func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// see params.go for network params

// SimParams has all the custom params for this sim
type SimParams struct {
	TwoThetas       bool    `desc:"if true, do 2 theta trials per action"`
	PctCortex       float32 `desc:"proportion of behavioral approach sequences driven by the cortex vs. hard-coded reflexive subcortical"`
	PctCortexMax    float32 `desc:"maximum PctCortex, when running on the schedule"`
	PctCortexStEpc  int     `desc:"epoch when PctCortex starts increasing"`
	PctCortexMaxEpc int     `desc:"epoch when PctCortexMax is reached"`
	PCAInterval     int     `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
	CortexDriving   bool    `desc:"true if cortex is driving this behavioral approach sequence"`
	ThetaStep       int     `desc:"theta counter -- only take an action every 2 trials"`
}

// Defaults sets default params
func (ss *SimParams) Defaults() {
	ss.TwoThetas = true
	ss.PctCortexMax = 1.0
	ss.PctCortexStEpc = 10
	ss.PctCortexMaxEpc = 50
	ss.PCAInterval = 10
	ss.ThetaStep = 0
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Sim          SimParams        `view:"no-inline" desc:"sim params"`
	Params       emer.Params      `view:"inline" desc:"all parameter management"`
	Loops        *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats     `desc:"contains computed statistic values"`
	Logs         elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Pats         *etable.Table    `view:"no-inline" desc:"the training patterns to use"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Context      axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `view:"inline" desc:"netview update parameters"`
	TestInterval int              `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Sim.Defaults()
	ss.Params.Params = ParamSets
	// ss.Params.ExtraSets = "WtScales"
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Pats = &etable.Table{}
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.TestInterval = 500
	ss.Context.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *Approach
	if len(ss.Envs) == 0 {
		trn = &Approach{}
		tst = &Approach{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*Approach)
		tst = ss.Envs.ByMode(etime.Test).(*Approach)
	}

	// note: names must be standard here!
	trn.Nm = etime.Train.String()
	trn.Defaults()
	trn.Config()
	trn.Validate()

	tst.Nm = etime.Test.String()
	tst.Defaults()
	tst.Config()
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ev := ss.Envs["Train"].(*Approach)
	net.InitName(net, "Boa")

	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	nAct := len(ev.ActMap)
	space := float32(2)

	pone2one := prjn.NewPoolOneToOne()
	one2one := prjn.NewOneToOne()
	full := prjn.NewFull()
	_ = pone2one

	ny := ev.NYReps
	nloc := ev.Locations

	rew, rwPred, snci := net.AddRWLayers("", relpos.Behind, space)
	_ = rew
	_ = rwPred
	snc := snci.(*axon.Layer)
	ach := net.AddRSalienceAChLayer("ACh")

	drives := net.AddLayer4D("Drives", 1, ev.NDrives, ny, 1, emer.Input)
	us, usPulv := net.AddInputPulv4D("US", 1, ev.NDrives, ny, 1, space)
	// cs, csp := net.AddInputPulv2D("CS", ev.PatSize.Y, ev.PatSize.X, space)
	// localist, for now:
	// cs, csp := net.AddInputPulv2D("CS", ny, ev.NDrives, space)
	cs := net.AddLayer2D("CS", ny, ev.NDrives, emer.Input)
	dist, distp := net.AddInputPulv2D("Dist", ny, ev.DistMax, space)
	time, timep := net.AddInputPulv2D("Time", ny, ev.TimeMax, space)
	// pos, posp := net.AddInputPulv2D("Pos", ny, nloc, space)
	pos := net.AddLayer2D("Pos", ny, nloc, emer.Input) // irrelevant here
	gate := net.AddLayer2D("Gate", ny, 2, emer.Input)  // signals gated or not

	vPmtxGo, vPmtxNo, _, _, _, vPstnp, vPstns, vPgpi := net.AddBG("Vp", 1, ev.NDrives, nuBgY, nuBgX, nuBgY, nuBgX, space)

	// todo: need m1d, driven by smad -- output pathway

	m1 := net.AddLayer2D("M1", nuCtxY, nuCtxX, emer.Hidden)
	vl := net.AddLayer2D("VL", ny, nAct, emer.Target)  // Action
	act := net.AddLayer2D("Act", ny, nAct, emer.Input) // Action
	m1P := net.AddPulvLayer2D("M1P", nuCtxY, nuCtxX)
	m1P.SetBuildConfig("DriveLayName", m1.Name())
	_ = vl
	_ = act

	blaa, blae, _, _, cemPos, _, pptg := net.AddAmygdala("", false, ev.NDrives, nuCtxY, nuCtxX, space)
	_ = cemPos
	_ = pptg

	ofc, ofcct := net.AddSuperCT4D("OFC", 1, ev.NDrives, nuCtxY, nuCtxX, space, one2one)
	// prjns are: super->PT, PT self, CT-> thal
	ofcpt, ofcmd := net.AddPTThalForSuper(ofc, ofcct, "MD", one2one, pone2one, pone2one, space)
	_ = ofcpt
	ofcct.SetClass("OFC CTCopy")
	// net.ConnectCTSelf(ofcct, pone2one) // much better for ofc not to have self prjns..
	// net.ConnectToPulv(ofc, ofcct, csp, full, full)
	net.ConnectToPulv(ofc, ofcct, usPulv, pone2one, pone2one)
	// Drives -> OFC then activates OFC -> VS -- OFC needs to be strongly BLA dependent
	// to reflect either current CS or maintained CS but not just echoing drive state.
	net.ConnectLayers(drives, ofc, pone2one, emer.Forward).SetClass("DrivesToOFC")
	// net.ConnectLayers(drives, ofcct, pone2one, emer.Forward).SetClass("DrivesToOFC")
	net.ConnectLayers(vPgpi, ofcmd, full, emer.Inhib).SetClass("BgFixed")
	// net.ConnectLayers(cs, ofc, full, emer.Forward) // let BLA handle it
	net.ConnectLayers(us, ofc, pone2one, emer.Forward)

	// todo: add ofcp and acc projections to it
	// todo: acc should have pos and negative stripes, with grounded prjns??

	acc, accct := net.AddSuperCT2D("ACC", nuCtxY+2, nuCtxX+2, space, one2one)
	// prjns are: super->PT, PT self, CT->thal
	accpt, accmd := net.AddPTThalForSuper(acc, accct, "MD", one2one, full, full, space)
	_ = accpt
	accct.SetClass("ACC CTCopy")
	net.ConnectCTSelf(accct, full)
	net.ConnectToPulv(acc, accct, distp, full, full)
	net.ConnectToPulv(acc, accct, timep, full, full)
	net.ConnectLayers(vPgpi, accmd, full, emer.Inhib).SetClass("BgFixed")

	net.ConnectLayers(dist, acc, full, emer.Forward)
	net.ConnectLayers(time, acc, full, emer.Forward)

	vPmtxGo.SetBuildConfig("ThalLay1Name", ofcmd.Name())
	vPmtxNo.SetBuildConfig("ThalLay1Name", ofcmd.Name())
	vPmtxGo.SetBuildConfig("ThalLay2Name", accmd.Name())
	vPmtxNo.SetBuildConfig("ThalLay2Name", accmd.Name())

	// m1P plus phase has action, Ctxt -> CT allows CT now to use that prev action

	alm, almct := net.AddSuperCT2D("ALM", nuCtxY+2, nuCtxX+2, space, one2one)
	// almpt, almthal := net.AddPTThalForSuper(alm, almct, "MD", one2one, full, full, space)
	almct.SetClass("ALM CTCopy")
	// _ = almpt
	// net.ConnectCTSelf(almct, full)
	net.ConnectToPulv(alm, almct, m1P, full, full)
	// net.ConnectToPulv(alm, almct, posp, full, full)
	// net.ConnectToPulv(alm, almct, distp, full, full)
	// net.ConnectLayers(vPgpi, almthal, full, emer.Inhib).SetClass("BgFixed")

	//	todo: add a PL layer, with Integ maint

	// contextualization based on action
	// net.BidirConnectLayers(ofc, alm, full)
	// net.BidirConnectLayers(acc, alm, full)
	// net.ConnectLayers(ofcpt, alm, full, emer.Forward)
	// net.ConnectLayers(accpt, alm, full, emer.Forward)

	// todo: blae is not connected properly at all yet

	ach.SetBuildConfig("SrcLay1Name", pptg.Name())

	// BLA
	net.ConnectToBLA(cs, blaa, full)
	net.ConnectToBLA(us, blaa, pone2one).SetClass("USToBLA")
	// net.ConnectToBLA(usp, blaa, pone2one).SetClass("USToBLA")
	// net.ConnectToBLA(drives, blaa, pone2one).SetClass("USToBLA")
	net.ConnectLayers(blaa, ofc, pone2one, emer.Forward)
	// todo: from deep maint layer
	// net.ConnectLayersPrjn(ofcpt, blae, pone2one, emer.Forward, &axon.BLAPrjn{})
	net.ConnectLayers(blae, blaa, pone2one, emer.Inhib).SetClass("BgFixed")
	// net.ConnectLayers(drives, blae, pone2one, emer.Forward)

	net.ConnectLayers(dist, alm, full, emer.Forward)
	net.ConnectLayers(time, alm, full, emer.Forward)
	net.ConnectLayers(ofcpt, alm, full, emer.Forward)
	net.ConnectLayers(accpt, alm, full, emer.Forward)
	net.ConnectLayers(gate, alm, full, emer.Forward)
	// net.ConnectLayers(pos, alm, full, emer.Forward)
	net.ConnectLayers(dist, m1, full, emer.Forward)
	net.ConnectLayers(time, m1, full, emer.Forward)
	net.ConnectLayers(gate, m1, full, emer.Forward) // this is key: direct to M1
	// neither of these other connections work nearly as well as explicit gate
	// net.ConnectLayers(ofcpt, m1, full, emer.Forward)
	// net.ConnectLayers(accpt, m1, full, emer.Forward)
	// net.ConnectLayers(ofcmd, m1, full, emer.Forward)
	// net.ConnectLayers(accmd, m1, full, emer.Forward)
	// key point: cs does not project directly to alm -- no simple S -> R mappings!?

	////////////////////////////////////////////////
	// BG / DA connections

	// net.ConnectLayers(almct, m1, full, emer.Forward) //  action output
	net.BidirConnectLayers(alm, m1, full) // todo: alm weaker?
	// net.ConnectLayers(alm, almpt, one2one, emer.Forward) // is weaker, provides some action sel but gating = stronger
	// net.ConnectLayers(alm, m1, full, emer.Forward)  //  note: non-gated!
	net.BidirConnectLayers(m1, vl, full)
	// net.BidirConnectLayers(alm, vl, full)
	// net.BidirConnectLayers(almct, vl, full)

	net.ConnectLayers(vl, alm, full, emer.Back)
	net.ConnectLayers(vl, almct, full, emer.Back)

	// same prjns to stn as mtxgo
	net.ConnectToMatrix(us, vPmtxGo, pone2one)
	net.ConnectToMatrix(blaa, vPmtxGo, pone2one).SetClass("BLAToBG")
	net.ConnectToMatrix(blaa, vPmtxNo, pone2one).SetClass("BLAToBG")
	net.ConnectLayers(blaa, vPstnp, full, emer.Forward)
	net.ConnectLayers(blaa, vPstns, full, emer.Forward)

	net.ConnectToMatrix(blae, vPmtxGo, pone2one)
	net.ConnectToMatrix(blae, vPmtxNo, pone2one)
	net.ConnectToMatrix(drives, vPmtxGo, pone2one).SetClass("DrivesToMtx")
	net.ConnectToMatrix(drives, vPmtxNo, pone2one).SetClass("DrivesToMtx")
	net.ConnectLayers(drives, vPstnp, full, emer.Forward) // probably not good: modulatory
	net.ConnectLayers(drives, vPstns, full, emer.Forward)
	net.ConnectToMatrix(ofc, vPmtxGo, pone2one)
	net.ConnectToMatrix(ofc, vPmtxNo, pone2one)
	net.ConnectLayers(ofc, vPstnp, full, emer.Forward)
	net.ConnectLayers(ofc, vPstns, full, emer.Forward)
	// net.ConnectToMatrix(ofcct, vPmtxGo, pone2one) // important for matrix to mainly use CS & BLA
	// net.ConnectToMatrix(ofcct, vPmtxNo, pone2one)
	// net.ConnectToMatrix(ofcpt, vPmtxGo, pone2one)
	// net.ConnectToMatrix(ofcpt, vPmtxNo, pone2one)
	net.ConnectToMatrix(acc, vPmtxGo, full)
	net.ConnectToMatrix(acc, vPmtxNo, full)
	net.ConnectLayers(acc, vPstnp, full, emer.Forward)
	net.ConnectLayers(acc, vPstns, full, emer.Forward)
	// net.ConnectToMatrix(accct, vPmtxGo, pone2one)
	// net.ConnectToMatrix(accct, vPmtxNo, pone2one)
	// net.ConnectToMatrix(accpt, vPmtxGo, pone2one)
	// net.ConnectToMatrix(accpt, vPmtxNo, pone2one)
	// net.ConnectToMatrix(alm, vPmtxGo, full) // not to MD
	// net.ConnectToMatrix(alm, vPmtxNo, full)

	net.ConnectToRWPrjn(ofc, rwPred, full)
	net.ConnectToRWPrjn(ofcct, rwPred, full)
	net.ConnectToRWPrjn(acc, rwPred, full)
	net.ConnectToRWPrjn(accct, rwPred, full)

	////////////////////////////////////////////////
	// position

	vPgpi.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: rew.Name(), YAlign: relpos.Front, Space: space})
	ach.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: snc.Name(), XAlign: relpos.Left, Space: space})

	drives.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: rew.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	us.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: drives.Name(), XAlign: relpos.Left, Space: space})
	cs.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: drives.Name(), YAlign: relpos.Front, Space: space})
	dist.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: cs.Name(), YAlign: relpos.Front, Space: space})
	time.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: distp.Name(), XAlign: relpos.Left, Space: space})
	pos.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: dist.Name(), YAlign: relpos.Front, Space: space})
	gate.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: pos.Name(), XAlign: relpos.Left, Space: space})

	m1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: pos.Name(), YAlign: relpos.Front, Space: space})
	m1P.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: m1.Name(), XAlign: relpos.Left, Space: space})
	vl.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: m1P.Name(), XAlign: relpos.Left, Space: space})
	act.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: vl.Name(), XAlign: relpos.Left, Space: space})

	blaa.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: drives.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	ofc.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: blaa.Name(), YAlign: relpos.Front, Space: space})
	acc.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: ofc.Name(), YAlign: relpos.Front, Space: space})
	alm.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: acc.Name(), YAlign: relpos.Front, Space: space})

	// net.NThreads = 2

	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.Params.SetObject("Network")
	ss.InitWts(net)
}

// InitWts configures initial weights according to structure
func (ss *Sim) InitWts(net *axon.Network) {
	net.InitWts()
	ss.ViewUpdt.RecordSyns() // note: critical to update weights here so DWt is visible
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.InitRndSeed()
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.Params.SetAll()
	ss.NewRun()
	ss.ViewUpdt.Update()
	ss.ViewUpdt.RecordSyns()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.RndSeeds.Set(run)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	man.AddStack(etime.Train).AddTime(etime.Run, 5).AddTime(etime.Epoch, 100).AddTime(etime.Trial, 100).AddTime(etime.Cycle, 200) // AddTime(etime.Phase, 2).

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, 100).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Context, &ss.ViewUpdt) // std algo code

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Replace("UpdateWeights", func() {
		ss.Net.DWt(&ss.Context)
		ss.ViewUpdt.RecordSyns() // note: critical to update weights here so DWt is visible
		ss.Net.WtFmDWt(&ss.Context)
	})

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("Env:Step", func() {
			// note: OnStart for env.Env, others may happen OnEnd
			ss.Envs[mode.String()].Step()
		})
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
		stack.Loops[etime.Trial].OnEnd.Add("StatCounters", ss.StatCounters)
		stack.Loops[etime.Trial].OnEnd.Add("TrialStats", ss.TrialStats)
	}

	// note: plusPhase is shared between all stacks!
	plusPhase, _ := man.Stacks[etime.Train].Loops[etime.Cycle].EventByName("PlusPhase")
	plusPhase.OnEvent.Add("TakeAction", func() {
		ss.TakeAction(ss.Net)
	})

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Add Testing
	// trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	// trainEpoch.OnStart.Add("TestAtInterval", func() {
	// 	if (ss.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.TestInterval == 0) {
	// 		// Note the +1 so that it doesn't occur at the 0th timestep.
	// 		ss.TestAll()
	// 	}
	// })

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Sim.PCAInterval > 0) && (trnEpc%ss.Sim.PCAInterval == 0) {
			// if ss.Args.Bool("mpi") {
			// 	ss.Logs.MPIGatherTableRows(etime.Analyze, etime.Trial, ss.Comm)
			// }
			axon.PCAStats(ss.Net.AsAxon(), &ss.Logs, &ss.Stats)
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Sim.PCAInterval > 0) && (trnEpc%ss.Sim.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
	})

	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PctCortex", func() {
		trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if trnEpc >= ss.Sim.PctCortexStEpc && trnEpc%5 == 0 {
			ss.Sim.PctCortex = ss.Sim.PctCortexMax * float32(trnEpc-ss.Sim.PctCortexStEpc) / float32(ss.Sim.PctCortexMaxEpc-ss.Sim.PctCortexStEpc)
			if ss.Sim.PctCortex > ss.Sim.PctCortexMax {
				ss.Sim.PctCortex = ss.Sim.PctCortexMax
			} else {
				mpi.Printf("PctCortex updated to: %g at epoch: %d\n", ss.Sim.PctCortex, trnEpc)
			}
		}
	})

	////////////////////////////////////////////
	// GUI

	if ss.Args.Bool("nogui") {
		man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
			ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		})
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt)
		axon.LooperUpdtPlots(man, &ss.GUI)
	}

	if Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) TakeAction(net *axon.Network) {
	if ss.Sim.TwoThetas && ss.Sim.ThetaStep == 0 {
		ss.Sim.ThetaStep++
		return
	}
	// fmt.Printf("Take Action\n")
	ss.Sim.ThetaStep = 0 // reset for next time

	ev := ss.Envs[ss.Context.Mode.String()].(*Approach)

	netAct, anm := ss.DecodeAct(ev)
	genAct := ev.ActGen()
	genActNm := ev.Acts[genAct]
	ss.Stats.SetString("NetAction", anm)
	ss.Stats.SetString("InstinctAction", genActNm)
	if netAct == genAct {
		ss.Stats.SetFloat("ActMatch", 1)
	} else {
		ss.Stats.SetFloat("ActMatch", 0)
	}

	actAct := genAct
	if ss.Sim.CortexDriving {
		actAct = netAct
	}
	actActNm := ev.Acts[actAct]
	ss.Stats.SetString("ActAction", actActNm)

	ev.Action(actActNm, nil)
	ss.ApplyAction(actAct)
	// ss.ApplyRew()
	// fmt.Printf("action: %s\n", ev.Acts[act])
}

// DecodeAct decodes the VL ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ev *Approach) (int, string) {
	vt := ss.Stats.SetLayerTensor(ss.Net, "VL", "ActM")
	return ev.DecodeAct(vt)
}

// ApplyRew applies updated reward
func (ss *Sim) ApplyRew() {
	net := ss.Net
	ev := ss.Envs[ss.Context.Mode.String()].(*Approach)
	lays := []string{"Rew"}
	for _, lnm := range lays {
		ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		itsr := ev.State(lnm)
		ly.ApplyExt(itsr)
	}
}

// ApplyUS applies US
func (ss *Sim) ApplyUS() {
	net := ss.Net
	ev := ss.Envs[ss.Context.Mode.String()].(*Approach)
	lays := []string{"US"}
	for _, lnm := range lays {
		ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		itsr := ev.State(lnm)
		ly.ApplyExt(itsr)
	}
}

func (ss *Sim) ApplyAction(act int) {
	net := ss.Net
	ev := ss.Envs[ss.Context.Mode.String()]
	ly := net.LayerByName("VL").(axon.AxonLayer).AsAxon()
	ly.SetType(emer.Input)
	ap := ev.State("Action")
	ly.ApplyExt(ap)
	ly.SetType(emer.Target)
	ly = net.LayerByName("Act").(axon.AxonLayer).AsAxon()
	ly.ApplyExt(ap)
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	ev := ss.Envs[ss.Context.Mode.String()].(*Approach)

	if ev.Time == 0 {
		ss.Sim.CortexDriving = erand.BoolProb(float64(ss.Sim.PctCortex), -1)
		net.InitActs() // this is still essential even with fully functioning decay below:
		// todo: need a more selective US gating mechanism!
		net.DecayStateByType(&ss.Context, 1, 1, axon.SuperLayer, axon.PTMaintLayer, axon.CTLayer, axon.VThalLayer)
		ev.RenderLocalist("Gate", 0)
	}

	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Pos", "Drives", "US", "CS", "Dist", "Time", "Rew", "Gate"}
	for _, lnm := range lays {
		ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		itsr := ev.State(lnm)
		ly.ApplyExt(itsr)
	}

	// this is key step to drive DA and US-ACh
	if ev.US != -1 {
		ss.Context.NeuroMod.SetRew(ev.Rew, true)
	} else {
		ss.Context.NeuroMod.SetRew(0, false)
	}

	// fmt.Printf("Rew: %g\n", ev.Rew)
	// ss.ApplyUS() // now full trial
	// ss.ApplyRew()
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	// ss.Envs.ByMode(etime.Train).Init(0)
	// ss.Envs.ByMode(etime.Test).Init(0)
	ss.Context.Reset()
	ss.Context.Mode = etime.Train
	ss.Sim.ThetaStep = 0
	ss.Sim.PctCortex = 0
	ss.InitWts(ss.Net)
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	// ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("Gated", 0)
	ss.Stats.SetFloat("Should", 0)
	ss.Stats.SetFloat("GateUS", 0)
	ss.Stats.SetFloat("GateCS", 0)
	ss.Stats.SetFloat("GatedEarly", 0)
	ss.Stats.SetFloat("GatedPostCS", 0)
	ss.Stats.SetFloat("WrongCSGate", 0)
	ss.Stats.SetFloat("Rew", 0)
	ss.Stats.SetString("NetAction", "")
	ss.Stats.SetString("InstinctAction", "")
	ss.Stats.SetString("ActAction", "")
	ss.Stats.SetFloat("ActMatch", 0)
	ss.Stats.SetFloat("AllGood", 0)
	lays := ss.Net.LayersByType(axon.PTMaintLayer)
	for _, lnm := range lays {
		ss.Stats.SetFloat("Maint"+lnm, 0)
		ss.Stats.SetFloat("MaintFail"+lnm, 0)
		ss.Stats.SetFloat("PreAct"+lnm, 0)
	}
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ss.Stats.SetFloat32("PctCortex", ss.Sim.PctCortex)
	// ss.Stats.SetFloat32("ACCPos", ss.Sim.ACCPos)
	// ss.Stats.SetFloat32("ACCNeg", ss.Sim.ACCNeg)
	// trlnm := fmt.Sprintf("pos: %g, neg: %g", ss.Sim.ACCPos, ss.Sim.ACCNeg)
	ss.Stats.SetString("TrialName", "trl")
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Cycle", "NetAction", "InstinctAction", "ActAction", "ActMatch", "Gated", "Should", "Rew"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	ss.GatedStats()
	ss.MaintStats()
	ss.Stats.SetFloat("DA", float64(ss.Context.NeuroMod.DA))
	ss.Stats.SetFloat("ACh", float64(ss.Context.NeuroMod.ACh))
	ss.Stats.SetFloat("AChRaw", float64(ss.Context.NeuroMod.AChRaw))
	ss.Stats.SetFloat("RewPred", float64(ss.Context.NeuroMod.RewPred))

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
	if fv := ss.Stats.Float("GatedPostCS"); !math.IsNaN(fv) {
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

// GatedStats updates the gated states
func (ss *Sim) GatedStats() {
	if ss.Sim.TwoThetas && ss.Sim.ThetaStep == 1 {
		return
	}
	// fmt.Printf("Gate Stats\n")
	net := ss.Net
	ev := ss.Envs[ss.Context.Mode.String()].(*Approach)
	mtxLy := net.LayerByName("VpMtxGo").(*axon.Layer)
	didGate := mtxLy.AnyGated()
	ss.Stats.SetFloat32("Gated", bools.ToFloat32(didGate))
	ss.Stats.SetFloat32("Should", bools.ToFloat32(ev.ShouldGate))
	ss.Stats.SetFloat32("GateUS", mat32.NaN())
	ss.Stats.SetFloat32("GateCS", mat32.NaN())
	ss.Stats.SetFloat32("GatedEarly", mat32.NaN())
	ss.Stats.SetFloat32("GatedPostCS", mat32.NaN())
	ss.Stats.SetFloat32("WrongCSGate", mat32.NaN())
	if didGate {
		ss.Stats.SetFloat32("WrongCSGate", bools.ToFloat32(ev.Drive != ev.USForPos()))
	}
	if ev.ShouldGate {
		if ev.US != -1 {
			ss.Stats.SetFloat32("GateUS", bools.ToFloat32(didGate))
		} else {
			ss.Stats.SetFloat32("GateCS", bools.ToFloat32(didGate))
		}
	} else {
		if ev.Dist < ev.DistMax-1 { // todo: not very robust
			ss.Stats.SetFloat32("GatedPostCS", bools.ToFloat32(didGate))
		} else {
			ss.Stats.SetFloat32("GatedEarly", bools.ToFloat32(didGate))
		}
	}
	ss.Stats.SetFloat32("Rew", ev.Rew)
}

// MaintStats updates the PFC maint stats
func (ss *Sim) MaintStats() {
	if ss.Sim.TwoThetas && ss.Sim.ThetaStep == 0 {
		return
	}
	// fmt.Printf("Maint Stats\n")
	ev := ss.Envs[ss.Context.Mode.String()].(*Approach)
	// should be maintaining while going forward
	isFwd := ev.LastAct == ev.ActMap["Forward"]
	isCons := ev.LastAct == ev.ActMap["Consume"]
	actThr := float32(0.05) // 0.1 too high
	net := ss.Net
	lays := net.LayersByType(axon.PTMaintLayer)
	hasMaint := false
	for _, lnm := range lays {
		mnm := "Maint" + lnm
		fnm := "MaintFail" + lnm
		pnm := "PreAct" + lnm
		ptly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		var mact float32
		if ptly.Is4D() {
			for pi := 1; pi < len(ptly.Pools); pi++ {
				avg := ptly.Pools[pi].AvgMax.Act.Plus.Avg
				if avg > mact {
					mact = avg
				}
			}
		} else {
			mact = ptly.Pools[0].AvgMax.Act.Plus.Avg
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
		ev.RenderLocalist("Gate", 1)
	} else {
		ev.RenderLocalist("Gate", 0)
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Phase, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	// ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "PctCortex")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "NetAction", "InstinctAction", "ActAction")
	// ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCPos")
	// ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCNeg")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	axon.LogAddPulvCorSimItems(&ss.Logs, ss.Net.AsAxon(), etime.Run, etime.Epoch, etime.Trial)

	// ss.ConfigActRFs()

	axon.LogAddDiagnosticItems(&ss.Logs, ss.Net.AsAxon(), etime.Epoch, etime.Trial)

	// todo: PCA items should apply to CT layers too -- pass a type here.
	axon.LogAddPCAItems(&ss.Logs, ss.Net.AsAxon(), etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Cycle)
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "Target")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.AllModes, etime.Cycle, "Target")

	ss.Logs.PlotItems("AllGood", "ActMatch", "GateCS", "WrongCSGate")
	// "MaintOFCPT", "MaintACCPT", "MaintFailOFCPT", "MaintFailACCPT"
	// "GateUS", "GatedEarly", "GatedPostCS", "Gated", "PctCortex",
	// "Rew", "DA", "MtxGo_ActAvg"

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net.AsAxon())
	// don't plot certain combinations we don't use
	// ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Phase)
	ss.Logs.NoPlot(etime.Test, etime.Phase)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	// ss.Logs.SetMeta(etime.Test, etime.Cycle, "LegendCol", "RunName")
}

func (ss *Sim) ConfigLogItems() {
	ss.Logs.AddStatAggItem("AllGood", "AllGood", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ActMatch", "ActMatch", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Gated", "Gated", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Should", "Should", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GateUS", "GateUS", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GateCS", "GateCS", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GatedEarly", "GatedEarly", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GatedPostCS", "GatedPostCS", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("WrongCSGate", "WrongCSGate", etime.Run, etime.Epoch, etime.Trial)

	lays := ss.Net.LayersByType(axon.PTMaintLayer)
	for _, lnm := range lays {
		nm := "Maint" + lnm
		ss.Logs.AddStatAggItem(nm, nm, etime.Run, etime.Epoch, etime.Trial)
		nm = "MaintFail" + lnm
		ss.Logs.AddStatAggItem(nm, nm, etime.Run, etime.Epoch, etime.Trial)
		nm = "PreAct" + lnm
		ss.Logs.AddStatAggItem(nm, nm, etime.Run, etime.Epoch, etime.Trial)
	}
	li := ss.Logs.AddStatAggItem("Rew", "Rew", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA", "DA", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("ACh", "ACh", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("AChRaw", "AChRaw", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("RewPred", "RewPred", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false

	ev := TheSim.Envs[etime.Train.String()].(*Approach)
	ss.Logs.AddItem(&elog.Item{
		Name:      "ActCor",
		Type:      etensor.FLOAT64,
		CellShape: []int{len(ev.Acts)},
		DimNames:  []string{"Acts"},
		// Plot:      true,
		Range:     minmax.F64{Min: 0},
		TensorIdx: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IdxView(ctx.Mode, etime.Trial)
				spl := split.GroupBy(ix, []string{"InstinctAction"})
				split.AggTry(spl, "ActMatch", agg.AggMean)
				ags := spl.AggsToTable(etable.ColNameOnly)
				ss.Logs.MiscTables["ActCor"] = ags
				ctx.SetTensor(ags.Cols[0]) // cors
			}}})
	for _, nm := range ev.Acts { // per-action % correct
		anm := nm // closure
		ss.Logs.AddItem(&elog.Item{
			Name: anm + "Cor",
			Type: etensor.FLOAT64,
			// Plot:  true,
			Range: minmax.F64{Min: 0},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ags := ss.Logs.MiscTables["ActCor"]
					rw := ags.RowsByString("InstinctAction", anm, etable.Equals, etable.UseCase)
					if len(rw) > 0 {
						ctx.SetFloat64(ags.CellFloat("ActMatch", rw[0]))
					}
				}}})
	}
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
	}
	ss.StatCounters()

	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "BOA = BG, OFC ACC Test"
	ss.GUI.MakeWindow(ss, "boa", title, `This project tests learning in the BG, OFC & ACC for basic approach learning to a CS associated with a US. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)

	nv.Scene().Camera.Pose.Pos.Set(0, 1.4, 2.6)
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	ss.GUI.ViewUpdt = &ss.ViewUpdt

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train, etime.Test})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "TestInit", Icon: "update",
		Tooltip: "reinitialize the testing control so it re-runs.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Loops.ResetCountersByMode(etime.Test)
			ss.GUI.UpdateWindow()
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
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
	return ss.GUI.Win
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.SetInt("epochs", 200)
	ss.Args.SetInt("runs", 10)
	ss.Args.Parse() // always parse
}

func (ss *Sim) CmdArgs() {
	ss.Args.ProcStd(&ss.Params)
	ss.Args.ProcStdLogs(&ss.Logs, &ss.Params, ss.Net.Name())
	ss.Args.SetBool("nogui", true)                                       // by definition if here
	ss.Stats.SetString("RunName", ss.Params.RunName(ss.Args.Int("run"))) // used for naming logs, stats, etc

	netdata := ss.Args.Bool("netdata")
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	mpi.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := &ss.Loops.GetLoop(etime.Train, etime.Run).Counter
	rc.Set(run)
	rc.Max = run + runs

	ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = ss.Args.Int("epochs")

	ss.NewRun()
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
