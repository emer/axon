// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
boa: This project tests BG, OFC & ACC learning in a CS-driven approach task.
*/
package main

import (
	"log"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/axon/pcore"
	"github.com/emer/axon/pvlv"
	"github.com/emer/axon/rl"
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
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
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
	PctCortex       float32 `desc:"proportion of action driven by the cortex vs. hard-coded reflexive subcortical"`
	PctCortexMax    float32 `desc:"maximum PctCortex, when running on the schedule"`
	PctCortexMaxEpc int     `desc:"epoch when PctCortexMax is reached"`
	PCAInterval     int     `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
}

// Defaults sets default params
func (ss *SimParams) Defaults() {
	ss.PctCortexMax = 0.0
	ss.PctCortexMaxEpc = 100
	ss.PCAInterval = 10
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *pcore.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Sim          SimParams        `view:"no-inline" desc:"sim params"`
	Params       emer.Params      `view:"inline" desc:"all parameter management"`
	Loops        *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats     `desc:"contains computed statistic values"`
	Logs         elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Pats         *etable.Table    `view:"no-inline" desc:"the training patterns to use"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Time         axon.Time        `desc:"axon timing parameters and state"`
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
	ss.Net = &pcore.Network{}
	ss.Sim.Defaults()
	ss.Params.Params = ParamSets
	ss.Params.ExtraSets = "WtScales"
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Pats = &etable.Table{}
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.TestInterval = 500
	ss.Time.Defaults()
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

func (ss *Sim) ConfigNet(net *pcore.Network) {
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

	rew, rp, snci := rl.AddRWLayers(net.AsAxon(), "", relpos.Behind, 2)
	_ = rew
	_ = rp
	snc := snci.(*rl.RWDaLayer)

	drives, drivesp := net.AddInputPulv4D("Drives", 1, ev.NDrives, ny, 1, space)
	us, usp := net.AddInputPulv4D("US", 1, ev.NDrives, ny, 1, space)
	// cs, csp := net.AddInputPulv2D("CS", ev.PatSize.Y, ev.PatSize.X, space)
	// localist, for now:
	cs, csp := net.AddInputPulv2D("CS", ny, ev.NDrives, space)
	_ = csp
	dist, distp := net.AddInputPulv2D("Dist", ny, ev.DistMax, space)
	time, timep := net.AddInputPulv2D("Time", ny, ev.TimeMax, space)
	pos, posp := net.AddInputPulv2D("Pos", ny, nloc, space)

	vPmtxGo, vPmtxNo, vPcini, _, _, _, vPstnp, vPstns, vPgpi := net.AddBG("Vp", 1, ev.NDrives, nuBgY, nuBgX, nuBgY, nuBgX, space)
	cin := vPcini.(*pcore.CINLayer)
	cin.RewLays.Add(snc.Name())

	// todo: need m1d, driven by smad -- output pathway

	m1 := net.AddLayer2D("M1", nuCtxY, nuCtxX, emer.Hidden)
	vl := net.AddLayer2D("VL", ny, nAct, emer.Target)  // Action
	act := net.AddLayer2D("Act", ny, nAct, emer.Input) // Action
	m1p := deep.AddPulvLayer2D(net.AsAxon(), "M1P", nuCtxY, nuCtxX)
	m1p.Driver = m1.Name()
	_ = vl
	_ = act

	blaa, blae := pvlv.AddBLALayers(net.AsAxon(), "BLA", true, ev.NDrives, nuCtxY, nuCtxX, relpos.Behind, space)

	// todo: rename sma -> ALM

	sma, smact := net.AddSuperCT2D("SMA", nuCtxY, nuCtxX, space, one2one)
	smapt, smathal := net.AddPTThalForSuper(sma, smact, "MD", one2one, full, full, space)
	smact.SetClass("SMA CTCopy")
	_ = smapt
	// net.ConnectCTSelf(smact, full)
	net.ConnectToPulv(sma, smact, m1p, full, full)
	net.ConnectToPulv(sma, smact, posp, full, full)
	net.ConnectToPulv(sma, smact, distp, full, full)
	net.ConnectLayers(vPgpi, smathal, full, emer.Inhib).SetClass("BgFixed")

	//	todo: add a PL layer, with Integ maint

	ofc, ofcct := net.AddSuperCT4D("OFC", 1, ev.NDrives, nuCtxY, nuCtxX, space, one2one)
	// prjns are: super->PT, PT self, CT-> thal
	ofcpt, ofcthal := net.AddPTThalForSuper(ofc, ofcct, "MD", pone2one, pone2one, pone2one, space)
	_ = ofcpt
	ofcct.SetClass("OFC CTInteg")
	// net.ConnectCTSelf(ofcct, pone2one) // much better for ofc not to have self prjns..
	// net.ConnectToPulv(ofc, ofcct, csp, full, full)
	net.ConnectToPulv(ofc, ofcct, usp, pone2one, pone2one)
	net.ConnectToPulv(ofc, ofcct, drivesp, pone2one, pone2one)
	net.ConnectLayers(vPgpi, ofcthal, full, emer.Inhib).SetClass("BgFixed")

	// todo: add ofcp and acc projections to it
	// todo: acc should have pos and negative stripes, with grounded prjns??

	acc, accct := net.AddSuperCT2D("ACC", nuCtxY, nuCtxX, space, one2one)
	// prjns are: super->PT, PT self, CT->thal
	accpt, accthal := net.AddPTThalForSuper(acc, accct, "MD", one2one, full, full, space)
	_ = accpt
	accct.SetClass("ACC CTInteg")
	net.ConnectCTSelf(accct, full)
	net.ConnectToPulv(acc, accct, distp, full, full)
	net.ConnectToPulv(acc, accct, timep, full, full)
	net.ConnectLayers(vPgpi, accthal, full, emer.Inhib).SetClass("BgFixed")

	net.ConnectLayers(dist, acc, full, emer.Forward)
	net.ConnectLayers(time, acc, full, emer.Forward)

	vPmtxGo.(*pcore.MatrixLayer).MtxThals.Add(accthal.Name(), ofcthal.Name())
	vPmtxNo.(*pcore.MatrixLayer).MtxThals.Add(accthal.Name(), ofcthal.Name())

	// m1p plus phase has action, Ctxt -> CT allows CT now to use that prev action

	// contextualization based on action
	net.BidirConnectLayers(ofc, sma, full)
	net.BidirConnectLayers(acc, sma, full)

	// Std corticocortical cons -- stim -> hid
	// net.ConnectLayers(cs, ofc, full, emer.Forward) // let BLA handle it
	net.ConnectLayers(us, ofc, pone2one, emer.Forward)
	net.ConnectLayers(drives, ofc, pone2one, emer.Forward).SetClass("BgFixed")

	// todo: blae is not connected properly at all yet

	// BLA
	net.ConnectLayersPrjn(cs, blaa, full, emer.Forward, &pvlv.BLAPrjn{})
	net.ConnectLayersPrjn(us, blaa, pone2one, emer.Forward, &pvlv.BLAPrjn{}).SetClass("USToBLA")
	// net.ConnectLayersPrjn(usp, blaa, pone2one, emer.Forward, &pvlv.BLAPrjn{}).SetClass("USToBLA")
	net.ConnectLayersPrjn(drives, blaa, pone2one, emer.Forward, &pvlv.BLAPrjn{}).SetClass("USToBLA")
	net.ConnectLayers(blaa, ofc, pone2one, emer.Forward)
	// todo: from deep maint layer
	// net.ConnectLayersPrjn(ofcpt, blae, pone2one, emer.Forward, &pvlv.BLAPrjn{})
	net.ConnectLayers(blae, blaa, pone2one, emer.Inhib).SetClass("BgFixed")
	// net.ConnectLayers(drives, blae, pone2one, emer.Forward)

	net.ConnectLayers(dist, sma, full, emer.Forward)
	net.ConnectLayers(time, sma, full, emer.Forward)
	net.ConnectLayers(pos, sma, full, emer.Forward)
	// key point: cs does not project directly to sma -- no simple S -> R mappings!?

	////////////////////////////////////////////////
	// BG / DA connections
	snc.SendDA.AddAllBut(net)

	net.ConnectLayers(smapt, m1, full, emer.Forward)     //  action output
	net.ConnectLayers(sma, smapt, one2one, emer.Forward) // is weaker, provides some action sel but gating = stronger
	// net.ConnectLayers(sma, m1, full, emer.Forward)  //  note: non-gated!
	net.BidirConnectLayers(m1, vl, full)

	// same prjns to stn as mtxgo
	net.ConnectToMatrix(us, vPmtxGo, pone2one)
	net.ConnectToMatrix(blaa, vPmtxGo, pone2one)
	net.ConnectToMatrix(blaa, vPmtxNo, pone2one)
	net.ConnectLayers(blaa, vPstnp, full, emer.Forward)
	net.ConnectLayers(blaa, vPstns, full, emer.Forward)

	net.ConnectToMatrix(blae, vPmtxGo, pone2one)
	net.ConnectToMatrix(blae, vPmtxNo, pone2one)
	net.ConnectToMatrix(drives, vPmtxGo, pone2one)
	net.ConnectToMatrix(drives, vPmtxNo, pone2one)
	net.ConnectLayers(drives, vPstnp, full, emer.Forward)
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
	// net.ConnectToMatrix(sma, vPmtxGo, full) // not to MD
	// net.ConnectToMatrix(sma, vPmtxNo, full)

	net.ConnectLayersPrjn(ofc, rp, full, emer.Forward, &rl.RWPrjn{})
	net.ConnectLayersPrjn(ofcct, rp, full, emer.Forward, &rl.RWPrjn{})
	net.ConnectLayersPrjn(acc, rp, full, emer.Forward, &rl.RWPrjn{})
	net.ConnectLayersPrjn(accct, rp, full, emer.Forward, &rl.RWPrjn{})

	////////////////////////////////////////////////
	// position

	vPgpi.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: rew.Name(), YAlign: relpos.Front, Space: space})

	drives.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: rew.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	us.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: drivesp.Name(), XAlign: relpos.Left, Space: space})
	cs.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: drives.Name(), YAlign: relpos.Front, Space: space})
	dist.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: cs.Name(), YAlign: relpos.Front, Space: space})
	time.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: distp.Name(), XAlign: relpos.Left, Space: space})
	pos.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: dist.Name(), YAlign: relpos.Front, Space: space})

	m1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: pos.Name(), YAlign: relpos.Front, Space: space})
	m1p.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: m1.Name(), XAlign: relpos.Left, Space: space})
	vl.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: m1p.Name(), XAlign: relpos.Left, Space: space})
	act.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: vl.Name(), XAlign: relpos.Left, Space: space})

	blaa.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: drives.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	ofc.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: blaa.Name(), YAlign: relpos.Front, Space: space})
	acc.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: ofc.Name(), YAlign: relpos.Front, Space: space})
	sma.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: acc.Name(), YAlign: relpos.Front, Space: space})

	net.Defaults()
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
}

// InitWts configures initial weights according to structure
func (ss *Sim) InitWts(net *pcore.Network) {
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

	axon.LooperStdPhases(man, &ss.Time, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Time, &ss.ViewUpdt) // std algo code

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Replace("UpdateWeights", func() {
		ss.Net.DWt(&ss.Time)
		ss.ViewUpdt.RecordSyns() // note: critical to update weights here so DWt is visible
		ss.Net.WtFmDWt(&ss.Time)
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
		if trnEpc > 1 && trnEpc%5 == 0 {
			ss.Sim.PctCortex = float32(trnEpc) / float32(ss.Sim.PctCortexMaxEpc)
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
func (ss *Sim) TakeAction(net *pcore.Network) {
	ev := ss.Envs[ss.Time.Mode].(*Approach)

	netAct, anm := ss.DecodeAct(ev)
	genAct := ev.ActGen()
	genActNm := ev.Acts[genAct]
	ss.Stats.SetString("NetAction", anm)
	ss.Stats.SetString("GenAction", genActNm)
	if netAct == genAct {
		ss.Stats.SetFloat("ActMatch", 1)
	} else {
		ss.Stats.SetFloat("ActMatch", 0)
	}

	actAct := genAct
	if erand.BoolProb(float64(ss.Sim.PctCortex), -1) {
		actAct = netAct
	}
	actActNm := ev.Acts[actAct]
	ss.Stats.SetString("ActAction", actActNm)

	ev.Action(actActNm, nil)

	ss.GatedStats()
	ss.ApplyAction(actAct)
	ss.ApplyRew()
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
	ev := ss.Envs[ss.Time.Mode].(*Approach)
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
	ev := ss.Envs[ss.Time.Mode].(*Approach)
	lays := []string{"US"}
	for _, lnm := range lays {
		ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		itsr := ev.State(lnm)
		ly.ApplyExt(itsr)
	}
}

// GatedStats updates the gated states based on gating -- when action is taken
func (ss *Sim) GatedStats() {
	net := ss.Net
	ev := ss.Envs[ss.Time.Mode].(*Approach)
	mtxLy := net.LayerByName("VpMtxGo").(*pcore.MatrixLayer)
	mtxLy.GatedFmAvgSpk()
	didGate := mtxLy.AnyGated()
	ss.Stats.SetFloat32("Gated", pcore.BoolToFloat32(didGate))
	ss.Stats.SetFloat32("Should", pcore.BoolToFloat32(ev.ShouldGate))
	ss.Stats.SetFloat32("GateUS", mat32.NaN())
	ss.Stats.SetFloat32("GateCS", mat32.NaN())
	ss.Stats.SetFloat32("NoGatePre", mat32.NaN())
	ss.Stats.SetFloat32("NoGatePost", mat32.NaN())
	if ev.ShouldGate {
		if ev.US != -1 {
			ss.Stats.SetFloat32("GateUS", pcore.BoolToFloat32(didGate))
		} else {
			ss.Stats.SetFloat32("GateCS", pcore.BoolToFloat32(didGate))
		}
	} else {
		if ev.Dist < ev.DistMax-1 { // todo: not very robust
			ss.Stats.SetFloat32("NoGatePost", pcore.BoolToFloat32(!didGate))
		} else {
			ss.Stats.SetFloat32("NoGatePre", pcore.BoolToFloat32(!didGate))
		}
	}
	ss.Stats.SetFloat32("Rew", ev.Rew)
}

func (ss *Sim) ApplyAction(act int) {
	net := ss.Net
	ev := ss.Envs[ss.Time.Mode]
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
	ev := ss.Envs[ss.Time.Mode].(*Approach)

	if ev.Time == 0 {
		net.InitActs()
		net.DecayStateByClass(1, 1, "Hidden", "PT", "CT") // US action gating
	}

	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Pos", "Drives", "US", "CS", "Dist", "Time", "Rew"}
	for _, lnm := range lays {
		ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		itsr := ev.State(lnm)
		ly.ApplyExt(itsr)
	}
	ss.ApplyUS() // now full trial
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	// ss.Envs.ByMode(etime.Train).Init(0)
	// ss.Envs.ByMode(etime.Test).Init(0)
	ss.Time.Reset()
	ss.Time.Mode = etime.Train.String()
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
	ss.Stats.SetFloat("NoGatePre", 0)
	ss.Stats.SetFloat("NoGatePost", 0)
	ss.Stats.SetFloat("Rew", 0)
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	var mode etime.Modes
	mode.FromString(ss.Time.Mode)
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	ss.Stats.SetFloat("PctCortex", float64(ss.Sim.PctCortex))
	// ss.Stats.SetFloat("ACCPos", float64(ss.Sim.ACCPos))
	// ss.Stats.SetFloat("ACCNeg", float64(ss.Sim.ACCNeg))
	// trlnm := fmt.Sprintf("pos: %g, neg: %g", ss.Sim.ACCPos, ss.Sim.ACCNeg)
	ss.Stats.SetString("TrialName", "trl")
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Cycle", "NetAction", "GenAction", "ActAction", "ActMatch", "Gated", "Should", "Rew"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	/*
		mode := etime.ModeFromString(ss.Time.Mode)
		trlog := ss.Logs.Log(mode, etime.Cycle)
		spkCyc := 0
		for row := 0; row < trlog.Rows; row++ {
			vts := trlog.CellTensorFloat1D("VThal_Spike", row, 0)
			if vts > 0 {
				spkCyc = row
				break
			}
		}
		ss.Stats.SetFloat("VThal_RT", float64(spkCyc)/200)
	*/
	da := ss.Net.LayerByName("DA").(axon.AxonLayer).AsAxon()
	ss.Stats.SetFloat("DA", float64(da.Neurons[0].Act))

}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Phase, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	// ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "PctCortex")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "NetAction", "GenAction", "ActAction")
	// ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCPos")
	// ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCNeg")

	ss.Logs.AddStatAggItem("ActMatch", "ActMatch", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Gated", "Gated", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Should", "Should", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GateUS", "GateUS", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GateCS", "GateCS", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("NoGatePre", "NoGatePre", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("NoGatePost", "NoGatePost", etime.Run, etime.Epoch, etime.Trial)
	li := ss.Logs.AddStatAggItem("Rew", "Rew", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA", "DA", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	deep.LogAddPulvCorSimItems(&ss.Logs, ss.Net.AsAxon(), etime.Run, etime.Epoch, etime.Trial)

	// ss.ConfigActRFs()

	axon.LogAddDiagnosticItems(&ss.Logs, ss.Net.AsAxon(), etime.Epoch, etime.Trial)

	// todo: PCA items should apply to CT layers too -- pass a type here.
	axon.LogAddPCAItems(&ss.Logs, ss.Net.AsAxon(), etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Cycle)
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "Target")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.AllModes, etime.Cycle, "Target")

	ss.Logs.PlotItems("ActMatch", "Gated", "GateUS", "GateCS", "NoGatePre", "NoGatePost") // "PctCortex", "Rew", "DA",  "MtxGo_ActAvg", "VThal_ActAvg", "VThal_RT")

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
	return
	// ss.Logs.AddStatAggItem("VThal_RT", "VThal_RT", etime.Run, etime.Epoch, etime.Trial)
	npools := 1
	poolshape := []int{npools}
	layers := ss.Net.LayersByClass("Matrix Thal")
	for _, lnm := range layers {
		clnm := lnm
		if clnm == "CIN" {
			continue
		}
		ss.Logs.AddItem(&elog.Item{
			Name:      clnm + "_ActAvg",
			Type:      etensor.FLOAT64,
			CellShape: poolshape,
			// Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					tsr := ss.Stats.F64Tensor("Log_ActAvg")
					tsr.SetShape(poolshape, nil, nil)
					ss.Stats.SetF64Tensor("Log_ActAvg", tsr)
					for pi := 0; pi < npools; pi++ {
						tsr.Values[pi] = float64(ly.Pools[pi+1].Inhib.Act.Avg)
					}
					ctx.SetTensor(tsr)
				}, etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					tsr := ss.Stats.F64Tensor("Log_ActAvg")
					tsr.SetShape(poolshape, nil, nil)
					ss.Stats.SetF64Tensor("Log_ActAvg", tsr)
					for pi := 0; pi < npools; pi++ {
						tsr.Values[pi] = float64(ly.Pools[pi+1].Inhib.Act.Avg)
					}
					ctx.SetTensor(tsr)
				}, etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
					tsr := ss.Stats.F64Tensor("Log_ActAvg")
					lyi := ctx.Layer(clnm)
					ly := lyi.(axon.AxonLayer).AsAxon()
					for pi := 0; pi < npools; pi++ {
						tsr.Values[pi] = float64(ly.SpkMaxAvgByPool(pi + 1))
					}
					ctx.SetTensor(tsr)
				}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}, etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:      clnm + "_Spike",
			Type:      etensor.FLOAT64,
			CellShape: poolshape,
			FixMin:    true,
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, etime.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					tsr := ss.Stats.F64Tensor("Log_ActAvg")
					tsr.SetShape(poolshape, nil, nil)
					ss.Stats.SetF64Tensor("Log_ActAvg", tsr)
					for pi := 0; pi < npools; pi++ {
						tsr.Values[pi] = float64(ly.SpikeAvgByPool(pi + 1))
					}
					ctx.SetTensor(tsr)
				}}})
	}
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Time.Mode = mode.String() // Also set specifically in a Loop callback.
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
