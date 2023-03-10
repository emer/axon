// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
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
	"github.com/emer/envs/cond"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
	"github.com/goki/vgpu/vgpu"
)

var (
	// Debug triggers various messages etc
	Debug = false
	// GPU runs with the GPU (for demo, testing -- not useful for such a small network)
	GPU = false
)

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

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	RunName  string           `view:"-" desc:"environment run name"`
	Net      *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params   emer.Params      `view:"inline" desc:"all parameter management"`
	Loops    *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats    estats.Stats     `desc:"contains computed statistic values"`
	Logs     elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Pats     *etable.Table    `view:"no-inline" desc:"the training patterns to use"`
	Envs     env.Envs         `view:"no-inline" desc:"Environments"`
	Context  axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.RunName = "PosAcq" // "PosAcq_B50"
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.Pats = &etable.Table{}
	ss.RndSeeds.Init(100) // max 100 runs
	ss.Context.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
	// ss.Defaults()
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
	var trn *cond.CondEnv
	if len(ss.Envs) == 0 {
		trn = &cond.CondEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	}

	// note: names must be standard here!
	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	trn.Config(ss.Args.Int("runs"), ss.RunName)
	trn.Validate()

	trn.Init(0)

	ss.Context.DrivePVLV.Drive.NActive = int32(cond.NUSs)

	// note: names must be in place when adding
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "PVLV")
	ev := ss.Envs["Train"].(*cond.CondEnv)
	ny := ev.NYReps
	nUSs := cond.NUSs

	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	popY := 4
	popX := 4
	space := float32(2)

	pone2one := prjn.NewPoolOneToOne()
	one2one := prjn.NewOneToOne()
	full := prjn.NewFull()
	_ = pone2one

	stim := ev.CurStates["StimIn"]
	ctxt := ev.CurStates["ContextIn"]
	// timeIn := ev.CurStates["USTimeIn"]

	vta, lhb := net.AddVTALHbLayers(relpos.Behind, space)
	ach := net.AddRSalienceAChLayer("ACh")

	vPmtxGo, vPmtxNo, _, _, _, vPstnp, vPstns, vPgpi := net.AddBG("Vp", 1, nUSs, nuBgY, nuBgX, nuBgY, nuBgX, space)

	vsPatchPosD1, vsPatchPosD2 := net.AddVSPatchLayers("", true, nUSs, nuBgY, nuBgX, relpos.Behind, space)
	vsPatchNegD1, vsPatchNegD2 := net.AddVSPatchLayers("", false, nUSs, nuBgY, nuBgX, relpos.Behind, space)
	_ = vsPatchNegD1
	_ = vsPatchNegD2

	drives, drivesP := net.AddDrivesPulvLayer(&ss.Context, popY, popX, space)
	usPos, usNeg, usPosP, usNegP := net.AddUSPulvLayers(nUSs, nUSs, ny, relpos.Behind, space)
	_ = usNegP
	_ = usPos
	_ = usNeg

	pvPos, pvNeg, pvPosP, pvNegP := net.AddPVPulvLayers(popY, popX, relpos.Behind, space)
	_ = pvNegP
	time, timeP := net.AddInputPulv4D("Time", 1, cond.MaxTime, ny, 1, space)

	cs, csP := net.AddInputPulv4D("StimIn", stim.Dim(0), stim.Dim(1), stim.Dim(2), stim.Dim(3), space)

	ctxIn := net.AddLayer4D("ContextIn", ctxt.Dim(0), ctxt.Dim(1), ctxt.Dim(2), ctxt.Dim(3), axon.InputLayer)
	_ = ctxIn
	// ustimeIn := net.AddLayer4D("USTimeIn", timeIn.Dim(0), timeIn.Dim(1), timeIn.Dim(2), timeIn.Dim(3), axon.InputLayer)
	// _ = ustimeIn

	gate := net.AddLayer2D("Gate", ny, 2, axon.InputLayer) // signals gated or not
	_ = gate

	blaPosA, blaPosE, blaNegA, blaNegE, cemPos, cemNeg, pptg := net.AddAmygdala("", true, nUSs, nuCtxY, nuCtxX, space)
	_ = cemPos
	_ = pptg
	_ = blaNegE
	_ = cemNeg
	blaPosA.SetBuildConfig("LayInhib1Name", blaNegA.Name())
	blaNegA.SetBuildConfig("LayInhib1Name", blaPosA.Name())

	ofc, ofcCT := net.AddSuperCT4D("OFC", 1, nUSs, nuCtxY, nuCtxX, space, one2one)
	// prjns are: super->PT, PT self, CT-> thal
	ofcPT, ofcMD := net.AddPTMaintThalForSuper(ofc, ofcCT, "MD", one2one, pone2one, pone2one, space)
	_ = ofcPT
	ofcCT.SetClass("OFC CTCopy")
	ofcPTPred := net.AddPTPredLayer(ofcPT, ofcCT, ofcMD, pone2one, pone2one, pone2one, space)
	_ = ofcPTPred
	_ = ofcPT
	ofcCT.SetClass("OFC CTCopy")
	// net.ConnectToPulv(ofc, ofcCT, usPulv, pone2one, pone2one)
	// Drives -> OFC then activates OFC -> VS -- OFC needs to be strongly BLA dependent
	// to reflect either current CS or maintained CS but not just echoing drive state.
	net.ConnectLayers(drives, ofc, pone2one, emer.Forward).SetClass("DrivesToOFC")
	// net.ConnectLayers(drives, ofcCT, pone2one, emer.Forward).SetClass("DrivesToOFC")
	net.ConnectLayers(vPgpi, ofcMD, full, emer.Inhib).SetClass("BgFixed")
	// net.ConnectLayers(cs, ofc, full, emer.Forward) // let BLA handle it
	net.ConnectLayers(time, ofc, full, emer.Forward).SetClass("TimeToOFC")
	net.ConnectLayers(pvPos, ofc, full, emer.Forward).SetClass("PVposToOFC")
	net.ConnectLayers(usPos, ofc, pone2one, emer.Forward)
	net.ConnectLayers(ofcPT, ofcCT, pone2one, emer.Forward)

	net.ConnectToPulv(ofc, ofcCT, drivesP, pone2one, pone2one)
	net.ConnectToPulv(ofc, ofcCT, usPosP, pone2one, pone2one)
	net.ConnectToPulv(ofc, ofcCT, pvPosP, pone2one, pone2one)
	net.ConnectToPulv(ofc, ofcCT, csP, full, full)
	net.ConnectToPulv(ofc, ofcCT, timeP, full, full)

	// net.ConnectPTPredToPulv(ofcPTPred, drivesP, pone2one, pone2one)
	// net.ConnectPTPredToPulv(ofcPTPred, usPosP, pone2one, pone2one)
	// net.ConnectPTPredToPulv(ofcPTPred, pvPosP, pone2one, pone2one)
	net.ConnectPTPredToPulv(ofcPTPred, csP, full, full)
	net.ConnectPTPredToPulv(ofcPTPred, timeP, full, full)

	vPmtxGo.SetBuildConfig("ThalLay1Name", ofcMD.Name())
	vPmtxNo.SetBuildConfig("ThalLay1Name", ofcMD.Name())

	// BLA
	net.ConnectToBLA(cs, blaPosA, full)
	net.ConnectToBLA(usPos, blaPosA, pone2one).SetClass("USToBLA")
	net.ConnectLayers(blaPosA, ofc, pone2one, emer.Forward)
	// todo: from deep maint layer
	// net.ConnectLayersPrjn(ofcPT, blaPosE, pone2one, emer.Forward, &axon.BLAPrjn{})
	net.ConnectLayers(blaPosE, blaPosA, pone2one, emer.Inhib).SetClass("BgFixed")
	net.ConnectLayers(ofcPTPred, blaPosE, pone2one, emer.Forward).SetClass("OFCToBLAExt")
	// net.ConnectLayers(drives, blaPosE, pone2one, emer.Forward)

	////////////////////////////////////////////////
	// BG / DA connections

	// same prjns to stn as mtxgo
	net.ConnectToMatrix(usPos, vPmtxGo, pone2one)
	net.ConnectToMatrix(blaPosA, vPmtxGo, pone2one).SetClass("BLAToBG")
	net.ConnectToMatrix(blaPosA, vPmtxNo, pone2one).SetClass("BLAToBG")
	net.ConnectLayers(blaPosA, vPstnp, full, emer.Forward)
	net.ConnectLayers(blaPosA, vPstns, full, emer.Forward)

	net.ConnectToMatrix(blaPosE, vPmtxGo, pone2one)
	net.ConnectToMatrix(blaPosE, vPmtxNo, pone2one)
	net.ConnectToMatrix(drives, vPmtxGo, pone2one).SetClass("DrivesToMtx")
	net.ConnectToMatrix(drives, vPmtxNo, pone2one).SetClass("DrivesToMtx")
	net.ConnectLayers(drives, vPstnp, full, emer.Forward) // probably not good: modulatory
	net.ConnectLayers(drives, vPstns, full, emer.Forward)
	net.ConnectToMatrix(ofc, vPmtxGo, pone2one)
	net.ConnectToMatrix(ofc, vPmtxNo, pone2one)
	net.ConnectLayers(ofc, vPstnp, full, emer.Forward)
	net.ConnectLayers(ofc, vPstns, full, emer.Forward)
	// net.ConnectToMatrix(ofcCT, vPmtxGo, pone2one) // important for matrix to mainly use CS & BLA
	// net.ConnectToMatrix(ofcCT, vPmtxNo, pone2one)
	// net.ConnectToMatrix(ofcPT, vPmtxGo, pone2one)
	// net.ConnectToMatrix(ofcPT, vPmtxNo, pone2one)

	// net.ConnectToRWPrjn(ofc, rwPred, full)
	// net.ConnectToRWPrjn(ofcCT, rwPred, full)

	net.ConnectToVSPatch(ofcPTPred, vsPatchPosD1, pone2one)
	net.ConnectToVSPatch(ofcPTPred, vsPatchPosD2, pone2one)
	// net.ConnectToVSPatch(ofcCT, vsPatchPosD1, pone2one) // only ofcPT is properly conditional on goal engaged
	// net.ConnectToVSPatch(ofcCT, vsPatchPosD2, pone2one)
	// net.ConnectToVSPatch(time, vsPatchPosD1, full)
	// net.ConnectToVSPatch(time, vsPatchPosD2, full)
	// net.ConnectToVSPatch(ofcPT, vsPatchNegD1, pone2one)
	// net.ConnectToVSPatch(ofcPT, vsPatchNegD2, pone2one)

	// net.ConnectToVSPatch(ustimeIn, vsPatchPosD1, full)
	// net.ConnectToVSPatch(ustimeIn, vsPatchPosD2, full)
	// net.ConnectToVSPatch(ustimeIn, vsPatchNegD1, full)
	// net.ConnectToVSPatch(ustimeIn, vsPatchNegD2, full)

	////////////////////////////////////////////////
	// position

	vPgpi.PlaceRightOf(vta, space)
	ach.PlaceBehind(lhb, space)

	vsPatchPosD1.PlaceRightOf(vPstns, space)
	vsPatchNegD2.PlaceRightOf(vsPatchPosD1, space)

	drives.PlaceAbove(vta)
	usPos.PlaceBehind(drivesP, space)
	usNeg.PlaceBehind(usPosP, space)

	pvPos.PlaceRightOf(drives, space)
	pvNeg.PlaceBehind(pvPosP, space)

	time.PlaceRightOf(pvPos, space)

	cs.PlaceRightOf(time, space)
	ctxIn.PlaceRightOf(cs, space)
	// ustimeIn.PlaceRightOf(ctxIn, space)

	blaPosA.PlaceAbove(drives)
	blaNegA.PlaceBehind(blaPosE, space)
	cemPos.PlaceBehind(blaNegE, space)
	cemNeg.PlaceBehind(cemPos, space)

	gate.PlaceRightOf(blaPosA, space)

	ofc.PlaceRightOf(gate, space)
	ofcMD.PlaceBehind(ofcPTPred, space)

	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.Params.SetObject("Network")
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if !ss.Args.Bool("nogui") {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
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

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Condition, 1).AddTime(etime.Block, 50).AddTime(etime.Sequence, 8).AddTime(etime.Trial, 5).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Context, &ss.ViewUpdt) // std algo code

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

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs, etime.Sequence)
	man.GetLoop(etime.Train, etime.Block).OnStart.Add("ResetLogTrial", func() {
		ss.Logs.ResetLog(etime.Train, etime.Trial)
	})

	// Save weights to file, to look at later
	// man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
	// 	ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
	// 	axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
	// })

	////////////////////////////////////////////
	// GUI

	if ss.Args.Bool("nogui") {
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net)
		axon.LooperUpdtPlots(man, &ss.GUI)
	}

	if Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// UpdateLoopMax gets the latest loop counter Max values from env
func (ss *Sim) UpdateLoopMax() {
	ev := ss.Envs[etime.Train.String()].(*cond.CondEnv)
	trn := ss.Loops.Stacks[etime.Train]
	trn.Loops[etime.Condition].Counter.Max = ev.Condition.Max
	trn.Loops[etime.Block].Counter.Max = ev.Block.Max
	trn.Loops[etime.Sequence].Counter.Max = ev.Trial.Max
	trn.Loops[etime.Trial].Counter.Max = ev.Tick.Max
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	ev := ss.Envs[ss.Context.Mode.String()].(*cond.CondEnv)
	ss.UpdateLoopMax()
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := ev.State(ly.Nm)
		if !kit.IfaceIsNil(pats) {
			ly.ApplyExt(pats)
		}
		switch lnm {
		case "StimIn":
			ly.Pools[0].Inhib.Clamped.SetBool(ev.CurTrial.CSOn)
		}
	}
	ss.ApplyPVLV(&ss.Context, &ev.CurTrial)
	net.ApplyExts(&ss.Context) // now required for GPU mode
}

// ApplyPVLV applies current PVLV values to Context.mDrivePVLV,
// from given trial data.
func (ss *Sim) ApplyPVLV(ctx *axon.Context, trl *cond.Trial) {
	dr := &ctx.DrivePVLV
	dr.InitUS()
	ctx.NeuroMod.HasRew.SetBool(false)
	if trl.USOn {
		if trl.Valence == cond.Pos {
			dr.SetPosUS(int32(trl.US), trl.USMag)
		} else {
			dr.SetNegUS(int32(trl.US), trl.USMag)
		}
		ctx.NeuroMod.HasRew.SetBool(true)
	}
	dr.InitDrives()
	dr.SetDrive(0, 1) // todo: need to get drive somehow -- add to env?
}

// InitEnvRun intializes a new environment run, as when the RunName is changed
// or at NewRun()
func (ss *Sim) InitEnvRun() {
	ev := ss.Envs["Train"].(*cond.CondEnv)
	ev.RunName = ss.RunName
	ev.Init(0)
	ss.LoadCondWeights(ev.CurRun.Weights) // only if nonempty
}

// LoadRunWeights loads weights specified in current run, if any
func (ss *Sim) LoadRunWeights() {
	ev := ss.Envs["Train"].(*cond.CondEnv)
	ss.LoadCondWeights(ev.CurRun.Weights) // only if nonempty
}

// LoadCondWeights loads weights saved after named condition, in wts/cond.wts.gz
func (ss *Sim) LoadCondWeights(cond string) {
	if cond == "" {
		return
	}
	wfn := "wts/" + cond + ".wts.gz"
	err := ss.Net.OpenWtsJSON(gi.FileName(wfn))
	if err != nil {
		log.Println(err)
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.InitEnvRun()
	ss.Context.Reset()
	ss.Context.Mode = etime.Train
	ss.Net.InitWts()
	ss.LoadRunWeights()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Condition)
	ss.Logs.ResetLog(etime.Train, etime.Block)
	ss.UpdateLoopMax()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ev := ss.Envs[ss.Context.Mode.String()].(*cond.CondEnv)
	ss.Stats.SetString("TrialName", ev.TrialName)
	ss.Stats.SetString("TrialType", ev.TrialType)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Condition", "Block", "Trial", "Trial", "TrialType", "TrialName", "Cycle"})
}

// TrialStats computes the tick-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	ctx := &ss.Context
	dr := &ctx.DrivePVLV
	ss.Stats.SetFloat32("DA", ctx.NeuroMod.DA)
	ss.Stats.SetFloat32("ACh", ctx.NeuroMod.ACh)
	ss.Stats.SetFloat32("VSPatchPos", dr.VSPatchVals.Pos)
	ss.Stats.SetFloat32("VSPatchNeg", dr.VSPatchVals.Neg)
	ss.Stats.SetFloat32("LHbDip", dr.VTA.Vals.LHbDip)
	ss.Stats.SetFloat32("LHbBurst", dr.VTA.Vals.LHbBurst)
	ss.Stats.SetFloat32("PVpos", dr.VTA.Vals.PVpos)
	ss.Stats.SetFloat32("PVneg", dr.VTA.Vals.PVneg)
	ss.Stats.SetFloat32("PPTg", dr.VTA.Vals.PPTg)
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.Train, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.Train, etime.Trial, "TrialName")
	ss.Logs.AddStatStringItem(etime.Train, etime.Trial, "TrialType")

	// ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	layers := ss.Net.AsAxon().LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Block, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net.AsAxon())

	ss.Logs.PlotItems("DA", "VSPatchPos", "LHbDip")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net.AsAxon())
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Epoch)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")

	ss.Logs.SetMeta(etime.Train, etime.Trial, "LegendCol", "Sequence")
}

func (ss *Sim) ConfigLogItems() {
	li := ss.Logs.AddStatAggItem("DA", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.Range.Min = -1
	li.Range.Max = 1
	li.FixMin = true
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("ACh", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.FixMin = false
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("VSPatchPos", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.Range.Min = -1
	li.Range.Max = 1
	li.FixMin = true
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("VSPatchNeg", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.FixMin = false
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("LHbDip", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("LHbBurst", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("PVpos", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("PVneg", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("PPTg", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
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
	// case time == etime.Trial:
	// 	row = ss.Stats.Int("Trial")
	case time == etime.Block:
		ss.BlockStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

func (ss *Sim) BlockStats() {
	stnm := "BlockByType"
	plt := ss.GUI.Plots[etime.ScopeKey(stnm)]

	ix := ss.Logs.IdxView(etime.Train, etime.Trial)
	spl := split.GroupBy(ix, []string{"TrialType", "Trial"})
	for _, ts := range ix.Table.ColNames {
		if ts == "TrialType" || ts == "TrialName" || ts == "Trial" {
			continue
		}
		split.Agg(spl, ts, agg.AggMean)
	}
	dt := spl.AggsToTable(etable.ColNameOnly)
	for ri := 0; ri < dt.Rows; ri++ {
		tt := dt.CellString("TrialType", ri)
		trl := int(dt.CellFloat("Trial", ri))
		dt.SetCellString("TrialType", ri, fmt.Sprintf("%s_%d", tt, trl))
	}
	dt.SetMetaData("DA:On", "+")
	dt.SetMetaData("VSPatchPos:On", "+")
	dt.SetMetaData("DA:FixMin", "+")
	dt.SetMetaData("DA:Min", "-1")
	dt.SetMetaData("DA:FixMax", "+")
	dt.SetMetaData("DA:Max", "1")
	ss.Logs.MiscTables[stnm] = dt
	plt.SetTable(dt)
	plt.Update()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Axon PVLV"
	ss.GUI.MakeWindow(ss, "pvlv", title, `This is the PVLV test model in Axon. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdt = &ss.ViewUpdt

	nv.Scene().Camera.Pose.Pos.Set(0, 1.4, 2.6)
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	ss.GUI.AddPlots(title, &ss.Logs)

	stnm := "BlockByType"
	dt := ss.Logs.MiscTable(stnm)
	plt := ss.GUI.TabView.AddNewTab(eplot.KiT_Plot2D, stnm+" Plot").(*eplot.Plot2D)
	ss.GUI.Plots[etime.ScopeKey(stnm)] = plt
	plt.Params.Title = stnm
	plt.Params.XAxisCol = "TrialType"

	plt.SetTable(dt)

	cb := gi.AddNewComboBox(ss.GUI.ToolBar, "runs")
	cb.ItemsFromStringList(cond.RunNames, false, 50)
	ri := 0
	for i, rn := range cond.RunNames {
		if rn == ss.RunName {
			ri = i
			break
		}
	}
	cb.SelectItem(ri)
	cb.ComboSig.Connect(ss.GUI.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.RunName = data.(string)
		ss.InitEnvRun()
	})

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
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/pvlv/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	if GPU {
		vgpu.Debug = Debug
		ss.Net.ConfigGPUwithGUI(&TheSim.Context) // must happen after gui or no gui
		gi.SetQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
	return ss.GUI.Win
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.AddInt("nzero", 2, "number of zero error epochs in a row to count as full training")
	ss.Args.AddInt("iticycles", 0, "number of cycles to run between trials (inter-trial-interval)")
	ss.Args.SetInt("epochs", 100)
	ss.Args.SetInt("runs", 5)
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

	ss.NewRun()
	if ss.Args.Bool("gpu") {
		ss.Net.ConfigGPUnoGUI(&TheSim.Context) // must happen after gui or no gui
	}
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
