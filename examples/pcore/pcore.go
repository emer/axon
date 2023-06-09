// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
pcore: This project simulates the inhibitory dynamics in the STN and GPe leading to integration of Go vs. NoGo signal in the basal ganglia.
*/
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
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/ki/bools"
	"github.com/goki/mat32"
	"github.com/goki/vgpu/vgpu"
)

var (
	// Debug triggers various messages etc
	Debug = false
	// GPU runs GUI with the GPU -- faster with NData = 16
	GPU = true
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.Config()
	if len(os.Args) > 1 {
		sim.RunNoGUI() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			sim.RunGUI()
		})
	}
}

// see params.go for network params

// SimParams has all the custom params for this sim
type SimParams struct {
	ZeroTest bool `desc:"test with no ACC activity at all -- params need to prevent gating in this situation too"`
	NPools   int  `view:"-" desc:"number of pools"`
	NData    int  `desc:"number of data-parallel items to process at once"`
	NTrials  int  `desc:"number of trials per epoch"`
}

// Defaults sets default params
func (ss *SimParams) Defaults() {
	ss.ZeroTest = false
	ss.NPools = 1
	ss.NData = 16
	ss.NTrials = 128
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net      *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Sim      SimParams        `view:"no-inline" desc:"sim params"`
	Params   emer.Params      `view:"inline" desc:"all parameter management"`
	Loops    *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats    estats.Stats     `desc:"contains computed statistic values"`
	Logs     elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Envs     env.Envs         `view:"no-inline" desc:"Environments"`
	Context  axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Sim.Defaults()
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	// ss.Params.ExtraSets = "WtScales" // todo: ensure params same without
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.Context.Defaults()
	ss.Context.PVLV.Drive.NActive = 2
	ss.Context.PVLV.Drive.NNegUSs = 1
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
	newEnv := (len(ss.Envs) == 0)

	for di := 0; di < ss.Sim.NData; di++ {
		var trn, tst *GoNoEnv
		if newEnv {
			trn = &GoNoEnv{}
			tst = &GoNoEnv{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*GoNoEnv)
			tst = ss.Envs.ByModeDi(etime.Test, di).(*GoNoEnv)
		}

		// note: names must be standard here!
		trn.Nm = env.ModeDi(etime.Train, di)
		trn.Defaults()
		trn.Config(etime.Train, ss.Sim.NPools, 73+int64(di)*73)
		trn.Validate()

		tst.Nm = env.ModeDi(etime.Test, di)
		tst.Defaults()
		tst.Config(etime.Test, ss.Sim.NPools, 181+int64(di)*181)
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

func (ss *Sim) ConfigPVLV(trn *GoNoEnv) {
	pv := &ss.Context.PVLV
	pv.Drive.NActive = 2
	pv.Drive.NNegUSs = 1
	pv.Effort.Gain = 0.05 // not using but anyway
	pv.Effort.Max = 20
	pv.Effort.MaxNovel = 8
	pv.Effort.MaxPostDip = 4
	pv.Urgency.U50 = 20 // 20 def
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*GoNoEnv)

	net.InitName(net, "PCore")
	net.SetMaxData(ctx, ss.Sim.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	np := ev.NPools
	nuY := ev.NUnitsY
	nuX := ev.NUnitsX
	space := float32(2)

	pone2one := prjn.NewPoolOneToOne()
	one2one := prjn.NewOneToOne()
	full := prjn.NewFull()
	_ = full
	mtxRndPrjn := prjn.NewPoolUnifRnd()
	mtxRndPrjn.PCon = 0.75
	_ = mtxRndPrjn

	mtxGo, mtxNo, gpeTA, stnp, stns, gpi := net.AddBG("", 1, np, nuY, nuX, nuY, nuX, space)
	_ = gpeTA
	gpeOut := net.AxonLayerByName("GPeOut")

	snc := net.AddLayer2D("SNc", 1, 1, axon.InputLayer)
	_ = snc

	urge := net.AddUrgencyLayer(5, 4)
	_ = urge

	accpos := net.AddLayer4D("ACCPos", 1, np, nuY, nuX, axon.InputLayer)
	accneg := net.AddLayer4D("ACCNeg", 1, np, nuY, nuX, axon.InputLayer)

	inly, inP := net.AddInputPulv4D("In", 1, np, nuY, nuX, space)

	pfc, pfcCT := net.AddSuperCT4D("PFC", "PFCPrjn", 1, np, nuY, nuX, space, one2one)
	// prjns are: super->PT, PT self, CT-> thal
	pfcPT, pfcVM := net.AddPTMaintThalForSuper(pfc, pfcCT, "VM", "PFCPrjn", one2one, pone2one, space)
	_ = pfcPT
	pfcCT.SetClass("PFC CTCopy")
	pfcCT.CTDefParamsMedium() // FSA

	net.ConnectLayers(inly, pfc, pone2one, axon.ForwardPrjn)
	net.ConnectToPulv(pfc, pfcCT, inP, pone2one, pone2one, "PFCPrjn")

	net.ConnectLayers(pfc, stnp, pone2one, axon.ForwardPrjn)
	net.ConnectLayers(pfc, stns, pone2one, axon.ForwardPrjn)

	net.ConnectLayers(gpi, pfcVM, pone2one, axon.InhibPrjn).SetClass("BgFixed")

	mtxGo.SetBuildConfig("ThalLay1Name", pfcVM.Name())
	mtxNo.SetBuildConfig("ThalLay1Name", pfcVM.Name())

	net.ConnectToMatrix(accpos, mtxGo, pone2one).SetClass("ACCPosToGo")
	net.ConnectToMatrix(accneg, mtxNo, pone2one).SetClass("ACCNegToNo")
	// cross connections:
	net.ConnectToMatrix(accpos, mtxNo, pone2one).SetClass("ACCPosToNo")
	net.ConnectToMatrix(accneg, mtxGo, pone2one).SetClass("ACCNegToGo")

	// pfc just has irrelevant info:
	// net.ConnectToMatrix(pfc, mtxGo, pone2one)
	// net.ConnectToMatrix(pfc, mtxNo, pone2one)

	net.ConnectToMatrix(urge, mtxGo, full)

	snc.PlaceRightOf(gpi, space)
	urge.PlaceRightOf(snc, space)
	gpeOut.PlaceAbove(gpi)
	stnp.PlaceRightOf(gpeTA, space)
	mtxGo.PlaceAbove(gpeOut)
	accpos.PlaceAbove(mtxGo)
	accneg.PlaceRightOf(accpos, space)
	inly.PlaceRightOf(accneg, space)
	pfc.PlaceRightOf(inly, space)
	pfcCT.PlaceRightOf(pfc, space)
	pfcPT.PlaceBehind(pfc, space)
	pfcVM.PlaceRightOf(pfcPT, space)

	err := net.Build(ctx)
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.Params.SetObject("Network")
	net.InitWts(ctx)
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
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdt.Update()
	ss.ViewUpdt.RecordSyns()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.RndSeeds.Set(run)
	ss.RndSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	ev := ss.Envs.ByModeDi(etime.Train, 0).(*GoNoEnv)
	trls := int(mat32.IntMultipleGE(float32(ss.Sim.NTrials), float32(ss.Sim.NData)))

	man.AddStack(etime.Train).AddTime(etime.Run, 5).AddTime(etime.Epoch, 30).AddTimeIncr(etime.Sequence, trls, ss.Sim.NData).AddTime(etime.Trial, 3).AddTime(etime.Cycle, 200)

	nTestInc := int(1.0/ev.TestInc) + 1
	totTstTrls := ev.TestReps * nTestInc * nTestInc

	testTrls := int(mat32.IntMultipleGE(float32(totTstTrls), float32(ss.Sim.NData)))

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTimeIncr(etime.Sequence, testTrls, ss.Sim.NData).AddTime(etime.Trial, 3).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

	man.GetLoop(etime.Test, etime.Sequence).OnStart.Add("TestInc", func() {
	})

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			seq := man.Stacks[mode].Loops[etime.Sequence].Counter.Cur
			trial := man.Stacks[mode].Loops[etime.Trial].Counter.Cur
			ss.ApplyInputs(mode, seq, trial)
		})
		stack.Loops[etime.Trial].OnEnd.Add("GatedAction", func() {
			trial := man.Stacks[mode].Loops[etime.Trial].Counter.Cur
			if trial == 1 {
				ss.GatedAction()
			}
		})
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs, etime.Sequence)

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfArgSet(ss.Net, &ss.Args, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI

	if ss.Args.Bool("nogui") {
		man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
			ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		})
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net)
		axon.LooperUpdtPlots(man, &ss.GUI)
		for _, m := range man.Stacks {
			m.Loops[etime.Cycle].OnEnd.Prepend("GUI:CounterUpdt", func() {
				ss.NetViewCounters()
			})
			m.Loops[etime.Trial].OnEnd.InsertAfter("GatedAction", "GUI:CounterUpdt", func() {
				ss.NetViewCounters()
			})
		}
	}

	if Debug {
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

	lays := []string{"ACCPos", "ACCNeg", "In"} // , "SNc"}
	if ss.Sim.ZeroTest {
		lays = []string{"In"}
	}

	for di := 0; di < ss.Sim.NData; di++ {
		idx := seq + di // sequence increments by NData automatically
		ev := ss.Envs.ByModeDi(mode, di).(*GoNoEnv)
		ev.Trial.Set(idx)
		if trial == 0 {
			ev.Step()
		}
		for _, lnm := range lays {
			ly := net.AxonLayerByName(lnm)
			itsr := ev.State(lnm)
			ly.ApplyExt(ctx, uint32(di), itsr)
		}
		ss.ApplyPVLV(ev, trial, uint32(di))
	}
	net.ApplyExts(ctx) // now required for GPU mode
}

// ApplyPVLV applies PVLV reward inputs
func (ss *Sim) ApplyPVLV(ev *GoNoEnv, trial int, di uint32) {
	ctx := &ss.Context
	ctx.PVLV.EffortUrgencyUpdt(ctx, di, &ss.Net.Rand, 1)
	if ctx.Mode == etime.Test {
		axon.UrgencyReset(ctx, di)
	}

	switch trial {
	case 0:
		axon.NeuroModSetRew(ctx, di, 0, false) // no rew
		axon.SetGlbV(ctx, di, axon.GvACh, 0)
	case 1:
		axon.NeuroModSetRew(ctx, di, 0, false) // no rew
		axon.SetGlbV(ctx, di, axon.GvACh, 1)
	case 2:
		axon.SetGlbV(ctx, di, axon.GvACh, 1)
		ss.GatedRew(ev, di)
	}
}

// GatedRew applies reward input based on gating action and input
func (ss *Sim) GatedRew(ev *GoNoEnv, di uint32) {
	rew := ev.Rew
	ss.SetRew(rew, di)
}

func (ss *Sim) SetRew(rew float32, di uint32) {
	ctx := &ss.Context
	ctx.PVLVInitUS(di)
	axon.NeuroModSetRew(ctx, di, rew, true)
	axon.SetGlbV(ctx, di, axon.GvDA, rew) // no reward prediction error
	if rew > 0 {
		ctx.PVLVSetUS(di, axon.Positive, 0, 1)
	} else if rew < 0 {
		ctx.PVLVSetUS(di, axon.Negative, 0, 1)
	}
}

// GatedAction records gating action and generates reward
// this happens at the end of Trial == 1 (2nd trial)
// so that the reward is present during the final trial when learning occurs.
func (ss *Sim) GatedAction() {
	ctx := &ss.Context
	mtxly := ss.Net.AxonLayerByName("MtxGo")
	vmly := ss.Net.AxonLayerByName("PFCVM")
	nan := mat32.NaN()
	for di := 0; di < ss.Sim.NData; di++ {
		ev := ss.Envs.ByModeDi(ctx.Mode, di).(*GoNoEnv)
		didGate := mtxly.AnyGated(uint32(di))
		action := "Gated"
		if !didGate {
			action = "NoGate"
		}
		ev.Action(action, nil)
		rt := vmly.LayerVals(uint32(di)).RT
		if rt > 0 {
			ss.Stats.SetFloat32Di("PFCVM_RT", di, rt/200)
		} else {
			ss.Stats.SetFloat32Di("PFCVM_RT", di, nan)
		}
		ss.Stats.SetFloat32Di("PFCVM_ActAvg", di, vmly.Pool(0, uint32(di)).AvgMax.SpkMax.Cycle.Avg)
		ss.Stats.SetFloat32Di("MtxGo_ActAvg", di, mtxly.Pool(0, uint32(di)).AvgMax.SpkMax.Cycle.Avg)
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRndSeed()
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
	ss.Stats.SetFloat("Gated", 0)
	ss.Stats.SetFloat("Should", 0)
	ss.Stats.SetFloat("Match", 0)
	ss.Stats.SetFloat("Rew", 0)
	ss.Stats.SetFloat("PFCVM_RT", 0.0)
	ss.Stats.SetFloat("PFCVM_ActAvg", 0.0)
	ss.Stats.SetFloat("MtxGo_ActAvg", 0.0)
	ss.Stats.SetFloat("ACCPos", 0.0)
	ss.Stats.SetFloat("ACCNeg", 0.0)
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters(di int) {
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	trlnm := fmt.Sprintf("%4f_%4f", ss.Stats.Float32("ACCPos"), ss.Stats.Float32("ACCNeg"))
	ss.Stats.SetString("TrialName", trlnm)
}

func (ss *Sim) NetViewCounters() {
	if ss.GUI.ViewUpdt.View == nil {
		return
	}
	di := ss.GUI.ViewUpdt.View.Di
	ss.TrialStats(di)
	ss.StatCounters(di)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Sequence", "Trial", "Di", "TrialName", "Cycle", "Gated", "Should", "Match", "Rew"})
}

// TrialStats records the trial-level statistics
func (ss *Sim) TrialStats(di int) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*GoNoEnv)
	ss.Stats.SetFloat32("ACCPos", ev.ACCPos)
	ss.Stats.SetFloat32("ACCNeg", ev.ACCNeg)
	ss.Stats.SetFloat32("Gated", bools.ToFloat32(ev.Gated))
	ss.Stats.SetFloat32("Should", bools.ToFloat32(ev.Should))
	ss.Stats.SetFloat32("Match", bools.ToFloat32(ev.Match))
	ss.Stats.SetFloat32("Rew", ev.Rew)
	ss.Stats.SetFloat32("PFCVM_RT", ss.Stats.Float32Di("PFCVM_RT", di))
	ss.Stats.SetFloat32("PFCVM_ActAvg", ss.Stats.Float32Di("PFCVM_ActAvg", di))
	ss.Stats.SetFloat32("MtxGo_ActAvg", ss.Stats.Float32Di("MtxGo_ActAvg", di))
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Sequence, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Sequence, "TrialName")
	ss.Logs.AddStatStringItem(etime.Test, etime.Sequence, "TrialName")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCPos")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCNeg")

	ss.Logs.AddStatAggItem("Gated", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("Should", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("Match", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("PFCVM_RT", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("PFCVM_ActAvg", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("MtxGo_ActAvg", etime.Run, etime.Epoch, etime.Sequence)
	li := ss.Logs.AddStatAggItem("Rew", etime.Run, etime.Epoch, etime.Sequence)
	li.FixMin = false
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Sequence)

	// axon.LogAddDiagnosticItems(&ss.Logs, ss.Net, etime.Epoch, etime.Trial)
	// axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net, etime.Test, etime.Cycle)

	ss.Logs.PlotItems("MtxGo_ActAvg", "PFCVM_ActAvg", "PFCVM_RT", "Gated", "Should", "Match", "Rew")

	ss.Logs.CreateTables()

	tsttrl := ss.Logs.Table(etime.Test, etime.Trial)
	if tsttrl != nil {
		tstst := tsttrl.Clone()
		ss.Logs.MiscTables["TestTrialStats"] = tstst
	}

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	// ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Run)
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
		for di := 0; di < ss.Sim.NData; di++ {
			ss.TrialStats(di)
			ss.StatCounters(di)
			ss.Logs.LogRowDi(mode, time, row, di)
		}
		return // don't do reg
	case time == etime.Epoch && mode == etime.Test:
		ss.TestStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

func (ss *Sim) TestStats() {
	tststnm := "TestTrialStats"
	plt := ss.GUI.Plots[etime.ScopeKey(tststnm)]
	ix := ss.Logs.IdxView(etime.Test, etime.Sequence)
	spl := split.GroupBy(ix, []string{"TrialName"})
	for _, ts := range ix.Table.ColNames {
		if ts == "TrialName" {
			continue
		}
		split.Agg(spl, ts, agg.AggMean)
	}
	tstst := spl.AggsToTable(etable.ColNameOnly)
	ss.Logs.MiscTables[tststnm] = tstst
	plt.SetTable(tstst)
	plt.Params.XAxisCol = "Sequence"
	plt.SetColParams("Gated", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Should", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Match", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "PCore Test"
	ss.GUI.MakeWindow(ss, "pcore", title, `This project simulates the inhibitory dynamics in the STN and GPe leading to integration of Go vs. NoGo signal in the basal ganglia. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.Phase, etime.Phase)

	// nv.Scene().Camera.Pose.Pos.Set(-0.028028872, 2.1134117, 2.3178313)
	// nv.Scene().Camera.LookAt(mat32.Vec3{0.00030842167, 0.045156803, -0.039506555}, mat32.Vec3{0, 1, 0})

	ss.GUI.ViewUpdt = &ss.ViewUpdt

	ss.GUI.AddPlots(title, &ss.Logs)

	tststnm := "TestTrialStats"
	tstst := ss.Logs.MiscTable(tststnm)
	plt := ss.GUI.TabView.AddNewTab(eplot.KiT_Plot2D, tststnm+" Plot").(*eplot.Plot2D)
	ss.GUI.Plots[etime.ScopeKey(tststnm)] = plt
	plt.Params.Title = tststnm
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(tstst)

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
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/pcore/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	if GPU {
		vgpu.Debug = Debug
		ss.Net.ConfigGPUwithGUI(&ss.Context) // must happen after gui or no gui
		gi.SetQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
	return ss.GUI.Win
}

func (ss *Sim) RunGUI() {
	ss.Init()
	win := ss.ConfigGui()
	win.StartEventLoop()
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.SetInt("epochs", 200)
	ss.Args.SetInt("runs", 10)
	ss.Args.AddInt("ndata", 16, "number of data items to run in parallel")
	ss.Args.AddInt("threads", 0, "number of parallel threads, for cpu computation (0 = use default)")
	ss.Args.Parse() // always parse
	if len(os.Args) > 1 {
		ss.Args.SetBool("nogui", true) // by definition if here
		ss.Sim.NData = ss.Args.Int("ndata")
		mpi.Printf("Set NData to: %d\n", ss.Sim.NData)
	}
}

func (ss *Sim) RunNoGUI() {
	ss.Args.ProcStd(&ss.Params)
	ss.Args.ProcStdLogs(&ss.Logs, &ss.Params, ss.Net.Name())
	ss.Args.SetBool("nogui", true)                                       // by definition if here
	ss.Stats.SetString("RunName", ss.Params.RunName(ss.Args.Int("run"))) // used for naming logs, stats, etc

	netdata := ss.Args.Bool("netdata")
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	ss.Init()

	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	mpi.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := &ss.Loops.GetLoop(etime.Train, etime.Run).Counter
	rc.Set(run)
	rc.Max = run + runs
	ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = ss.Args.Int("epochs")
	if ss.Args.Bool("gpu") {
		ss.Net.ConfigGPUnoGUI(&ss.Context) // must happen after gui or no gui
	}
	ss.Net.SetNThreads(ss.Args.Int("threads"))
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	ss.NewRun()
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
