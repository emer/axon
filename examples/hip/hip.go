// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip runs a hippocampus model for testing parameters and new learning ideas
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/emer/axon/axon"
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
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/metric"
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
		gimain.Main(func() {
			sim.RunGUI()
		})
	} else {
		sim.RunNoGUI()
	}
}

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Config Config         `desc:"simulation configuration parameters -- set by .toml config file and / or args"`
	Net    *axon.Network  `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params emer.NetParams `view:"inline" desc:"all parameter management"`

	Loops        *looper.Manager `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats    `desc:"contains computed statistic values"`
	Logs         elog.Logs       `desc:"Contains all the logs and information about the logs.'"`
	PretrainMode bool            `desc:"if true, run in pretrain mode"`

	PoolVocab    patgen.Vocab     `view:"no-inline" desc:"pool patterns vocabulary"`
	TrainAB      *etable.Table    `view:"no-inline" desc:"AB training patterns to use"`
	TrainAC      *etable.Table    `view:"no-inline" desc:"AC training patterns to use"`
	TestAB       *etable.Table    `view:"no-inline" desc:"AB testing patterns to use"`
	TestAC       *etable.Table    `view:"no-inline" desc:"AC testing patterns to use"`
	PreTrainLure *etable.Table    `view:"no-inline" desc:"Lure pretrain patterns to use"`
	TestLure     *etable.Table    `view:"no-inline" desc:"Lure testing patterns to use"`
	TrainAll     *etable.Table    `view:"no-inline" desc:"all training patterns -- for pretrain"`
	TestABAC     *etable.Table    `view:"no-inline" desc:"TestAB + TestAC"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Context      axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Config.Defaults()
	econfig.Config(&ss.Config, "config.toml")

	ss.Net = &axon.Network{}
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()

	ss.PoolVocab = patgen.Vocab{}
	ss.TrainAB = &etable.Table{}
	ss.TrainAC = &etable.Table{}
	ss.TestAB = &etable.Table{}
	ss.TestAC = &etable.Table{}
	ss.PreTrainLure = &etable.Table{}
	ss.TestLure = &etable.Table{}
	ss.TrainAll = &etable.Table{}
	ss.TestABAC = &etable.Table{}
	ss.PretrainMode = false

	ss.RndSeeds.Init(100) // max 100 runs
	ss.InitRndSeed(0)
	ss.Context.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigPats()
	// ss.OpenPats()
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
	var trn, tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*env.FixedTable)
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	// note: names must be standard here!
	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	trn.Config(etable.NewIdxView(ss.TrainAB))
	trn.Validate()

	tst.Nm = etime.Test.String()
	tst.Dsc = "testing params and state"
	tst.Config(etable.NewIdxView(ss.TestABAC))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	hip := &ss.Config.Hip
	net.InitName(net, "Hip_bench")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	in := net.AddLayer4D("Input", hip.EC3NPool.Y, hip.EC3NPool.X, hip.EC3NNrn.Y, hip.EC3NNrn.X, axon.InputLayer)
	inToEc2 := prjn.NewUnifRnd()
	inToEc2.PCon = hip.InToEc2PCon
	onetoone := prjn.NewOneToOne()
	ec2, ec3, _, _, _, _ := net.AddHip(ctx, &ss.Config.Hip, 2)
	net.ConnectLayers(in, ec2, inToEc2, axon.ForwardPrjn)
	net.ConnectLayers(in, ec3, onetoone, axon.ForwardPrjn)
	ec2.PlaceAbove(in)

	err := net.Build(ctx)
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWts(ctx)
	net.InitTopoSWts()
}

func (ss *Sim) ApplyParams() {
	ss.Params.Network = ss.Net
	ss.Params.SetAll()
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
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdt.Update()
	ss.ViewUpdt.RecordSyns()
}

func (ss *Sim) TestInit() {
	ss.Loops.ResetCountersByMode(etime.Test)
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed(run int) {
	rand.Seed(ss.RndSeeds[run])
	ss.RndSeeds.Set(run)
	ss.RndSeeds.Set(run, &ss.Net.Rand)
	patgen.NewRand(ss.RndSeeds[run])
}

func ConfigLoopsHip(ctx *axon.Context, man *looper.Manager, net *axon.Network, mossyDel, mossyDelTest, thetaLow, thetaHigh float32, pretrain *bool) {

	var tmpVals []float32

	input := net.AxonLayerByName("Input")
	ec5 := net.AxonLayerByName("EC5")
	ca1 := net.AxonLayerByName("CA1")
	ca3 := net.AxonLayerByName("CA3")
	dg := net.AxonLayerByName("DG")
	dgFmEc2, _ := dg.SendNameTry("EC2")
	ca1FmEc3, _ := ca1.SendNameTry("EC3")
	ca1FmCa3, _ := ca1.SendNameTry("CA3")
	ca3FmDg, _ := ca3.SendNameTry("DG")
	ca3FmEc2, _ := ca3.SendNameTry("EC2")
	ca3FmCa3, _ := ca3.SendNameTry("CA3")

	dgPjScale := ca3FmDg.(*axon.Prjn).Params.PrjnScale.Rel
	ca1FmCa3Abs := ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Abs

	startOfQ1 := looper.NewEvent("Q0", 0, func() {
		if *pretrain {
			dgFmEc2.(*axon.Prjn).Params.Learn.Learn = 0
			ca3FmEc2.(*axon.Prjn).Params.Learn.Learn = 0
			ca3FmCa3.(*axon.Prjn).Params.Learn.Learn = 0
			ca1FmCa3.(*axon.Prjn).Params.Learn.Learn = 0
			ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Abs = 0
		} else {
			dgFmEc2.(*axon.Prjn).Params.Learn.Learn = 1
			ca3FmEc2.(*axon.Prjn).Params.Learn.Learn = 1
			ca3FmCa3.(*axon.Prjn).Params.Learn.Learn = 1
			ca1FmCa3.(*axon.Prjn).Params.Learn.Learn = 1
			ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Abs = ca1FmCa3Abs
		}
		ca1FmEc3.(*axon.Prjn).Params.PrjnScale.Rel = thetaHigh
		ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Rel = thetaLow

		ca3FmDg.(*axon.Prjn).Params.PrjnScale.Rel = dgPjScale - mossyDel // turn off DG input to CA3 in first quarter

		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})
	endOfQ1 := looper.NewEvent("Q1", 50, func() {
		ca1FmEc3.(*axon.Prjn).Params.PrjnScale.Rel = thetaLow
		ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Rel = thetaHigh
		if man.Mode == etime.Test {
			ca3FmDg.(*axon.Prjn).Params.PrjnScale.Rel = dgPjScale - mossyDelTest // testing
		}
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	}) // 50ms
	endOfQ3 := looper.NewEvent("Q3", 150, func() {
		ca3FmDg.(*axon.Prjn).Params.PrjnScale.Rel = dgPjScale // restore at the beginning of plus phase for CA3 EDL
		ca1FmEc3.(*axon.Prjn).Params.PrjnScale.Rel = thetaHigh
		ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Rel = thetaLow
		if man.Mode == etime.Train { // clamp EC5 from Input
			for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
				input.UnitVals(&tmpVals, "Act", int(di))
				ec5.ApplyExt1D32(ctx, di, tmpVals)
			}
		}
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})
	endOfQ4 := looper.NewEvent("Q4", 200, func() {
		ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Rel = thetaHigh
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})

	man.AddEventAllModes(etime.Cycle, startOfQ1, endOfQ1, endOfQ3, endOfQ4)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	trls := int(mat32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	man.AddStack(etime.Train).AddTime(etime.Run, ss.Config.Run.Runs).AddTime(etime.Epoch, ss.Config.Run.Epochs).AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTimeIncr(etime.Trial, 2*trls, ss.Config.Run.NData).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

	ConfigLoopsHip(&ss.Context, man, ss.Net, ss.Config.Hip.MossyDel, ss.Config.Hip.MossyDelTest, ss.Config.Hip.ThetaLow, ss.Config.Hip.ThetaHigh, &ss.PretrainMode)

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnEnd.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()

			// switch to AC
			trn := ss.Envs.ByMode(etime.Train).(*env.FixedTable)
			tstEpcLog := ss.Logs.Tables[etime.Scope(etime.Test, etime.Epoch)]
			epc := ss.Stats.Int("Epoch")
			abMem := float32(tstEpcLog.Table.CellFloat("ABMem", epc))
			if (trn.Table.Table.MetaData["name"] == "TrainAB") && (abMem >= ss.Config.Run.StopMem || epc == ss.Config.Run.Epochs/2) {
				ss.Stats.SetInt("FirstPerfect", epc)
				trn.Config(etable.NewIdxView(ss.TrainAC))
				trn.Validate()
			}
		}
	})

	// early stop
	man.GetLoop(etime.Train, etime.Epoch).IsDone["ACMemStop"] = func() bool {
		// This is calculated in TrialStats
		tstEpcLog := ss.Logs.Tables[etime.Scope(etime.Test, etime.Epoch)]
		acMem := float32(tstEpcLog.Table.CellFloat("ACMem", ss.Stats.Int("Epoch")))
		stop := acMem >= ss.Config.Run.StopMem
		return stop
	}

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		axon.LogTestErrors(&ss.Logs)
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWts, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI

	if !ss.Config.GUI {
		man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
			ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		})
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net)
		axon.LooperUpdtPlots(man, &ss.GUI)
		for _, m := range man.Stacks {
			m.Loops[etime.Cycle].OnEnd.InsertBefore("GUI:UpdateNetView", "GUI:CounterUpdt", func() {
				ss.NetViewCounters(etime.Cycle)
			})
			m.Loops[etime.Trial].OnEnd.InsertBefore("GUI:UpdateNetView", "GUI:CounterUpdt", func() {
				ss.NetViewCounters(etime.Trial)
			})
		}
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
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	net.InitExt(ctx)
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		ev.Step()
		// note: must save env state for logging / stats due to data parallel re-use of same env
		ss.Stats.SetStringDi("TrialName", int(di), ev.TrialName.Cur)
		for _, lnm := range lays {
			ly := ss.Net.AxonLayerByName(lnm)
			pats := ev.State(ly.Nm)
			if pats != nil {
				ly.ApplyExt(ctx, di, pats)
			}
		}
	}
	net.ApplyExts(ctx) // now required for GPU mode
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRndSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	ss.ConfigPats()
	ss.ConfigEnv()
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWts(ctx)
	ss.InitStats()
	ss.StatCounters(0)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

/////////////////////////////////////////////////////////////////////////
//   Pats

func (ss *Sim) ConfigPats() {
	hp := &ss.Config.Hip
	ecY := hp.EC3NPool.Y
	ecX := hp.EC3NPool.X
	plY := hp.EC3NNrn.Y // good idea to get shorter vars when used frequently
	plX := hp.EC3NNrn.X // makes much more readable
	npats := ss.Config.Run.NTrials
	pctAct := hp.ECPctAct
	minDiff := ss.Config.Pat.MinDiffPct
	nOn := patgen.NFmPct(pctAct, plY*plX)
	ctxtflip := patgen.NFmPct(ss.Config.Pat.CtxtFlipPct, nOn)
	patgen.AddVocabEmpty(ss.PoolVocab, "empty", npats, plY, plX)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "C", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt", 3, plY, plX, pctAct, minDiff) // totally diff

	for i := 0; i < (ecY-1)*ecX*3; i++ { // 12 contexts! 1: 1 row of stimuli pats; 3: 3 diff ctxt bases
		list := i / ((ecY - 1) * ecX)
		ctxtNm := fmt.Sprintf("ctxt%d", i+1)
		tsr, _ := patgen.AddVocabRepeat(ss.PoolVocab, ctxtNm, npats, "ctxt", list)
		patgen.FlipBitsRows(tsr, ctxtflip, ctxtflip, 1, 0)
		//todo: also support drifting
		//solution 2: drift based on last trial (will require sequential learning)
		//patgen.VocabDrift(ss.PoolVocab, ss.NFlipBits, "ctxt"+strconv.Itoa(i+1))
	}

	patgen.InitPats(ss.TrainAB, "TrainAB", "TrainAB Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAB, ss.PoolVocab, "Input", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(ss.TrainAB, ss.PoolVocab, "EC5", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(ss.TestAB, "TestAB", "TestAB Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(ss.TestAB, ss.PoolVocab, "EC5", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(ss.TrainAC, "TrainAC", "TrainAC Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAC, ss.PoolVocab, "Input", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(ss.TrainAC, ss.PoolVocab, "EC5", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(ss.TestAC, "TestAC", "TestAC Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(ss.TestAC, ss.PoolVocab, "EC5", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(ss.PreTrainLure, "PreTrainLure", "PreTrainLure Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.PreTrainLure, ss.PoolVocab, "Input", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(ss.PreTrainLure, ss.PoolVocab, "EC5", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"})   // arbitrary ctxt here

	patgen.InitPats(ss.TestLure, "TestLure", "TestLure Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA", "empty", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "EC5", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"})      // arbitrary ctxt here

	ss.TrainAll = ss.TrainAB.Clone()
	ss.TrainAll.AppendRows(ss.TrainAC)
	ss.TrainAll.AppendRows(ss.PreTrainLure)
	ss.TrainAll.MetaData["name"] = "TrainAll"
	ss.TrainAll.MetaData["desc"] = "All Training Patterns"

	ss.TestABAC = ss.TestAB.Clone()
	ss.TestABAC.AppendRows(ss.TestAC)
	ss.TestABAC.MetaData["name"] = "TestABAC"
	ss.TestABAC.MetaData["desc"] = "All Testing Patterns"
}

func (ss *Sim) OpenPats() {
	dt := ss.TrainAB
	dt.SetMetaData("name", "TrainAB")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.tsv", etable.Tab)
	if err != nil {
		log.Println(err)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("UnitErr", 0.0)
	ss.Stats.SetFloat("CorSim", 0.0)
	ss.Stats.SetFloat("TrgOnWasOffAll", 0.0)
	ss.Stats.SetFloat("TrgOnWasOffCmp", 0.0)
	ss.Stats.SetFloat("TrgOffWasOn", 0.0)
	ss.Stats.SetFloat("ABMem", 0.0)
	ss.Stats.SetFloat("ACMem", 0.0)
	ss.Stats.SetFloat("Mem", 0.0)
	ss.Stats.SetInt("FirstPerfect", -1) // first epoch at when AB Mem is perfect
	ss.Stats.SetInt("RecallItem", -1)   // item recalled in EC5 completion pool
	ss.Stats.SetFloat("ABRecMem", 0.0)  // similar to ABMem but using correlation on completion pool
	ss.Stats.SetFloat("ACRecMem", 0.0)

	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters(di int) {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl+di)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	ss.Stats.SetString("TrialName", ss.Stats.StringDi("TrialName", di))
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
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di", "TrialName", "Cycle", "UnitErr", "TrlErr", "CorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
	out := ss.Net.AxonLayerByName("EC5")

	ss.Stats.SetFloat("CorSim", float64(out.Vals[di].CorSim.Cor))
	ss.Stats.SetFloat("UnitErr", out.PctUnitErr(&ss.Context)[di])
	ss.MemStats(ss.Loops.Mode, di)

	if ss.Stats.Float("UnitErr") > ss.Config.Hip.MemThr {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

// MemStats computes ActM vs. Target on ECout with binary counts
// must be called at end of 3rd quarter so that Target values are
// for the entire full pattern as opposed to the plus-phase target
// values clamped from ECin activations
func (ss *Sim) MemStats(mode etime.Modes, di int) {
	memthr := ss.Config.Hip.MemThr
	ecout := ss.Net.AxonLayerByName("EC5")
	inp := ss.Net.AxonLayerByName("Input") // note: must be input b/c ECin can be active
	nn := ecout.Shape().Len()
	actThr := float32(0.2)
	trgOnWasOffAll := 0.0 // all units
	trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
	trgOffWasOn := 0.0    // should have been off
	cmpN := 0.0           // completion target
	trgOnN := 0.0
	trgOffN := 0.0
	actMi, _ := ecout.UnitVarIdx("ActM")
	targi, _ := ecout.UnitVarIdx("Target")

	ss.Stats.SetFloat("ABMem", math.NaN())
	ss.Stats.SetFloat("ACMem", math.NaN())
	ss.Stats.SetFloat("ABRecMem", math.NaN())
	ss.Stats.SetFloat("ACRecMem", math.NaN())

	trialnm := ss.Stats.StringDi("TrialName", di)
	isAB := strings.Contains(trialnm, "AB")

	for ni := 0; ni < nn; ni++ {
		actm := ecout.UnitVal1D(actMi, ni, di)
		trg := ecout.UnitVal1D(targi, ni, di) // full pattern target
		inact := inp.UnitVal1D(actMi, ni, di)
		if trg < actThr { // trgOff
			trgOffN += 1
			if actm > actThr {
				trgOffWasOn += 1
			}
		} else { // trgOn
			trgOnN += 1
			if inact < actThr { // missing in ECin -- completion target
				cmpN += 1
				if actm < actThr {
					trgOnWasOffAll += 1
					trgOnWasOffCmp += 1
				}
			} else {
				if actm < actThr {
					trgOnWasOffAll += 1
				}
			}
		}
	}
	trgOnWasOffAll /= trgOnN
	trgOffWasOn /= trgOffN
	if mode == etime.Train { // no compare
		if trgOnWasOffAll < memthr && trgOffWasOn < memthr {
			ss.Stats.SetFloat("Mem", 1)
		} else {
			ss.Stats.SetFloat("Mem", 0)
		}
	} else { // test
		if cmpN > 0 { // should be
			trgOnWasOffCmp /= cmpN
			if trgOnWasOffCmp < memthr && trgOffWasOn < memthr {
				ss.Stats.SetFloat("Mem", 1)
				if isAB {
					ss.Stats.SetFloat("ABMem", 1)
				} else {
					ss.Stats.SetFloat("ACMem", 1)
				}
			} else {
				ss.Stats.SetFloat("Mem", 0)
				if isAB {
					ss.Stats.SetFloat("ABMem", 0)
				} else {
					ss.Stats.SetFloat("ACMem", 0)
				}
			}
		}
	}
	ss.Stats.SetFloat("TrgOnWasOffAll", trgOnWasOffAll)
	ss.Stats.SetFloat("TrgOnWasOffCmp", trgOnWasOffCmp)
	ss.Stats.SetFloat("TrgOffWasOn", trgOffWasOn)

	// take completion pool to do CosDiff
	var recallPat etensor.Float32
	ecout.UnitValsTensor(&recallPat, "ActM", di)
	mostSimilar := -1
	highestCosDiff := float32(0)
	var cosDiff float32
	var patToComplete *etensor.Float32
	var correctIdx int
	if isAB {
		patToComplete, _ = ss.PoolVocab.ByNameTry("B")
		correctIdx, _ = strconv.Atoi(strings.Split(trialnm, "AB")[1])
	} else {
		patToComplete, _ = ss.PoolVocab.ByNameTry("C")
		correctIdx, _ = strconv.Atoi(strings.Split(trialnm, "AC")[1])
	}
	for i := 0; i < patToComplete.Shp[0]; i++ { // for each item in the list
		cosDiff = metric.Correlation32(recallPat.SubSpace([]int{0, 1}).(*etensor.Float32).Values, patToComplete.SubSpace([]int{i}).(*etensor.Float32).Values)
		if cosDiff > highestCosDiff {
			highestCosDiff = cosDiff
			mostSimilar = i
		}
	}

	ss.Stats.SetInt("RecallItem", mostSimilar)
	if isAB {
		ss.Stats.SetFloat("ABRecMem", bools.ToFloat64(mostSimilar == correctIdx))
	} else {
		ss.Stats.SetFloat("ACRecMem", bools.ToFloat64(mostSimilar == correctIdx))
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) AddLogItems() {
	itemNames := []string{"CorSim", "UnitErr", "PctCor", "PctErr", "TrgOnWasOffAll", "TrgOnWasOffCmp", "TrgOffWasOn", "Mem", "ABMem", "ACMem", "ABRecMem", "ACRecMem"}
	for _, st := range itemNames {
		stnm := st
		tonm := "Tst" + st
		ss.Logs.AddItem(&elog.Item{
			Name: tonm,
			Type: etensor.FLOAT64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetFloat64(ctx.ItemFloat(etime.Test, etime.Epoch, stnm))
				},
				etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
					ctx.SetFloat64(ctx.ItemFloat(etime.Test, etime.Epoch, stnm)) // take the last epoch
					// ctx.SetAgg(ctx.Mode, etime.Epoch, agg.AggMax) // agg.AggMax for max over epochs
				}}})
	}
}

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddStatAggItem("CorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("TrgOnWasOffAll", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("TrgOnWasOffCmp", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("TrgOffWasOn", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ABMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ACMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Mem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ABRecMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ACRecMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatIntNoAggItem(etime.Train, etime.Run, "FirstPerfect")
	ss.Logs.AddStatIntNoAggItem(etime.Train, etime.Trial, "RecallItem")
	ss.Logs.AddStatIntNoAggItem(etime.Test, etime.Trial, "RecallItem")
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	// ss.Logs.AddCopyFromFloatItems(etime.Train, etime.Epoch, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr", "TrgOnWasOffAll", "TrgOnWasOffCmp", "TrgOffWasOn", "Mem")
	ss.AddLogItems()

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	// axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net, etime.Test, etime.Cycle)
	ss.Logs.AddLayerTensorItems(ss.Net, "ActM", etime.Test, etime.Trial, "TargetLayer")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "TargetLayer")

	ss.Logs.PlotItems("TrgOnWasOffAll", "TrgOnWasOffCmp", "ABMem", "ACMem", "ABRecMem", "ACRecMem", "TstTrgOnWasOffAll", "TstTrgOnWasOffCmp", "TstMem", "TstABMem", "TstACMem", "TstABRecMem", "TstACRecMem")

	ss.Logs.CreateTables()
	ss.Logs.SetMeta(etime.Train, etime.Run, "TrgOnWasOffAll:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TrgOnWasOffCmp:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ABMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ACMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ABRecMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ACRecMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstTrgOnWasOffAll:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstTrgOnWasOffCmp:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstACMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstACRecMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "FirstPerfect:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Run, "Type", "Bar")
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		for di := 0; di < int(ctx.NetIdxs.NData); di++ {
			ss.TrialStats(di)
			ss.StatCounters(di)
			ss.Logs.LogRowDi(mode, time, row, di)
		}
		return // don't do reg below
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Axon Hippocampus"
	ss.GUI.MakeWindow(ss, "hip", title, `Benchmarking`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.Phase, etime.Phase)
	ss.GUI.ViewUpdt = &ss.ViewUpdt

	nv.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Init", Icon: "update",
		Tooltip: "Call ResetCountersByMode with test mode and update GUI.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.TestInit()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train, etime.Test})

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
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/ra25/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		// vgpu.Debug = ss.Config.Debug // when debugging GPU..
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

	econfig.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Train, etime.Trial, "trl", netName, runName)
	econfig.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	econfig.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)
	econfig.SetLogFile(&ss.Logs, ss.Config.Log.TestEpoch, etime.Test, etime.Epoch, "tst_epc", netName, runName)
	econfig.SetLogFile(&ss.Logs, ss.Config.Log.TestTrial, etime.Test, etime.Trial, "tst_trl", netName, runName)

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	// for standalone no gui run
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context) // must happen after gui or no gui
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	ss.Init()

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.Runs, ss.Config.Run.Run)
	ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)

	ss.Loops.Run(etime.Train)

	// for factor run
	// ss.TwoFactorRun()

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}

var ConfigFiles = []string{"smallhip", "medhip"}

var ListSizes = []int{20}

// TwoFactorRun runs outer-loop crossed with inner-loop params
func (ss *Sim) TwoFactorRun() {
	for _, config := range ConfigFiles {
		for _, listSize := range ListSizes {

			ss.Net.GPU.Destroy()
			ss.Net = &axon.Network{}
			ss.Params.Network = ss.Net

			// setting name for this factor combo
			ss.Params.Tag = fmt.Sprintf("%s_%d", config, listSize)
			ss.Stats.SetString("RunName", ss.Params.RunName(ss.Config.Run.Run))

			ss.Config.Run.NTrials = listSize
			econfig.OpenWithIncludes(&ss.Config, config+".toml")

			// reconfig for this factor combo
			ss.InitRndSeed(0)
			ss.ConfigPats()
			ss.ConfigEnv()
			ss.ConfigNet(ss.Net)
			ss.ConfigLoops()

			if ss.Config.Run.GPU {
				ss.Net.ConfigGPUnoGUI(&ss.Context) // must happen after gui or no gui
			}
			mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

			ss.Init()

			mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.Runs, ss.Config.Run.Run)
			ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)

			// print our info for checking purposes
			fmt.Println("CA3 shape: ", ss.Net.AxonLayerByName("CA3").Shp.Shp)
			fmt.Println("EC2 shape: ", ss.Net.AxonLayerByName("EC2").Shp.Shp)
			fmt.Println("# of pairs: ", ss.TrainAB.Rows)

			ss.Loops.Run(etime.Train)
		}
	}
}
