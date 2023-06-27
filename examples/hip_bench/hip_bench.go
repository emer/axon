// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip_bench runs a hippocampus model for testing parameters and new learning ideas
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
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etable"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
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

// see params.go for params

// SimParams has all the custom params for this sim
type SimParams struct {
	NData        int `desc:"number of data-parallel items to process at once"`
	NTrials      int `desc:"number of trials per epoch"`
	TestInterval int `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	// PCAInterval  int `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
}

// Defaults sets default params
func (ss *SimParams) Defaults() {
	ss.NData = 1
	ss.NTrials = 10
	ss.TestInterval = 1
	// ss.PCAInterval = 5
}

// PatParams have the pattern parameters
type PatParams struct {
	ListSize    int     `desc:"number of A-B, A-C patterns each"`
	MinDiffPct  float32 `desc:"minimum difference between item random patterns, as a proportion (0-1) of total active"`
	DriftCtxt   bool    `desc:"use drifting context representations -- otherwise does bit flips from prototype"`
	CtxtFlipPct float32 `desc:"proportion (0-1) of active bits to flip for each context pattern, relative to a prototype, for non-drifting"`
	DriftPct    float32 `desc:"percentage of active bits that drift, per step, for drifting context"`
}

func (pp *PatParams) Defaults() {
	pp.ListSize = 10 // 20 def
	pp.MinDiffPct = 0.5
	pp.CtxtFlipPct = .25
}

// HipParams have the hippocampus size and connectivity parameters
type HipParams struct {
	EC2Size      evec.Vec2i `desc:"size of EC2"`
	ECSize       evec.Vec2i `desc:"size of EC in terms of overall pools (outer dimension)"`
	ECPool       evec.Vec2i `desc:"size of one EC pool"`
	CA1Pool      evec.Vec2i `desc:"size of one CA1 pool"`
	CA3Size      evec.Vec2i `desc:"size of CA3"`
	DGRatio      float32    `desc:"size of DG / CA3"`
	DGSize       evec.Vec2i `inactive:"+" desc:"size of DG"`
	lateralPCon  float32    `desc:"percent connectivity in EC2 lateral"`
	EC2PCon      float32    `desc:"percent connectivity from Input to EC2"`
	EC3ToEC2PCon float32    `desc:"percent connectivity from EC3 to EC2"`
	DGPCon       float32    `desc:"percent connectivity into DG"`
	CA3PCon      float32    `desc:"percent connectivity into CA3"`
	CA1PCon      float32    `desc:"percent connectivity from CA3 into CA1"`
	MossyPCon    float32    `desc:"percent connectivity into CA3 from DG"`
	ECPctAct     float32    `desc:"percent activation in EC pool"`
	MossyDel     float32    `desc:"delta in mossy effective strength between minus and plus phase"`
	MossyDelTest float32    `desc:"delta in mossy strength for testing (relative to base param)"`
	MemThr       float64    `desc:"memory threshold"`
}

func (hp *HipParams) Defaults() {
	// size
	hp.EC2Size.Set(21, 21) // 21
	hp.ECSize.Set(2, 3)
	hp.ECPool.Set(7, 7)
	hp.CA1Pool.Set(10, 10) // using MedHip now
	hp.CA3Size.Set(20, 20) // using MedHip now
	hp.DGRatio = 2.236     // c.f. Ketz et al., 2013

	// ratio
	hp.DGPCon = 0.25 // .35 is sig worse, .2 learns faster but AB recall is worse
	hp.CA3PCon = 0.25
	hp.CA1PCon = 0.25
	hp.MossyPCon = 0.02 // .02 > .05 > .01 (for small net)
	hp.ECPctAct = 0.2
	hp.lateralPCon = 0.75
	hp.EC2PCon = 0.25      // 0.005 for no binding
	hp.EC3ToEC2PCon = 0.1 // 0.1 for EC3-EC2 in WintererMaierWoznyEtAl17, not sure about Input-EC2

	hp.MossyDel = 4     // 4 -- best is 4 del on 4 rel baseline
	hp.MossyDelTest = 3 // for rel = 4: 3 > 2 > 0 > 4 -- 4 is very bad -- need a small amount.. 0 for NoDynMF and orig

	hp.MemThr = 0.34
}

func (hp *HipParams) Update() {
	hp.DGSize.X = int(float32(hp.CA3Size.X) * hp.DGRatio)
	hp.DGSize.Y = int(float32(hp.CA3Size.Y) * hp.DGRatio)
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net    *axon.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Sim    SimParams     `desc:"misc params specific to this simulation"`
	Hip    HipParams     `desc:"hippocampus sizing parameters"`
	Pat    PatParams     `desc:"parameters for the input patterns"`
	Params emer.Params   `view:"inline" desc:"all parameter management"`

	Loops *looper.Manager `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats estats.Stats    `desc:"contains computed statistic values"`
	Logs  elog.Logs       `desc:"Contains all the logs and information about the logs.'"`

	PoolVocab    patgen.Vocab     `view:"no-inline" desc:"pool patterns vocabulary"`
	TrainAB      *etable.Table    `view:"no-inline" desc:"AB training patterns to use"`
	TrainAC      *etable.Table    `view:"no-inline" desc:"AC training patterns to use"`
	TestAB       *etable.Table    `view:"no-inline" desc:"AB testing patterns to use"`
	TestAC       *etable.Table    `view:"no-inline" desc:"AC testing patterns to use"`
	PreTrainLure *etable.Table    `view:"no-inline" desc:"Lure pretrain patterns to use"`
	TestLure     *etable.Table    `view:"no-inline" desc:"Lure testing patterns to use"`
	TrainAll     *etable.Table    `view:"no-inline" desc:"all training patterns -- for pretrain"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Context      axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Sim.Defaults()
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()

	ss.PoolVocab = patgen.Vocab{}
	ss.TrainAB = &etable.Table{}
	ss.TrainAC = &etable.Table{}
	ss.TestAB = &etable.Table{}
	ss.TestAC = &etable.Table{}
	ss.PreTrainLure = &etable.Table{}
	ss.TestLure = &etable.Table{}
	ss.TrainAll = &etable.Table{}

	ss.RndSeeds.Init(100) // max 100 runs
	ss.Context.Defaults()
	ss.Pat.Defaults() // ??? where to put this?
	ss.Hip.Defaults() // ??? where to put this?
	ss.ConfigArgs()   // do this first, has key defaults
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigPats()
	// ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	ss.Hip.Update() // update DG size
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
	tst.Config(etable.NewIdxView(ss.TestAB))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	// ss.Params.AddLayers([]string{"EC2", "EC3", "DG", "CA3", "CA1"}, "Hidden")
	// ss.Params.SetObject("NetSize")

	net.InitName(net, "Hip_bench")
	net.SetMaxData(ctx, ss.Sim.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	// inp := net.AddLayer2D("Input", 5, 5, axon.InputLayer)
	// hid1 := net.AddLayer2D("Hidden1", ss.Params.LayY("Hidden1", 10), ss.Params.LayX("Hidden1", 10), axon.SuperLayer)
	// hid2 := net.AddLayer2D("Hidden2", ss.Params.LayY("Hidden2", 10), ss.Params.LayX("Hidden2", 10), axon.SuperLayer)
	// out := net.AddLayer2D("Output", 5, 5, axon.TargetLayer)
	hp := &ss.Hip
	in := net.AddLayer4D("Input", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, axon.InputLayer)
	ec2 := net.AddLayer2D("EC2", hp.EC2Size.Y, hp.EC2Size.X, axon.SuperLayer)
	ec3 := net.AddLayer4D("EC3", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, axon.SuperLayer)
	ec5 := net.AddLayer4D("EC5", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, axon.TargetLayer) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", hp.ECSize.Y, hp.ECSize.X, hp.CA1Pool.Y, hp.CA1Pool.X, axon.SuperLayer)
	dg := net.AddLayer2D("DG", hp.DGSize.Y, hp.DGSize.X, axon.SuperLayer)
	ca3 := net.AddLayer2D("CA3", hp.CA3Size.Y, hp.CA3Size.X, axon.SuperLayer)

	ec3.SetClass("EC")
	ec5.SetClass("EC")

	ec2.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, Space: 2})
	ec3.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "EC2", YAlign: relpos.Front, Space: 2})
	ec5.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "EC3", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "EC2", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "DG", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "CA3", YAlign: relpos.Front, Space: 2})

	// use this to position layers relative to each other
	// hid2.PlaceRightOf(hid1, 2)

	// note: see emergent/prjn module for all the options on how to connect
	// NewFull returns a new prjn.Full connectivity pattern
	onetoone := prjn.NewOneToOne()
	pool1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()
	inToEc2 := prjn.NewUnifRnd()
	inToEc2.PCon = hp.EC2PCon
	Ec3ToEc2 := prjn.NewUnifRnd()
	Ec3ToEc2.PCon = hp.EC3ToEC2PCon
	mossy := prjn.NewUnifRnd()
	mossy.PCon = hp.MossyPCon

	// circle inhibitory lateral
	lat := prjn.NewCircle()
	lat.TopoWts = true
	lat.Radius = 2
	lat.Sigma = 2

	net.ConnectLayers(in, ec2, inToEc2, axon.ForwardPrjn)
	net.ConnectLayers(in, ec3, onetoone, axon.ForwardPrjn)
	net.ConnectLayers(ec3, ec2, Ec3ToEc2, axon.ForwardPrjn)
	net.ConnectLayers(ec5, ec3, onetoone, axon.BackPrjn)

	inh := net.ConnectLayers(ec2, ec2, lat, axon.InhibPrjn)
	inh.SetClass("InhibLateral")

	// MSP
	net.ConnectLayers(ec3, ca1, pool1to1, axon.ForwardPrjn).SetClass("EcCa1Prjn")
	net.ConnectLayers(ca1, ec5, pool1to1, axon.ForwardPrjn).SetClass("EcCa1Prjn")
	net.ConnectLayers(ec5, ca1, pool1to1, axon.ForwardPrjn).SetClass("EcCa1Prjn")

	// TSP
	ppathDG := prjn.NewUnifRnd()
	ppathDG.PCon = hp.DGPCon
	ppathCA3 := prjn.NewUnifRnd()
	ppathCA3.PCon = hp.CA3PCon
	ca3ToCa1 := prjn.NewUnifRnd()
	ca3ToCa1.PCon = hp.CA1PCon
	net.ConnectLayers(ec2, dg, ppathDG, axon.ForwardPrjn).SetClass("HippoCHL")
	net.ConnectLayers(ec2, ca3, ppathCA3, axon.ForwardPrjn).SetClass("PPath")
	net.ConnectLayers(ca3, ca3, full, axon.LateralPrjn).SetClass("PPath")
	net.ConnectLayers(dg, ca3, mossy, axon.ForwardPrjn).SetClass("HippoCHL")
	net.ConnectLayers(ca3, ca1, ca3ToCa1, axon.ForwardPrjn).SetClass("HippoCHL")

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// out.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	err := net.Build(ctx)
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	net.SetNThreads(4) // useful with NData
	ss.Params.SetObject("Network")
	ss.Params.SetObject("Sim")
	ss.Params.SetObject("Hip")
	ss.Params.SetObject("Pat")
	net.InitWts(ctx)
	net.InitTopoSWts()
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

func (ss *Sim) TestInit() {
	ss.Loops.ResetCountersByMode(etime.Test)
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.RndSeeds.Set(run)
	ss.RndSeeds.Set(run, &ss.Net.Rand)
}

func ConfigLoopsHip(ctx *axon.Context, man *looper.Manager, net *axon.Network, mossyDel, mossyDelTest float32) {
	

	var tmpVals []float32

	input := net.AxonLayerByName("Input")
	ec5 := net.AxonLayerByName("EC5")
	ca1 := net.AxonLayerByName("CA1")
	ca3 := net.AxonLayerByName("CA3")
	ca1FmEc3, _ := ca1.SendNameTry("EC3")
	ca1FmCa3, _ := ca1.SendNameTry("CA3")
	ca3FmDg, _ := ca3.SendNameTry("DG")

	dgPjScale := ca3FmDg.(*axon.Prjn).Params.PrjnScale.Rel

	// if man.Mode == etime.Train {
	// 	ec5.Params.LayType = axon.TargetLayer
	// } else {
	// 	ec5.Params.LayType = axon.CompareLayer
	// }

	startOfQ1 := looper.NewEvent("Q0", 0, func() {
		ca1FmEc3.(*axon.Prjn).Params.PrjnScale.Rel = 1
		ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Rel = 0.3

		ca3FmDg.(*axon.Prjn).Params.PrjnScale.Rel = dgPjScale - mossyDel // turn off DG input to CA3 in first quarter

		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})
	endOfQ1 := looper.NewEvent("Q1", 50, func() {
		ca1FmEc3.(*axon.Prjn).Params.PrjnScale.Rel = 0.3
		ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Rel = 1
		if man.Mode == etime.Train {
			ca3FmDg.(*axon.Prjn).Params.PrjnScale.Rel = dgPjScale // restore after 1st quarter
		} else {
			ca3FmDg.(*axon.Prjn).Params.PrjnScale.Rel = dgPjScale - mossyDelTest // testing
		}
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	}) // 50ms
	endOfQ3 := looper.NewEvent("Q3", 150, func() {
		ca1FmEc3.(*axon.Prjn).Params.PrjnScale.Rel = 1
		ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Rel = 0.3
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
		ca3FmDg.(*axon.Prjn).Params.PrjnScale.Rel = dgPjScale // restore
		ca1FmCa3.(*axon.Prjn).Params.PrjnScale.Rel = 1
		net.InitGScale(ctx) // update computed scaling factors
		net.GPU.SyncParamsToGPU()
	})

	man.AddEventAllModes(etime.Cycle, startOfQ1, endOfQ1, endOfQ3, endOfQ4)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	trls := int(mat32.IntMultipleGE(float32(ss.Sim.NTrials), float32(ss.Sim.NData)))

	man.AddStack(etime.Train).AddTime(etime.Run, 5).AddTime(etime.Epoch, 200).AddTimeIncr(etime.Trial, trls, ss.Sim.NData).AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTimeIncr(etime.Trial, trls, ss.Sim.NData).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

	ConfigLoopsHip(&ss.Context, man, ss.Net, 4, 3)

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
			// fmt.println(man.GetLoop(mode, etime.Trial))
		})
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnEnd.Add("TestAtInterval", func() {
		if (ss.Sim.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Sim.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		axon.LogTestErrors(&ss.Logs)
	})
	// man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
	// 	trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	// 	if ss.Sim.PCAInterval > 0 && trnEpc%ss.Sim.PCAInterval == 0 {
	// 		axon.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
	// 		ss.Logs.ResetLog(etime.Analyze, etime.Trial)
	// 	}
	// })

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	// man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
	// 	trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	// 	if (ss.Sim.PCAInterval > 0) && (trnEpc%ss.Sim.PCAInterval == 0) {
	// 		ss.Log(etime.Analyze, etime.Trial)
	// 	}
	// })

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

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
			m.Loops[etime.Trial].OnEnd.Prepend("GUI:CounterUpdt", func() {
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
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode)
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	net.InitExt(ctx)
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		ev.Step()
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
	ss.InitRndSeed()
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
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
	hp := &ss.Hip
	ecY := hp.ECSize.Y
	ecX := hp.ECSize.X
	plY := hp.ECPool.Y // good idea to get shorter vars when used frequently
	plX := hp.ECPool.X // makes much more readable
	npats := ss.Pat.ListSize
	pctAct := hp.ECPctAct
	minDiff := ss.Pat.MinDiffPct
	nOn := patgen.NFmPct(pctAct, plY*plX)
	ctxtflip := patgen.NFmPct(ss.Pat.CtxtFlipPct, nOn)
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
	ss.Stats.SetFloat("Mem", 0.0)

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
	ev := ss.Envs.ByMode(ctx.Mode)
	ss.Stats.SetString("TrialName", ev.(*env.FixedTable).TrialName.Cur)
}

func (ss *Sim) NetViewCounters() {
	if ss.GUI.ViewUpdt.View == nil {
		return
	}
	di := ss.GUI.ViewUpdt.View.Di
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

	if ss.Stats.Float("UnitErr") > 0.34 {
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
	memthr := ss.Hip.MemThr
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
	// actQ1i, _ := ecout.UnitVarIdx("ActSt1") // where to see this?
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
		// fmt.Println("cmpN", cmpN)
		fmt.Println("trgOnWasOffCmp", trgOnWasOffCmp)
		if cmpN > 0 { // should be
			trgOnWasOffCmp /= cmpN
			if trgOnWasOffCmp < memthr && trgOffWasOn < memthr {
				ss.Stats.SetFloat("Mem", 1)
			} else {
				ss.Stats.SetFloat("Mem", 0)
			}
		}
	}
	ss.Stats.SetFloat("TrgOnWasOffAll", trgOnWasOffAll)
	ss.Stats.SetFloat("TrgOnWasOffCmp", trgOnWasOffCmp)
	ss.Stats.SetFloat("TrgOffWasOn", trgOffWasOn)
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

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
	ss.Logs.AddStatAggItem("Mem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddCopyFromFloatItems(etime.Train, etime.Epoch, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr", "TrgOnWasOffAll", "TrgOnWasOffCmp", "TrgOffWasOn", "Mem")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	// axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net, etime.Test, etime.Cycle)
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer", "TargetLayer")

	ss.Logs.PlotItems("FirstZero", "LastZero", "TrgOnWasOffAll", "TrgOnWasOffCmp", "Mem")

	ss.Logs.CreateTables()
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
		for di := 0; di < ss.Sim.NData; di++ {
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
	ss.GUI.MakeWindow(ss, "hip_bench", title, `Benchmarking`)
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

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Init", Icon: "update",
		Tooltip: "Call ResetCountersByMode with test mode and update GUI.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.TestInit()
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Test})

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
	ss.Args.AddInt("nzero", 2, "number of zero error epochs in a row to count as full training")
	ss.Args.AddInt("ndata", 16, "number of data items to run in parallel")
	ss.Args.AddInt("threads", 0, "number of parallel threads, for cpu computation (0 = use default)")
	ss.Args.SetInt("epochs", 100)
	ss.Args.SetInt("runs", 5)
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

	ss.Net.GPU.Destroy() // safe even if no GPU
}
