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
	"math/rand"
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
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/popcode"
	"github.com/emer/emergent/prjn"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
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

// see params.go for network params

// SimParams has all the custom params for this sim
type SimParams struct {
	NPools     int          `view:"-" desc:"number of pools"`
	NUnitsY    int          `view:"-" desc:"number of units within each pool, Y"`
	NUnitsX    int          `view:"-" desc:"number of units within each pool, X"`
	NUnits     int          `view:"-" desc:"total number of units within each pool"`
	NoInc      bool         `desc:"do not auto-increment ACCPos / Neg values during test -- also set by Test1 button"`
	ACCPos     float32      `desc:"activation of ACC positive valence -- drives go"`
	ACCNeg     float32      `desc:"activation of ACC neg valence -- drives nogo"`
	ACCPosInc  float32      `desc:"across-units multiplier in activation of ACC positive valence -- e.g., .9 daecrements subsequent units by 10%"`
	ACCNegInc  float32      `desc:"across-units multiplier in activation of ACC neg valence, e.g., 1.1 increments subsequent units by 10%"`
	PosNegThr  float32      `desc:"threshold on diff between ACCPos - ACCNeg for counting as a Go trial"`
	SNc        float32      `desc:"dopamine level - computed for learning"`
	TestInc    float32      `desc:"increment in testing activation for test all"`
	TestReps   int          `desc:"number of repetitions per testing level"`
	InitMtxWts bool         `desc:"initialize matrix Go / No weights to follow Pos / Neg inputs -- else .5 even and must be learned"`
	InN        int          `desc:"number of different values to learn in input layer"`
	InCtr      int          `inactive:"+" desc:"input counter -- gives PFC network something to do"`
	PopCode    popcode.OneD `desc:"pop code the values in ACCPos and Neg"`
}

// Defaults sets default params
func (ss *SimParams) Defaults() {
	ss.NoInc = false
	ss.NPools = 1
	ss.NUnitsY = 5
	ss.NUnitsX = 5
	ss.NUnits = ss.NUnitsY * ss.NUnitsX
	ss.ACCPos = 1
	ss.ACCNeg = .2
	ss.ACCPosInc = 1 // 0.8
	ss.ACCNegInc = 1
	ss.PosNegThr = 0
	ss.TestInc = 0.1
	ss.TestReps = 25
	ss.InN = 5
	ss.PopCode.Defaults()
	ss.PopCode.SetRange(-0.2, 1.2, 0.1)
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
	// ss.Params.ExtraSets = "WtScales" // todo: ensure params same without
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Pats = &etable.Table{}
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.TestInterval = 500
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
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{ss.Sim.NUnitsY, ss.Sim.NUnitsX}, []string{"Y", "X"}},
	}
	dt.SetFromSchema(sch, 10)
	non := int(float32(ss.Sim.NUnits) * .15)
	patgen.PermutedBinaryMinDiff(dt.Cols[1].(*etensor.Float32), non, 1, 0, non/2)

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
	// trn.Config(etable.NewIdxView(ss.Pats))
	trn.Validate()

	tst.Nm = etime.Test.String()
	tst.Dsc = "testing params and state"
	// tst.Config(etable.NewIdxView(ss.Pats))
	tst.Sequential = true
	tst.Validate()

	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// trn.Table = splits.Splits[0]
	// tst.Table = splits.Splits[1]

	// trn.Init(0)
	// tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "PCore")
	np := ss.Sim.NPools
	nuY := ss.Sim.NUnitsY
	nuX := ss.Sim.NUnitsX
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

	pfc, pfcCT := net.AddSuperCT4D("PFC", 1, np, nuY, nuX, space, one2one)
	// prjns are: super->PT, PT self, CT-> thal
	pfcPT, pfcVM := net.AddPTMaintThalForSuper(pfc, pfcCT, "VM", one2one, pone2one, space)
	_ = pfcPT
	pfcCT.SetClass("PFC CTCopy")
	pfcCT.CTDefParamsMedium() // FSA

	net.ConnectLayers(inly, pfc, pone2one, axon.ForwardPrjn)
	net.ConnectToPulv(pfc, pfcCT, inP, pone2one, pone2one)

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

	if !ss.Sim.InitMtxWts {
		ss.ViewUpdt.RecordSyns() // note: critical to update weights here so DWt is visible
		return
	}

	mtxgo := net.AxonLayerByName("MtxGo")
	mtxno := net.AxonLayerByName("MtxNo")

	for _, pj := range mtxgo.RcvPrjns {
		slay := pj.Send
		if slay.Nm == "PFC" {
			continue
		}
		sy := &pj.Syns[0]
		if slay.Nm == "ACCPos" {
			sy.Wt = float32(erand.UniformMeanRange(0.75, 0.25, -1))
		} else {
			sy.Wt = 0
		}
		sy.LWt = pj.Params.SWt.LWtFmWts(sy.Wt, sy.SWt)
	}

	for _, pj := range mtxno.RcvPrjns {
		slay := pj.Send
		if slay.Nm == "PFC" {
			continue
		}
		sy := &pj.Syns[0]
		if slay.Nm == "ACCNeg" {
			sy.Wt = float32(erand.UniformMeanRange(0.75, 0.25, -1))
		} else {
			sy.Wt = 0
		}
		sy.LWt = pj.Params.SWt.LWtFmWts(sy.Wt, sy.SWt)
	}
	ss.ViewUpdt.RecordSyns() // note: critical to update weights here so DWt is visible

	// ab, err := Asset("pcore.wts") // embedded in executable
	// if err != nil {
	// 	log.Println(err)
	// }
	// net.ReadWtsJSON(bytes.NewBuffer(ab))
	// net.OpenWtsJSON("pcore.wts")
	// below is one-time conversion from c++ weights
	// net.OpenWtsCpp("CatsDogsNet.wts")
	// net.SaveWtsJSON("pcore.wts")
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

	man.AddStack(etime.Train).AddTime(etime.Run, 5).AddTime(etime.Epoch, 50).AddTime(etime.Sequence, 50).AddTime(etime.Trial, 3).AddTime(etime.Cycle, 200)

	nTestInc := int(1.0/ss.Sim.TestInc) + 1

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Sequence, ss.Sim.TestReps*nTestInc*nTestInc).AddTime(etime.Trial, 3).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

	man.GetLoop(etime.Test, etime.Sequence).OnStart.Add("TestInc", func() {
		if ss.Sim.NoInc {
			return
		}
		seq := man.Stacks[etime.Test].Loops[etime.Sequence].Counter.Cur
		repn := seq / ss.Sim.TestReps
		pos := repn / nTestInc
		neg := repn % nTestInc
		ss.Sim.ACCPos = float32(pos) * ss.Sim.TestInc
		ss.Sim.ACCNeg = float32(neg) * ss.Sim.TestInc
	})

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		// stack.Loops[etime.Trial].OnStart.Add("Env:Step", func() {
		// 	// note: OnStart for env.Env, others may happen OnEnd
		// 	ss.Envs[mode.String()].Step()
		// })
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			trial := man.Stacks[mode].Loops[etime.Trial].Counter.Cur
			ss.ApplyInputs(mode, trial)
		})
		stack.Loops[etime.Trial].OnEnd.Add("StatCounters", ss.StatCounters)
		stack.Loops[etime.Trial].OnEnd.Add("TrialStats", ss.TrialStats)
	}

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

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

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
func (ss *Sim) ApplyInputs(mode etime.Modes, trial int) {
	net := ss.Net
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	if mode == etime.Train && trial == 0 {
		if !ss.Sim.NoInc {
			ss.Sim.ACCPos = rand.Float32()
			ss.Sim.ACCNeg = rand.Float32()
		}
		ss.Sim.SNc = ss.Sim.ACCPos - ss.Sim.ACCNeg
	}

	np := ss.Sim.NPools
	nu := ss.Sim.NUnits
	itsr := etensor.Float32{}
	itsr.SetShape([]int{np * nu}, nil, nil)

	lays := []string{"ACCPos", "ACCNeg", "In"}
	vals := []float32{ss.Sim.ACCPos, ss.Sim.ACCNeg, 1}
	for li, lnm := range lays {
		ly := net.AxonLayerByName(lnm)
		for j := 0; j < np; j++ {
			// np = different pools have changing increments
			// such that first pool is best choice
			io := j * nu
			var poolVal float32
			switch lnm {
			case "ACCPos":
				poolVal = vals[li] * mat32.Pow(ss.Sim.ACCPosInc, float32(j))
			case "ACCNeg":
				poolVal = vals[li] * mat32.Pow(ss.Sim.ACCNegInc, float32(j))
			default:
				poolVal = float32(ss.Sim.InCtr) / float32(ss.Sim.InN)
			}
			sv := itsr.Values[io : io+nu]
			ss.Sim.PopCode.Encode(&sv, poolVal, nu, false)
			// for i := 0; i < nu; i++ {
			// 	pval := float32(ss.Pats.CellTensorFloat1D("Input", row, i))
			// 	itsr.Values[io+i] = pval * vals[li]
		}
		ly.ApplyExt(&itsr)
	}
	ss.Sim.InCtr++
	if ss.Sim.InCtr > ss.Sim.InN {
		ss.Sim.InCtr = 0
	}
	ss.ApplyPVLV(trial)
	net.ApplyExts(&ss.Context) // now required for GPU mode
}

// ApplyPVLV applies PVLV reward inputs
func (ss *Sim) ApplyPVLV(trial int) {
	ss.Context.PVLV.EffortUrgencyUpdt(1)
	if ss.Context.Mode == etime.Test {
		ss.Context.PVLV.Urgency.Reset()
	}

	switch trial {
	case 0:
		ss.Context.NeuroMod.SetRew(0, false) // no rew
		ss.Context.NeuroMod.ACh = 0
	case 1:
		ss.Context.NeuroMod.SetRew(0, false) // no rew
		ss.Context.NeuroMod.ACh = 1
	case 2:
		ss.Context.NeuroMod.ACh = 1
		ss.GatedRew()
	}
}

// GatedRew applies reward input based on gating action and input
func (ss *Sim) GatedRew() {
	rew := ss.Stats.Float32("Rew")
	ss.SetRew(rew)
}

func (ss *Sim) SetRew(rew float32) {
	net := ss.Net
	ss.Context.NeuroMod.SetRew(rew, true)
	ss.Context.NeuroMod.DA = rew // no reward prediction error
	ss.Context.PVLV.InitUS()
	if rew > 0 {
		ss.Context.PVLV.SetPosUS(1, rew)
	} else if rew < 0 {
		ss.Context.PVLV.SetNegUS(1, rew)
	}

	itsr := etensor.Float32{}
	itsr.SetShape([]int{1}, nil, nil)
	itsr.Values[0] = rew
	sncly := net.AxonLayerByName("SNc")
	sncly.ApplyExt(&itsr)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	// ss.Envs.ByMode(etime.Train).Init(0)
	// ss.Envs.ByMode(etime.Test).Init(0)
	ss.Sim.InCtr = 0
	ss.Context.Reset()
	ss.Context.Mode = etime.Train
	ss.InitWts(ss.Net)
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	// ss.Envs.ByMode(etime.Test).Init(0)
	ss.Sim.NoInc = false
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("PFCVM_RT", 0.0)
	ss.Stats.SetFloat("Gated", 0)
	ss.Stats.SetFloat("Should", 0)
	ss.Stats.SetFloat("Match", 0)
	ss.Stats.SetFloat("Rew", 0)
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
	ss.Stats.SetFloat32("ACCPos", ss.Sim.ACCPos)
	ss.Stats.SetFloat32("ACCNeg", ss.Sim.ACCNeg)
	trlnm := fmt.Sprintf("%4f_%4f", ss.Sim.ACCPos, ss.Sim.ACCNeg)
	ss.Stats.SetString("TrialName", trlnm)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Sequence", "Trial", "TrialName", "Cycle", "Gated", "Should", "Match", "Rew"})
}

// TrialStats computes the trial-level statistics.
func (ss *Sim) TrialStats() {
	nan := mat32.NaN()
	ss.Stats.SetFloat32("Gated", nan)
	ss.Stats.SetFloat32("Should", nan)
	ss.Stats.SetFloat32("Match", nan)
	ss.Stats.SetFloat32("Rew", nan)
	ss.Stats.SetFloat32("PFCVM_RT", nan)

	trial := ss.Loops.Stacks[ss.Context.Mode].Loops[etime.Trial].Counter.Cur
	if trial != 1 {
		return
	}
	ss.RTStat()
	ss.GatedStats()
}

func (ss *Sim) RTStat() {
	net := ss.Net
	vtly := net.AxonLayerByName("PFCVM")
	gated := vtly.AnyGated()
	if !gated {
		return
	}
	mode := ss.Context.Mode
	trlog := ss.Logs.Log(mode, etime.Cycle)
	spkCyc := 0
	for row := 0; row < trlog.Rows; row++ {
		vts := trlog.CellTensorFloat1D("PFCVM_Spike", row, 0)
		if vts > 0.05 {
			spkCyc = row
			break
		}
	}
	ss.Stats.SetFloat("PFCVM_RT", float64(spkCyc)/200)
}

// GatedStats records gating stats
func (ss *Sim) GatedStats() {
	mtxly := ss.Net.AxonLayerByName("MtxGo")
	didGate, _ := mtxly.MatrixGated(&ss.Context) // will also be called later
	pndiff := (ss.Sim.ACCPos - ss.Sim.ACCNeg) - ss.Sim.PosNegThr
	shouldGate := pndiff > 0
	match := false
	var rew float32
	switch {
	case shouldGate && didGate:
		rew = 1
		match = true
	case shouldGate && !didGate:
		rew = -1
	case !shouldGate && didGate:
		rew = -1
	case !shouldGate && !didGate:
		rew = 1
		match = true
	}

	ss.Stats.SetFloat32("Gated", bools.ToFloat32(didGate))
	ss.Stats.SetFloat32("Should", bools.ToFloat32(shouldGate))
	ss.Stats.SetFloat32("Match", bools.ToFloat32(match))
	ss.Stats.SetFloat32("Rew", rew)
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Sequence, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Sequence, "TrialName")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCPos")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCNeg")

	ss.Logs.AddStatAggItem("Gated", "Gated", etime.Run, etime.Epoch, etime.Sequence, etime.Trial)
	ss.Logs.AddStatAggItem("Should", "Should", etime.Run, etime.Epoch, etime.Sequence, etime.Trial)
	ss.Logs.AddStatAggItem("Match", "Match", etime.Run, etime.Epoch, etime.Sequence, etime.Trial)
	li := ss.Logs.AddStatAggItem("Rew", "Rew", etime.Run, etime.Epoch, etime.Sequence, etime.Trial)
	li.FixMin = false
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

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

func (ss *Sim) ConfigLogItems() {
	ss.Logs.AddStatAggItem("PFCVM_RT", "PFCVM_RT", etime.Run, etime.Epoch, etime.Sequence, etime.Trial)
	layers := ss.Net.LayersByType(axon.MatrixLayer, axon.BGThalLayer)
	npools := []int{ss.Sim.NPools}
	for _, lnm := range layers {
		clnm := lnm
		if clnm == "LDT" {
			continue
		}
		ss.Logs.AddItem(&elog.Item{
			Name:      clnm + "_ActAvg",
			Type:      etensor.FLOAT64,
			CellShape: npools,
			// Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					npools := []int{ss.Sim.NPools}
					tsr := ss.Stats.F64Tensor("Log_ActAvg")
					tsr.SetShape(npools, nil, nil)
					ss.Stats.SetF64Tensor("Log_ActAvg", tsr)
					for pi := 0; pi < ss.Sim.NPools; pi++ {
						tsr.Values[pi] = float64(ly.AvgMaxVarByPool("Act", pi+1).Avg)
					}
					ctx.SetTensor(tsr)
				}, etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					npools := []int{ss.Sim.NPools}
					tsr := ss.Stats.F64Tensor("Log_ActAvg")
					tsr.SetShape(npools, nil, nil)
					ss.Stats.SetF64Tensor("Log_ActAvg", tsr)
					for pi := 0; pi < ss.Sim.NPools; pi++ {
						tsr.Values[pi] = float64(ly.AvgMaxVarByPool("Act", pi+1).Avg)
					}
					ctx.SetTensor(tsr)
				}, etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
					tsr := ss.Stats.F64Tensor("Log_ActAvg")
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					for pi := 0; pi < ss.Sim.NPools; pi++ {
						tsr.Values[pi] = float64(ly.AvgMaxVarByPool("SpkMax", pi+1).Avg)
					}
					ctx.SetTensor(tsr)
				}, etime.Scope(etime.AllModes, etime.Sequence): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Sequence, agg.AggMean)
				}, etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:      clnm + "_Spike",
			Type:      etensor.FLOAT64,
			CellShape: npools,
			FixMin:    true,
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, etime.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					npools := []int{ss.Sim.NPools}
					tsr := ss.Stats.F64Tensor("Log_ActAvg")
					tsr.SetShape(npools, nil, nil)
					ss.Stats.SetF64Tensor("Log_ActAvg", tsr)
					for pi := 0; pi < ss.Sim.NPools; pi++ {
						tsr.Values[pi] = float64(ly.AvgMaxVarByPool("SpkMax", pi+1).Avg)
					}
					ctx.SetTensor(tsr)
				}}})
	}
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode != etime.Analyze {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
	}
	ss.StatCounters()

	trial := ss.Loops.Stacks[mode].Loops[etime.Trial].Counter.Cur
	if trial == 0 && time <= etime.Trial {
		return // no logging on first phase
	}

	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
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
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)

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

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test1", Icon: "play",
		Tooltip: "run one test trial with current ACCPos, ACCNeg params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Loops.ResetCountersByMode(etime.Test)
			ss.Sim.NoInc = true
			ss.Loops.Step(etime.Train, 1, etime.Trial)
			ss.Sim.NoInc = false
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
