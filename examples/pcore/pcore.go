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
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
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
	// GPU runs GUI with the GPU -- faster with NData = 16
	GPU = false
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
	NPools     int          `view:"-" desc:"number of pools"`
	NUnitsY    int          `view:"-" desc:"number of units within each pool, Y"`
	NUnitsX    int          `view:"-" desc:"number of units within each pool, X"`
	NUnits     int          `view:"-" desc:"total number of units within each pool"`
	NoInc      bool         `desc:"do not auto-increment ACCPos / Neg values during test -- also set by Test1 button"`
	ZeroTest   bool         `desc:"test with no ACC activity at all -- params need to prevent gating in this situation too"`
	ACCPos     float32      `desc:"activation of ACC positive valence -- drives go"`
	ACCNeg     float32      `desc:"activation of ACC neg valence -- drives nogo"`
	ACCPosInc  float32      `desc:"across-units multiplier in activation of ACC positive valence -- e.g., .9 daecrements subsequent units by 10%"`
	ACCNegInc  float32      `desc:"across-units multiplier in activation of ACC neg valence, e.g., 1.1 increments subsequent units by 10%"`
	PosNegThr  float32      `desc:"threshold on diff between ACCPos - ACCNeg for counting as a Go trial"`
	SNc        float32      `desc:"dopamine level - computed for learning"`
	NData      int          `desc:"number of data-parallel items to process at once"`
	NTrials    int          `desc:"number of trials per epoch"`
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
	ss.ZeroTest = false
	ss.NPools = 1
	ss.NUnitsY = 5
	ss.NUnitsX = 5
	ss.NUnits = ss.NUnitsY * ss.NUnitsX
	ss.ACCPos = 1
	ss.ACCNeg = .2
	ss.ACCPosInc = 1 // 0.8
	ss.ACCNegInc = 1
	ss.PosNegThr = 0
	ss.NData = 1
	ss.NTrials = 128
	ss.TestInc = 0.1
	ss.TestReps = 16
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
	Net      *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Sim      SimParams        `view:"no-inline" desc:"sim params"`
	Params   emer.Params      `view:"inline" desc:"all parameter management"`
	Loops    *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats    estats.Stats     `desc:"contains computed statistic values"`
	Logs     elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Pats     *etable.Table    `view:"no-inline" desc:"the training patterns to use"`
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
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	net.InitName(net, "PCore")
	net.SetMaxData(ctx, ss.Sim.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

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

	trls := int(mat32.IntMultipleGE(float32(ss.Sim.NTrials), float32(ss.Sim.NData)))

	man.AddStack(etime.Train).AddTime(etime.Run, 5).AddTime(etime.Epoch, 30).AddTimeIncr(etime.Sequence, trls, ss.Sim.NData).AddTime(etime.Trial, 3).AddTime(etime.Cycle, 200)

	nTestInc := int(1.0/ss.Sim.TestInc) + 1
	totTstTrls := ss.Sim.TestReps * nTestInc * nTestInc

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
		stack.Loops[etime.Trial].OnEnd.Add("GatedStats", func() {
			trial := man.Stacks[mode].Loops[etime.Trial].Counter.Cur
			if trial == 2 {
				ss.GatedStats()
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
			m.Loops[etime.Trial].OnEnd.InsertAfter("GatedStats", "GUI:CounterUpdt", func() {
				ss.NetViewCounters()
			})
		}
	}

	if Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// SetACC sets the ACC values, either random (training)
// or for testing, based on incrementing index
func (ss *Sim) SetACC(mode etime.Modes, idx, di int) {
	nTestInc := int(1.0/ss.Sim.TestInc) + 1
	if !ss.Sim.NoInc {
		if mode == etime.Test {
			repn := idx / ss.Sim.TestReps
			pos := repn / nTestInc
			neg := repn % nTestInc
			ss.Sim.ACCPos = float32(pos) * ss.Sim.TestInc
			ss.Sim.ACCNeg = float32(neg) * ss.Sim.TestInc
			// fmt.Printf("idx: %d  di: %d  repn: %d  pos: %d  neg: %d\n", idx, di, repn, pos, neg)
		} else {
			ss.Sim.ACCPos = rand.Float32()
			ss.Sim.ACCNeg = rand.Float32()
		}
	}
	ss.Sim.SNc = ss.Sim.ACCPos - ss.Sim.ACCNeg
	ss.Stats.SetFloat32Di("ACCPos", di, ss.Sim.ACCPos)
	ss.Stats.SetFloat32Di("ACCNeg", di, ss.Sim.ACCNeg)
}

// GetACC gets the previously stored ACC values
func (ss *Sim) GetACC(di int) {
	ss.Sim.ACCPos = ss.Stats.Float32Di("ACCPos", di)
	ss.Sim.ACCNeg = ss.Stats.Float32Di("ACCNeg", di)
	ss.Sim.SNc = ss.Sim.ACCPos - ss.Sim.ACCNeg
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(mode etime.Modes, seq, trial int) {
	ctx := &ss.Context
	net := ss.Net
	ss.Net.InitExt(ctx)

	np := ss.Sim.NPools
	nu := ss.Sim.NUnits
	itsr := etensor.Float32{}
	itsr.SetShape([]int{np * nu}, nil, nil)

	lays := []string{"ACCPos", "ACCNeg", "In"}
	if ss.Sim.ZeroTest {
		lays = []string{"In"}
	}

	for di := 0; di < ss.Sim.NData; di++ {
		idx := seq + di
		if trial == 0 {
			ss.SetACC(mode, idx, di)
		} else {
			ss.GetACC(di)
		}
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
			}
			ly.ApplyExt(ctx, uint32(di), &itsr)
		}
		ss.ApplyPVLV(trial, uint32(di))
		ss.Sim.InCtr++
		if ss.Sim.InCtr > ss.Sim.InN {
			ss.Sim.InCtr = 0
		}
	}
	net.ApplyExts(ctx) // now required for GPU mode
}

// ApplyPVLV applies PVLV reward inputs
func (ss *Sim) ApplyPVLV(trial int, di uint32) {
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
		ss.GatedRew(di)
	}
}

// GatedRew applies reward input based on gating action and input
func (ss *Sim) GatedRew(di uint32) {
	rew := ss.Stats.Float32Di("Rew", int(di))
	ss.SetRew(rew, di)
}

func (ss *Sim) SetRew(rew float32, di uint32) {
	ctx := &ss.Context
	ctx.PVLVInitUS(di)
	net := ss.Net
	axon.NeuroModSetRew(ctx, di, rew, true)
	axon.SetGlbV(ctx, di, axon.GvDA, rew) // no reward prediction error
	if rew > 0 {
		ctx.PVLVSetUS(di, axon.Positive, 0, 1)
	} else if rew < 0 {
		ctx.PVLVSetUS(di, axon.Negative, 0, 1)
	}

	itsr := etensor.Float32{}
	itsr.SetShape([]int{1}, nil, nil)
	itsr.Values[0] = rew
	sncly := net.AxonLayerByName("SNc")
	sncly.ApplyExt(ctx, di, &itsr)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRndSeed()
	ss.Sim.InCtr = 0
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
func (ss *Sim) StatCounters(di int) {
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ss.GetACC(di)
	trlnm := fmt.Sprintf("%4f_%4f", ss.Sim.ACCPos, ss.Sim.ACCNeg)
	ss.Stats.SetString("TrialName", trlnm)
}

func (ss *Sim) NetViewCounters() {
	if ss.GUI.ViewUpdt.View == nil {
		return
	}
	di := ss.GUI.ViewUpdt.View.Di
	ss.StatCounters(di)
	ss.TrialStats(di)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Sequence", "Trial", "Di", "TrialName", "Cycle", "Gated", "Should", "Match", "Rew"})
}

// TrialStats records the trial-level statistics -- all saved in GatedStats
func (ss *Sim) TrialStats(di int) {
	ss.Stats.SetFloat32("ACCPos", ss.Stats.Float32Di("ACCPos", di))
	ss.Stats.SetFloat32("ACCNeg", ss.Stats.Float32Di("ACCNeg", di))
	ss.Stats.SetFloat32("Gated", ss.Stats.Float32Di("Gated", di))
	ss.Stats.SetFloat32("Should", ss.Stats.Float32Di("Should", di))
	ss.Stats.SetFloat32("Match", ss.Stats.Float32Di("Match", di))
	ss.Stats.SetFloat32("Rew", ss.Stats.Float32Di("Rew", di))
	ss.Stats.SetFloat32("PFCVM_RT", ss.Stats.Float32Di("PFCM_RT", di))
}

func (ss *Sim) RTStat(di int) {
	return
	// todo: need a GPU-based mech that sets global var
	ctx := &ss.Context
	net := ss.Net
	vtly := net.AxonLayerByName("PFCVM")
	gated := vtly.AnyGated(uint32(di))
	if !gated {
		return
	}
	mode := ctx.Mode
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

// GatedStats records gating stats, during presentation
func (ss *Sim) GatedStats() {
	for di := 0; di < ss.Sim.NData; di++ {
		ss.GetACC(di)
		ss.GatedStatsDi(di)
	}
}

func (ss *Sim) GatedStatsDi(di int) {
	pndiff := (ss.Sim.ACCPos - ss.Sim.ACCNeg) - ss.Sim.PosNegThr
	shouldGate := pndiff > 0
	mtxly := ss.Net.AxonLayerByName("MtxGo")
	didGate := mtxly.AnyGated(uint32(di))
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

	ss.Stats.SetFloat32Di("Should", di, bools.ToFloat32(shouldGate))
	ss.Stats.SetFloat32Di("Gated", di, bools.ToFloat32(didGate))
	ss.Stats.SetFloat32Di("Match", di, bools.ToFloat32(match))
	ss.Stats.SetFloat32Di("Rew", di, rew)
	ss.Stats.SetFloat32Di("PFCVM_RT", di, .1)
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

	ss.Logs.AddStatAggItem("Gated", "Gated", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("Should", "Should", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("Match", "Match", etime.Run, etime.Epoch, etime.Sequence)
	li := ss.Logs.AddStatAggItem("Rew", "Rew", etime.Run, etime.Epoch, etime.Sequence)
	li.FixMin = false
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Sequence)

	// axon.LogAddDiagnosticItems(&ss.Logs, ss.Net, etime.Epoch, etime.Trial)
	// axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net, etime.Test, etime.Cycle)

	// ss.Logs.PlotItems("MtxGo_ActAvg", "PFCVM_ActAvg", "PFCVM_RT", "Gated", "Should", "Match", "Rew")
	ss.Logs.PlotItems("Gated", "Should", "Match", "Rew")

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
