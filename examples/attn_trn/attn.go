// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
attn_trn: test of trn-based attention in basic V1, V2, LIP localist network with gabor inputs.

Tests for effects of contextual normalization in Reynolds & Heeger, 2009 framework.
In terms of differential sizes of attentional spotlight vs. stimulus size.
*/
package main

// TODO: fix or delete

func main() {}

/*
import (
	"fmt"
	"log"
	"strconv"

	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"cogentcore.org/core/math32/vecint"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/plot/plotcore"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/split"
	"cogentcore.org/core/core"
	"cogentcore.org/core/views"
	"cogentcore.org/core/tree"
	"cogentcore.org/core/math32"
)

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// TestType is the type of testing patterns
type TestType int32 //enums:enum

const (
	// AttnSize tests effects of attentional spotlight relative to stimulus size
	AttnSize TestType = iota

	// AttnSizeDebug smaller debugging version of AttnSize
	AttnSizeDebug

	// AttnSizeC2Up contrast level 2 and above
	AttnSizeC2Up

	// Popout tests unique feature popout
	Popout
	TestTypeN
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Path", Desc: "no learning",
				Params: params.Params{
					"Path.Learn.Learn":    "false",
					"Path.SWts.Init.Mean": "0.8",
					"Path.SWts.Init.Var":  "0",
					"Path.SWts.Init.Sym":  "false", // for lesions, just in case
				}},
			{Sel: "Layer", Desc: "pool etc",
				Params: params.Params{
					"Layer.Inhib.Pool.Gi":        "1.0",
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Pool.FFEx0":     "0.18",
					"Layer.Inhib.Pool.FFEx":      "0.1",
					"Layer.Inhib.Layer.On":       "false", // TRC drives layer-level
					"Layer.Inhib.ActAvg.Nominal": "0.01",
					"Layer.Inhib.Layer.FFEx0":    "0.18",
					"Layer.Inhib.Layer.FFEx":     "0.1",
					"Layer.Acts.Decay.Act":       "1",
					"Layer.Acts.Decay.Glong":     "1",
					"Layer.Acts.Decay.KNa":       "1",
					"Layer.Acts.KNa.On":          "false", // turn off by default
					"Layer.Acts.Noise.Dist":      "Gaussian",
					"Layer.Acts.Noise.Var":       "0.002",
					"Layer.Acts.Noise.Type":      "NoNoise", // "GeNoise",
				}},
			{Sel: "SuperLayer", Desc: "pool etc",
				Params: params.Params{
					"Layer.Inhib.Layer.On":       "true",
					"Layer.Inhib.Layer.Gi":       "1.2",
					"Layer.Inhib.Layer.FFEx0":    "0.01", // must be < FF0
					"Layer.Inhib.Layer.FFEx":     "0",    // some effect randomly
					"Layer.Inhib.Layer.FF0":      "0.01", // doesn't have any effect until < .02
					"Layer.Inhib.Pool.Gi":        "1.5",
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Pool.FFEx0":     "0.18",
					"Layer.Inhib.Pool.FFEx":      "0", // 10? no big effects
					"Layer.Inhib.ActAvg.Nominal": "0.05",
					"Layer.Acts.Attn.On":         "true",
					"Layer.Acts.Attn.Min":        "0.2", // 0.5
					"Layer.Inhib.Topo.On":        "true",
					"Layer.Inhib.Topo.Width":     "4",
					"Layer.Inhib.Topo.Sigma":     "1.0",
					"Layer.Inhib.Topo.Gi":        "0.05",
					"Layer.Inhib.Topo.FF0":       "0.15",
					"Layer.Acts.Noise.Dist":      "Gaussian",
					"Layer.Acts.Noise.Var":       "0.02",    // .02
					"Layer.Acts.Noise.Type":      "GeNoise", // "GeNoise",
				}},
			{Sel: "TRCALayer", Desc: "topo etc pool etc",
				Params: params.Params{
					"Layer.Inhib.Pool.On":        "false",
					"Layer.Inhib.Layer.On":       "true",
					"Layer.Inhib.Layer.Gi":       "1.2",
					"Layer.Inhib.Layer.FFEx0":    "0.18",
					"Layer.Inhib.Layer.FFEx":     "0",
					"Layer.Inhib.ActAvg.Nominal": "0.2",
					"Layer.Inhib.Topo.On":        "true",
					"Layer.Inhib.Topo.Width":     "4",
					"Layer.Inhib.Topo.Sigma":     "1.0",
					"Layer.Inhib.Topo.Gi":        "0.03",
					"Layer.Inhib.Topo.FF0":       "0.18",
					"Layer.SendAttn.Thr":         "0.1",
				}},
			{Sel: "#V2CTA", Desc: "topo etc pool etc",
				Params: params.Params{
					"Layer.Inhib.Pool.On":        "false",
					"Layer.Inhib.Layer.On":       "true",
					"Layer.Inhib.Layer.Gi":       "1.0",
					"Layer.Inhib.Layer.FFEx0":    "0.15",
					"Layer.Inhib.Layer.FFEx":     "20",
					"Layer.Inhib.ActAvg.Nominal": "0.3",
				}},
			{Sel: "#LIP", Desc: "pool etc",
				Params: params.Params{
					"Layer.Inhib.Pool.On":        "false",
					"Layer.Inhib.Layer.Gi":       "1.5",
					"Layer.Inhib.Layer.On":       "true", // TRN drives all layer-level
					"Layer.Inhib.ActAvg.Nominal": "0.3",
				}},
			{Sel: "TRNLayer", Desc: "trn just does whole layer",
				Params: params.Params{
					"Layer.Inhib.Pool.On":        "false",
					"Layer.Inhib.Layer.On":       "true",
					"Layer.Inhib.Layer.Gi":       "2.0",
					"Layer.Inhib.ActAvg.Nominal": ".03",
					"Layer.Acts.Dt.GTau":         "3",
				}},
			{Sel: ".BackPath", Desc: "weaker output",
				Params: params.Params{
					"Path.PathScale.Rel": "0.1",
				}},
			{Sel: "#V2ToV2CTA", Desc: "",
				Params: params.Params{
					"Path.PathScale.Rel": "0.2", // 0.8
				}},
			{Sel: "#LIPToV2CTA", Desc: "",
				Params: params.Params{
					"Path.PathScale.Rel": "0.5", // 0.5
				}},
			{Sel: ".Inhib", Desc: "",
				Params: params.Params{
					"Path.PathScale.Abs": "1",
				}},
			{Sel: "#LIPToV2TA", Desc: "",
				Params: params.Params{
					"Path.PathScale.Rel": "0.3", // 0.3
				}},
			{Sel: "#LIPToV2", Desc: "",
				Params: params.Params{
					"Path.PathScale.Rel": "0.0",
				}},
			{Sel: "#V2CTAToV2CTA", Desc: "lateral within V2CTA",
				Params: params.Params{
					"Path.PathScale.Rel": "0.0", // 0.4
				}},
			{Sel: "#V1ToV2", Desc: "",
				Params: params.Params{
					"Path.PathScale.Abs": "1.2",
				}},
		},
	}},
	{Name: "KNaAdapt", Desc: "Turn on KNa adaptation", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "KNa adapt on",
				Params: params.Params{
					"Layer.Acts.KNa.On": "true",
				}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// number of cycles per trial
	Cycles int `default:"200"`

	// number of runs to run to collect stats
	Runs int `default:"10"`

	// sodium (Na) gated potassium (K) channels that cause neurons to fatigue over time
	KNaAdapt bool `default:"true"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *axon.Network `display:"no-inline"`

	//
	Path3x3Skp1 *paths.PoolTile `display:"Standard same-to-same size topographic pathway"`

	//
	Path5x5Skp1 *paths.PoolTile `display:"Standard same-to-same size topographic pathway"`

	// select which type of test (input patterns) to use
	Test TestType

	// testing trial-level log data -- click to see record of network's response to each input
	TstTrlLog *table.Table `display:"no-inline"`

	// aggregated testing data
	TstRunLog *table.Table `display:"no-inline"`

	// aggregate stats on testing data
	TstStats *table.Table `display:"no-inline"`

	// full collection of param sets -- not really interesting for this model
	Params params.Sets `display:"no-inline"`

	// Testing environment -- manages iterating over testing
	TestEnv AttnEnv

	// axon timing parameters and state
	Context axon.Context

	// whether to update the network view while running
	ViewOn bool

	// at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster
	ViewUpdate axon.TimeScales

	// layer to measure attentional effects on
	AttnLay string

	// names of layers to record activations etc of during testing
	TstRecLays []string

	// max activation in center of stimulus 1 (attended, stronger)
	S1Act float32

	// max activation in center of stimulus 2 (ignored, weaker)
	S2Act float32

	// percent modulation = (S1Act - S2Act) / S1Act
	PctMod float32

	// main GUI window
	Win *core.Window `display:"-"`

	// the network viewer
	NetView *netview.NetView `display:"-"`

	// the master toolbar
	ToolBar *core.ToolBar `display:"-"`

	// the test-trial plot
	TstTrlPlot *plotcore.PlotEditor `display:"-"`

	// the test-trial plot
	TstRunPlot *plotcore.PlotEditor `display:"-"`

	// for holding layer values
	ValuesTsrs map[string]*tensor.Float32 `display:"-"`

	// true if sim is running
	IsRunning bool `display:"-"`

	// flag to stop running
	StopNow bool `display:"-"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Defaults()
	ss.Net = &axon.Network{}
	ss.Test = AttnSize
	ss.TstTrlLog = &table.Table{}
	ss.TstRunLog = &table.Table{}
	ss.TstStats = &table.Table{}
	ss.Params = ParamSets
	ss.ViewOn = true
	ss.ViewUpdate = axon.AlphaCycle // axon.Cycle // axon.FastSpike
	ss.TstRecLays = []string{"V2"}

	ss.Path3x3Skp1 = paths.NewPoolTile()
	ss.Path3x3Skp1.Size.Set(3, 3)
	ss.Path3x3Skp1.Skip.Set(1, 1)
	ss.Path3x3Skp1.Start.Set(-1, -1)
	ss.Path3x3Skp1.TopoRange.Min = 0.8 // note: none of these make a very big diff
	ss.Path3x3Skp1.GaussInPool.On = false
	ss.Path3x3Skp1.GaussFull.CtrMove = 0

	ss.Path5x5Skp1 = paths.NewPoolTile()
	ss.Path5x5Skp1.Size.Set(5, 5)
	ss.Path5x5Skp1.Skip.Set(1, 1)
	ss.Path5x5Skp1.Start.Set(-2, -2)
	ss.Path5x5Skp1.TopoRange.Min = 0.8 // note: none of these make a very big diff
	ss.Path5x5Skp1.GaussInPool.On = false
	ss.Path5x5Skp1.GaussFull.CtrMove = 0

	ss.Defaults()
}

// Defaults sets default params
func (ss *Sim) Defaults() {
	ss.AttnLay = "V2"
	ss.Cycles = 200
	ss.Runs = 25
	ss.KNaAdapt = false
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstRunLog(ss.TstRunLog)
}

func (ss *Sim) ConfigEnv() {
	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Defaults()
	ss.TestEnv.V1Pools.Set(16, 16)
	ss.TestEnv.V1Feats.Set(4, 2)
	ss.UpdateEnv()
	ss.TestEnv.Config()
	ss.TestEnv.Validate()
	ss.TestEnv.Init(0)
}

func (ss *Sim) UpdateEnv() {
	switch ss.Test {
	case AttnSize:
		ss.TestEnv.Stims = StimAttnSizeAll
	case AttnSizeDebug:
		ss.TestEnv.Stims = StimAttnSizeDebug
	case AttnSizeC2Up:
		ss.TestEnv.Stims = StimAttnSizeC2Up
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "AttnNet")
	psz := ss.TestEnv.V1Pools
	fsz := ss.TestEnv.V1Feats
	v1 := net.AddLayer4D("V1", psz.Y, psz.X, fsz.Y, fsz.X, axon.InputLayer)
	v2 := net.AddSuperLayer4D("V2", psz.Y, psz.X, fsz.Y, fsz.X)
	lip := net.AddLayer4D("LIP", psz.Y, psz.X, 1, 1, axon.InputLayer)
	v2cta := net.AddLayer4D("V2CTA", psz.Y, psz.X, 1, 1, axon.SuperLayer)
	v2ta := net.AddTRCALayer4D("V2TA", psz.Y, psz.X, 1, 1)

	v2ta.SendAttn.ToLays.Add("V2")

	v2ta.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "V2", YAlign: relpos.Front, Space: 1})
	v2cta.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "V2TA", XAlign: relpos.Left, Space: 2})
	lip.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "V2", YAlign: relpos.Front, XAlign: relpos.Left})

	one2one := paths.NewOneToOne()
	pone2one := paths.NewPoolOneToOne()
	circle := paths.NewCircle()
	circle.Radius = 6
	circle.TopoWts = true
	circle.Sigma = 1

	// net.ConnectLayers(v2ct, v2p, one2one, emer.Forward)

	net.ConnectLayers(v1, v2, one2one, emer.Forward)
	net.ConnectLayers(v2, v2ta, ss.Path5x5Skp1, emer.Forward) // or v2cta
	// net.ConnectLayers(v2, v2, ss.Path5x5Skp1, emer.Inhib)
	// net.ConnectLayers(v2ta, v2ta, circle, emer.Inhib)
	// net.ConnectLayers(v2cta, v2cta, circle, emer.Lateral)
	// net.ConnectLayers(v2cta, v2ta, ss.Path5x5Skp1, emer.Forward)
	net.ConnectLayers(lip, v2, pone2one, emer.Back)
	net.ConnectLayers(lip, v2cta, pone2one, emer.Back) // ss.Path5x5Skp1
	net.ConnectLayers(lip, v2ta, pone2one, emer.Back)  // ss.Path5x5Skp1 was ponetoone

	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	ss.InitWts()
}

// InitWts initialize weights
func (ss *Sim) InitWts() {
	net := ss.Net
	net.InitWts()
	net.InitTopoSWts() //  sets all wt scales
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.UpdateEnv()
	ss.TestEnv.Init(0)
	ss.Context.Reset()
	// ss.Context.CycPerQtr = 55 // 220 total
	ss.InitWts()
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.TstTrlLog.SetNumRows(0)
	ss.UpdateView(false)
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	nm := ss.TestEnv.String()
	return fmt.Sprintf("Trial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TestEnv.Trial.Cur, ss.Context.Cycle, nm)
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters())
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

func (ss *Sim) UpdateViewTime(train bool, viewUpdate axon.TimeScales) {
	switch viewUpdate {
	case axon.Cycle:
		ss.UpdateView(train)
	case axon.FastSpike:
		if ss.Context.Cycle%10 == 0 {
			ss.UpdateView(train)
		}
	case axon.GammaCycle:
		if ss.Context.Cycle%25 == 0 {
			ss.UpdateView(train)
		}
	case axon.AlphaCycle:
		if ss.Context.Cycle%100 == 0 {
			ss.UpdateView(train)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// ThetaCyc runs one theta cycle (200 msec) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFromDWt calls are made.
// Handles netview updating within scope, and calls TrainStats()
func (ss *Sim) ThetaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdate := ss.ViewUpdate

	plusCyc := 50
	minusCyc := ss.Cycles - plusCyc

	ss.Net.NewState()
	ss.Context.NewState()
	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Context)
		// ss.LogTstCyc(ss.TstCycLog, ss.Context.Cycle)
		ss.Context.CycleInc()
		switch ss.Context.Cycle { // save states at beta-frequency -- not used computationally
		case 75:
			ss.Net.ActSt1(&ss.Context)
		case 100:
			ss.Net.ActSt2(&ss.Context)
		}
		if cyc == minusCyc-1 { // do before view update
			ss.Net.MinusPhase(&ss.Context)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdate)
		}
	}
	ss.Context.NewPhase()
	if viewUpdate == axon.Phase {
		ss.UpdateView(train)
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Context)
		// ss.LogTstCyc(ss.TstCycLog, ss.Context.Cycle)
		ss.Context.CycleInc()
		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Context)
			// ss.Net.CTCtxt(&ss.Context) // update context at end
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdate)
		}
	}
	if viewUpdate == axon.Phase || viewUpdate == axon.AlphaCycle || viewUpdate == axon.ThetaCycle {
		ss.UpdateView(train)
	}

	// if ss.TstCycPlot != nil {
	// 	ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	// }
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"V1", "LIP"}
	for _, lnm := range lays {
		ly := ss.Net.AxonLayerByName(lnm)
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func (ss *Sim) StimMaxAct(stm *Stim, lnm string) float32 {
	ly := ss.Net.AxonLayerByName(lnm)
	sz := vecint.Vector2i{ly.Shp.DimSize(1), ly.Shp.DimSize(0)}
	pt := stm.PosXY(sz)
	cx := int(pt.X)
	cy := int(pt.Y)
	max := float32(0)
	for dy := -1; dy <= 1; dy++ {
		y := cy + dy
		if y < 0 || y >= sz.Y {
			continue
		}
		for dx := -1; dx <= 1; dx++ {
			x := cx + dx
			if x < 0 || x >= sz.X {
				continue
			}
			pi := y*sz.X + x
			pl := &ly.Pools[pi+1]
			max = math32.Max(max, pl.Inhib.Act.Max)
		}
	}
	return max
}

func (ss *Sim) StimAvgAct(stm *Stim, lnm string) float32 {
	ly := ss.Net.AxonLayerByName(lnm)
	sz := vecint.Vector2i{ly.Shp.DimSize(1), ly.Shp.DimSize(0)}
	pt := stm.PosXY(sz)
	cx := int(math32.Round(pt.X)) - 1
	cy := int(math32.Round(pt.Y)) - 1
	if cx > sz.X/2 {
		cx++
	}
	// fmt.Printf("cx: %d  cy: %d\n", cx, cy)
	avg := float32(0)
	thr := float32(0.1)
	_ = thr
	hwd := 1
	for dy := 0; dy <= hwd; dy++ {
		y := cy + dy
		if y < 0 || y >= sz.Y {
			continue
		}
		for dx := 0; dx <= hwd; dx++ {
			x := cx + dx
			if x < 0 || x >= sz.X {
				continue
			}
			pi := y*sz.X + x
			pl := &ly.Pools[pi+1]
			for ni := pl.StIndex; ni < pl.EdIndex; ni++ {
				nrn := &ly.Neurons[ni]
				if nrn.Act >= thr {
					// avg += nrn.Attn
					avg += nrn.Act
				}
			}
		}
	}
	// if n > 0 {
	// 	avg /= float32(n)
	// }
	return avg / float32(4)
}

func (ss *Sim) TrialStats() {
	ss.S1Act = ss.StimAvgAct(&ss.TestEnv.CurStim.Stims[0], ss.AttnLay)
	ss.S2Act = ss.StimAvgAct(&ss.TestEnv.CurStim.Stims[1], ss.AttnLay)
	if ss.S1Act > 0 {
		ss.PctMod = (ss.S1Act - ss.S2Act) / ss.S1Act
		ss.PctMod = math32.Max(ss.PctMod, 0)
	} else {
		ss.PctMod = 0
	}
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWts saves the network weights -- when called with views.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWts(filename core.Filename) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial() {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewUpdate > axon.AlphaCycle {
			ss.UpdateView(false)
		}
		return
	}

	ss.Net.InitActs()
	ss.ApplyInputs(&ss.TestEnv)
	ss.ThetaCyc(false)
	// ss.ThetaCyc(false) // 2x
	ss.TrialStats()
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestTrialGUI runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrialGUI() {
	ss.TestTrial()
	ss.Stopped()
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.SetParams("", false) // in case params were changed
	ss.UpdateEnv()
	ss.TestEnv.Init(0)
	for {
		ss.TestTrial()
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

// TestRuns runs through the full set of testing items
func (ss *Sim) TestRuns() {
	ss.SetParams("", false) // in case params were changed
	ss.UpdateEnv()
	ss.TestEnv.Init(0)
	for {
		ss.TestTrial()
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if ss.StopNow {
			break
		}
		if chg {
			ss.TestEnv.Run.Incr()
			if ss.TestEnv.Run.Cur >= ss.Runs {
				break
			}
		}
	}
	ss.TestStats()
}

// RunTestRuns runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestRuns() {
	ss.StopNow = false
	ss.TestRuns()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}

	// spo := ss.Params.SetByName("Base").SheetByName("Network").SelByName(".SpatToObj")
	// spo.Params.SetParamByName("Path.PathScale.Rel", fmt.Sprintf("%g", ss.SpatToObj))
	// vsp := ss.Params.SetByName("Base").SheetByName("Network").SelByName("#V1ToSpat1")
	// vsp.Params.SetParamByName("Path.PathScale.Rel", fmt.Sprintf("%g", ss.V1ToSpat1))

	err := ss.SetParamsSet("Base", sheet, setMsg)

	if ss.KNaAdapt {
		err = ss.SetParamsSet("KNaAdapt", sheet, setMsg)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

//////////////////////////////////////////////
//  TstTrlLog

// ValuesTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValuesTsr(name string) *tensor.Float32 {
	if ss.ValuesTsrs == nil {
		ss.ValuesTsrs = make(map[string]*tensor.Float32)
	}
	tsr, ok := ss.ValuesTsrs[name]
	if !ok {
		tsr = &tensor.Float32{}
		ss.ValuesTsrs[name] = tsr
	}
	return tsr
}

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *table.Table) {
	row := dt.Rows
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	trl := ss.TestEnv.Trial.Cur
	rn := ss.TestEnv.Run.Cur

	dt.SetFloat("Run", row, float64(rn))
	dt.SetFloat("Trial", row, float64(trl))
	dt.SetString("TrialName", row, ss.TestEnv.String())
	dt.SetFloat("Cycle", row, float64(ss.Context.Cycle))
	dt.SetFloat("S1Act", row, float64(ss.S1Act))
	dt.SetFloat("S2Act", row, float64(ss.S2Act))
	dt.SetFloat("PctMod", row, float64(ss.PctMod))

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValuesTsr(lnm)
		ly := ss.Net.AxonLayerByName(lnm)
		ly.UnitValuesTensor(tsr, "Act")
		dt.SetTensor(lnm, row, tsr)
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *table.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.AddIntColumn("Run")
	dt.AddIntColumn("Trial")
	dt.AddStringColumn("TrialName")
	dt.AddIntColumn("Cycle")
	dt.AddFloat64Column("S1Act")
	dt.AddFloat64Column("S2Act")
	dt.AddFloat64Column("PctMod")
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.AxonLayerByName(lnm)
		dt.AddFloat64Column(lnm, ly.Shp.Sizes)
	}
	dt.SetNumRows(0)
}

func (ss *Sim) ConfigTstTrlPlot(plt *plotcore.PlotEditor, dt *table.Table) *plotcore.PlotEditor {
	plt.Options.Title = "Attn Test Trial Plot"
	plt.Options.XAxis = "Trial"
	plt.SetTable(dt)
	plt.Options.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColumnOptions("Run", plotcore.Off, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColumnOptions("Trial", plotcore.Off, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColumnOptions("TrialName", plotcore.Off, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColumnOptions("Cycle", plotcore.Off, plotcore.FixMin, 0, plotcore.FixMax, 220)
	plt.SetColumnOptions("S1Act", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("S2Act", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("PctMod", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)

	for _, lnm := range ss.TstRecLays {
		cp := plt.SetColumnOptions(lnm, plotcore.Off, plotcore.FixMin, 0, plotcore.FixMax, 1)
		cp.TensorIndex = -1 // plot all
	}
	return plt
}

func (ss *Sim) TestStats() {
	dt := ss.TstTrlLog
	runix := table.NewIndexView(dt)
	spl := split.GroupBy(runix, []string{"Trial"})
	split.AggColumn(spl, "TrialName", stats.Mean)
	split.AggColumn(spl, "S1Act", stats.Mean)
	split.AggColumn(spl, "S2Act", stats.Mean)
	split.AggColumn(spl, "PctMod", stats.Mean)
	ss.TstStats = spl.AggsToTable(table.ColumnNameOnly)
	ss.TstRunLog = ss.TstStats.Clone()
	ss.TstRunPlot.SetTable(ss.TstRunLog)
}

func (ss *Sim) ConfigTstRunLog(dt *table.Table) {
	dt.SetMetaData("name", "TstRunLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.AddIntColumn("Trial")
	dt.AddStringColumn("TrialName")
	dt.AddFloat64Column("S1Act")
	dt.AddFloat64Column("S2Act")
	dt.AddFloat64Column("PctMod")
	dt.SetNumRows(0)
}

func (ss *Sim) ConfigTstRunPlot(plt *plotcore.PlotEditor, dt *table.Table) *plotcore.PlotEditor {
	plt.Options.Title = "Attn Test Run Plot"
	plt.Options.XAxis = "Trial"
	plt.SetTable(dt)
	plt.Options.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColumnOptions("Trial", plotcore.Off, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColumnOptions("TrialName", plotcore.Off, plotcore.FixMin, 0, plotcore.FloatMax, 0)
	plt.SetColumnOptions("S1Act", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("S2Act", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	plt.SetColumnOptions("PctMod", plotcore.On, plotcore.FixMin, 0, plotcore.FixMax, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Scene().Camera.Pose.Pos.Set(0, 1.14, 2.13)
	nv.Scene().Camera.LookAt(math32.Vec3(0, -0.14, 0), math32.Vec3(0, 1, 0))
	// nv.SetMaxRecs(300)
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Window {
	width := 1600
	height := 1200

	core.SetAppName("attn")
	core.SetAppAbout(`attn: This simulation illustrates how object recognition (ventral, what) and spatial (dorsal, where) pathways interact to produce spatial attention effects, and accurately capture the effects of brain damage to the spatial pathway. See <a href="https://github.com/emer/axon/blob/master/examples/attn_trn/README.md">README.md on GitHub</a>.</p>`)

	win := core.NewMainWindow("attn", "Attention", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := core.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := core.AddNewSplitView(mfr, "split")
	split.Dim = math32.X
	split.SetStretchMax()

	sv := views.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := core.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(plotcore.KiT_PlotView, "TstTrlPlot").(*plotcore.PlotEditor)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(plotcore.KiT_PlotView, "TstRunPlot").(*plotcore.PlotEditor)
	ss.TstRunPlot = ss.ConfigTstRunPlot(plt, ss.TstRunLog)

	split.SetSplits(.2, .8)

	tbar.AddAction(core.ActOpts{Label: "Init", Icon: icons.Update, Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(core.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TestTrialGUI()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Test Runs", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials x runs times for stats.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestRuns()
		}
	})

	tbar.AddSeparator("msep")

	tbar.AddAction(core.ActOpts{Label: "Lesion", Icon: "cut", Tooltip: "Lesion spatial pathways.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		views.CallMethod(ss, "Lesion", vp)
	})

	tbar.AddAction(core.ActOpts{Label: "Defaults", Icon: icons.Update, Tooltip: "Restore default parameters.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Defaults()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			core.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/attn_trn/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := core.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*core.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*core.Action)
	emen.Menu.AddCopyCutPaste(win)

		inQuitPrompt := false
		core.SetQuitReqFunc(func() {
			if inQuitPrompt {
				return
			}
			inQuitPrompt = true
			core.PromptDialog(vp, core.DlgOpts{Title: "Really Quit?",
				Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, core.AddOk, core.AddCancel,
				win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
					if sig == int64(core.DialogAccepted) {
						core.Quit()
					} else {
						inQuitPrompt = false
					}
				})
		})

		// core.SetQuitCleanFunc(func() {
		// 	fmt.Printf("Doing final Quit cleanup here..\n")
		// })

		inClosePrompt := false
		win.SetCloseReqFunc(func(w *core.Window) {
			if inClosePrompt {
				return
			}
			inClosePrompt = true
			core.PromptDialog(vp, core.DlgOpts{Title: "Really Close Window?",
				Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, core.AddOk, core.AddCancel,
				win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
					if sig == int64(core.DialogAccepted) {
						core.Quit()
					} else {
						inClosePrompt = false
					}
				})
		})

	win.SetCloseCleanFunc(func(w *core.Window) {
		go core.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = tree.Props{
	"CallMethods": tree.PropSlice{
		{"SaveWts", tree.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": tree.PropSlice{
				{"File Name", tree.Props{
					"ext": ".wts",
				}},
			},
		}},
		{"Lesion", tree.Props{
			"desc": "lesions given set of layers (or unlesions for NoLesion) and locations and number of units (Half = partial = 1/2 units, Full = both units)",
			"icon": "cut",
			"Args": tree.PropSlice{
				{"Layers", tree.Props{}},
				{"Locations", tree.Props{}},
				{"Units", tree.Props{}},
			},
		}},
	},
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGUI()
	win.StartEventLoop()
}
*/
