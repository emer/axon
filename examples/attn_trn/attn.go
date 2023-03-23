// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

/*
attn_trn: test of trn-based attention in basic V1, V2, LIP localist network with gabor inputs.

Tests for effects of contextual normalization in Reynolds & Heeger, 2009 framework.
In terms of differential sizes of attentional spotlight vs. stimulus size.
*/
package main

import (
	"fmt"
	"log"
	"strconv"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
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
type TestType int32

//go:generate stringer -type=TestType

var KiT_TestType = kit.Enums.AddEnum(TestTypeN, kit.NotBitFlag, nil)

func (ev TestType) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *TestType) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

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
			{Sel: "Prjn", Desc: "no learning",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0",
					"Prjn.SWt.Init.Sym":  "false", // for lesions, just in case
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
					"Layer.Act.Decay.Act":        "1",
					"Layer.Act.Decay.Glong":      "1",
					"Layer.Act.Decay.KNa":        "1",
					"Layer.Act.KNa.On":           "false", // turn off by default
					"Layer.Act.Noise.Dist":       "Gaussian",
					"Layer.Act.Noise.Var":        "0.002",
					"Layer.Act.Noise.Type":       "NoNoise", // "GeNoise",
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
					"Layer.Act.Attn.On":          "true",
					"Layer.Act.Attn.Min":         "0.2", // 0.5
					"Layer.Inhib.Topo.On":        "true",
					"Layer.Inhib.Topo.Width":     "4",
					"Layer.Inhib.Topo.Sigma":     "1.0",
					"Layer.Inhib.Topo.Gi":        "0.05",
					"Layer.Inhib.Topo.FF0":       "0.15",
					"Layer.Act.Noise.Dist":       "Gaussian",
					"Layer.Act.Noise.Var":        "0.02",    // .02
					"Layer.Act.Noise.Type":       "GeNoise", // "GeNoise",
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
					"Layer.Act.Dt.GTau":          "3",
				}},
			{Sel: ".BackPrjn", Desc: "weaker output",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: "#V2ToV2CTA", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.8
				}},
			{Sel: "#LIPToV2CTA", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5", // 0.5
				}},
			{Sel: ".Inhib", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: "#LIPToV2TA", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3
				}},
			{Sel: "#LIPToV2", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.0",
				}},
			{Sel: "#V2CTAToV2CTA", Desc: "lateral within V2CTA",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.0", // 0.4
				}},
			{Sel: "#V1ToV2", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.2",
				}},
		},
	}},
	{Name: "KNaAdapt", Desc: "Turn on KNa adaptation", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "KNa adapt on",
				Params: params.Params{
					"Layer.Act.KNa.On": "true",
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
	Cycles      int             `def:"200" desc:"number of cycles per trial"`
	Runs        int             `def:"10" desc:"number of runs to run to collect stats"`
	KNaAdapt    bool            `def:"true" desc:"sodium (Na) gated potassium (K) channels that cause neurons to fatigue over time"`
	Net         *axon.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Prjn3x3Skp1 *prjn.PoolTile  `view:"Standard same-to-same size topographic projection"`
	Prjn5x5Skp1 *prjn.PoolTile  `view:"Standard same-to-same size topographic projection"`
	Test        TestType        `desc:"select which type of test (input patterns) to use"`
	TstTrlLog   *etable.Table   `view:"no-inline" desc:"testing trial-level log data -- click to see record of network's response to each input"`
	TstRunLog   *etable.Table   `view:"no-inline" desc:"aggregated testing data"`
	TstStats    *etable.Table   `view:"no-inline" desc:"aggregate stats on testing data"`
	Params      params.Sets     `view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"`
	TestEnv     AttnEnv         `desc:"Testing environment -- manages iterating over testing"`
	Context     axon.Context    `desc:"axon timing parameters and state"`
	ViewOn      bool            `desc:"whether to update the network view while running"`
	ViewUpdt    axon.TimeScales `desc:"at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster"`
	AttnLay     string          `desc:"layer to measure attentional effects on"`
	TstRecLays  []string        `desc:"names of layers to record activations etc of during testing"`

	S1Act  float32 `desc:"max activation in center of stimulus 1 (attended, stronger)"`
	S2Act  float32 `desc:"max activation in center of stimulus 2 (ignored, weaker)"`
	PctMod float32 `desc:"percent modulation = (S1Act - S2Act) / S1Act"`

	// internal state - view:"-"
	Win        *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView    *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar    *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TstTrlPlot *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	TstRunPlot *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	ValsTsrs   map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	IsRunning  bool                        `view:"-" desc:"true if sim is running"`
	StopNow    bool                        `view:"-" desc:"flag to stop running"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Defaults()
	ss.Net = &axon.Network{}
	ss.Test = AttnSize
	ss.TstTrlLog = &etable.Table{}
	ss.TstRunLog = &etable.Table{}
	ss.TstStats = &etable.Table{}
	ss.Params = ParamSets
	ss.ViewOn = true
	ss.ViewUpdt = axon.AlphaCycle // axon.Cycle // axon.FastSpike
	ss.TstRecLays = []string{"V2"}

	ss.Prjn3x3Skp1 = prjn.NewPoolTile()
	ss.Prjn3x3Skp1.Size.Set(3, 3)
	ss.Prjn3x3Skp1.Skip.Set(1, 1)
	ss.Prjn3x3Skp1.Start.Set(-1, -1)
	ss.Prjn3x3Skp1.TopoRange.Min = 0.8 // note: none of these make a very big diff
	ss.Prjn3x3Skp1.GaussInPool.On = false
	ss.Prjn3x3Skp1.GaussFull.CtrMove = 0

	ss.Prjn5x5Skp1 = prjn.NewPoolTile()
	ss.Prjn5x5Skp1.Size.Set(5, 5)
	ss.Prjn5x5Skp1.Skip.Set(1, 1)
	ss.Prjn5x5Skp1.Start.Set(-2, -2)
	ss.Prjn5x5Skp1.TopoRange.Min = 0.8 // note: none of these make a very big diff
	ss.Prjn5x5Skp1.GaussInPool.On = false
	ss.Prjn5x5Skp1.GaussFull.CtrMove = 0

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

	one2one := prjn.NewOneToOne()
	pone2one := prjn.NewPoolOneToOne()
	circle := prjn.NewCircle()
	circle.Radius = 6
	circle.TopoWts = true
	circle.Sigma = 1

	// net.ConnectLayers(v2ct, v2p, one2one, emer.Forward)

	net.ConnectLayers(v1, v2, one2one, emer.Forward)
	net.ConnectLayers(v2, v2ta, ss.Prjn5x5Skp1, emer.Forward) // or v2cta
	// net.ConnectLayers(v2, v2, ss.Prjn5x5Skp1, emer.Inhib)
	// net.ConnectLayers(v2ta, v2ta, circle, emer.Inhib)
	// net.ConnectLayers(v2cta, v2cta, circle, emer.Lateral)
	// net.ConnectLayers(v2cta, v2ta, ss.Prjn5x5Skp1, emer.Forward)
	net.ConnectLayers(lip, v2, pone2one, emer.Back)
	net.ConnectLayers(lip, v2cta, pone2one, emer.Back) // ss.Prjn5x5Skp1
	net.ConnectLayers(lip, v2ta, pone2one, emer.Back)  // ss.Prjn5x5Skp1 was ponetoone

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

func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
	switch viewUpdt {
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
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope, and calls TrainStats()
func (ss *Sim) ThetaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.ViewUpdt

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
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.Context.NewPhase()
	if viewUpdt == axon.Phase {
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
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
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
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func (ss *Sim) StimMaxAct(stm *Stim, lnm string) float32 {
	ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	sz := evec.Vec2i{ly.Shp.Dim(1), ly.Shp.Dim(0)}
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
			max = mat32.Max(max, pl.Inhib.Act.Max)
		}
	}
	return max
}

func (ss *Sim) StimAvgAct(stm *Stim, lnm string) float32 {
	ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	sz := evec.Vec2i{ly.Shp.Dim(1), ly.Shp.Dim(0)}
	pt := stm.PosXY(sz)
	cx := int(mat32.Round(pt.X)) - 1
	cy := int(mat32.Round(pt.Y)) - 1
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
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
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
		ss.PctMod = mat32.Max(ss.PctMod, 0)
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

// SaveWts saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWts(filename gi.FileName) {
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
		if ss.ViewUpdt > axon.AlphaCycle {
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
	// spo.Params.SetParamByName("Prjn.PrjnScale.Rel", fmt.Sprintf("%g", ss.SpatToObj))
	// vsp := ss.Params.SetByName("Base").SheetByName("Network").SelByName("#V1ToSpat1")
	// vsp.Params.SetParamByName("Prjn.PrjnScale.Rel", fmt.Sprintf("%g", ss.V1ToSpat1))

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

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	row := dt.Rows
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	trl := ss.TestEnv.Trial.Cur
	rn := ss.TestEnv.Run.Cur

	dt.SetCellFloat("Run", row, float64(rn))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.String())
	dt.SetCellFloat("Cycle", row, float64(ss.Context.Cycle))
	dt.SetCellFloat("S1Act", row, float64(ss.S1Act))
	dt.SetCellFloat("S2Act", row, float64(ss.S2Act))
	dt.SetCellFloat("PctMod", row, float64(ss.PctMod))

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValsTsr(lnm)
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		ly.UnitValsTensor(tsr, "Act")
		dt.SetCellTensor(lnm, row, tsr)
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Cycle", etensor.INT64, nil, nil},
		{"S1Act", etensor.FLOAT64, nil, nil},
		{"S2Act", etensor.FLOAT64, nil, nil},
		{"PctMod", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Attn Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	plt.Params.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 220)
	plt.SetColParams("S1Act", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("S2Act", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctMod", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.TstRecLays {
		cp := plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		cp.TensorIdx = -1 // plot all
	}
	return plt
}

func (ss *Sim) TestStats() {
	dt := ss.TstTrlLog
	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Trial"})
	split.Agg(spl, "TrialName", agg.AggMean)
	split.Agg(spl, "S1Act", agg.AggMean)
	split.Agg(spl, "S2Act", agg.AggMean)
	split.Agg(spl, "PctMod", agg.AggMean)
	ss.TstStats = spl.AggsToTable(etable.ColNameOnly)
	ss.TstRunLog = ss.TstStats.Clone()
	ss.TstRunPlot.SetTable(ss.TstRunLog)
}

func (ss *Sim) ConfigTstRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstRunLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"S1Act", etensor.FLOAT64, nil, nil},
		{"S2Act", etensor.FLOAT64, nil, nil},
		{"PctMod", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Attn Test Run Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	plt.Params.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("S1Act", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("S2Act", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctMod", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Scene().Camera.Pose.Pos.Set(0, 1.14, 2.13)
	nv.Scene().Camera.LookAt(mat32.Vec3{0, -0.14, 0}, mat32.Vec3{0, 1, 0})
	// nv.SetMaxRecs(300)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("attn")
	gi.SetAppAbout(`attn: This simulation illustrates how object recognition (ventral, what) and spatial (dorsal, where) pathways interact to produce spatial attention effects, and accurately capture the effects of brain damage to the spatial pathway. See <a href="https://github.com/emer/axon/blob/master/examples/attn_trn/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("attn", "Attention", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstRunPlot").(*eplot.Plot2D)
	ss.TstRunPlot = ss.ConfigTstRunPlot(plt, ss.TstRunLog)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TestTrialGUI()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Runs", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials x runs times for stats.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestRuns()
		}
	})

	tbar.AddSeparator("msep")

	tbar.AddAction(gi.ActOpts{Label: "Lesion", Icon: "cut", Tooltip: "Lesion spatial pathways.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		giv.CallMethod(ss, "Lesion", vp)
	})

	tbar.AddAction(gi.ActOpts{Label: "Defaults", Icon: "update", Tooltip: "Restore default parameters.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Defaults()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/attn_trn/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	/*
		inQuitPrompt := false
		gi.SetQuitReqFunc(func() {
			if inQuitPrompt {
				return
			}
			inQuitPrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
				Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inQuitPrompt = false
					}
				})
		})

		// gi.SetQuitCleanFunc(func() {
		// 	fmt.Printf("Doing final Quit cleanup here..\n")
		// })

		inClosePrompt := false
		win.SetCloseReqFunc(func(w *gi.Window) {
			if inClosePrompt {
				return
			}
			inClosePrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
				Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inClosePrompt = false
					}
				})
		})
	*/

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWts", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts",
				}},
			},
		}},
		{"Lesion", ki.Props{
			"desc": "lesions given set of layers (or unlesions for NoLesion) and locations and number of units (Half = partial = 1/2 units, Full = both units)",
			"icon": "cut",
			"Args": ki.PropSlice{
				{"Layers", ki.Props{}},
				{"Locations", ki.Props{}},
				{"Units", ki.Props{}},
			},
		}},
	},
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
