// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// lvis explores how a hierarchy of areas in the ventral stream
// of visual processing (up to inferotemporal (IT) cortex) can produce
// robust object recognition that is invariant to changes in position,
// size, etc of retinal input images.
package deepvision

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"slices"

	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/metric"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensorcore"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/decoder"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/paths"
)

// Modes are the looping modes (Stacks) for running and statistics.
type Modes int32 //enums:enum
const (
	Train Modes = iota
	Test
	NovelTrain
)

// Levels are the looping levels for running and statistics.
type Levels int32 //enums:enum
const (
	Cycle Levels = iota
	Trial
	Epoch
	Run
)

// StatsPhase is the phase of stats processing for given mode, level.
// Accumulated values are reset at Start, added each Step.
type StatsPhase int32 //enums:enum
const (
	Start StatsPhase = iota
	Step
)

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config *Config `new-window:"+"`

	// Net is the network: click to view / edit parameters for layers, paths, etc.
	Net *axon.Network `new-window:"+" display:"no-inline"`

	// Params manages network parameter setting.
	Params axon.Params `display:"inline"`

	// Paths are all the specialized pathways for the network.
	Paths Paths `new-window:"+" display:"no-inline"`

	// Decoder is used as a comparison vs. the Output layer.
	Decoder decoder.SoftMax

	// Loops are the control loops for running the sim, in different Modes
	// across stacks of Levels.
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// Envs provides mode-string based storage of environments.
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// TrainUpdate has Train mode netview update parameters.
	TrainUpdate axon.NetViewUpdate `display:"inline"`

	// TestUpdate has Test mode netview update parameters.
	TestUpdate axon.NetViewUpdate `display:"inline"`

	// Root is the root tensorfs directory, where all stats and other misc sim data goes.
	Root *tensorfs.Node `display:"-"`

	// Stats has the stats directory within Root.
	Stats *tensorfs.Node `display:"-"`

	// Current has the current stats values within Stats.
	Current *tensorfs.Node `display:"-"`

	// StatFuncs are statistics functions called at given mode and level,
	// to perform all stats computations. phase = Start does init at start of given level,
	// and all initialization / configuration (called during Init too).
	StatFuncs []func(mode Modes, level Levels, phase StatsPhase) `display:"-"`

	// GUI manages all the GUI elements
	GUI egui.GUI `display:"-"`

	// RandSeeds is a list of random seeds to use for each run.
	RandSeeds randx.Seeds `display:"-"`
}

func (ss *Sim) SetConfig(cfg *Config) { ss.Config = cfg }
func (ss *Sim) Body() *core.Body      { return ss.GUI.Body }

func (ss *Sim) ConfigSim() {
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.Paths.Defaults()
	ss.Net = axon.NewNetwork(ss.Config.Name)
	ss.Params.Config(LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag, reflect.ValueOf(ss))
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	if ss.Config.GPU {
		gpu.SelectAdapter = ss.Config.Run.GPUDevice
		axon.GPUInit()
		axon.UseGPU = true
	}
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLoops()
	ss.ConfigStats()
	// if ss.Config.Run.GPU {
	// 	fmt.Println(axon.GPUSystem.Vars().StringDoc())
	// }
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
	ss.RSAInit()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	newEnv := (len(ss.Envs) == 0)
	ndata := ss.Config.Run.NData
	var objdata *table.Table

	for di := 0; di < ndata; di++ {
		var trn, tst *Obj3DSacEnv
		if newEnv {
			trn = &Obj3DSacEnv{}
			tst = &Obj3DSacEnv{}
		} else {
			trn = ss.Envs.ByModeDi(Train, di).(*Obj3DSacEnv)
			tst = ss.Envs.ByModeDi(Test, di).(*Obj3DSacEnv)
		}

		trn.Name = env.ModeDi(Train, di)
		trn.Defaults()
		trn.NData = ndata
		trn.Di = di
		trn.RandSeed = 73 + int64(di)*73
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
		}
		trn.Config()
		if di == 0 {
			trn.OpenTable()
			objdata = trn.Table
		} else {
			trn.Table = objdata
		}

		tst.Name = env.ModeDi(Test, di)
		tst.Defaults()
		tst.NData = ndata
		tst.Di = di
		tst.RandSeed = 181 + int64(di)*181
		// tst.Test = true
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(tst, ss.Config.Env.Env)
		}
		tst.Config()
		tst.Table = objdata

		// if ss.Config.Run.MPI {
		// 	if ss.Config.Debug {
		// 		mpi.Printf("Did Env MPIAlloc\n")
		// 	}
		// 	trn.MPIAlloc()
		// 	tst.MPIAlloc()
		// }

		trn.Init(0)
		tst.Init(0)

		ss.Envs.Add(trn, tst)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.Context().SetThetaCycles(int32(ss.Config.Run.Cycles)).
		SetPlusCycles(int32(ss.Config.Run.PlusCycles)).
		SetSlowInterval(int32(ss.Config.Run.SlowInterval)).
		SetAdaptGiInterval(int32(ss.Config.Run.AdaptGiInterval))
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	space := float32(4)
	// one2one := paths.NewOneToOne()
	full := paths.NewFull()
	pool1to1 := paths.NewPoolOneToOne()
	pts := &ss.Paths
	rndcut := paths.NewUniformRand()
	rndcut.PCon = 0.1
	_ = rndcut

	// trn := ss.Envs.ByModeDi(Train, 0).(*Obj3DSacEnv)

	sample2 := func(ly *axon.Layer) {
		ly.SetSampleShape(emer.CenterPoolIndexes(ly, 2), emer.CenterPoolShape(ly, 2))
	}

	// LIP network
	v1m := net.AddLayer4D("V1m", axon.InputLayer, 8, 8, 5, 4).AddClass("V1m")
	v1h := net.AddLayer4D("V1h", axon.InputLayer, 16, 16, 5, 4).AddClass("V1h")

	sample2(v1m)
	sample2(v1h)

	eyepos := net.AddLayer2D("EyePos", axon.InputLayer, 21, 21).AddClass("PopCode")
	sacplan := net.AddLayer2D("SacPlan", axon.InputLayer, 11, 11).AddClass("PopCode")
	sac := net.AddLayer2D("Saccade", axon.InputLayer, 11, 11).AddClass("PopCode")
	objvel := net.AddLayer2D("ObjVel", axon.InputLayer, 11, 11).AddClass("PopCode")

	// 2,2 for mt > 1,1
	mtpos := net.AddLayer4D("MTpos", axon.SuperLayer, 8, 8, 2, 2).AddClass("MTpos")
	mtposP := net.AddPulvForLayer(mtpos, space).AddClass("MTpos")

	lip, lipCT := net.AddSuperCT4D("LIP", "LIPCtxt", 8, 8, 4, 4, space, pts.PT3x3Skp1) // 4x4 == 5x5
	// net.ConnectCTSelf(lipCT, full, "LIPSelf") // maint + ctself: bad
	// net.ConnectLayers(lipCT, lipCT, pts.PT3x3Skp1, axon.CTCtxtPath).AddClass("CTSelfCtxt")

	sample2(lip)
	sample2(lipCT)

	net.ConnectLayers(v1m, mtpos, pool1to1, axon.ForwardPath).AddClass("Fixed")
	net.ConnectLayers(mtpos, lip, pool1to1, axon.ForwardPath).AddClass("Fixed")

	net.ConnectToPulv(lip, lipCT, mtposP, full, pool1to1, "") // full >> pts.PT3x3Skp1

	// these are important for good performance on pure LIP version:
	net.ConnectLayers(eyepos, lip, full, axon.ForwardPath)
	net.ConnectLayers(sacplan, lip, full, axon.ForwardPath)
	net.ConnectLayers(objvel, lip, full, axon.ForwardPath)

	net.ConnectLayers(eyepos, lipCT, full, axon.ForwardPath)
	net.ConnectLayers(sac, lipCT, full, axon.ForwardPath)
	net.ConnectLayers(objvel, lipCT, full, axon.ForwardPath)

	net.ConnectLayers(sac, lip, full, axon.ForwardPath)

	// 	Positioning

	v1h.PlaceRightOf(v1m, space)

	mtpos.PlaceAbove(v1m)
	mtposP.PlaceRightOf(mtpos, space)
	lip.PlaceBehind(mtpos, space*2)
	lipCT.PlaceBehind(lip, space)

	eyepos.PlaceRightOf(mtposP, space)
	sacplan.PlaceBehind(eyepos, space)
	sac.PlaceBehind(sacplan, space)
	objvel.PlaceBehind(sac, space)

	var v1mP, v2, v2CT, v3, v3CT, dp, dpCT, v3P, v4, v4CT, teo, teoCT, v4P, te, teCT, teoP *axon.Layer

	//////// V2
	if !ss.Config.Run.V2Plus {
		goto build
	}
	v1mP = net.AddPulvForLayer(v1m, space).AddClass("V1m")
	v2, v2CT = net.AddSuperCT4D("V2", "", 8, 8, 10, 10, space, pts.PT3x3Skp1) // 3x3 >> p1to1
	sample2(v2)
	sample2(v2CT)

	net.ConnectToPulv(v2, v2CT, v1mP, pts.PT3x3Skp1, pts.PT3x3Skp1, "FromV1mP") // 3x3 >> p1to1

	// old has v2selfct 3x3s1, not good here:
	// net.ConnectLayers(v2CT, v2CT, pts.PT3x3Skp1, axon.CTCtxtPath).AddClass("CTSelfCtxt")

	net.ConnectLayers(v1m, v2, pts.PT3x3Skp1, axon.ForwardPath).AddClass("V1V2")
	net.ConnectLayers(v1h, v2, pts.PT4x4Skp2, axon.ForwardPath).AddClass("V1V2")

	// net.ConnectLayers(v2CT, lipCT, pool1to1, axon.ForwardPath).AddClass("FwdWeak") // harmful
	net.ConnectLayers(lipCT, v2CT, pool1to1, axon.BackPath) // critical!

	net.ConnectLayers(v2, lip, pool1to1, axon.ForwardPath).AddClass("FwdWeak") // good later
	net.ConnectLayers(lip, v2, pool1to1, axon.BackPath)                        // helpful

	v2.PlaceAbove(v1m)
	mtpos.PlaceAbove(v2)

	//////// V3
	if !ss.Config.Run.V3Plus {
		goto build
	}
	v3, v3CT = net.AddSuperCT4D("V3", "", 4, 4, 10, 10, space, pts.PT3x3Skp1)
	sample2(v3)
	sample2(v3CT)

	// orig 4x4
	net.ConnectToPulv(v3, v3CT, v1mP, pts.PT4x4Skp2Recip, pts.PT4x4Skp2, "FromV1mP")

	// old has v3selfct 3x3s1: not bad up to .2, but no benefit
	// net.ConnectLayers(v3CT, v3CT, pts.PT3x3Skp1, axon.CTCtxtPath).AddClass("CTSelfCtxt")

	// orig 4x4skp2
	net.ConnectLayers(v2, v3, pts.PT4x4Skp2, axon.ForwardPath)
	net.ConnectLayers(v3, v2, pts.PT4x4Skp2Recip, axon.BackPath)

	// net.ConnectLayers(v3CT, lipCT, pts.PT2x2Skp2Recip, axon.ForwardPath).AddClass("FwdWeak") // bad for lip
	// net.ConnectLayers(lipCT, v3CT, pts.PT2x2Skp2, axon.BackPath) // bad; 2x2 orig

	// missing in orig, slower at start but needed for later:
	net.ConnectLayers(v2CT, v3CT, pts.PT4x4Skp2, axon.ForwardPath).AddClass("FwdWeak")

	// todo: strong .5 in orig
	// net.ConnectLayers(v3CT, v2CT, pts.PT4x4Skp2Recip, axon.BackPath) // yes top-down CT
	// orig has a "leak" from super -> CT here: (2x2 == 4x4) --
	// good: mostly prevents "sag" at end
	net.ConnectLayers(v3, v2CT, pts.PT2x2Skp2Recip, axon.BackPath)

	// orig 2x2:
	net.ConnectLayers(v3, lip, pts.PT2x2Skp2Recip, axon.ForwardPath).AddClass("FwdWeak")
	net.ConnectLayers(lip, v3, pts.PT2x2Skp2, axon.BackPath)

	net.ConnectLayers(v1m, v3, rndcut, axon.ForwardPath).AddClass("V1SC")   // shortcut!
	net.ConnectLayers(v1m, v3CT, rndcut, axon.ForwardPath).AddClass("V1SC") // shortcut! // CT def good

	v3.PlaceRightOf(v2, space)

	//////// DP
	if ss.Config.Run.DP { // now a significant benefit in V1mP performance!

		dp, dpCT = net.AddSuperCT4D("DP", "", 4, 4, 10, 10, space, pts.PT3x3Skp1)
		sample2(dp)
		sample2(dpCT)

		net.ConnectToPulv(dp, dpCT, v1mP, pts.PT4x4Skp2Recip, pts.PT4x4Skp2, "FromV1mP")

		// todo test:
		// net.ConnectLayers(dpCT, dpCT, pts.PT3x3Skp1, axon.CTCtxtPath).AddClass("CTSelfCtxt")
		// maint is maybe better:
		// net.ConnectCTSelf(dpCT, pts.PT3x3Skp1, "DPCTSelf")

		// net.ConnectLayers(v2, dp, pts.PT4x4Skp2, axon.ForwardPath) // better without
		// net.ConnectLayers(dp, v2, pts.PT4x4Skp2Recip, axon.BackPath)

		net.ConnectLayers(v3, dp, pts.PT3x3Skp1, axon.ForwardPath) // v3 > v2 connectivity
		net.ConnectLayers(dp, v3, pts.PT3x3Skp1, axon.BackPath)

		v3P = net.AddPulvForLayer(v3, space).AddClass("V3")
		net.ConnectToPulv(dp, dpCT, v3P, pts.PT3x3Skp1, pts.PT3x3Skp1, "FromV3P")
		net.ConnectLayers(v2CT, v3P, pts.PT4x4Skp2, axon.ForwardPath) // fwd CT, but not recip!

		// no DP <-> LIP?
		// no FF CT -> CT?
		// net.ConnectLayers(v2CT, dpCT, pts.PT4x4Skp2, axon.ForwardPath).AddClass("FwdWeak")
		// net.ConnectLayers(v3, dp, pts.PT3x3Skp1, axon.ForwardPath).AddClass("FwdWeak")

		// net.ConnectLayers(dpCT, v2CT, pts.PT4x4Skp2Recip, axon.BackPath) // tiny bit worse

		// leak from super to CT:
		net.ConnectLayers(dp, v2CT, pts.PT2x2Skp2Recip, axon.BackPath) // strong .5 in orig
		net.ConnectLayers(dp, v3CT, pts.PT3x3Skp1, axon.BackPath)

		// net.ConnectLayers(dpCT, v3CT, full, axon.BackPath)

		net.ConnectLayers(v1m, dp, rndcut, axon.ForwardPath).AddClass("V1SC")   // shortcut!
		net.ConnectLayers(v1m, dpCT, rndcut, axon.ForwardPath).AddClass("V1SC") // shortcut! // CT good

		v3P.PlaceBehind(v3CT, space)
		dp.PlaceBehind(v3P, space)
	}

	//////// V4
	if !ss.Config.Run.V4Plus {
		goto build
	}
	v4, v4CT = net.AddSuperCT4D("V4", "", 4, 4, 10, 10, space, pts.PT3x3Skp1) // 3x3 >> p1to1?? orig 1to1
	sample2(v4)
	sample2(v4CT)

	net.ConnectToPulv(v4, v4CT, v1mP, pts.PT4x4Skp2Recip, pts.PT4x4Skp2, "FromV1mP") // 3x3 >> p1to1??
	// no V4 -> v1mP: sig worse overall
	// net.ConnectLayers(v1mP, v4, pts.PT4x4Skp2, axon.BackPath).AddClass("FromPulv", "FromV1mP")
	// net.ConnectLayers(v1mP, v4CT, pts.PT4x4Skp2, axon.BackPath).AddClass("FromPulv", "FromV1mP")

	// orig has v4selfct 3x3s1
	// net.ConnectLayers(v4CT, v4CT, pts.PT3x3Skp1, axon.CTCtxtPath).AddClass("CTSelfCtxt")
	// maint is maybe better:
	net.ConnectCTSelf(v4CT, pts.PT3x3Skp1, "V4CTSelf")

	net.ConnectLayers(v2, v4, pts.PT4x4Skp2, axon.ForwardPath)
	net.ConnectLayers(v4, v2, pts.PT4x4Skp2Recip, axon.BackPath)

	// not useful:
	// net.ConnectLayers(v4, v3, pts.PT3x3Skp1, axon.BackPath) // v4 -> v3 but not v3 -> v4

	// no V4 <-> LIP
	// no FF CT -> CT?
	// net.ConnectLayers(v2CT, v4CT, pts.PT4x4Skp2, axon.ForwardPath).AddClass("FwdWeak")
	// net.ConnectLayers(v3, v4, pts.PT3x3Skp1, axon.ForwardPath).AddClass("FwdWeak")

	// net.ConnectLayers(v4CT, v2CT, pts.PT4x4Skp2Recip, axon.BackPath) // tiny bit worse

	// leak from super to CT:
	net.ConnectLayers(v4, v2CT, pts.PT2x2Skp2Recip, axon.BackPath) // strong .5 in orig

	net.ConnectLayers(v1m, v4, rndcut, axon.ForwardPath).AddClass("V1SC")   // shortcut!
	net.ConnectLayers(v1m, v4CT, rndcut, axon.ForwardPath).AddClass("V1SC") // shortcut!

	v4.PlaceRightOf(v3, space)

	//////// TEO
	if !ss.Config.Run.TEOPlus {
		goto build
	}
	teo, teoCT = net.AddSuperCT4D("TEO", "", 4, 4, 10, 10, space, pool1to1)
	sample2(teo)
	sample2(teoCT)

	net.LateralConnectLayer(teo, pool1to1).AddClass("TEOSelfMaint")

	// orig has teoselfct 3x3s1 -- todo try, also one with maint
	// net.ConnectLayers(teoCT, teoCT, pool1to1, axon.CTCtxtPath).AddClass("CTSelfCtxt")
	net.ConnectCTSelf(teoCT, pool1to1, "TEOCTSelf") // 1to1? blows up later

	net.ConnectLayers(v4, teo, pts.PT3x3Skp1, axon.ForwardPath)
	net.ConnectLayers(teo, v4, pts.PT3x3Skp1, axon.BackPath)

	net.ConnectLayers(teo, v3, pts.PT3x3Skp1, axon.BackPath) // teo -> v3 but not v3 -> teo

	// net.ConnectLayers(v4CT, teoCT, pts.PT3x3Skp1, axon.ForwardPath).AddClass("FwdWeak")

	// net.ConnectLayers(teoCT, v2CT, pts.PT4x4Skp2Recip, axon.BackPath) // not needed..
	net.ConnectLayers(teoCT, v4CT, pts.PT4x4Skp2Recip, axon.BackPath)

	v4P = net.AddPulvForLayer(v4, space).AddClass("V4")
	net.ConnectToPulv(teo, teoCT, v4P, pts.PT4x4Skp2Recip, pts.PT4x4Skp2, "FromV4P")

	// orig has a "leak" from super -> CT here, helps stabilize reps
	// net.ConnectLayers(teo, v2CT, pts.PT4x4Skp2Recip, axon.BackPath) // maybe not
	// net.ConnectLayers(teo, v3CT, pts.PT3x3Skp1, axon.BackPath)
	// net.ConnectLayers(teo, v4CT, pts.PT3x3Skp1, axon.BackPath) // no diff really

	net.ConnectLayers(v1m, teo, rndcut, axon.ForwardPath).AddClass("V1SC")   // shortcut!
	net.ConnectLayers(v1m, teoCT, rndcut, axon.ForwardPath).AddClass("V1SC") // shortcut!

	v4P.PlaceBehind(v4CT, space)
	teo.PlaceRightOf(eyepos, space)

	//////// TE
	if !ss.Config.Run.TE {
		goto build
	}
	te, teCT = net.AddSuperCT4D("TE", "", 2, 2, 10, 10, space, pool1to1)
	sample2(te)
	sample2(teCT)

	net.LateralConnectLayer(te, pool1to1).AddClass("TESelfMaint")

	// net.ConnectLayers(teCT, teCT, pool1to1, axon.CTCtxtPath).AddClass("CTSelfCtxt")
	net.ConnectCTSelf(teCT, pool1to1, "TECTSelf") // maint plus does better

	net.ConnectLayers(teo, te, full, axon.ForwardPath)
	net.ConnectLayers(te, teo, full, axon.BackPath)

	net.ConnectLayers(te, v4, full, axon.BackPath) // te -> v3 but not v3 -> te

	net.ConnectLayers(teCT, v4CT, full, axon.BackPath)
	net.ConnectLayers(teCT, teoCT, full, axon.BackPath)

	teoP = net.AddPulvForLayer(teo, space).AddClass("TEO")
	net.ConnectToPulv(te, teCT, teoP, full, full, "FromTEOP")

	net.ConnectLayers(v1m, te, rndcut, axon.ForwardPath).AddClass("V1SC")   // shortcut!
	net.ConnectLayers(v1m, teCT, rndcut, axon.ForwardPath).AddClass("V1SC") // shortcut!

	teoP.PlaceBehind(teoCT, space)
	te.PlaceRightOf(teo, space)

build:
	net.Build()
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	ss.InitWeights(net)

	mpi.Println(net.SizeReport(false))

	// adding each additional layer type improves decoding..
	// layers := []emer.Layer{v4f16, v4f8, teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8}
	// layers := []emer.Layer{out}
	// todo: decoder
	// ss.Decoder.InitLayer(len(trn.Images.Cats), layers)
	// ss.Decoder.Lrate = 0.05 // 0.05 > 0.1 > 0.2 for larger number of objs!
	// if ss.Config.Run.MPI {
	// 	ss.Decoder.Comm = ss.Comm
	// }
}

func (ss *Sim) SetTopoScales(net *axon.Network, send, recv string, pooltile *paths.PoolTile) {
	return // TODO:
	// slay := net.LayerByName(send)
	// rlay := net.LayerByName(recv)
	// pt, _ := rlay.RecvPathBySendName(send)
	// scales := &tensor.Float32{}
	// pooltile.TopoWeights(&slay.Shape, &rlay.Shape, scales)
	// TODO: this function does not exist:
	// pt.SetScalesRPool(scales)
}

func (ss *Sim) InitWeights(net *axon.Network) {
	// net.InitTopoScales() //  sets all wt scales
	pts := &ss.Paths

	// these are not set automatically b/c prjn is Full, not PoolTile
	ss.SetTopoScales(net, "EyePos", "LIP", pts.PTGaussTopo)
	ss.SetTopoScales(net, "SacPlan", "LIP", pts.PTSigTopo)
	ss.SetTopoScales(net, "ObjVel", "LIP", pts.PTSigTopo)

	ss.SetTopoScales(net, "LIP", "LIPCT", pts.PT3x3Skp1)
	ss.SetTopoScales(net, "EyePos", "LIPCT", pts.PTGaussTopo)
	ss.SetTopoScales(net, "Saccade", "LIPCT", pts.PTSigTopo)
	ss.SetTopoScales(net, "ObjVel", "LIPCT", pts.PTSigTopo)

	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.Script = ss.Config.Params.Script
	ss.Params.ApplyAll(ss.Net)
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.SetRunName()
	ss.InitRandSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.ApplyParams()
	ss.StatsInit()
	ss.NewRun()
	ss.TrainUpdate.RecordSyns()
	ss.TrainUpdate.Update(Train, Trial)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// NetViewUpdater returns the NetViewUpdate for given mode.
func (ss *Sim) NetViewUpdater(mode enums.Enum) *axon.NetViewUpdate {
	if mode.Int64() == Train.Int64() {
		return &ss.TrainUpdate
	}
	return &ss.TestUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trials := int(math32.IntMultipleGE(float32(ss.Config.Run.Trials), float32(ss.Config.Run.NData)))
	cycles := ss.Config.Run.Cycles
	plusPhase := ss.Config.Run.PlusCycles

	ls.AddStack(Train, Trial).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, cycles-plusPhase, cycles-1, Cycle, Trial, Train)

	ls.Stacks[Train].OnInit.Add("Init", ss.Init)

	ls.AddOnStartToLoop(Trial, "ApplyInputs", func(mode enums.Enum) {
		ss.ApplyInputs(mode.(Modes))
	})

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	trainEpoch := ls.Loop(Train, Epoch)

	trainEpoch.OnStart.Add("SaveWeightsAt", func() {
		epc := trainEpoch.Counter.Cur
		for _, se := range ss.Config.Log.SaveWeightsAt {
			if epc != se {
				continue
			}
			ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, epc)
			axon.SaveWeights(ss.Net, ctrString, ss.RunName())
			ss.RSASaveRActs("RSARActs_" + ss.RunName() + "_" + ctrString + ".tar.gz")
		}
	})

	trainEpoch.OnStart.Add("TurnOnAdaptGi", func() {
		epc := trainEpoch.Counter.Cur
		if epc != 250 {
			return
		}
		lays := ss.Net.LayersByType(axon.CTLayer)
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			if ly == nil {
				continue
			}
			ly.Params.Inhib.ActAvg.AdaptGi.SetBool(true)
		}
		axon.ToGPUParams()
		fmt.Println("At epoch:", epc, "turned on AdaptGi in CT layers")
	})

	// trainEpoch.OnStart.Add("TestAtInterval", func() {
	// 	if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
	// 		ss.TestAll()
	// 	}
	// })

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.RunName())
		if ss.Config.Log.SaveWeights {
			ss.RSASaveRActs("RSARActs_" + ss.RunName() + "_" + ctrString + ".tar.gz")
		}
	})

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
		ls.Stacks[Test].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment for given mode.
// Any other start-of-trial logic can also be put here.
func (ss *Sim) ApplyInputs(mode Modes) {
	net := ss.Net
	ctx := ss.Net.Context()
	ndata := int(net.Context().NData)
	curModeDir := ss.Current.Dir(mode.String())
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	net.InitExt()
	for di := uint32(0); di < ctx.NData; di++ {
		ev := ss.Envs.ByModeDi(mode, int(di)).(*Obj3DSacEnv)
		ev.Step()
		curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), int(di))
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			st := ev.State(ly.Name)
			if st != nil {
				ly.ApplyExt(uint32(di), st)
			}
		}
	}
	net.ApplyExts()
	ss.UpdateImage()
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	ss.InitRandSeed(ss.Loops.Loop(Train, Run).Counter.Cur)
	for di := 0; di < int(ctx.NData); di++ {
		ss.Envs.ByModeDi(Train, di).Init(0)
		ss.Envs.ByModeDi(Test, di).Init(0)
	}
	ctx.Reset()
	ss.ApplyParams() // must reapply due to changes @250
	ss.Net.InitWeights()
	if ss.Config.Run.StartWeights != "" {
		ss.Net.OpenWeightsJSON(core.Filename(ss.Config.Run.StartWeights))
		mpi.Printf("Starting with initial weights from: %s\n", ss.Config.Run.StartWeights)
	}
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ctx := ss.Net.Context()
	for di := 0; di < int(ctx.NData); di++ {
		ss.Envs.ByModeDi(Test, di).Init(0)
	}
	ss.Loops.ResetAndRun(Test)
	ss.Loops.Mode = Train // important because this is called from Train Run: go back.
}

//////// Stats

// AddStat adds a stat compute function.
func (ss *Sim) AddStat(f func(mode Modes, level Levels, phase StatsPhase)) {
	ss.StatFuncs = append(ss.StatFuncs, f)
}

// StatsStart is called by Looper at the start of given level, for each iteration.
// It needs to call RunStats Start at the next level down.
// e.g., each Epoch is the start of the full set of Trial Steps.
func (ss *Sim) StatsStart(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level <= Trial {
		return
	}
	ss.RunStats(mode, level-1, Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level == Cycle {
		return
	}
	ss.RunStats(mode, level, Step)
	tensorfs.DirTable(axon.StatsNode(ss.Stats, mode, level), nil).WriteToLog()
}

// RunStats runs the StatFuncs for given mode, level and phase.
func (ss *Sim) RunStats(mode Modes, level Levels, phase StatsPhase) {
	for _, sf := range ss.StatFuncs {
		sf(mode, level, phase)
	}
	if phase == Step && ss.GUI.Tabs != nil {
		nm := mode.String() + " " + level.String() + " Plot"
		ss.GUI.Tabs.AsLab().GoUpdatePlot(nm)
	}
}

// SetRunName sets the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) SetRunName() string {
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Current.StringValue("RunName", 1).SetString1D(runName, 0)
	return runName
}

// RunName returns the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) RunName() string {
	return ss.Current.StringValue("RunName", 1).String1D(0)
}

// StatsInit initializes all the stats by calling Start across all modes and levels.
func (ss *Sim) StatsInit() {
	for md, st := range ss.Loops.Stacks {
		mode := md.(Modes)
		for _, lev := range st.Order {
			level := lev.(Levels)
			if level == Cycle {
				continue
			}
			ss.RunStats(mode, level, Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
		// ev := ss.Envs.ByMode(Train).(*Obj3DSacEnv)
		// tbs.TensorGrid("Image", &ev.Vis.ImgTsr)
		tbs.SelectTabIndex(idx)
	}
}

// ConfigStats handles configures functions to do all stats computation
// in the tensorfs system.
func (ss *Sim) ConfigStats() {
	net := ss.Net
	ss.Stats = ss.Root.Dir("Stats")
	ss.Current = ss.Stats.Dir("Current")

	ss.SetRunName()

	// last arg(s) are levels to exclude
	counterFunc := axon.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		counterFunc(mode, level, phase == Start)
	})
	runNameFunc := axon.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		runNameFunc(mode, level, phase == Start)
	})
	trialNameFunc := axon.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trialNameFunc(mode, level, phase == Start)
	})

	perTrlFunc := axon.StatPerTrialMSec(ss.Stats, Train, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		perTrlFunc(mode, level, phase == Start)
	})

	corSimFunc := ss.StatCorSim()
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		corSimFunc(mode, level, phase)
	})

	prevCorFunc := ss.StatPrevCorSim()
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		prevCorFunc(mode, level, phase)
	})

	if ss.Config.Run.V2Plus {
		slays := net.LayersByType(axon.SuperLayer)
		slays = slices.DeleteFunc(slays, func(s string) bool {
			return s == "MTpos"
		})
		slays = append(slays, "V1m")
		rsaFunc := ss.StatRSA(slays...)
		ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
			rsaFunc(mode, level, phase)
		})
	}

	lays := net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.PulvinarLayer, axon.InputLayer)
	actGeFunc := axon.StatLayerActGe(ss.Stats, net, Train, Trial, Run, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		actGeFunc(mode, level, phase == Start)
	})

	giMultFunc := axon.StatLayerGiMult(ss.Stats, net, Train, Epoch, Run, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		giMultFunc(mode, level, phase == Start)
	})

	pcaFunc := axon.StatPCA(ss.Stats, ss.Current, net, ss.Config.Run.PCAInterval, Train, Trial, Run, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trnEpc := ss.Loops.Loop(Train, Epoch).Counter.Cur
		pcaFunc(mode, level, phase == Start, trnEpc)
	})
}

// StatCorSim returns a Stats function that records 1 - [LayerPhaseDiff] stats,
// i.e., Correlation-based similarity, for given layer names.
func (ss *Sim) StatCorSim() func(mode Modes, level Levels, phase StatsPhase) {
	net := ss.Net
	layers := net.LayersByType(axon.PulvinarLayer)
	ticks := []string{"", "0", "Foc", "Sac"}
	return func(mode Modes, level Levels, phase StatsPhase) {
		if level < Trial {
			return
		}
		modeDir := ss.Stats.Dir(mode.String())
		curModeDir := ss.Current.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		subDir := modeDir.Dir((level - 1).String())
		ndata := int(net.Context().NData)
		for _, lnm := range layers {
			for _, t := range ticks {
				ly := net.LayerByName(lnm)
				li := ly.Params.Index
				name := lnm + "_CorSim" + t
				tsr := levelDir.Float64(name)
				if phase == Start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0).SetMax(1)
						s.On = true
					})
					continue
				}
				switch level {
				case Trial:
					for di := range ndata {
						ev := ss.Envs.ByModeDi(mode, di).(*Obj3DSacEnv)
						tick := ev.Tick.Cur
						nan := math.NaN()
						stat := 1.0 - float64(axon.LayerStates.Value(int(li), int(di), int(axon.LayerPhaseDiff)))
						switch t {
						case "":
						case "0":
							if tick != 0 {
								stat = nan
							}
						case "Foc":
							if tick == 0 || tick%2 == 0 {
								stat = nan
							}
						case "Sac":
							if tick == 0 || tick%2 == 1 {
								stat = nan
							}
						}
						curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
						tsr.AppendRowFloat(float64(stat))
					}
				case Run:
					tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
				default:
					tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
				}
			}
		}
	}
}

// StatPrevCorSim returns a Stats function that compute correlations
// between previous trial activity state and current minus phase and
// plus phase state. This is important for predictive learning.
// Also the super layer stats track overall representation change over time.
func (ss *Sim) StatPrevCorSim() func(mode Modes, level Levels, phase StatsPhase) {
	net := ss.Net
	layers := net.LayersByType(axon.PulvinarLayer, axon.SuperLayer)
	ticks := []string{"", "0", "Foc", "Sac"}
	statNames := []string{"PrevToM", "PrevToP"}
	return func(mode Modes, level Levels, phase StatsPhase) {
		if level < Trial {
			return
		}
		modeDir := ss.Stats.Dir(mode.String())
		curModeDir := ss.Current.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		subDir := modeDir.Dir((level - 1).String())
		ndata := int(net.Context().NData)
		for _, lnm := range layers {
			for _, t := range ticks {
				for si, statName := range statNames {
					ly := net.LayerByName(lnm)
					name := lnm + "_" + statName + t
					tsr := levelDir.Float64(name)
					if phase == Start {
						tsr.SetNumRows(0)
						plot.SetFirstStyler(tsr, func(s *plot.Style) {
							s.Range.SetMin(0).SetMax(1)
						})
						continue
					}
					switch level {
					case Trial:
						// note: current lnm + _var is standard reusable unit vals buffer
						actM := curModeDir.Float64(lnm+"_ActM", ly.GetSampleShape().Sizes...)
						actP := curModeDir.Float64(lnm+"_ActP", ly.GetSampleShape().Sizes...)
						// note: CaD is sufficiently stable that it is fine to compare with ActM and ActP
						prev := curModeDir.Float64(lnm+"_CaDPrev", ly.GetSampleShape().Sizes...)
						for di := range ndata {
							ev := ss.Envs.ByModeDi(mode, di).(*Obj3DSacEnv)
							tick := ev.Tick.Cur
							nan := math.NaN()
							ly.UnitValuesSampleTensor(prev, "CaDPrev", di)
							prev.SetShapeSizes(prev.Len()) // set to 1D -- inexpensive and faster for computation
							var stat float64
							switch si {
							case 0:
								ly.UnitValuesSampleTensor(actM, "ActM", di)
								actM.SetShapeSizes(actM.Len())
								cov := metric.Correlation(actM, prev)
								stat = cov.Float1D(0)
							case 1:
								ly.UnitValuesSampleTensor(actP, "ActP", di)
								actP.SetShapeSizes(actP.Len())
								cov := metric.Correlation(actP, prev)
								stat = cov.Float1D(0)
							}
							switch t {
							case "":
							case "0":
								if tick != 0 {
									stat = nan
								}
							case "Foc":
								if tick == 0 || tick%2 == 0 {
									stat = nan
								}
							case "Sac":
								if tick == 0 || tick%2 == 1 {
									stat = nan
								}
							}
							curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
							tsr.AppendRowFloat(stat)
						}
					case Run:
						tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
					default:
						tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
					}
				}
			}
		}
	}
}

// StatCounters returns counters string to show at bottom of netview.
func (ss *Sim) StatCounters(mode, level enums.Enum) string {
	counters := ss.Loops.Stacks[mode].CountersString()
	vu := ss.NetViewUpdater(mode)
	if vu == nil || vu.View == nil {
		return counters
	}
	di := vu.View.Di
	counters += fmt.Sprintf(" Di: %d", di)
	curModeDir := ss.Current.Dir(mode.String())
	if curModeDir.Node("TrialName") == nil {
		return counters
	}
	counters += fmt.Sprintf(" TrialName: %s", curModeDir.StringValue("TrialName").String1D(di))
	ev := ss.Envs.ByModeDi(mode, di).(*Obj3DSacEnv)
	counters += fmt.Sprintf(" Tick: %d", ev.Tick.Cur)
	if level == Cycle {
		return counters
	}
	statNames := []string{"MTposP_CorSim", "V1mP_CorSim"}
	for _, name := range statNames {
		if curModeDir.Node(name) != nil {
			counters += fmt.Sprintf(" %s: %.4g", name, curModeDir.Float64(name).Float1D(di))
		}
	}
	return counters
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.StopLevel = Trial
	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 2 * ss.Config.Run.Cycles
	nv.Options.Raster.Max = ss.Config.Run.Cycles
	nv.Options.LayerNameSize = 0.03
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.TestUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.3, 2.15)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, -.1, .05), math32.Vec3(0, 1, 0))

	ss.StatsInit()

	trn := ss.Envs.ByModeDi(Train, 0).(*Obj3DSacEnv)
	img := &trn.Img.Tsr
	tensorcore.AddGridStylerTo(img, func(s *tensorcore.GridStyle) {
		s.Image = true
	})
	ss.GUI.Tabs.TensorGrid("Image", img)

	ss.RSAGUI()

	ss.GUI.Tabs.SelectTabIndex(0)
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) UpdateImage() {
	if !ss.Config.GUI {
		return
	}
	ss.GUI.Tabs.TabUpdateRender("Image")
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "Open RAvgs",
		Icon:    icons.Open,
		Tooltip: "Open running-average activation data from tar file, and run stats on data.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.CallFunc(ss.GUI.Body, ss.RSAOpenRActs)
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL(ss.Config.URL)
		},
	})
}

func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}

	runName := ss.SetRunName()
	netName := ss.Net.Name
	cfg := &ss.Config.Log
	axon.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Train, cfg.Test})

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.Runs, ss.Config.Run.Run)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)

	ss.Loops.Run(Train)

	axon.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
	axon.GPURelease()
}
