// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"fmt"
	"strings"

	"github.com/Astera-org/axon/axon"
	"github.com/Astera-org/axon/rl"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// MatrixParams has parameters for Dorsal Striatum Matrix computation
// These are the main Go / NoGo gating units in BG.
type MatrixParams struct {
	GPHasPools   bool    `desc:"do the GP pathways that we drive have separate pools that compete for selecting one out of multiple options in parallel (true) or is it a single big competition for Go vs. No (false)"`
	InvertNoGate bool    `desc:"invert the direction of learning if not gated -- allows negative DA to increase gating when gating didn't happen.  Does not work with GPHasPools at present."`
	GateThr      float32 `desc:"threshold on layer Avg SpkMax for Matrix Go and Thal layers to count as having gated"`
	BurstGain    float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
	DipGain      float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
	ModGain      float32 `desc:"gain factor multiplying the modulator input GeSyn conductances -- total modulation has a maximum of 1"`
	AChInhib     float32 `desc:"strength of extra Gi current multiplied by MaxACh-ACh (ACh > Max = 0) -- ACh is disinhibitory on striatal firing"`
	MaxACh       float32 `desc:"level of ACh at or above which AChInhib goes to 0 -- ACh typically ranges between 0-1"`
}

func (mp *MatrixParams) Defaults() {
	mp.GateThr = 0.01
	mp.BurstGain = 1
	mp.DipGain = 1
	mp.ModGain = 1
	mp.AChInhib = 0
	mp.MaxACh = 0.5
}

// GiFmACh returns inhibitory conductance from ach value, where ACh is 0 at baseline
// and goes up to 1 at US or CS -- effect is disinhibitory on MSNs
func (mp *MatrixParams) GiFmACh(ach float32) float32 {
	ai := mp.MaxACh - ach
	if ai < 0 {
		ai = 0
	}
	return mp.AChInhib * ai
}

// MatrixLayer represents the matrisome medium spiny neurons (MSNs)
// that are the main Go / NoGo gating units in BG.  D1R = Go, D2R = NoGo.
// The Gated value for each pool is updated in the PlusPhase and can be
// called separately too.
type MatrixLayer struct {
	rl.Layer
	DaR      DaReceptors   `desc:"dominant type of dopamine receptor -- D1R for Go pathway, D2R for NoGo"`
	Matrix   MatrixParams  `view:"inline" desc:"matrix parameters"`
	MtxThals emer.LayNames `desc:"first layer here is other corresponding MatrixLayer (Go vs. NoGo), and rest are thalamus layers that are affected by this layer"`
	USLayers emer.LayNames `desc:"layer(s) that represent the presence of a US -- if the Max act of these layers is above .1, then USActive flag is set, which conditions learning behavior: weight changes are updated only at time of US based on current DA * trace of current and prior gating activity.  If this list is empty, then a US-independent form of learning is used."`
	HasMod   bool          `inactive:"+" desc:"has modulatory projections, flagged with Modulator setting -- automatically detected"`
	Gated    []bool        `inactive:"+" desc:"indicates whether gated, based on both Go Matrix Avg SpkMax values and thalamic activity -- is a single bool value unless GpHasPools is true"`
	DALrn    float32       `inactive:"+" desc:"effective learning dopamine value for this layer: reflects DaR and Gains"`
	ACh      float32       `inactive:"+" desc:"acetylcholine value from CIN cholinergic interneurons (or shared sources depending on configuration) reflecting unsigned reward salience -- non-prediction-discounted absolute value of US or CS predictions thereof -- used for modulating inhibition and learning, and resetting trace in US-independent version.  Unlike actual CIN (TAN = tonically active neurons) ACh is 0 at baseline and goes up (to 1 max) for salient events -- we invert for inhibition effects."`
	USActive bool          `inactive:"+" desc:"marks presence of US as a function of activity over USLayers -- controls learning logic as described there"`
	Mods     []float32     `desc:"modulatory values from Modulator projection(s) for each neuron"`
}

var KiT_MatrixLayer = kit.Types.AddType(&MatrixLayer{}, LayerProps)

// Defaults in param.Sheet format
// Sel: "MatrixLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Pool.On":      "false",
// 		"Layer.Inhib.Layer.On":     "true",
// 		"Layer.Inhib.Layer.Gi":     "0.5",
// 		"Layer.Inhib.Layer.FB":     "0.0",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 	}}

func (ly *MatrixLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Matrix.Defaults()
	ly.Typ = Matrix

	// special inhib params
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Inhib.Pool.On = false
	ly.Inhib.Layer.On = true
	ly.Inhib.Layer.Gi = 0.5
	ly.Inhib.Layer.FB = 0
	ly.Inhib.Pool.FB = 0
	ly.Inhib.Pool.Gi = 0.5
	ly.Inhib.ActAvg.Init = 0.25

	// important: user needs to adjust wt scale of some PFC inputs vs others:
	// drivers vs. modulators

	for _, pji := range ly.RcvPrjns {
		pj := pji.(axon.AxonPrjn).AsAxon()
		pj.SWt.Init.SPct = 0
		if _, ok := pj.Send.(*GPLayer); ok { // From GPe TA or In
			pj.PrjnScale.Abs = 1
			pj.Learn.Learn = false
			pj.SWt.Adapt.SigGain = 1
			pj.SWt.Init.Mean = 0.75
			pj.SWt.Init.Var = 0.0
			pj.SWt.Init.Sym = false
			if strings.HasSuffix(pj.Send.Name(), "GPeIn") { // GPeInToMtx
				pj.PrjnScale.Abs = 0.5 // counterbalance for GPeTA to reduce oscillations
			} else if strings.HasSuffix(pj.Send.Name(), "GPeTA") { // GPeTAToMtx
				if strings.HasSuffix(ly.Nm, "MtxGo") {
					pj.PrjnScale.Abs = 2 // was .8
				} else {
					pj.PrjnScale.Abs = 1 // was .3 GPeTAToMtxNo must be weaker to prevent oscillations, even with GPeIn offset
				}
			}
		}
	}

	ly.UpdateParams()
}

// AChLayer interface:

func (ly *MatrixLayer) GetACh() float32    { return ly.ACh }
func (ly *MatrixLayer) SetACh(ach float32) { ly.ACh = ach }

func (ly *MatrixLayer) Class() string {
	return "Matrix " + ly.Cls
}

func (ly *MatrixLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	if ly.Matrix.GPHasPools {
		ly.Gated = make([]bool, len(ly.Pools))
	} else {
		ly.Gated = make([]bool, 1)
	}
	err = ly.MtxThals.Validate(ly.Network, "MatrixLayer.MtxThals")
	err = ly.USLayers.Validate(ly.Network, "MatrixLayer.USLayers")

	ly.HasMod = false
	for _, p := range ly.RcvPrjns {
		pj, ok := p.(*MatrixPrjn)
		if ok && pj.Trace.Modulator {
			ly.HasMod = true
			break
		}
	}
	ly.Mods = make([]float32, len(ly.Neurons))

	return err
}

// InitMods initializes the Mods modulator values
func (ly *MatrixLayer) InitMods() {
	if !ly.HasMod {
		return
	}
	for ni := range ly.Mods {
		ly.Mods[ni] = 0
	}
}

func (ly *MatrixLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DALrn = 0
	ly.ACh = 0
	ly.USActive = false
	for i := range ly.Gated {
		ly.Gated[i] = false
	}
	ly.InitMods()
}

func (ly *MatrixLayer) DecayState(decay, glong float32) {
	ly.Layer.DecayState(decay, glong)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Learn.DecayCaLrnSpk(nrn, glong) // ?
	}
	ly.InitMods()
}

// USActiveFmUS updates the USActive flag based on USLayers state
func (ly *MatrixLayer) USActiveFmUS(ctime *axon.Time) {
	ly.USActive = false
	if len(ly.USLayers) == 0 {
		return
	}
	mx := rl.MaxAbsActFmLayers(ly.Network, ly.USLayers)
	if mx > 0.1 {
		ly.USActive = true
	}
}

// GiFmACh sets inhibitory conductance from ACh value, where ACh is 0 at baseline
// and goes up to 1 at US or CS -- effect is disinhibitory on MSNs
func (ly *MatrixLayer) GiFmACh(ctime *axon.Time) {
	gi := ly.Matrix.GiFmACh(ly.ACh)
	if gi == 0 {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.GiSyn += gi
	}
}

func (ly *MatrixLayer) GiFmSpikes(ctime *axon.Time) {
	ly.Layer.GiFmSpikes(ctime)
	ly.GiFmACh(ctime)
}

func (ly *MatrixLayer) GInteg(ni int, nrn *axon.Neuron, ctime *axon.Time) {
	if !ly.HasMod {
		ly.Layer.GInteg(ni, nrn, ctime)
		return
	}
	ly.GFmSpikeRaw(ni, nrn, ctime)
	var mod float32
	var modRaw float32
	for _, p := range ly.RcvPrjns {
		pj, ok := p.(*MatrixPrjn)
		if ok && pj.Trace.Modulator {
			gv := pj.GVals[ni]
			mod += gv.GSyn
			modRaw += gv.GRaw
		}
	}
	nrn.GeRaw -= modRaw // don't include in gating signal -- pure modulator
	nrn.GeSyn -= mod
	mod *= ly.Matrix.ModGain
	if mod > 0 {
		mod = 1
	}
	ly.Mods[ni] = mod
	nrn.GeRaw *= mod
	nrn.GeSyn *= mod
	ly.GFmRawSyn(ni, nrn, ctime)
	ly.GiInteg(ni, nrn, ctime)
}

func (ly *MatrixLayer) SpikeFmG(ni int, nrn *axon.Neuron, ctime *axon.Time) {
	ly.Layer.SpikeFmG(ni, nrn, ctime)
	if ly.DaR == D1R {
		return
	}
	// for NoGo layer, if Go has spiked, then need NoGo too
	// if nrn.SpkMax > 0 {
	// 	return
	// }

	if ctime.Cycle >= ly.Act.Dt.MaxCycStart {
		if nrn.Ge > nrn.SpkMax {
			nrn.SpkMax = nrn.Ge
		}
	}
}

// todo: replace with ki/bools.ToFloat32
// BoolToFloat32 -- the lack of ternary conditional expressions
// is *only* Go decision I disagree about
func BoolToFloat32(b bool) float32 {
	if b {
		return 1
	}
	return 0
}

// AnyGated returns true if any of the pools gated
func (ly *MatrixLayer) AnyGated() bool {
	for _, gt := range ly.Gated {
		if gt {
			return true
		}
	}
	return false
}

// PlusPhase does updating at end of the plus phase
// calls DAActLrn
func (ly *MatrixLayer) PlusPhase(ctime *axon.Time) {
	ly.Layer.PlusPhase(ctime)
	ly.USActiveFmUS(ctime)
	ly.GatedFmAvgSpk()
	ly.DAActLrn()

	smax := ly.AvgMaxVarByPool("SpkMax", 0).Max
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		mlr := ly.Learn.RLrate.RLrateSigDeriv(nrn.SpkMax, smax)
		// dlr := ly.Learn.RLrate.RLrateDiff(nrn.CaSpkP, nrn.CaSpkD) // not useful
		nrn.RLrate = mlr
	}
}

// GatedFmAvgSpk updates Gated based on Avg SpkMax activity in Go Matrix and
// ThalLayers listed in MtxThals
func (ly *MatrixLayer) GatedFmAvgSpk() {
	lays, _ := ly.MtxThals.Layers(ly.Network)
	if ly.DaR != D1R { // copy from go
		goMtx := lays[0].(*MatrixLayer)
		for i := range ly.Gated {
			ly.Gated[i] = goMtx.Gated[i]
		}
		return
	}
	mtxGated := false
	if ly.Is4D() {
		for pi := 1; pi < len(ly.Pools); pi++ {
			spkavg := ly.AvgMaxVarByPool("SpkMax", pi).Avg
			gthr := spkavg > ly.Matrix.GateThr
			if gthr {
				// fmt.Printf("mtx %s gated spkavg: %g pool: %d\n", ly.Name(), spkavg, pi)
				mtxGated = true
			}
			if ly.Matrix.GPHasPools {
				ly.Gated[pi] = gthr
			}
		}
		if ly.Matrix.GPHasPools {
			ly.Gated[0] = mtxGated
		}
	} else {
		spkavg := ly.AvgMaxVarByPool("SpkMax", 0).Avg
		if spkavg > ly.Matrix.GateThr {
			mtxGated = true
		}
	}

	lays = lays[1:]
	thalGated := false
	for _, tly := range lays {
		thLy := tly.(*ThalLayer)
		lgt := thLy.GatedFmAvgSpk(ly.Matrix.GateThr)
		if lgt {
			thalGated = true
		}
	}

	mtxGated = mtxGated && thalGated

	if ly.Matrix.GPHasPools {
		if !thalGated {
			for i := range ly.Gated {
				ly.Gated[i] = false // veto
			}
		}
	} else {
		ly.Gated[0] = mtxGated
	}
}

// DAActLrn sets effective learning dopamine value from given raw DA value,
// applying Burst and Dip Gain factors, and then reversing sign for D2R
// and also for InvertNoGate -- must have done GatedFmThal before this.
func (ly *MatrixLayer) DAActLrn() {
	da := ly.DA
	if da > 0 {
		da *= ly.Matrix.BurstGain
	} else {
		da *= ly.Matrix.DipGain
	}
	if ly.DaR == D2R {
		da *= -1
	}
	ly.DALrn = da
}

///////////////////////////////////////////////////////////////////
// Unit var access

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *MatrixLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + 3
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *MatrixLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	nvi := 0
	switch varNm {
	case "DALrn":
		nvi = 0
	case "Gated":
		nvi = 1
	case "ACh":
		nvi = 2
	default:
		return -1, fmt.Errorf("pcore.NeuronVars: variable named: %s not found", varNm)
	}
	nn := ly.Layer.UnitVarNum()
	return nn + nvi, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *MatrixLayer) UnitVal1D(varIdx int, idx int) float32 {
	if varIdx < 0 {
		return mat32.NaN()
	}
	nn := ly.Layer.UnitVarNum()
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return mat32.NaN()
	}
	varIdx -= nn
	switch varIdx {
	case 0:
		return ly.DALrn
	case 1:
		gt := ly.Gated[0]
		if len(ly.Gated) == len(ly.Pools) {
			gt = ly.Gated[ly.Neurons[idx].SubPool]
		}
		return BoolToFloat32(gt)
	case 2:
		return ly.ACh
	default:
		return mat32.NaN()
	}
	return 0
}
