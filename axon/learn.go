// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"math/rand"

	"github.com/emer/etable/minmax"
	"github.com/goki/mat32"
)

///////////////////////////////////////////////////////////////////////
//  learn.go contains the learning params and functions for axon

// axon.LearnNeurParams manages learning-related parameters at the neuron-level.
// This is mainly the running average activations that drive learning
type LearnNeurParams struct {
	ActAvg    LrnActAvgParams `view:"inline" desc:"parameters for computing running average activations that drive learning"`
	TrgAvgAct TrgAvgActParams `view:"inline" desc:"synaptic scaling parameters for regulating overall average activity compared to neuron's own target level"`
	RLrate    RLrateParams    `view:"inline" desc:"recv neuron learning rate modulation params"`
}

func (ln *LearnNeurParams) Update() {
	ln.ActAvg.Update()
	ln.TrgAvgAct.Update()
	ln.RLrate.Update()
}

func (ln *LearnNeurParams) Defaults() {
	ln.ActAvg.Defaults()
	ln.TrgAvgAct.Defaults()
	ln.RLrate.Defaults()
}

// InitActAvg initializes the running-average activation values that drive learning.
// Called by InitWts (at start of learning).
func (ln *LearnNeurParams) InitActAvg(nrn *Neuron) {
	nrn.AvgSS = ln.ActAvg.Init
	nrn.AvgS = ln.ActAvg.Init
	nrn.AvgM = ln.ActAvg.Init
	nrn.AvgSLrn = 0
	nrn.AvgMLrn = 0
}

// AvgsFmAct updates the running averages based on current learning activation.
// Computed after new activation for current cycle is updated.
func (ln *LearnNeurParams) AvgsFmAct(nrn *Neuron) {
	ln.ActAvg.AvgsFmAct(ln.ActAvg.SpikeG*nrn.Spike, &nrn.AvgSS, &nrn.AvgS, &nrn.AvgM, &nrn.AvgSLrn, &nrn.AvgMLrn)
}

// LrnActAvgParams has rate constants for averaging over activations
// at different time scales, to produce the running average activation
// values that then drive learning in the XCAL learning rules.
// Is driven directly by spikes that increment running-average at super-short
// timescale.  Time cycle of 50 msec quarters / theta window learning works
// Cyc:50, SS:35 S:8, M:40 (best)
// Cyc:25, SS:20, S:4, M:20
type LrnActAvgParams struct {
	SpikeG float32 `def:"8" desc:"gain multiplier on spike: how much spike drives AvgSS value"`
	MinLrn float32 `def:"0.02" desc:"minimum learning activation -- below this goes to zero"`
	SSTau  float32 `def:"40" min:"1" desc:"time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life), for continuously updating the super-short time-scale AvgSS value -- this is provides a pre-integration step before integrating into the AvgS short time scale -- it is particularly important for spiking -- in general 4 is the largest value without starting to impair learning, but a value of 7 can be combined with m_in_s = 0 with somewhat worse results"`
	STau   float32 `def:"10" min:"1" desc:"time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life), for continuously updating the short time-scale AvgS value from the super-short AvgSS value (cascade mode) -- AvgS represents the plus phase learning signal that reflects the most recent past information"`
	MTau   float32 `def:"40" min:"1" desc:"time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life), for continuously updating the medium time-scale AvgM value from the short AvgS value (cascade mode) -- AvgM represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome) -- the default value of 10 generally cannot be exceeded without impairing learning"`
	LrnM   float32 `def:"0.1,0" min:"0" max:"1" desc:"how much of the medium term average activation to mix in with the short (plus phase) to compute the Neuron AvgSLrn variable that is used for the unit's short-term average in learning. This is important to ensure that when unit turns off in plus phase (short time scale), enough medium-phase trace remains so that learning signal doesn't just go all the way to 0, at which point no learning would take place -- typically need faster time constant for updating S such that this trace of the M signal is lost -- can set SSTau=7 and set this to 0 but learning is generally somewhat worse"`
	Init   float32 `def:"0.15" min:"0" max:"1" desc:"initial value for average"`

	SSDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	SDt  float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	MDt  float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	LrnS float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"1-LrnM"`
}

// AvgsFmAct computes averages based on current act
func (aa *LrnActAvgParams) AvgsFmAct(act float32, avgSS, avgS, avgM, avgSLrn, avgMLrn *float32) {
	*avgSS += aa.SSDt * (act - *avgSS)
	*avgS += aa.SDt * (*avgSS - *avgS)
	*avgM += aa.MDt * (*avgS - *avgM)
	*avgMLrn = *avgM

	thrS := *avgS

	if *avgMLrn < aa.MinLrn && thrS < aa.MinLrn {
		*avgMLrn = 0
		thrS = 0
	}

	*avgSLrn = aa.LrnS*thrS + aa.LrnM**avgMLrn
}

func (aa *LrnActAvgParams) Update() {
	aa.SSDt = 1 / aa.SSTau
	aa.SDt = 1 / aa.STau
	aa.MDt = 1 / aa.MTau
	aa.LrnS = 1 - aa.LrnM
}

func (aa *LrnActAvgParams) Defaults() {
	aa.SpikeG = 8
	aa.MinLrn = 0.02
	aa.SSTau = 40 // 20 for 25 cycle qtr
	aa.STau = 10
	aa.MTau = 40 // 20 for 25 cycle qtr
	aa.LrnM = 0.1
	aa.Init = 0.15
	aa.Update()

}

//////////////////////////////////////////////////////////////////////////////////////
//  TrgAvgActParams

// TrgAvgActParams govern the target and actual long-term average activity in neurons.
// Target value is adapted by unit-wise error and difference in actual vs. target
// drives synaptic scaling.
type TrgAvgActParams struct {
	ErrLrate     float32    `def:"0.02,0.01" desc:"learning rate for adjustments to Trg value based on unit-level error signal.  Population TrgAvg values are renormalized to fixed overall average in TrgRange.  Generally use .02 for smaller networks, and 0.01 for larger networks."`
	SynScaleRate float32    `def:"0.01,0.005" desc:"rate parameter for how much to scale synaptic weights in proportion to the AvgDif between target and actual proportion activity.  Use faster 0.01 rate for smaller models, 0.005 for larger models."`
	TrgRange     minmax.F32 `desc:"[default .5 to 2] range of target normalized average activations -- individual neurons are assigned values within this range to TrgAvg, and clamped within this range."`
	Permute      bool       `def:"true" desc:"permute the order of TrgAvg values within layer -- otherwise they are just assigned in order from highest to lowest for easy visualization -- generally must be true if any topographic weights are being used"`
	Pool         bool       `desc:"use pool-level target values if pool-level inhibition and 4D pooled layers are present -- if pool sizes are relatively small, then may not be useful to distribute targets just within pool"`
}

func (ta *TrgAvgActParams) Update() {
}

func (ta *TrgAvgActParams) Defaults() {
	ta.ErrLrate = 0.02
	ta.SynScaleRate = 0.01
	ta.TrgRange.Set(0.5, 2)
	ta.Permute = true
	ta.Pool = true
	ta.Update()
}

//////////////////////////////////////////////////////////////////////////////////////
//  RLrateParams

// RLrateParams recv neuron learning rate modulation parameters.
// RLrate is computed as |AvgS - AvgM| / Max(AvgS, AvgM) subject to thresholding
type RLrateParams struct {
	On        bool    `def:"true" desc:"use learning rate modulation"`
	ActThr    float32 `def:"0.2" desc:"threshold on Max(AvgS, AvgM) below which Min lrate applies -- must be > 0 to prevent div by zero"`
	ActDifThr float32 `def:"0" desc:"threshold on recv neuron error delta, i.e., |AvgS - AvgM| below which lrate is at Min value"`
	Min       float32 `def:"0.01" desc:"minimum learning rate value when below ActDifThr"`
}

func (rl *RLrateParams) Update() {
}

func (rl *RLrateParams) Defaults() {
	rl.On = true
	rl.ActThr = 0.2
	rl.ActDifThr = 0.0
	rl.Min = 0.01
}

// RLrate returns the learning rate as a function of AvgS and AvgM values
func (rl *RLrateParams) RLrate(avgS, avgM float32) float32 {
	max := mat32.Max(avgS, avgM)
	if max > rl.ActThr { // avoid div by 0
		dif := mat32.Abs(avgS - avgM)
		if dif < rl.ActDifThr {
			return rl.Min
		}
		return dif / max
	}
	return rl.Min
}

///////////////////////////////////////////////////////////////////////
//  SWtParams

// SWtParams manages structural, slowly adapting weight values (SWt),
// in terms of initialization and updating over course of learning.
// SWts impose initial and slowly adapting constraints on neuron connectivity
// to encourage differentiation of neuron representations and overall good behavior
// in terms of not hogging the representational space.
// The TrgAvg activity constraint is not enforced through SWt -- it needs to be
// more dynamic and supported by the regular learned weights.
type SWtParams struct {
	Init  SWtInitParams  `desc:"initialization of SWt values"`
	Adapt SWtAdaptParams `desc:"adaptation of SWt values in response to LWt learning"`
	Limit minmax.F32     `def:"{0.2 0.8}" view:"inline" desc:"range limits for SWt values"`
}

func (sp *SWtParams) Defaults() {
	sp.Init.Defaults()
	sp.Adapt.Defaults()
	sp.Limit.Set(0.2, 0.8)
}

func (sp *SWtParams) Update() {
	sp.Init.Update()
	sp.Adapt.Update()
}

// WtVal returns the effective Wt value given the SWt and LWt values
func (sp *SWtParams) WtVal(swt, lwt float32) float32 {
	return swt * sp.SigFmLinWt(lwt)
}

// ClipSWt returns SWt value clipped to valid range
func (sp *SWtParams) ClipSWt(swt float32) float32 {
	return sp.Limit.ClipVal(swt)
}

// ClipWt returns Wt value clipped to 0-1 range
func (sp *SWtParams) ClipWt(wt float32) float32 {
	if wt > 1 {
		return 1
	}
	if wt < 0 {
		return 0
	}
	return wt
}

// SigFun is the sigmoid function for value w in 0-1 range, with gain and offset params
func SigFun(w, gain, off float32) float32 {
	if w <= 0 {
		return 0
	}
	if w >= 1 {
		return 1
	}
	return (1 / (1 + mat32.Pow((off*(1-w))/w, gain)))
}

// SigFun61 is the sigmoid function for value w in 0-1 range, with default gain = 6, offset = 1 params
func SigFun61(w float32) float32 {
	if w <= 0 {
		return 0
	}
	if w >= 1 {
		return 1
	}
	pw := (1 - w) / w
	return (1 / (1 + pw*pw*pw*pw*pw*pw))
}

// SigInvFun is the inverse of the sigmoid function
func SigInvFun(w, gain, off float32) float32 {
	if w <= 0 {
		return 0
	}
	if w >= 1 {
		return 1
	}
	return 1.0 / (1.0 + mat32.Pow((1.0-w)/w, 1/gain)/off)
}

// SigInvFun61 is the inverse of the sigmoid function, with default gain = 6, offset = 1 params
func SigInvFun61(w float32) float32 {
	if w <= 0 {
		return 0
	}
	if w >= 1 {
		return 1
	}
	rval := 1.0 / (1.0 + mat32.Pow((1.0-w)/w, 1.0/6.0))
	return rval
}

// SigFmLinWt returns sigmoidal contrast-enhanced weight from linear weight,
// centered at 1 in preparation for multiplying times SWt
func (sp *SWtParams) SigFmLinWt(lw float32) float32 {
	var wt float32
	switch {
	case sp.Adapt.SigGain == 1:
		wt = lw
	case sp.Adapt.SigGain == 6:
		wt = SigFun61(lw)
	default:
		wt = SigFun(lw, sp.Adapt.SigGain, 1)
	}
	return 2 * wt // center at 1 instead of .5
}

// LinFmSigWt returns linear weight from sigmoidal contrast-enhanced weight.
// wt is in range 0-2 centered at 1 -- return value is in 0-1 range, centered at .5
func (sp *SWtParams) LinFmSigWt(wt float32) float32 {
	wt *= 0.5
	if sp.Adapt.SigGain == 1 {
		return wt
	}
	if sp.Adapt.SigGain == 6 {
		return SigInvFun61(wt)
	}
	return SigInvFun(wt, sp.Adapt.SigGain, 1)
}

// LWtFmWts returns linear, learning LWt from wt and swt.
// LWt is set to reproduce given Wt relative to given SWt base value.
func (sp *SWtParams) LWtFmWts(wt, swt float32) float32 {
	rwt := wt / swt
	return sp.LinFmSigWt(rwt)
}

// WtFmDWt updates the synaptic weights from accumulated weight changes.
// wt is the sigmoidal contrast-enhanced weight and lwt is the linear weight value.
func (sp *SWtParams) WtFmDWt(dwt, wt, lwt *float32, swt float32) {
	if *dwt == 0 {
		if *wt == 0 { // restore failed wts
			*wt = sp.WtVal(swt, *lwt)
		}
		return
	}
	// note: softbound happened at dwt stage
	*lwt += *dwt
	if *lwt < 0 {
		*lwt = 0
	} else if *lwt > 1 {
		*lwt = 1
	}
	*wt = sp.WtVal(swt, *lwt)
	*dwt = 0
}

// SWtInitParams for initial SWt values
type SWtInitParams struct {
	SPct float32 `min:"0" max:"1" def:"0,1,0.5" desc:"how much of the initial random weights are captured in the SWt values -- rest goes into the LWt values.  1 gives the strongest initial biasing effect, for larger models that need more structural support. 0.5 should work for most models where stronger constraints are not needed."`
	Mean float32 `def:"0.5,0.4" desc:"target mean weight values across receiving neuron's projection -- the mean SWt values are constrained to remain at this value.  some projections may benefit from lower mean of .4"`
	Var  float32 `def:"0.25" desc:"initial variance in weight values, prior to constraints."`
	Sym  bool    `def:"true" desc:"symmetrize the initial weight values with those in reciprocal projection -- typically true for bidirectional excitatory connections"`
}

func (sp *SWtInitParams) Defaults() {
	sp.SPct = 0.5
	sp.Mean = 0.5
	sp.Var = 0.25
	sp.Sym = true
}

func (sp *SWtInitParams) Update() {
}

// RndVar returns the random variance in weight value (zero mean) based on Var param
func (sp *SWtInitParams) RndVar() float32 {
	return sp.Var * 2 * (rand.Float32() - 0.5)
}

// SWtAdaptParams manages adaptation of SWt values
type SWtAdaptParams struct {
	On       bool    `desc:"if true, adaptation is active -- if false, SWt values are not updated, in which case it is generally good to have Init.SPct=0 too."`
	Lrate    float32 `viewif:"On" def:"0.1,0.01,0.001" desc:"learning rate multiplier on the accumulated DWt values (which already have fast Lrate applied) to incorporate into SWt during slow outer loop updating -- lower values impose stronger constraints, for larger networks that need more structural support, e.g., 0.001 is better after 1,000 epochs in large models.  0.1 is fine for smaller models."`
	SigGain  float32 `viewif:"On" def:"6" desc:"gain of sigmoidal constrast enhancement function used to transform learned, linear LWt values into Wt values"`
	DreamVar float32 `viewif:"On" def:"0,0.01,0.02" desc:"extra random variability to add to LWts after every SWt update, which theoretically happens at night -- hence the association with dreaming.  0.01 is max for a small network that still allows learning, 0.02 works well for larger networks that can benefit more.  generally avoid adding to projections to output layers."`
}

func (sp *SWtAdaptParams) Defaults() {
	sp.On = true
	sp.Lrate = 0.1
	sp.SigGain = 6
	sp.DreamVar = 0.0
}

func (sp *SWtAdaptParams) Update() {
}

// RndVar returns the random variance (zero mean) based on DreamVar param
func (sp *SWtAdaptParams) RndVar() float32 {
	return sp.DreamVar * 2 * (rand.Float32() - 0.5)
}

///////////////////////////////////////////////////////////////////////
//  LearnSynParams

// LearnSynParams manages learning-related parameters at the synapse-level.
type LearnSynParams struct {
	Learn bool        `desc:"enable learning for this projection"`
	Lrate LrateParams `desc:"learning rate parameters, supporting two levels of modulation on top of base learning rate."`
	XCal  XCalParams  `view:"inline" desc:"parameters for the XCal learning rule"`
}

func (ls *LearnSynParams) Update() {
	ls.Lrate.Update()
	ls.XCal.Update()
}

func (ls *LearnSynParams) Defaults() {
	ls.Learn = true
	ls.Lrate.Defaults()
	ls.XCal.Defaults()
}

// CHLdWt returns the error-driven weight change component for the
// temporally eXtended Contrastive Attractor Learning (XCAL), CHL version
func (ls *LearnSynParams) CHLdWt(suAvgSLrn, suAvgMLrn, ruAvgSLrn, ruAvgMLrn float32) float32 {
	srs := suAvgSLrn * ruAvgSLrn
	srm := suAvgMLrn * ruAvgMLrn
	return ls.XCal.DWt(srs, srm)
}

// LrateParams manages learning rate parameters
type LrateParams struct {
	Base  float32 `def:"0.04,0.01" desc:"base learning rate for this projection -- can be modulated by other factors below."`
	Sched float32 `desc:"scheduled learning rate multiplier, simulating reduction in plasticity over aging"`
	Mod   float32 `desc:"dynamic learning rate modulation due to neuromodulatory or other such factors"`
	Eff   float32 `inactive:"+" desc:"effective actual learning rate multiplier used in computing DWt: Eff = eMod * Sched * Base"`
}

func (ls *LrateParams) Defaults() {
	ls.Base = 0.04
	ls.Sched = 1
	ls.Mod = 1
	ls.Update()
}

func (ls *LrateParams) Update() {
	ls.Eff = ls.Mod * ls.Sched * ls.Base
}

// Init initializes modulation values back to 1 and updates Eff
func (ls *LrateParams) Init() {
	ls.Sched = 1
	ls.Mod = 1
	ls.Update()
}

//////////////////////////////////////////////////////////////////////////////////////
//  XCalParams

// XCalParams are parameters for temporally eXtended Contrastive Attractor Learning function (XCAL)
// which is the standard learning equation for axon .
type XCalParams struct {
	SubMean float32 `def:"1" desc:"amount of the mean dWt to subtract -- 1.0 = full zero-sum dWt -- only on non-zero DWts (see DWtThr)"`
	DWtThr  float32 `def:"0.0001" desc:"threshold on DWt to be included in SubMean process -- this is *prior* to lrate multiplier"`
	DRev    float32 `def:"0.1" min:"0" max:"0.99" desc:"proportional point within LTD range where magnitude reverses to go back down to zero at zero -- err-driven svm component does better with smaller values"`
	DThr    float32 `def:"0.0001,0.01" min:"0" desc:"minimum LTD threshold value below which no weight change occurs -- this is now *relative* to the threshold"`
	LrnThr  float32 `def:"0.01" desc:"xcal learning threshold -- don't learn when sending unit activation is below this value in both phases -- due to the nature of the learning function being 0 when the sr coproduct is 0, it should not affect learning in any substantial way -- nonstandard learning algorithms that have different properties should ignore it"`

	DRevRatio float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"-(1-DRev)/DRev -- multiplication factor in learning rule -- builds in the minus sign!"`
}

func (xc *XCalParams) Update() {
	if xc.DRev > 0 {
		xc.DRevRatio = -(1 - xc.DRev) / xc.DRev
	} else {
		xc.DRevRatio = -1
	}
}

func (xc *XCalParams) Defaults() {
	xc.SubMean = 1
	xc.DWtThr = 0.0001
	xc.DRev = 0.1
	xc.DThr = 0.0001
	xc.LrnThr = 0.01
	xc.Update()
}

// DWt is the XCAL function for weight change -- the "check mark" function -- no DGain, no ThrPMin
func (xc *XCalParams) DWt(srval, thrP float32) float32 {
	var dwt float32
	if srval < xc.DThr {
		dwt = 0
	} else if srval > thrP*xc.DRev {
		dwt = (srval - thrP)
	} else {
		dwt = srval * xc.DRevRatio
	}
	return dwt
}

//////////////////////////////////////////////////////////////////////////////////////
//  LrateMod

// LrateMod implements global learning rate modulation, based on a performance-based
// factor, for example error.  Increasing levels of the factor = higher learning rate.
// This can be added to a Sim and called prior to DWt() to dynamically change lrate
// based on overall network performance.
type LrateMod struct {
	On    bool       `desc:"toggle use of this modulation factor"`
	Base  float32    `viewif:"On" min:"0" max:"1" desc:"baseline learning rate -- what you get for correct cases"`
	Range minmax.F32 `viewif:"On" desc:"defines the range over which modulation occurs for the modulator factor -- Min and below get the Base level of learning rate modulation, Max and above get a modulation of 1"`
}

func (lr *LrateMod) Defaults() {
	lr.On = true
	lr.Base = 0.2
	lr.Range.Set(0.2, 0.8)
}

func (lr *LrateMod) Update() {
}

// Mod returns the learning rate modulation factor as a function
// of any kind of normalized modulation factor, e.g., an error measure.
// If fact <= Range.Min, returns Base
// If fact >= Range.Max, returns 1
// otherwise, returns proportional value between Base..1
func (lr *LrateMod) Mod(fact float32) float32 {
	lrm := lr.Range.NormVal(fact)    // clips to 0-1 range
	mod := lr.Base + lrm*(1-lr.Base) // resulting mod is in Base-1 range
	return mod
}

// LrateMod calls LrateMod on given network, using computed Mod factor
// based on given normalized modulation factor
// (0 = no error = Base learning rate, 1 = maximum error).
// returns modulation factor applied.
func (lr *LrateMod) LrateMod(net *Network, fact float32) float32 {
	if lr.Range.Max == 0 {
		lr.Defaults()
	}
	if !lr.On {
		return 1
	}
	mod := lr.Mod(fact)
	net.LrateMod(mod)
	return mod
}
