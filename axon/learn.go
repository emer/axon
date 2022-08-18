// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"math/rand"

	"github.com/emer/axon/chans"
	"github.com/emer/axon/kinase"
	"github.com/emer/etable/minmax"
	"github.com/goki/mat32"
)

///////////////////////////////////////////////////////////////////////
//  learn.go contains the learning params and functions for axon

// axon.LearnNeurParams manages learning-related parameters at the neuron-level.
// This is mainly the running average activations that drive learning
type LearnNeurParams struct {
	NeurCa    NeurCaParams     `view:"inline" desc:"parameters for computing simple spike-driven calcium signaling variables"`
	LrnNMDA   chans.NMDAParams `view:"inline" desc:"NMDA channel parameters used for learning, vs. the ones driving activation -- allows exploration of learning parameters independent of their effects on active maintenance contributions of NMDA, and may be supported by different receptor subtypes"`
	TrgAvgAct TrgAvgActParams  `view:"inline" desc:"synaptic scaling parameters for regulating overall average activity compared to neuron's own target level"`
	RLrate    RLrateParams     `view:"inline" desc:"recv neuron learning rate modulation params -- an additional error-based modulation of learning for receiver side: RLrate = |SpkCaP - SpkCaD| / Max(SpkCaP, SpkCaD)"`
}

func (ln *LearnNeurParams) Update() {
	ln.NeurCa.Update()
	ln.LrnNMDA.Update()
	ln.TrgAvgAct.Update()
	ln.RLrate.Update()
}

func (ln *LearnNeurParams) Defaults() {
	ln.NeurCa.Defaults()
	ln.LrnNMDA.Defaults()
	ln.LrnNMDA.ITau = 1
	ln.LrnNMDA.Tau = 50 // 50 seems sig better than 100, which is needed for active maintenance -- this is major diff
	ln.LrnNMDA.Update()
	ln.TrgAvgAct.Defaults()
	ln.RLrate.Defaults()
}

// InitNeurCa initializes the running-average activation values that drive learning.
// Called by InitWts (at start of learning).
func (ln *LearnNeurParams) InitNeurCa(nrn *Neuron) {
	nrn.CaSyn = 0
	nrn.CaSpkM = 0
	nrn.CaSpkP = 0
	nrn.CaSpkD = 0
	nrn.CaM = 0
	nrn.CaP = 0
	nrn.CaD = 0
	nrn.CaDiff = 0
}

// DecayNeurCa decays neuron-level calcium by given factor (between trials)
func (ln *LearnNeurParams) DecayNeurCa(nrn *Neuron, decay float32) {
	if !ln.NeurCa.Decay {
		return
	}
	nrn.CaSyn -= decay * nrn.CaSyn
	nrn.CaSpkM -= decay * nrn.CaSpkM
	nrn.CaSpkP -= decay * nrn.CaSpkP
	nrn.CaSpkD -= decay * nrn.CaSpkD

	nrn.CaM -= decay * nrn.CaM
	nrn.CaP -= decay * nrn.CaP
	nrn.CaD -= decay * nrn.CaD

	nrn.GnmdaLrn -= decay * nrn.GnmdaLrn
	nrn.NmdaCa -= decay * nrn.NmdaCa
	nrn.VgccCa -= decay * nrn.VgccCa
	nrn.CaLrn -= decay * nrn.CaLrn

	nrn.SnmdaO -= decay * nrn.SnmdaO
	nrn.SnmdaI -= decay * nrn.SnmdaI
}

// CaLrn updates the CaLrn integrated Ca from NmdaCa + VgccCa,
// computing NmdaCa via LrnNMDA channel, and VgccCa
// is either computed from spikes or Vgcca current.
// CaLrn is normalized by CaMax
func (ln *LearnNeurParams) CaLrn(nrn *Neuron, geExt float32) {
	nrn.GnmdaLrn = ln.LrnNMDA.NMDASyn(nrn.GnmdaLrn, nrn.GeRaw+geExt)
	gnmda := ln.LrnNMDA.Gnmda(nrn.GnmdaLrn, nrn.VmDend)
	nrn.NmdaCa = gnmda * ln.LrnNMDA.CaFmV(nrn.VmDend)
	ln.LrnNMDA.SnmdaFmSpike(nrn.Spike, &nrn.SnmdaO, &nrn.SnmdaI)

	if ln.NeurCa.SpkVGCC {
		if ln.NeurCa.MTauCaLrn {
			nrn.VgccCa = ln.NeurCa.SpkVGCCa * nrn.Spike
		} else {
			nrn.VgccCa = ln.NeurCa.SpkVGCCa * nrn.CaSpkM
		}
	}
	nrn.CaLrn = ln.NeurCa.CaNorm(nrn.NmdaCa + nrn.VgccCa)
}

// CaFmSpike updates the simple spike-based calcium signaling vals.
// Computed after new activation for current cycle is updated.
func (ln *LearnNeurParams) CaFmSpike(nrn *Neuron) {
	ln.NeurCa.CaFmSpike(nrn)
}

// NeurCaParams parameterizes the neuron-level spike-triggered calcium
// signals for the NeurSpkCa version of the Kinase learning rule.
// Spikes trigger decaying traces of Ca integrated in a cascading fashion
// at multiple time scales, with P = LTP / plus-phase and D = LTD / minus phase
// driving key subtraction for error-driven learning rule.
type NeurCaParams struct {
	SpkVGCC   bool    `desc:"use spikes to generate VGCC instead of actual VGCC current -- see SpkVGCCa for calcium contribution"`
	MTauCaLrn bool    `desc:"apply MTau to all of CaLrn -- otherwise just applies to CaSpkM which is then integrated directly with CaLrn into CaM"`
	SpkVGCCa  float32 `def:"10" desc:"gain factor for spikes in computing Ca contribution to CaM from CaSpkM, which is also affected by SpikeG"`
	SpikeG    float32 `def:"8" desc:"gain multiplier on spike: how much spike drives CaM value for CaSpkM, which also drives learning CaM via SpkVGCC mode"`
	SynTau    float32 `def:"30" min:"1" desc:"spike-driven calcium trace at sender and recv neurons for synapse-level learning rules (CaSyn), time constant in cycles (msec)"`
	MTau      float32 `def:"5" min:"1" desc:"spike-driven calcium CaM mean Ca (calmodulin) time constant in cycles (msec), with a value of 10 roughly tracking the biophysical dynamics of Ca.`
	PTau      float32 `def:"40" min:"1" desc:"LTP spike-driven Ca factor (CaP) time constant in cycles (msec), simulating CaMKII in the Kinase framework, with 40 on top of MTau = 10 roughly tracking the biophysical rise time.  Computationally, CaP represents the plus phase learning signal that reflects the most recent past information"`
	DTau      float32 `def:"40" min:"1" desc:"LTD spike-driven Ca factor (CaD) time constant in cycles (msec), simulating DAPK1 in Kinase framework.  Computationally, CaD represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome)"`
	CaMax     float32 `def:"50" desc:"maximum typical calcium level -- used for normalizing CaLrn, which then drives learning as it is integrated into CaM and CaP"`
	Decay     bool    `def:"false" desc:"if true, decay Ca values along with other longer duration state variables at the ThetaCycle boundary"`

	SynDt   float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	MDt     float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	PDt     float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	DDt     float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	SynSpkG float32 `view:"+" json:"-" xml:"-" inactive:"+" desc:"Ca gain factor for SynSpkCa learning rule, to compensate for the effect of SynTau, which increases Ca as it gets larger.  is 1 for SynTau = 30"`
}

func (np *NeurCaParams) Update() {
	np.SynDt = 1 / np.SynTau
	np.MDt = 1 / np.MTau
	np.PDt = 1 / np.PTau
	np.DDt = 1 / np.DTau
	np.SynSpkG = mat32.Sqrt(30) / mat32.Sqrt(np.SynTau)
}

func (np *NeurCaParams) Defaults() {
	np.SpkVGCCa = 10
	np.SpikeG = 8
	np.SynTau = 30
	np.MTau = 5
	np.PTau = 40
	np.DTau = 40
	np.CaMax = 50
	np.Decay = false // todo: true?
	np.Update()
}

// CaFmSpike computes Ca* calcium signals based on current spike, for NeurSpkCa
func (np *NeurCaParams) CaFmSpike(nrn *Neuron) {
	nsp := np.SpikeG * nrn.Spike
	nrn.CaSyn += np.SynDt * (nsp - nrn.CaSyn)
	nrn.CaSpkM += np.MDt * (nsp - nrn.CaSpkM)
	nrn.CaSpkP += np.PDt * (nrn.CaSpkM - nrn.CaSpkP)
	nrn.CaSpkD += np.DDt * (nrn.CaSpkP - nrn.CaSpkD)

	if np.MTauCaLrn {
		nrn.CaM += np.MDt * (nrn.CaLrn - nrn.CaM)
	} else {
		nrn.CaM = nrn.CaLrn
	}
	nrn.CaP += np.PDt * (nrn.CaM - nrn.CaP)
	nrn.CaD += np.DDt * (nrn.CaP - nrn.CaD)
	nrn.CaDiff = nrn.CaP - nrn.CaD
}

// CaNorm normalizes and thresholds the calcium level according to CaMax, CaThr
func (np *NeurCaParams) CaNorm(ca float32) float32 {
	return ca / np.CaMax
}

// SynSpkCa computes synaptic spiking Ca from send and recv neuron CaSyn vals
func (np *NeurCaParams) SynSpkCa(snCaSyn, rnCaSyn float32) float32 {
	return np.SynSpkG * snCaSyn * rnCaSyn
}

//////////////////////////////////////////////////////////////////////////////////////
//  TrgAvgActParams

// TrgAvgActParams govern the target and actual long-term average activity in neurons.
// Target value is adapted by unit-wise error and difference in actual vs. target
// drives synaptic scaling.
type TrgAvgActParams struct {
	On           bool       `desc:"whether to use target average activity mechanism to scale synaptic weights"`
	ErrLrate     float32    `viewif:"On" def:"0.02,0.01" desc:"learning rate for adjustments to Trg value based on unit-level error signal.  Population TrgAvg values are renormalized to fixed overall average in TrgRange.  Generally use .02 for smaller networks, and 0.01 for larger networks."`
	SynScaleRate float32    `viewif:"On" def:"0.01,0.005" desc:"rate parameter for how much to scale synaptic weights in proportion to the AvgDif between target and actual proportion activity.  Use faster 0.01 rate for smaller models, 0.005 for larger models."`
	TrgRange     minmax.F32 `viewif:"On" desc:"[default .5 to 2] range of target normalized average activations -- individual neurons are assigned values within this range to TrgAvg, and clamped within this range."`
	Permute      bool       `viewif:"On" def:"true" desc:"permute the order of TrgAvg values within layer -- otherwise they are just assigned in order from highest to lowest for easy visualization -- generally must be true if any topographic weights are being used"`
	Pool         bool       `viewif:"On" desc:"use pool-level target values if pool-level inhibition and 4D pooled layers are present -- if pool sizes are relatively small, then may not be useful to distribute targets just within pool"`
}

func (ta *TrgAvgActParams) Update() {
}

func (ta *TrgAvgActParams) Defaults() {
	ta.On = true
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
// RLrate is computed as |CaP - CaD| / Max(CaP, CaD) subject to thresholding
type RLrateParams struct {
	On         bool    `def:"true" desc:"use learning rate modulation"`
	ActThr     float32 `def:"0.1" desc:"threshold on Max(CaP, CaD) below which Min lrate applies -- must be > 0 to prevent div by zero"`
	ActDiffThr float32 `def:"0.02" desc:"threshold on recv neuron error delta, i.e., |CaP - CaD| below which lrate is at Min value"`
	Min        float32 `def:"0.001" desc:"minimum learning rate value when below ActDiffThr"`
}

func (rl *RLrateParams) Update() {
}

func (rl *RLrateParams) Defaults() {
	rl.On = true
	rl.ActThr = 0.1
	rl.ActDiffThr = 0.02
	rl.Min = 0.001
	rl.Update()
}

// RLrate returns the learning rate as a function of CaP and CaD values
func (rl *RLrateParams) RLrate(scap, scad, corSimAvg float32) float32 {
	if !rl.On {
		return 1.0
	}
	max := mat32.Max(scap, scad)
	if max > rl.ActThr { // avoid div by 0
		dif := mat32.Abs(scap - scad)
		if dif < rl.ActDiffThr {
			return rl.Min
		}
		return (dif / max)
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
	Init  SWtInitParams  `view:"inline" desc:"initialization of SWt values"`
	Adapt SWtAdaptParams `view:"inline" desc:"adaptation of SWt values in response to LWt learning"`
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
// centered at 1 and normed in range +/- 1 around that
// in preparation for multiplying times SWt
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
	return 2.0 * wt // center at 1 instead of .5
}

// LinFmSigWt returns linear weight from sigmoidal contrast-enhanced weight.
// wt is centered at 1, and normed in range +/- 1 around that,
// return value is in 0-1 range, centered at .5
func (sp *SWtParams) LinFmSigWt(wt float32) float32 {
	wt *= 0.5
	if wt < 0 {
		wt = 0
	} else if wt > 1 {
		wt = 1
	}
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

// todo: change below to not use pointers

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

// InitSynCa initializes synaptic calcium state, including CaUpT
func InitSynCa(sy *Synapse) {
	sy.CaUpT = -1
	sy.Ca = 0
	sy.CaM = 0
	sy.CaP = 0
	sy.CaD = 0
}

// DecaySynCa decays synaptic calcium by given factor (between trials)
func DecaySynCa(sy *Synapse, decay float32) {
	sy.CaM -= decay * sy.CaM
	sy.CaP -= decay * sy.CaP
	sy.CaD -= decay * sy.CaD
}

// InitWtsSyn initializes weight values based on WtInit randomness parameters
// for an individual synapse.
// It also updates the linear weight value based on the sigmoidal weight value.
func (sp *SWtParams) InitWtsSyn(sy *Synapse, mean, spct float32) {
	wtv := sp.Init.RndVar()
	sy.Wt = mean + wtv
	sy.SWt = sp.ClipSWt(mean + spct*wtv)
	sy.LWt = sp.LWtFmWts(sy.Wt, sy.SWt)
	sy.DWt = 0
	sy.DSWt = 0
	InitSynCa(sy)
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
	Lrate    float32 `viewif:"On" def:"0.1,0.01,0.001,0.0002" desc:"learning rate multiplier on the accumulated DWt values (which already have fast Lrate applied) to incorporate into SWt during slow outer loop updating -- lower values impose stronger constraints, for larger networks that need more structural support, e.g., 0.001 is better after 1,000 epochs in large models.  0.1 is fine for smaller models."`
	SigGain  float32 `viewif:"On" def:"6" desc:"gain of sigmoidal constrast enhancement function used to transform learned, linear LWt values into Wt values"`
	DreamVar float32 `viewif:"On" def:"0,0.01,0.02" desc:"extra random variability to add to LWts after every SWt update, which theoretically happens at night -- hence the association with dreaming.  0.01 is max for a small network that still allows learning, 0.02 works well for larger networks that can benefit more.  generally avoid adding to projections to output layers."`
}

func (sp *SWtAdaptParams) Defaults() {
	sp.On = true
	sp.Lrate = 0.1
	sp.SigGain = 6
	sp.DreamVar = 0.0
	sp.Update()
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
	Learn    bool            `desc:"enable learning for this projection"`
	Lrate    LrateParams     `desc:"learning rate parameters, supporting two levels of modulation on top of base learning rate."`
	Trace    TraceParams     `desc:"trace-based learning parameters"`
	KinaseCa kinase.CaParams `view:"inline" desc:"kinase calcium Ca integration parameters"`
}

func (ls *LearnSynParams) Update() {
	ls.Lrate.Update()
	ls.Trace.Update()
	ls.KinaseCa.Update()
}

func (ls *LearnSynParams) Defaults() {
	ls.Learn = true
	ls.Lrate.Defaults()
	ls.Trace.Defaults()
	ls.KinaseCa.Defaults()
}

// CHLdWt returns the error-driven weight change component for a
// CHL contrastive hebbian learning rule, optionally using the checkmark
// temporally eXtended Contrastive Attractor Learning (XCAL) function
func (ls *LearnSynParams) CHLdWt(suCaP, suCaD, ruCaP, ruCaD float32) float32 {
	srp := suCaP * ruCaP
	srd := suCaD * ruCaD
	return srp - srd
}

// DeltaDWt returns the error-driven weight change component for a
// simple delta between a minus and plus phase factor, optionally using the checkmark
// temporally eXtended Contrastive Attractor Learning (XCAL) function
func (ls *LearnSynParams) DeltaDWt(plus, minus float32) float32 {
	return plus - minus
}

// LrateParams manages learning rate parameters
type LrateParams struct {
	Base  float32 `def:"0.04,0.1,0.2" desc:"base learning rate for this projection -- can be modulated by other factors below -- for larger networks, use slower rates such as 0.04, smaller networks can use faster 0.2."`
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

// TraceParams manages learning rate parameters
type TraceParams struct {
	Tau float32 `desc:"time constant for integrating trace over theta cycle timescales -- governs the decay rate of syanptic trace"`
	Dt  float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
}

func (tp *TraceParams) Defaults() {
	tp.Tau = 1
	tp.Update()
}

func (tp *TraceParams) Update() {
	tp.Dt = 1.0 / tp.Tau
}

// TrFmCa returns updated trace factor as function of a synaptic calcium factor
func (tp *TraceParams) TrFmCa(tr float32, ca float32) float32 {
	tr += tp.Dt * (ca - tr)
	return tr
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
