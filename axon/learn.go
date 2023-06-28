// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/axon/chans"
	"github.com/emer/axon/kinase"
	"github.com/emer/emergent/erand"
	"github.com/emer/etable/minmax"
	"github.com/goki/gosl/slbool"
	"github.com/goki/mat32"
)

///////////////////////////////////////////////////////////////////////
//  learn.go contains the learning params and functions for axon

//gosl: hlsl learn_neur
// #include "kinase.hlsl"
// #include "neuromod.hlsl"
//gosl: end learn_neur

//gosl: start learn_neur

// CaLrnParams parameterizes the neuron-level calcium signals driving learning:
// CaLrn = NMDA + VGCC Ca sources, where VGCC can be simulated from spiking or
// use the more complex and dynamic VGCC channel directly.
// CaLrn is then integrated in a cascading manner at multiple time scales:
// CaM (as in calmodulin), CaP (ltP, CaMKII, plus phase), CaD (ltD, DAPK1, minus phase).
type CaLrnParams struct {
	Norm      float32     `def:"80" desc:"denomenator used for normalizing CaLrn, so the max is roughly 1 - 1.5 or so, which works best in terms of previous standard learning rules, and overall learning performance"`
	SpkVGCC   slbool.Bool `def:"true" desc:"use spikes to generate VGCC instead of actual VGCC current -- see SpkVGCCa for calcium contribution from each spike"`
	SpkVgccCa float32     `def:"35" desc:"multiplier on spike for computing Ca contribution to CaLrn in SpkVGCC mode"`
	VgccTau   float32     `def:"10" desc:"time constant of decay for VgccCa calcium -- it is highly transient around spikes, so decay and diffusion factors are more important than for long-lasting NMDA factor.  VgccCa is integrated separately int VgccCaInt prior to adding into NMDA Ca in CaLrn"`

	Dt kinase.CaDtParams `view:"inline" desc:"time constants for integrating CaLrn across M, P and D cascading levels"`

	UpdtThr float32 `def:"0.01,0.02,0.5" desc:"Threshold on CaSpkP CaSpkD value for updating synapse-level Ca values (SynCa) -- this is purely a performance optimization that excludes random infrequent spikes -- 0.05 works well on larger networks but not smaller, which require the .01 default."`

	VgccDt  float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	NormInv float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"= 1 / Norm"`

	pad int32
}

func (np *CaLrnParams) Defaults() {
	np.Norm = 80
	np.SpkVGCC.SetBool(true)
	np.SpkVgccCa = 35
	np.UpdtThr = 0.01
	np.VgccTau = 10
	np.Dt.Defaults()
	np.Dt.MTau = 2
	np.Update()
}

func (np *CaLrnParams) Update() {
	np.Dt.Update()
	np.VgccDt = 1 / np.VgccTau
	np.NormInv = 1 / np.Norm
}

// VgccCa updates the simulated VGCC calcium from spiking, if that option is selected,
// and performs time-integration of VgccCa
func (np *CaLrnParams) VgccCaFmSpike(ctx *Context, ni, di uint32) {
	if np.SpkVGCC.IsTrue() {
		SetNrnV(ctx, ni, di, VgccCa, np.SpkVgccCa*NrnV(ctx, ni, di, Spike))
	}
	AddNrnV(ctx, ni, di, VgccCaInt, NrnV(ctx, ni, di, VgccCa)-np.VgccDt*NrnV(ctx, ni, di, VgccCaInt))
	// Dt only affects decay, not rise time
}

// CaLrns updates the CaLrn value and its cascaded values, based on NMDA, VGCC Ca
// it first calls VgccCa to update the spike-driven version of that variable, and
// perform its time-integration.
func (np *CaLrnParams) CaLrns(ctx *Context, ni, di uint32) {
	np.VgccCaFmSpike(ctx, ni, di)
	SetNrnV(ctx, ni, di, CaLrn, np.NormInv*(NrnV(ctx, ni, di, NmdaCa)+NrnV(ctx, ni, di, VgccCaInt)))
	AddNrnV(ctx, ni, di, NrnCaM, np.Dt.MDt*(NrnV(ctx, ni, di, CaLrn)-NrnV(ctx, ni, di, NrnCaM)))
	AddNrnV(ctx, ni, di, NrnCaP, np.Dt.PDt*(NrnV(ctx, ni, di, NrnCaM)-NrnV(ctx, ni, di, NrnCaP)))
	AddNrnV(ctx, ni, di, NrnCaD, np.Dt.DDt*(NrnV(ctx, ni, di, NrnCaP)-NrnV(ctx, ni, di, NrnCaD)))
	SetNrnV(ctx, ni, di, CaDiff, NrnV(ctx, ni, di, NrnCaP)-NrnV(ctx, ni, di, NrnCaD))
}

// CaSpkParams parameterizes the neuron-level spike-driven calcium
// signals, starting with CaSyn that is integrated at the neuron level
// and drives synapse-level, pre * post Ca integration, which provides the Tr
// trace that multiplies error signals, and drives learning directly for Target layers.
// CaSpk* values are integrated separately at the Neuron level and used for UpdtThr
// and RLRate as a proxy for the activation (spiking) based learning signal.
type CaSpkParams struct {
	SpikeG float32 `def:"8,12" desc:"gain multiplier on spike for computing CaSpk: increasing this directly affects the magnitude of the trace values, learning rate in Target layers, and other factors that depend on CaSpk values: RLRate, UpdtThr.  Prjn.KinaseCa.SpikeG provides an additional gain factor specific to the synapse-level trace factors, without affecting neuron-level CaSpk values.  Larger networks require higher gain factors at the neuron level -- 12, vs 8 for smaller."`
	SynTau float32 `def:"30" min:"1" desc:"time constant for integrating spike-driven calcium trace at sender and recv neurons, CaSyn, which then drives synapse-level integration of the joint pre * post synapse-level activity, in cycles (msec).  Note: if this param is changed, then there will be a change in effective learning rate that can be compensated for by multiplying PrjnParams.Learn.KinaseCa.SpikeG by sqrt(30 / sqrt(SynTau)"`

	SynDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`

	pad int32

	Dt kinase.CaDtParams `view:"inline" desc:"time constants for integrating CaSpk across M, P and D cascading levels -- these are typically the same as in CaLrn and Prjn level for synaptic integration, except for the M factor."`
}

func (np *CaSpkParams) Defaults() {
	np.SpikeG = 8
	np.SynTau = 30
	np.Dt.Defaults()
	np.Update()
}

func (np *CaSpkParams) Update() {
	np.Dt.Update()
	np.SynDt = 1 / np.SynTau
}

// CaFmSpike computes CaSpk* and CaSyn calcium signals based on current spike.
func (np *CaSpkParams) CaFmSpike(ctx *Context, ni, di uint32) {
	nsp := np.SpikeG * NrnV(ctx, ni, di, Spike)
	AddNrnV(ctx, ni, di, CaSyn, np.SynDt*(nsp-NrnV(ctx, ni, di, CaSyn)))
	AddNrnV(ctx, ni, di, CaSpkM, np.Dt.MDt*(nsp-NrnV(ctx, ni, di, CaSpkM)))
	AddNrnV(ctx, ni, di, CaSpkP, np.Dt.PDt*(NrnV(ctx, ni, di, CaSpkM)-NrnV(ctx, ni, di, CaSpkP)))
	AddNrnV(ctx, ni, di, CaSpkD, np.Dt.DDt*(NrnV(ctx, ni, di, CaSpkP)-NrnV(ctx, ni, di, CaSpkD)))
}

//////////////////////////////////////////////////////////////////////////////////////
//  TrgAvgActParams

// TrgAvgActParams govern the target and actual long-term average activity in neurons.
// Target value is adapted by neuron-wise error and difference in actual vs. target.
// drives synaptic scaling at a slow timescale (Network.SlowInterval).
type TrgAvgActParams struct {
	On           slbool.Bool `desc:"whether to use target average activity mechanism to scale synaptic weights"`
	ErrLRate     float32     `viewif:"On" def:"0.02" desc:"learning rate for adjustments to Trg value based on unit-level error signal.  Population TrgAvg values are renormalized to fixed overall average in TrgRange. Generally, deviating from the default doesn't make much difference."`
	SynScaleRate float32     `viewif:"On" def:"0.005,0.0002" desc:"rate parameter for how much to scale synaptic weights in proportion to the AvgDif between target and actual proportion activity -- this determines the effective strength of the constraint, and larger models may need more than the weaker default value."`
	SubMean      float32     `viewif:"On" def:"0,1" desc:"amount of mean trg change to subtract -- 1 = full zero sum.  1 works best in general -- but in some cases it may be better to start with 0 and then increase using network SetSubMean method at a later point."`
	TrgRange     minmax.F32  `viewif:"On" def:"{0.5 2}" desc:"range of target normalized average activations -- individual neurons are assigned values within this range to TrgAvg, and clamped within this range."`
	Permute      slbool.Bool `viewif:"On" def:"true" desc:"permute the order of TrgAvg values within layer -- otherwise they are just assigned in order from highest to lowest for easy visualization -- generally must be true if any topographic weights are being used"`
	Pool         slbool.Bool `viewif:"On" desc:"use pool-level target values if pool-level inhibition and 4D pooled layers are present -- if pool sizes are relatively small, then may not be useful to distribute targets just within pool"`

	pad, pad1 int32
}

func (ta *TrgAvgActParams) Update() {
}

func (ta *TrgAvgActParams) Defaults() {
	ta.On.SetBool(true)
	ta.ErrLRate = 0.02
	ta.SynScaleRate = 0.005
	ta.SubMean = 1 // 1 in general beneficial
	ta.TrgRange.Set(0.5, 2)
	ta.Permute.SetBool(true)
	ta.Pool.SetBool(true)
	ta.Update()
}

//////////////////////////////////////////////////////////////////////////////////////
//  RLRateParams

// RLRateParams are recv neuron learning rate modulation parameters.
// Has two factors: the derivative of the sigmoid based on CaSpkD
// activity levels, and based on the phase-wise differences in activity (Diff).
type RLRateParams struct {
	On         slbool.Bool `def:"true" desc:"use learning rate modulation"`
	SigmoidMin float32     `viewif:"On" def:"0.05,1" desc:"minimum learning rate multiplier for sigmoidal act (1-act) factor -- prevents lrate from going too low for extreme values.  Set to 1 to disable Sigmoid derivative factor, which is default for Target layers."`
	Diff       slbool.Bool `viewif:"On" desc:"modulate learning rate as a function of plus - minus differences"`
	SpkThr     float32     `viewif:"On&&Diff" def:"0.1" desc:"threshold on Max(CaSpkP, CaSpkD) below which Min lrate applies -- must be > 0 to prevent div by zero"`
	DiffThr    float32     `viewif:"On&&Diff" def:"0.02" desc:"threshold on recv neuron error delta, i.e., |CaSpkP - CaSpkD| below which lrate is at Min value"`
	Min        float32     `viewif:"On&&Diff" def:"0.001" desc:"for Diff component, minimum learning rate value when below ActDiffThr"`

	pad, pad1 int32
}

func (rl *RLRateParams) Update() {
}

func (rl *RLRateParams) Defaults() {
	rl.On.SetBool(true)
	rl.SigmoidMin = 0.05
	rl.Diff.SetBool(true)
	rl.SpkThr = 0.1
	rl.DiffThr = 0.02
	rl.Min = 0.001
	rl.Update()
}

// RLRateSigDeriv returns the sigmoid derivative learning rate
// factor as a function of spiking activity, with mid-range values having
// full learning and extreme values a reduced learning rate:
// deriv = act * (1 - act)
// The activity should be CaSpkP and the layer maximum is used
// to normalize that to a 0-1 range.
func (rl *RLRateParams) RLRateSigDeriv(act float32, laymax float32) float32 {
	if rl.On.IsFalse() || laymax == 0 {
		return 1.0
	}
	ca := act / laymax
	lr := 4.0 * ca * (1 - ca) // .5 * .5 = .25 = peak
	if lr < rl.SigmoidMin {
		lr = rl.SigmoidMin
	}
	return lr
}

// RLRateDiff returns the learning rate as a function of difference between
// CaSpkP and CaSpkD values
func (rl *RLRateParams) RLRateDiff(scap, scad float32) float32 {
	if rl.On.IsFalse() || rl.Diff.IsFalse() {
		return 1.0
	}
	smax := mat32.Max(scap, scad)
	if smax > rl.SpkThr { // avoid div by 0
		dif := mat32.Abs(scap - scad)
		if dif < rl.DiffThr {
			return rl.Min
		}
		return (dif / smax)
	}
	return rl.Min
}

// axon.LearnNeurParams manages learning-related parameters at the neuron-level.
// This is mainly the running average activations that drive learning
type LearnNeurParams struct {
	CaLearn   CaLrnParams      `view:"inline" desc:"parameterizes the neuron-level calcium signals driving learning: CaLrn = NMDA + VGCC Ca sources, where VGCC can be simulated from spiking or use the more complex and dynamic VGCC channel directly.  CaLrn is then integrated in a cascading manner at multiple time scales: CaM (as in calmodulin), CaP (ltP, CaMKII, plus phase), CaD (ltD, DAPK1, minus phase)."`
	CaSpk     CaSpkParams      `view:"inline" desc:"parameterizes the neuron-level spike-driven calcium signals, starting with CaSyn that is integrated at the neuron level, and drives synapse-level, pre * post Ca integration, which provides the Tr trace that multiplies error signals, and drives learning directly for Target layers. CaSpk* values are integrated separately at the Neuron level and used for UpdtThr and RLRate as a proxy for the activation (spiking) based learning signal."`
	LrnNMDA   chans.NMDAParams `view:"inline" desc:"NMDA channel parameters used for learning, vs. the ones driving activation -- allows exploration of learning parameters independent of their effects on active maintenance contributions of NMDA, and may be supported by different receptor subtypes"`
	TrgAvgAct TrgAvgActParams  `view:"inline" desc:"synaptic scaling parameters for regulating overall average activity compared to neuron's own target level"`
	RLRate    RLRateParams     `view:"inline" desc:"recv neuron learning rate modulation params -- an additional error-based modulation of learning for receiver side: RLRate = |CaSpkP - CaSpkD| / Max(CaSpkP, CaSpkD)"`
	NeuroMod  NeuroModParams   `view:"inline" desc:"neuromodulation effects on learning rate and activity, as a function of layer-level DA and ACh values, which are updated from global Context values, and computed from reinforcement learning algorithms"`
}

func (ln *LearnNeurParams) Update() {
	ln.CaLearn.Update()
	ln.CaSpk.Update()
	ln.LrnNMDA.Update()
	ln.TrgAvgAct.Update()
	ln.RLRate.Update()
	ln.NeuroMod.Update()
}

func (ln *LearnNeurParams) Defaults() {
	ln.CaLearn.Defaults()
	ln.CaSpk.Defaults()
	ln.LrnNMDA.Defaults()
	ln.LrnNMDA.ITau = 1
	ln.LrnNMDA.Update()
	ln.TrgAvgAct.Defaults()
	ln.RLRate.Defaults()
	ln.NeuroMod.Defaults()
}

// InitCaLrnSpk initializes the neuron-level calcium learning and spking variables.
// Called by InitWts (at start of learning).
func (ln *LearnNeurParams) InitNeurCa(ctx *Context, ni, di uint32) {
	SetNrnV(ctx, ni, di, GnmdaLrn, 0)
	SetNrnV(ctx, ni, di, NmdaCa, 0)

	SetNrnV(ctx, ni, di, VgccCa, 0)
	SetNrnV(ctx, ni, di, VgccCaInt, 0)

	SetNrnV(ctx, ni, di, CaLrn, 0)

	SetNrnV(ctx, ni, di, CaSyn, 0)
	SetNrnV(ctx, ni, di, CaSpkM, 0)
	SetNrnV(ctx, ni, di, CaSpkP, 0)
	SetNrnV(ctx, ni, di, CaSpkD, 0)
	SetNrnV(ctx, ni, di, CaSpkPM, 0)

	SetNrnV(ctx, ni, di, NrnCaM, 0)
	SetNrnV(ctx, ni, di, NrnCaP, 0)
	SetNrnV(ctx, ni, di, NrnCaD, 0)
	SetNrnV(ctx, ni, di, CaDiff, 0)
}

// LrnNMDAFmRaw updates the separate NMDA conductance and calcium values
// based on GeTot = GeRaw + external ge conductance.  These are the variables
// that drive learning -- can be the same as activation but also can be different
// for testing learning Ca effects independent of activation effects.
func (ln *LearnNeurParams) LrnNMDAFmRaw(ctx *Context, ni, di uint32, geTot float32) {
	if geTot < 0 {
		geTot = 0
	}
	vmd := NrnV(ctx, ni, di, VmDend)
	SetNrnV(ctx, ni, di, GnmdaLrn, ln.LrnNMDA.NMDASyn(NrnV(ctx, ni, di, GnmdaLrn), geTot))
	gnmda := ln.LrnNMDA.Gnmda(NrnV(ctx, ni, di, GnmdaLrn), vmd)
	SetNrnV(ctx, ni, di, NmdaCa, gnmda*ln.LrnNMDA.CaFmV(vmd))
}

// CaFmSpike updates all spike-driven calcium variables, including CaLrn and CaSpk.
// Computed after new activation for current cycle is updated.
func (ln *LearnNeurParams) CaFmSpike(ctx *Context, ni, di uint32) {
	ln.CaSpk.CaFmSpike(ctx, ni, di)
	ln.CaLearn.CaLrns(ctx, ni, di)
}

//gosl: end learn_neur

///////////////////////////////////////////////////////////////////////
// Prjn level learning params

//gosl: hlsl learn
// #include "minmax.hlsl"
// #include "kinase.hlsl"
// #include "synapse.hlsl"
// #include "neuron.hlsl"
//gosl: end learn

//gosl: start learn

///////////////////////////////////////////////////////////////////////
//  SWtParams

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

// SWtInitParams for initial SWt values
type SWtInitParams struct {
	SPct float32     `min:"0" max:"1" def:"0,1,0.5" desc:"how much of the initial random weights are captured in the SWt values -- rest goes into the LWt values.  1 gives the strongest initial biasing effect, for larger models that need more structural support. 0.5 should work for most models where stronger constraints are not needed."`
	Mean float32     `def:"0.5,0.4" desc:"target mean weight values across receiving neuron's projection -- the mean SWt values are constrained to remain at this value.  some projections may benefit from lower mean of .4"`
	Var  float32     `def:"0.25" desc:"initial variance in weight values, prior to constraints."`
	Sym  slbool.Bool `def:"true" desc:"symmetrize the initial weight values with those in reciprocal projection -- typically true for bidirectional excitatory connections"`
}

func (sp *SWtInitParams) Defaults() {
	sp.SPct = 0.5
	sp.Mean = 0.5
	sp.Var = 0.25
	sp.Sym.SetBool(true)
}

func (sp *SWtInitParams) Update() {
}

// SWtAdaptParams manages adaptation of SWt values
type SWtAdaptParams struct {
	On      slbool.Bool `desc:"if true, adaptation is active -- if false, SWt values are not updated, in which case it is generally good to have Init.SPct=0 too."`
	LRate   float32     `viewif:"On" def:"0.1,0.01,0.001,0.0002" desc:"learning rate multiplier on the accumulated DWt values (which already have fast LRate applied) to incorporate into SWt during slow outer loop updating -- lower values impose stronger constraints, for larger networks that need more structural support, e.g., 0.001 is better after 1,000 epochs in large models.  0.1 is fine for smaller models."`
	SubMean float32     `viewif:"On" def:"1" desc:"amount of mean to subtract from SWt delta when updating -- generally best to set to 1"`
	SigGain float32     `viewif:"On" def:"6" desc:"gain of sigmoidal constrast enhancement function used to transform learned, linear LWt values into Wt values"`
}

func (sp *SWtAdaptParams) Defaults() {
	sp.On.SetBool(true)
	sp.LRate = 0.1
	sp.SubMean = 1
	sp.SigGain = 6
	sp.Update()
}

func (sp *SWtAdaptParams) Update() {
}

//gosl: end learn

// RndVar returns the random variance in weight value (zero mean) based on Var param
func (sp *SWtInitParams) RndVar(rnd erand.Rand) float32 {
	return sp.Var * 2.0 * (rnd.Float32(-1) - 0.5)
}

// // RndVar returns the random variance (zero mean) based on DreamVar param
// func (sp *SWtAdaptParams) RndVar(rnd erand.Rand) float32 {
// 	return sp.DreamVar * 2.0 * (rnd.Float32(-1) - 0.5)
// }

//gosl: start learn

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

// SigFmLinWt returns sigmoidal contrast-enhanced weight from linear weight,
// centered at 1 and normed in range +/- 1 around that
// in preparation for multiplying times SWt
func (sp *SWtParams) SigFmLinWt(lw float32) float32 {
	var wt float32
	if sp.Adapt.SigGain == 1 {
		wt = lw
	} else if sp.Adapt.SigGain == 6 {
		wt = SigFun61(lw)
	} else {
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

// WtFmDWt updates the synaptic weights from accumulated weight changes.
// wt is the sigmoidal contrast-enhanced weight and lwt is the linear weight value.
func (sp *SWtParams) WtFmDWt(wt, lwt *float32, dwt, swt float32) {
	if dwt == 0 {
		if *wt == 0 { // restore failed wts
			*wt = sp.WtVal(swt, *lwt)
		}
		return
	}
	// note: softbound happened at dwt stage
	*lwt += dwt
	if *lwt < 0 {
		*lwt = 0
	} else if *lwt > 1 {
		*lwt = 1
	}
	*wt = sp.WtVal(swt, *lwt)
}

// InitSynCa initializes synaptic calcium state, including CaUpT
func InitSynCa(ctx *Context, syni, di uint32) {
	SetSynCaV(ctx, syni, di, CaUpT, 0)
	SetSynCaV(ctx, syni, di, CaM, 0)
	SetSynCaV(ctx, syni, di, CaP, 0)
	SetSynCaV(ctx, syni, di, CaD, 0)
}

// DecaySynCa decays synaptic calcium by given factor (between trials)
// Not used by default.
func DecaySynCa(ctx *Context, syni, di uint32, decay float32) {
	AddSynCaV(ctx, syni, di, CaM, -decay*SynCaV(ctx, syni, di, CaM))
	AddSynCaV(ctx, syni, di, CaP, -decay*SynCaV(ctx, syni, di, CaP))
	AddSynCaV(ctx, syni, di, CaD, -decay*SynCaV(ctx, syni, di, CaD))
}

//gosl: end learn

// InitWtsSyn initializes weight values based on WtInit randomness parameters
// for an individual synapse.
// It also updates the linear weight value based on the sigmoidal weight value.
func (sp *SWtParams) InitWtsSyn(ctx *Context, syni uint32, rnd erand.Rand, mean, spct float32) {
	wtv := sp.Init.RndVar(rnd)
	wt := mean + wtv
	SetSynV(ctx, syni, Wt, wt)
	SetSynV(ctx, syni, SWt, sp.ClipSWt(mean+spct*wtv))
	if spct == 0 { // this is critical for weak init wt, SPCt = 0 prjns
		SetSynV(ctx, syni, SWt, 0.5)
	}
	SetSynV(ctx, syni, LWt, sp.LWtFmWts(wt, SynV(ctx, syni, SWt)))
	SetSynV(ctx, syni, DWt, 0)
	SetSynV(ctx, syni, DSWt, 0)
}

//gosl: start learn

// LRateParams manages learning rate parameters
type LRateParams struct {
	Base  float32 `def:"0.04,0.1,0.2" desc:"base learning rate for this projection -- can be modulated by other factors below -- for larger networks, use slower rates such as 0.04, smaller networks can use faster 0.2."`
	Sched float32 `desc:"scheduled learning rate multiplier, simulating reduction in plasticity over aging"`
	Mod   float32 `desc:"dynamic learning rate modulation due to neuromodulatory or other such factors"`
	Eff   float32 `inactive:"+" desc:"effective actual learning rate multiplier used in computing DWt: Eff = eMod * Sched * Base"`
}

func (ls *LRateParams) Defaults() {
	ls.Base = 0.04
	ls.Sched = 1
	ls.Mod = 1
	ls.Update()
}

func (ls *LRateParams) Update() {
	ls.UpdateEff()
}

func (ls *LRateParams) UpdateEff() {
	ls.Eff = ls.Mod * ls.Sched * ls.Base
}

// Init initializes modulation values back to 1 and updates Eff
func (ls *LRateParams) Init() {
	ls.Sched = 1
	ls.Mod = 1
	ls.UpdateEff()
}

// TraceParams manages learning rate parameters
type TraceParams struct {
	Tau      float32 `def:"1,2,4" desc:"time constant for integrating trace over theta cycle timescales -- governs the decay rate of syanptic trace"`
	SubMean  float32 `def:"0,1" desc:"amount of the mean dWt to subtract, producing a zero-sum effect -- 1.0 = full zero-sum dWt -- only on non-zero DWts.  typically set to 0 for standard trace learning projections, although some require it for stability over the long haul.  can use SetSubMean to set to 1 after significant early learning has occurred with 0.  Some special prjn types (e.g., Hebb) benefit from SubMean = 1 always"`
	LearnThr float32 `desc:"threshold for learning, depending on different algorithms -- in Matrix and VSPatch it applies to normalized GeIntMax value -- setting this relatively high encourages sparser representations"`
	LTDFactor float32 `desc:"factor for computing LTD -- typically 1.0 for standard trace learning, and higher for more LTD (e.g., hippocampus)"`
	Dt       float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`

	pad, pad1, pad2 float32
}

func (tp *TraceParams) Defaults() {
	tp.Tau = 1
	tp.SubMean = 0
	tp.LearnThr = 0
	tp.LTDFactor = 1
	tp.Update()
}

func (tp *TraceParams) Update() {
	tp.Dt = 1.0 / tp.Tau
}

// TrFmCa returns updated trace factor as function of a
// synaptic calcium update factor and current trace
func (tp *TraceParams) TrFmCa(tr float32, ca float32) float32 {
	tr += tp.Dt * (ca - tr)
	return tr
}

//////////////////////////////////////////////////////////////////////////////////////
//  LRateMod

// LRateMod implements global learning rate modulation, based on a performance-based
// factor, for example error.  Increasing levels of the factor = higher learning rate.
// This can be added to a Sim and called prior to DWt() to dynamically change lrate
// based on overall network performance.
type LRateMod struct {
	On   slbool.Bool `desc:"toggle use of this modulation factor"`
	Base float32     `viewif:"On" min:"0" max:"1" desc:"baseline learning rate -- what you get for correct cases"`

	pad, pad1 int32

	Range minmax.F32 `viewif:"On" desc:"defines the range over which modulation occurs for the modulator factor -- Min and below get the Base level of learning rate modulation, Max and above get a modulation of 1"`
}

func (lr *LRateMod) Defaults() {
	lr.On.SetBool(true)
	lr.Base = 0.2
	lr.Range.Set(0.2, 0.8)
}

func (lr *LRateMod) Update() {
}

// Mod returns the learning rate modulation factor as a function
// of any kind of normalized modulation factor, e.g., an error measure.
// If fact <= Range.Min, returns Base
// If fact >= Range.Max, returns 1
// otherwise, returns proportional value between Base..1
func (lr *LRateMod) Mod(fact float32) float32 {
	lrm := lr.Range.NormVal(fact)    // clips to 0-1 range
	mod := lr.Base + lrm*(1-lr.Base) // resulting mod is in Base-1 range
	return mod
}

//gosl: end learn

// LRateMod calls LRateMod on given network, using computed Mod factor
// based on given normalized modulation factor
// (0 = no error = Base learning rate, 1 = maximum error).
// returns modulation factor applied.
func (lr *LRateMod) LRateMod(net *Network, fact float32) float32 {
	if lr.Range.Max == 0 {
		lr.Defaults()
	}
	if lr.On.IsFalse() {
		return 1
	}
	mod := lr.Mod(fact)
	net.LRateMod(mod)
	return mod
}

//gosl: start learn

///////////////////////////////////////////////////////////////////////
//  LearnSynParams

// LearnSynParams manages learning-related parameters at the synapse-level.
type LearnSynParams struct {
	Learn slbool.Bool `desc:"enable learning for this projection"`

	pad, pad1, pad2 int32

	LRate    LRateParams     `viewif:"Learn" desc:"learning rate parameters, supporting two levels of modulation on top of base learning rate."`
	Trace    TraceParams     `viewif:"Learn" desc:"trace-based learning parameters"`
	KinaseCa kinase.CaParams `viewif:"Learn" view:"inline" desc:"kinase calcium Ca integration parameters"`
}

func (ls *LearnSynParams) Update() {
	ls.LRate.Update()
	ls.Trace.Update()
	ls.KinaseCa.Update()
}

func (ls *LearnSynParams) Defaults() {
	ls.Learn.SetBool(true)
	ls.LRate.Defaults()
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

//gosl: end learn
