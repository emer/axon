// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"math/rand"

	"github.com/emer/axon/chans"
	"github.com/emer/axon/glong"
	"github.com/emer/axon/knadapt"
	"github.com/emer/emergent/erand"
	"github.com/emer/etable/minmax"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

///////////////////////////////////////////////////////////////////////
//  act.go contains the activation params and functions for axon

// axon.ActParams contains all the activation computation params and functions
// for basic Axon, at the neuron level .
// This is included in axon.Layer to drive the computation.
type ActParams struct {
	Spike   SpikeParams       `view:"inline" desc:"Spiking function parameters"`
	Init    ActInitParams     `view:"inline" desc:"initial values for key network state variables -- initialized in InitActs called by InitWts, and provides target values for DecayState"`
	Decay   DecayParams       `view:"inline" desc:"amount to decay between AlphaCycles, simulating passage of time and effects of saccades etc, especially important for environments with random temporal structure (e.g., most standard neural net training corpora) "`
	Dt      DtParams          `view:"inline" desc:"time and rate constants for temporal derivatives / updating of activation state"`
	Gbar    chans.Chans       `view:"inline" desc:"[Defaults: 1, .2, 1, 1] maximal conductances levels for channels"`
	Erev    chans.Chans       `view:"inline" desc:"[Defaults: 1, .3, .25, .1] reversal potentials for each channel"`
	GTarg   GTargParams       `view:"inline" desc:"target conductance levels for excitation and inhibition, driving adaptation of GScale.Scale conductance scaling"`
	Clamp   ClampParams       `view:"inline" desc:"how external inputs drive neural activations"`
	Noise   ActNoiseParams    `view:"inline" desc:"how, where, when, and how much noise to add"`
	VmRange minmax.F32        `view:"inline" desc:"range for Vm membrane potential -- [0.1, 1.0] -- important to keep just at extreme range of reversal potentials to prevent numerical instability"`
	KNa     knadapt.Params    `view:"no-inline" desc:"sodium-gated potassium channel adaptation parameters -- activates an inhibitory leak-like current as a function of neural activity (firing = Na influx) at three different time-scales (M-type = fast, Slick = medium, Slack = slow)"`
	NMDA    glong.NMDAParams  `view:"inline" desc:"NMDA channel parameters plus more general params"`
	GABAB   glong.GABABParams `view:"inline" desc:"GABA-B / GIRK channel parameters"`
}

func (ac *ActParams) Defaults() {
	ac.Spike.Defaults()
	ac.Init.Defaults()
	ac.Decay.Defaults()
	ac.Dt.Defaults()
	ac.Gbar.SetAll(1.0, 0.2, 1.0, 1.0) // E, L, I, K: gbar l = 0.2 > 0.1
	ac.Erev.SetAll(1.0, 0.3, 0.1, 0.1) // E, L, I, K: K = hyperpolarized -90mv
	ac.GTarg.Defaults()
	ac.Clamp.Defaults()
	ac.Noise.Defaults()
	ac.VmRange.Set(0.1, 1.0)
	ac.KNa.Defaults()
	ac.KNa.On = true // generally beneficial
	ac.NMDA.Defaults()
	ac.NMDA.Gbar = 0.03 // 0.3 best.
	ac.GABAB.Defaults()
	ac.Update()
}

// Update must be called after any changes to parameters
func (ac *ActParams) Update() {
	ac.Spike.Update()
	ac.Init.Update()
	ac.Decay.Update()
	ac.Dt.Update()
	ac.GTarg.Update()
	ac.Clamp.Update()
	ac.Noise.Update()
	ac.KNa.Update()
}

///////////////////////////////////////////////////////////////////////
//  Init

// DecayState decays the activation state toward initial values
// in proportion to given decay parameter.  Special case values
// such as Glong and KNa are also decayed with their
// separately parameterized values.
// Called with ac.Decay.Act by Layer during NewState
func (ac *ActParams) DecayState(nrn *Neuron, decay float32) {
	// always reset these -- otherwise get insanely large values that take forever to update
	nrn.ISI = -1
	nrn.ISIAvg = -1

	if decay > 0 { // no-op for most, but not all..
		nrn.Spike = 0
		nrn.Act -= decay * (nrn.Act - ac.Init.Act)
		nrn.ActInt -= decay * (nrn.ActInt - ac.Init.Act)
		nrn.Ge -= decay * (nrn.Ge - ac.Init.Ge)
		nrn.Gi -= decay * (nrn.Gi - ac.Init.Gi)
		nrn.Gk -= decay * nrn.Gk

		nrn.Vm -= decay * (nrn.Vm - ac.Init.Vm)

		nrn.GiSyn -= decay * nrn.GiSyn
		nrn.GiSelf -= decay * nrn.GiSelf
	}

	nrn.VmDend -= ac.Decay.Glong * (nrn.VmDend - ac.Init.Vm)

	nrn.Gnmda -= ac.Decay.Glong * nrn.Gnmda
	nrn.NMDA -= ac.Decay.Glong * nrn.NMDA
	nrn.NMDASyn -= ac.Decay.Glong * nrn.NMDASyn

	nrn.GgabaB -= ac.Decay.Glong * nrn.GgabaB
	nrn.GABAB -= ac.Decay.Glong * nrn.GABAB
	nrn.GABABx -= ac.Decay.Glong * nrn.GABABx

	nrn.GknaFast -= ac.Decay.KNa * nrn.GknaFast
	nrn.GknaMed -= ac.Decay.KNa * nrn.GknaMed
	nrn.GknaSlow -= ac.Decay.KNa * nrn.GknaSlow

	nrn.ActDel = 0
	nrn.Inet = 0
	nrn.GeRaw = 0
	nrn.GiRaw = 0
}

// InitActs initializes activation state in neuron -- called during InitWts but otherwise not
// automatically called (DecayState is used instead)
func (ac *ActParams) InitActs(nrn *Neuron) {
	nrn.Spike = 0
	nrn.ISI = -1
	nrn.ISIAvg = -1
	nrn.Act = ac.Init.Act
	nrn.ActInt = ac.Init.Act
	nrn.Ge = ac.Init.Ge
	nrn.Gi = ac.Init.Gi
	nrn.Gk = 0
	nrn.Inet = 0
	nrn.Vm = ac.Init.Vm
	nrn.VmDend = ac.Init.Vm
	nrn.Targ = 0
	nrn.Ext = 0

	nrn.ActDel = 0

	nrn.GiSyn = 0
	nrn.GiSelf = 0
	nrn.GeRaw = 0
	nrn.GiRaw = 0

	nrn.GknaFast = 0
	nrn.GknaMed = 0
	nrn.GknaSlow = 0

	nrn.Gnmda = 0
	nrn.NMDA = 0
	nrn.NMDASyn = 0

	nrn.GgabaB = 0
	nrn.GABAB = 0
	nrn.GABABx = 0

	ac.InitLongActs(nrn)
}

// InitLongActs initializes longer time-scale activation states in neuron
// (ActPrv, ActSt*, ActM, ActP, ActDif)
// Called from InitActs, which is called from InitWts, but otherwise not automatically called
// (DecayState is used instead)
func (ac *ActParams) InitLongActs(nrn *Neuron) {
	nrn.ActPrv = 0
	nrn.ActSt1 = 0
	nrn.ActSt2 = 0
	nrn.ActM = 0
	nrn.ActP = 0
	nrn.ActDif = 0
	nrn.GeM = 0
}

///////////////////////////////////////////////////////////////////////
//  Cycle

// BurstGe returns extra bursting excitatory conductance based on params
func (ac *ActParams) BurstGe(cyc int, actm float32) float32 {
	if ac.Clamp.Burst && actm < ac.Clamp.BurstThr && cyc < ac.Clamp.BurstCyc {
		return ac.Clamp.BurstGe
	}
	return 0
}

// GeFmRaw integrates Ge excitatory conductance from GeRaw value
// (can add other terms to geRaw prior to calling this)
func (ac *ActParams) GeFmRaw(nrn *Neuron, geRaw float32, cyc int, actm float32) {
	if ac.Clamp.Type == AddGeClamp && nrn.HasFlag(NeurHasExt) {
		geRaw += nrn.Ext * ac.Clamp.Ge
	}

	if ac.Clamp.Type == GeClamp && nrn.HasFlag(NeurHasExt) {
		ge := ac.Clamp.Ge + ac.BurstGe(cyc, actm)
		nrn.Ge = nrn.Ext * ge
	} else {
		ac.Dt.GeFmRaw(geRaw, &nrn.Ge, ac.Init.Ge)
	}

	// first place noise is required -- generate here!
	if ac.Noise.Type != NoNoise && !ac.Noise.Fixed && ac.Noise.Dist != erand.Mean {
		nrn.Noise = float32(ac.Noise.Gen(-1))
	}
	if ac.Noise.Type == GeNoise {
		nrn.Ge += nrn.Noise
	}
}

// GiFmRaw integrates GiSyn inhibitory synaptic conductance from GiRaw value
// (can add other terms to geRaw prior to calling this)
func (ac *ActParams) GiFmRaw(nrn *Neuron, giRaw float32) {
	ac.Dt.GiFmRaw(giRaw, &nrn.GiSyn, ac.Init.Gi)
	nrn.GiSyn = mat32.Max(nrn.GiSyn, 0) // negative inhib G doesn't make any sense
}

// InetFmG computes net current from conductances and Vm
func (ac *ActParams) InetFmG(vm, ge, gi, gk float32) float32 {
	return ge*(ac.Erev.E-vm) + ac.Gbar.L*(ac.Erev.L-vm) + gi*(ac.Erev.I-vm) + gk*(ac.Erev.K-vm)
}

// VmFmG computes membrane potential Vm from conductances Ge, Gi, and Gk.
func (ac *ActParams) VmFmG(nrn *Neuron) {
	updtVm := true
	if ac.Spike.Tr > 0 && nrn.ISI >= 0 && nrn.ISI < float32(ac.Spike.Tr) {
		updtVm = false // don't update the spiking vm during refract
	}

	nwVm := nrn.Vm
	ge := nrn.Ge * ac.Gbar.E
	gi := nrn.Gi * ac.Gbar.I
	gk := nrn.Gk * ac.Gbar.K
	if updtVm {
		vmEff := nrn.Vm
		// midpoint method: take a half-step in vmEff
		inet1 := ac.InetFmG(vmEff, ge, gi, gk)
		vmEff += .5 * ac.Dt.VmDt * inet1 // go half way
		inet2 := ac.InetFmG(vmEff, ge, gi, gk)
		// add spike current if relevant
		if ac.Spike.Exp {
			inet2 += ac.Gbar.L * ac.Spike.ExpSlope *
				mat32.FastExp((vmEff-ac.Spike.Thr)/ac.Spike.ExpSlope)
		}
		nwVm += ac.Dt.VmDt * inet2
		nrn.Inet = inet2
	}
	{ // always update VmDend
		vmEff := nrn.VmDend
		// midpoint method: take a half-step in vmEff
		inet1 := ac.InetFmG(vmEff, ge, gi, gk)
		vmEff += .5 * ac.Dt.VmDendDt * inet1 // go half way
		inet2 := ac.InetFmG(vmEff, ge, gi, gk)
		nrn.VmDend = ac.VmRange.ClipVal(nrn.VmDend + ac.Dt.VmDendDt*inet2)
	}

	if ac.Noise.Type == VmNoise {
		nwVm += nrn.Noise
	}
	nrn.Vm = ac.VmRange.ClipVal(nwVm)
}

// ActFmG computes Spike from Vm and ISI-based activation
func (ac *ActParams) ActFmG(nrn *Neuron) {
	if ac.HasRateClamp(nrn) {
		ac.RateClamp(nrn)
		return
	}
	var thr float32
	if ac.Spike.Exp {
		thr = ac.Spike.ExpThr
	} else {
		thr = ac.Spike.Thr
	}
	if nrn.Vm >= thr {
		nrn.Spike = 1
		nrn.Vm = ac.Spike.VmR
		nrn.Inet = 0
		if nrn.ISIAvg == -1 {
			nrn.ISIAvg = -2
		} else if nrn.ISI > 0 { // must have spiked to update
			ac.Spike.AvgFmISI(&nrn.ISIAvg, nrn.ISI+1)
		}
		nrn.ISI = 0
	} else {
		nrn.Spike = 0
		if nrn.ISI >= 0 {
			nrn.ISI += 1
		}
		if nrn.ISIAvg >= 0 && nrn.ISI > 0 && nrn.ISI > 1.2*nrn.ISIAvg {
			ac.Spike.AvgFmISI(&nrn.ISIAvg, nrn.ISI)
		}
	}

	nwAct := ac.Spike.ActFmISI(nrn.ISIAvg, .001, ac.Dt.Integ)
	if nwAct > 1 {
		nwAct = 1
	}
	nwAct = nrn.Act + ac.Dt.VmDt*(nwAct-nrn.Act)
	nrn.ActDel = nwAct - nrn.Act
	nrn.Act = nwAct
	if ac.KNa.On {
		ac.KNa.GcFmSpike(&nrn.GknaFast, &nrn.GknaMed, &nrn.GknaSlow, nrn.Spike > .5)
		nrn.Gk = nrn.GknaFast + nrn.GknaMed + nrn.GknaSlow
	}
}

// HasRateClamp returns true if this neuron has external input that should be hard clamped
func (ac *ActParams) HasRateClamp(nrn *Neuron) bool {
	return ac.Clamp.Type == RateClamp && nrn.HasFlag(NeurHasExt)
}

// RateClamp drives Poisson rate spiking according to external input.
// Also adds any Noise *if* noise is set to ActNoise.
func (ac *ActParams) RateClamp(nrn *Neuron) {
	ext := nrn.Ext
	if ac.Noise.Type == ActNoise {
		ext += nrn.Noise
	}
	if nrn.ISI > 1 {
		nrn.ISI -= 1
		nrn.Spike = 0
	} else {
		if ext <= 0.0001 {
			nrn.ISI = 0
			nrn.ISIAvg = -1
			nrn.Spike = 0
		} else {
			nrn.Spike = 1
			nrn.ISI = (1000 * float32(rand.ExpFloat64())) / (ac.Clamp.Rate * ext)
			nrn.ISI = mat32.Max(float32(ac.Spike.Tr), nrn.ISI)
			if nrn.ISIAvg == -1 {
				nrn.ISIAvg = -2
			} else if nrn.ISI > 0 { // must have spiked to update
				ac.Spike.AvgFmISI(&nrn.ISIAvg, nrn.ISI+1)
			}
		}
	}

	// keep everything else clamped
	nrn.Vm = ac.Init.Vm
	nrn.VmDend = ac.Init.Vm
	nrn.Inet = 0

	nwAct := ac.Spike.ActFmISI(nrn.ISIAvg, .001, ac.Dt.Integ)
	if nwAct > 1 {
		nwAct = 1
	}
	nwAct = nrn.Act + ac.Dt.VmDt*(nwAct-nrn.Act)
	nrn.ActDel = nwAct - nrn.Act
	nrn.Act = nwAct
	if ac.KNa.On {
		ac.KNa.GcFmSpike(&nrn.GknaFast, &nrn.GknaMed, &nrn.GknaSlow, nrn.Spike > .5)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  SpikeParams

// SpikeParams contains spiking activation function params.
// Implements a basic thresholded Vm model, and optionally
// the AdEx adaptive exponential function (adapt is KNaAdapt)
type SpikeParams struct {
	Thr      float32 `def:"0.5" desc:"threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization"`
	VmR      float32 `def:"0.3" desc:"post-spiking membrane potential to reset to, produces refractory effect if lower than VmInit -- 0.3 is apropriate biologically-based value for AdEx (Brette & Gurstner, 2005) parameters"`
	Tr       int     `def:"3" desc:"post-spiking explicit refractory period, in cycles -- prevents Vm updating for this number of cycles post firing"`
	Exp      bool    `def:"true" desc:"if true, turn on exponential excitatory current that drives Vm rapidly upward for spiking as it gets past its nominal firing threshold (Thr) -- nicely captures the Hodgkin Huxley dynamics of Na and K channels -- uses Brette & Gurstner 2005 AdEx formulation"`
	ExpSlope float32 `viewif:"Exp" def:"0.02" desc:"slope in Vm (2 mV = .02 in normalized units) for extra exponential excitatory current that drives Vm rapidly upward for spiking as it gets past its nominal firing threshold (Thr) -- nicely captures the Hodgkin Huxley dynamics of Na and K channels -- uses Brette & Gurstner 2005 AdEx formulation"`
	ExpThr   float32 `viewif:"Exp" def:"1" desc:"membrane potential threshold for actually triggering a spike when using the exponential mechanism"`
	MaxHz    float32 `def:"180" min:"1" desc:"for translating spiking interval (rate) into rate-code activation equivalent (and vice-versa, for clamped layers), what is the maximum firing rate associated with a maximum activation value (max act is typically 1.0 -- depends on act_range)"`
	ISITau   float32 `def:"5" min:"1" desc:"constant for integrating the spiking interval in estimating spiking rate"`
	ISIDt    float32 `view:"-" desc:"rate = 1 / tau"`
}

func (sk *SpikeParams) Defaults() {
	sk.Thr = 0.5
	sk.VmR = 0.3
	sk.Tr = 3
	sk.Exp = true
	sk.ExpSlope = 0.02
	sk.ExpThr = 1.0
	sk.MaxHz = 180
	sk.ISITau = 5
	sk.Update()
}

func (sk *SpikeParams) Update() {
	sk.ISIDt = 1 / sk.ISITau
}

// ActToISI compute spiking interval from a given rate-coded activation,
// based on time increment (.001 = 1msec default), Act.Dt.Integ
func (sk *SpikeParams) ActToISI(act, timeInc, integ float32) float32 {
	if act == 0 {
		return 0
	}
	return (1 / (timeInc * integ * act * sk.MaxHz))
}

// ActFmISI computes rate-code activation from estimated spiking interval
func (sk *SpikeParams) ActFmISI(isi, timeInc, integ float32) float32 {
	if isi <= 0 {
		return 0
	}
	maxInt := 1.0 / (timeInc * integ * sk.MaxHz) // interval at max hz..
	return maxInt / isi                          // normalized
}

// AvgFmISI updates spiking ISI from current isi interval value
func (sk *SpikeParams) AvgFmISI(avg *float32, isi float32) {
	if *avg <= 0 {
		*avg = isi
	} else if isi < 0.8**avg {
		*avg = isi // if significantly less than we take that
	} else { // integrate on slower
		*avg += sk.ISIDt * (isi - *avg) // running avg updt
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  ActInitParams

// ActInitParams are initial values for key network state variables.
// Initialized in InitActs called by InitWts, and provides target values for DecayState.
type ActInitParams struct {
	Vm  float32 `def:"0.3" desc:"initial membrane potential -- see Erev.L for the resting potential (typically .3)"`
	Act float32 `def:"0" desc:"initial activation value -- typically 0"`
	Ge  float32 `def:"0" desc:"baseline level of excitatory conductance (net input) -- Ge is initialized to this value, and it is added in as a constant background level of excitatory input -- captures all the other inputs not represented in the model, and intrinsic excitability, etc"`
	Gi  float32 `def:"0" desc:"baseline level of inhibitory conductance (net input) -- Gi is initialized to this value, and it is added in as a constant background level of inhibitory input -- captures all the other inputs not represented in the model"`
}

func (ai *ActInitParams) Update() {
}

func (ai *ActInitParams) Defaults() {
	ai.Vm = 0.3
	ai.Act = 0
	ai.Ge = 0
	ai.Gi = 0
}

//////////////////////////////////////////////////////////////////////////////////////
//  DecayParams

// DecayParams control the decay of activation state in the DecayState function
// called in NewState when a new state is to be processed.
type DecayParams struct {
	Act   float32 `def:"0,0.2,0.5,1" max:"1" min:"0" desc:"proportion to decay most activation state variables toward initial values at start of every AlphaCycle (except those controlled separately below) -- if 1 it is effectively equivalent to full clear, resetting other derived values.  ISI is reset every AlphaCycle to get a fresh sample of activations (doesn't affect direct computation -- only readout)."`
	Glong float32 `def:"0,0.6" max:"1" min:"0" desc:"proportion to decay long-lasting conductances, NMDA and GABA, and also the dendritic membrane potential -- when using random stimulus order, it is important to decay this significantly to allow a fresh start -- but set Act to 0 to enable ongoing activity to keep neurons in their sensitive regime."`
	KNa   float32 `max:"1" min:"0" desc:"decay of Kna adaptation values -- has a separate decay because often useful to have this not decay at all even if decay is on."`
}

func (ai *DecayParams) Update() {
}

func (ai *DecayParams) Defaults() {
	ai.Act = 0.2
	ai.Glong = 0.6
	ai.KNa = 0
}

//////////////////////////////////////////////////////////////////////////////////////
//  DtParams

// DtParams are time and rate constants for temporal derivatives in Axon (Vm, G)
type DtParams struct {
	Integ      float32 `def:"1,0.5" min:"0" desc:"overall rate constant for numerical integration, for all equations at the unit level -- all time constants are specified in millisecond units, with one cycle = 1 msec -- if you instead want to make one cycle = 2 msec, you can do this globally by setting this integ value to 2 (etc).  However, stability issues will likely arise if you go too high.  For improved numerical stability, you may even need to reduce this value to 0.5 or possibly even lower (typically however this is not necessary).  MUST also coordinate this with network.time_inc variable to ensure that global network.time reflects simulated time accurately"`
	VmTau      float32 `def:"2.81" min:"1" desc:"membrane potential time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life) -- reflects the capacitance of the neuron in principle -- biological default for AdEx spiking model C = 281 pF = 2.81 normalized"`
	VmDendTau  float32 `def:"5" min:"1" desc:"dendritic membrane potential integration time constant"`
	GeTau      float32 `def:"5" min:"1" desc:"time constant for decay of excitatory AMPA receptor conductance."`
	GiTau      float32 `def:"7" min:"1" desc:"time constant for decay of inhibitory GABAa receptor conductance."`
	IntTau     float32 `def:"20" min:"1" desc:"time constant for integrating AvgS values over time, used in computing ActInt, and for GeM from Ge -- this is used for scoring performance, not for learning, in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life), "`
	LongAvgTau float32 `def:"20" desc:"time constant for integrating slower long-time-scale averages, such as nrn.ActAvg, ly.ActAvg.AvgMaxGeM, Pool.ActsMAvg, ActsPAvg in trials (tau is roughly how long it takes for value to change significantly) -- set lower for smaller models"`

	VmDt      float32 `view:"-" json:"-" xml:"-" desc:"nominal rate = Integ / tau"`
	VmDendDt  float32 `view:"-" json:"-" xml:"-" desc:"nominal rate = Integ / tau"`
	GeDt      float32 `view:"-" json:"-" xml:"-" desc:"rate = Integ / tau"`
	GiDt      float32 `view:"-" json:"-" xml:"-" desc:"rate = Integ / tau"`
	IntDt     float32 `view:"-" json:"-" xml:"-" desc:"rate = Integ / tau"`
	LongAvgDt float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (dp *DtParams) Update() {
	dp.VmDt = dp.Integ / dp.VmTau
	dp.VmDendDt = dp.Integ / dp.VmDendTau
	dp.GeDt = dp.Integ / dp.GeTau
	dp.GiDt = dp.Integ / dp.GiTau
	dp.IntDt = dp.Integ / dp.IntTau
	dp.LongAvgDt = 1 / dp.LongAvgTau
}

func (dp *DtParams) Defaults() {
	dp.Integ = 1
	dp.VmTau = 2.81
	dp.VmDendTau = 5
	dp.GeTau = 5
	dp.GiTau = 7
	dp.IntTau = 20
	dp.LongAvgTau = 20
	dp.Update()
}

// GeFmRaw updates ge from raw input, decaying with time constant, back to min baseline value
func (dp *DtParams) GeFmRaw(geRaw float32, ge *float32, min float32) {
	*ge += geRaw - dp.GeDt*(*ge-min)
}

// GiFmRaw updates gi from raw input, decaying with time constant, back to min baseline value
func (dp *DtParams) GiFmRaw(giRaw float32, gi *float32, min float32) {
	*gi += giRaw - dp.GiDt*(*gi-min)
}

// AvgVarUpdt updates the average and variance from current value, using LongAvgDt
func (dp *DtParams) AvgVarUpdt(avg, vr *float32, val float32) {
	if *avg == 0 { // first time -- set
		*avg = val
		*vr = 0
	} else {
		del := val - *avg
		incr := dp.LongAvgDt * del
		*avg += incr
		// following is magic exponentially-weighted incremental variance formula
		// derived by Finch, 2009: Incremental calculation of weighted mean and variance
		if *vr == 0 {
			*vr = 2 * (1 - dp.LongAvgDt) * del * incr
		} else {
			*vr = (1 - dp.LongAvgDt) * (*vr + del*incr)
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
// GTargParams

// GTargParams are target conductance levels for excitation and inhibition,
// driving adaptation of GScale.Scale conductance scaling
type GTargParams struct {
	GeMax float32 `def:"1" min:"0" desc:"target maximum excitatory conductance in the minus phase: GeM"`
	GiMax float32 `def:"1" min:"0" desc:"target maximum inhibitory conductance in the minus phase: GiM -- for actual synaptic inhibitory neuron inputs (GiSyn) not FFFB computed inhibition"`
}

func (gt *GTargParams) Update() {
}

func (gt *GTargParams) Defaults() {
	gt.GeMax = 1
	gt.GiMax = 1
	gt.Update()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Noise

// ActNoiseTypes are different types / locations of random noise for activations
type ActNoiseTypes int

//go:generate stringer -type=ActNoiseTypes

var KiT_ActNoiseTypes = kit.Enums.AddEnum(ActNoiseTypesN, kit.NotBitFlag, nil)

func (ev ActNoiseTypes) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *ActNoiseTypes) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// NoNoise means no noise added
	NoNoise ActNoiseTypes = iota

	// VmNoise means noise is added to the membrane potential.
	VmNoise

	// GeNoise means noise is added to the excitatory conductance (Ge).
	GeNoise

	// ActNoise means noise is added to the final rate code activation
	ActNoise

	// GeMultNoise means that noise is multiplicative on the Ge excitatory conductance values
	GeMultNoise

	ActNoiseTypesN
)

// ActNoiseParams contains parameters for activation-level noise
type ActNoiseParams struct {
	erand.RndParams
	Type  ActNoiseTypes `desc:"where and how to add processing noise"`
	Fixed bool          `desc:"keep the same noise value over the entire alpha cycle -- prevents noise from being washed out and produces a stable effect that can be better used for learning -- this is strongly recommended for most learning situations"`
}

func (an *ActNoiseParams) Update() {
}

func (an *ActNoiseParams) Defaults() {
	an.Fixed = false
}

//////////////////////////////////////////////////////////////////////////////////////
//  ClampParams

// ClampTypes are different types of clamping
type ClampTypes int

//go:generate stringer -type=ClampTypes

var KiT_ClampTypes = kit.Enums.AddEnum(ClampTypesN, kit.NotBitFlag, nil)

func (ev ClampTypes) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *ClampTypes) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// GeClamp drives a constant excitatory input given by Ge value
	// ignoring any other source of Ge input -- like a current clamp.
	// This works best in general by allowing more natural temporal dynamics.
	GeClamp ClampTypes = iota

	// RateClamp drives a poisson firing rate in proportion to clamped value.
	RateClamp

	// AddGeClamp adds a constant extra Ge value on top of existing Ge inputs
	AddGeClamp

	ClampTypesN
)

// ClampParams are for specifying how external inputs are clamped onto network activation values
type ClampParams struct {
	ErrThr   float32    `def:"0.5" desc:"threshold on neuron Act activity to count as active for computing error relative to target in PctErr method"`
	Type     ClampTypes `def:"GeClamp" desc:"type of clamping to use -- GeClamp provides a more natural input with initial spike onset related to input strength, and some adaptation effects, etc"`
	Rate     float32    `viewif:"Type=RateClamp" def:"180" desc:"for RateClamp mode, maximum spiking rate in Hz for Poisson spike generator (multiplies clamped input value to get rate)"`
	Ge       float32    `viewif:"Type!=RateClamp" def:"0.2,0.6" desc:"amount of Ge driven for clamping, for GeClamp and AddGeClamp"`
	Burst    bool       `viewif:"Type=GeClamp" desc:"activate bursting at start of clamping window"`
	BurstThr float32    `def:"0.5" viewif:"Burst&&Type=GeClamp" desc:"for Target layers, if ActM < this threshold then the neuron bursts -- otherwise the burst is adapted and doesn't apply -- amplifies errors -- set to 1 to always burst (e.g., for input layers)"`
	BurstCyc int        `def:"0,20" viewif:"Burst&&Type=GeClamp" desc:"duration of extra bursting -- for Target layers, at start of plus phase, else start of alpha cycle"`
	BurstGe  float32    `def:"2" viewif:"Burst&&Type=GeClamp" desc:"extra bursting Ge during BurstCyc cycles -- added to Ge -- 2 for maximum kick -- 1 and 1.5 should have slightly weaker effects.  Can potentially be useful to set Act.Spike.Tr = 2 to speed up bursting."`
}

func (cp *ClampParams) Update() {
}

func (cp *ClampParams) Defaults() {
	cp.ErrThr = 0.5
	cp.Type = GeClamp
	cp.Rate = 180
	cp.Ge = 0.6
	cp.Burst = false // maybe not necessary
	cp.BurstCyc = 20
	cp.BurstThr = 0.5
	cp.BurstGe = 2.0
}

//////////////////////////////////////////////////////////////////////////////////////
//  SynComParams

/// SynComParams are synaptic communication parameters: delay and probability of failure
type SynComParams struct {
	Delay      int     `desc:"synaptic delay for inputs arriving at this projection -- IMPORTANT: if you change this, you must rebuild network!"`
	PFail      float32 `desc:"probability of synaptic transmission failure -- if > 0, then weights are turned off at random as a function of PFail * (1-Min(Wt/Max, 1))^2"`
	PFailWtMax float32 `desc:"maximum weight value that experiences no synaptic failure -- weights at or above this level never fail to communicate, while probability of failure increases parabolically below this level -- enter 0 to have a uniform probability of failure regardless of weight size"`
}

func (sc *SynComParams) Defaults() {
	sc.Delay = 2
	sc.PFail = 0 // 0.5 works?
	sc.PFailWtMax = 0.8
}

func (sc *SynComParams) Update() {
}

// WtFailP returns probability of weight (synapse) failure given current weight value
func (sc *SynComParams) WtFailP(wt float32) float32 {
	if sc.PFailWtMax == 0 {
		return sc.PFail
	}
	if wt >= sc.PFailWtMax {
		return 0
	}
	weff := 1 - wt/sc.PFailWtMax
	return sc.PFail * weff * weff
}

// WtFail returns true if synapse should fail
func (sc *SynComParams) WtFail(wt float32) bool {
	fp := sc.WtFailP(wt)
	if fp == 0 {
		return false
	}
	return erand.BoolP(fp)
}

// Fail updates failure status of given weight
func (sc *SynComParams) Fail(wt *float32) {
	if sc.PFail > 0 {
		if sc.WtFail(*wt) {
			*wt = 0
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  PrjnScaleParams

// PrjnScaleParams are projection scaling parameters: modulates overall strength of projection,
// using both absolute and relative factors.
// Also includes ability to adapt Scale factors to maintain AvgMaxGeM / GiM max conductances
// according to Acts.GTarg target values.
type PrjnScaleParams struct {
	Rel        float32 `min:"0" desc:"[Defaults: Forward=1, Back=0.2] relative scaling that shifts balance between different projections -- this is subject to normalization across all other projections into receiving neuron, and determines the GScale.Targ for adapting scaling"`
	Init       float32 `def:"1" min:"0" desc:"adjustment factor for the initial scaling -- can be used to adjust for idiosyncrasies not accommodated by the standard scaling -- typically Adapt should compensate for most cases"`
	Adapt      bool    `def:"true" desc:"Adapt the 'GScale' scaling value so the ActAvg.AvgMaxGeM / GiM running-average value for this projections remains in the target range, specified in Acts.GTarg"`
	ScaleLrate float32 `viewif:"Adapt" def:"0.5" desc:"learning rate for adapting the GScale value, as function of target value -- lrate is also multiplied by the GScale.Orig to compensate for significant differences in overall scale of these scaling factors -- fastest value with some smoothing at .5 works well."`
	HiTol      float32 `def:"0" viewif:"Adapt" desc:"tolerance for higher than target AvgMaxGeM / GiM as a proportion of that target value (0 = exactly the target, 0.2 = 20% higher than target) -- only once activations move outside this tolerance are scale values adapted"`
	LoTol      float32 `def:"0.8" viewif:"Adapt" desc:"tolerance for lower than target AvgMaxGeM / GiM as a proportion of that target value (0 = exactly the target, 0.8 = 80% lower than target) -- only once activations move outside this tolerance are scale values adapted"`
	AvgTau     float32 `def:"500" desc:"time constant for integrating projection-level averages for this scaling process: Prjn.GScale.AvgAvg, AvgMax (tau is roughly how long it takes for value to change significantly) -- these are updated at the cycle level and thus require a much slower rate constant compared to other such variables integrated at the AlphaCycle level."`

	AvgDt float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (ws *PrjnScaleParams) Defaults() {
	ws.Rel = 1
	ws.Init = 1
	ws.Adapt = true
	ws.ScaleLrate = 0.5
	ws.HiTol = 0
	ws.LoTol = 0.8
	ws.AvgTau = 500
	ws.Update()
}

func (ws *PrjnScaleParams) Update() {
	ws.AvgDt = 1 / ws.AvgTau
}

// SLayActScale computes scaling factor based on sending layer activity level (savg), number of units
// in sending layer (snu), and number of recv connections (ncon).
// Uses a fixed sem_extra standard-error-of-the-mean (SEM) extra value of 2
// to add to the average expected number of active connections to receive,
// for purposes of computing scaling factors with partial connectivity
// For 25% layer activity, binomial SEM = sqrt(p(1-p)) = .43, so 3x = 1.3 so 2 is a reasonable default.
func (ws *PrjnScaleParams) SLayActScale(savg, snu, ncon float32) float32 {
	ncon = mat32.Max(ncon, 1) // prjn Avg can be < 1 in some cases
	semExtra := 2
	slayActN := int(mat32.Round(savg * snu)) // sending layer actual # active
	slayActN = ints.MaxInt(slayActN, 1)
	var sc float32
	if ncon == snu {
		sc = 1 / float32(slayActN)
	} else {
		maxActN := int(mat32.Min(ncon, float32(slayActN))) // max number we could get
		avgActN := int(mat32.Round(savg * ncon))           // recv average actual # active if uniform
		avgActN = ints.MaxInt(avgActN, 1)
		expActN := avgActN + semExtra // expected
		expActN = ints.MinInt(expActN, maxActN)
		sc = 1 / float32(expActN)
	}
	return sc
}

// FullScale returns full scaling factor, which is product of Init * Rel * SLayActScale
func (ws *PrjnScaleParams) FullScale(savg, snu, ncon float32) float32 {
	return ws.Init * ws.Rel * ws.SLayActScale(savg, snu, ncon)
}
