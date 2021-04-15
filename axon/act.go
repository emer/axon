// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"math/rand"

	"github.com/chewxy/math32"
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
	Init    ActInitParams     `view:"inline" desc:"initial values for key network state variables -- initialized at start of trial with InitActs or DecayActs"`
	Dt      DtParams          `view:"inline" desc:"time and rate constants for temporal derivatives / updating of activation state"`
	Gbar    chans.Chans       `view:"inline" desc:"[Defaults: 1, .2, 1, 1] maximal conductances levels for channels"`
	Erev    chans.Chans       `view:"inline" desc:"[Defaults: 1, .3, .25, .1] reversal potentials for each channel"`
	Clamp   ClampParams       `view:"inline" desc:"how external inputs drive neural activations"`
	Noise   ActNoiseParams    `view:"inline" desc:"how, where, when, and how much noise to add"`
	VmRange minmax.F32        `view:"inline" desc:"range for Vm membrane potential -- [0, 2.0] by default"`
	KNa     knadapt.Params    `view:"no-inline" desc:"sodium-gated potassium channel adaptation parameters -- activates an inhibitory leak-like current as a function of neural activity (firing = Na influx) at three different time-scales (M-type = fast, Slick = medium, Slack = slow)"`
	NMDA    glong.NMDAParams  `view:"inline" desc:"NMDA channel parameters plus more general params"`
	GABAB   glong.GABABParams `view:"inline" desc:"GABA-B / GIRK channel parameters"`
}

func (ac *ActParams) Defaults() {
	ac.Spike.Defaults()
	ac.Init.Defaults()
	ac.Dt.Defaults()
	ac.Gbar.SetAll(1.0, 0.2, 1.0, 1.0)
	ac.Erev.SetAll(1.0, 0.3, 0.25, 0.1) // K = hyperpolarized -90mv
	ac.Clamp.Defaults()
	ac.Noise.Defaults()
	ac.VmRange.Max = 2.0
	ac.KNa.Defaults()
	ac.KNa.On = true
	ac.NMDA.Defaults()
	ac.NMDA.Gbar = 0.01 // a bit weaker by default
	ac.GABAB.Defaults()
	ac.Update()
}

// Update must be called after any changes to parameters
func (ac *ActParams) Update() {
	ac.Spike.Update()
	ac.Init.Update()
	ac.Dt.Update()
	ac.Clamp.Update()
	ac.Noise.Update()
	ac.KNa.Update()
}

///////////////////////////////////////////////////////////////////////
//  Init

// DecayState decays the activation state toward initial values in proportion to given decay parameter
// Called with ac.Init.Decay by Layer during AlphaCycInit
func (ac *ActParams) DecayState(nrn *Neuron, decay float32) {
	if decay > 0 { // no-op for most, but not all..
		nrn.Act -= decay * (nrn.Act - ac.Init.Act)
		nrn.Ge -= decay * (nrn.Ge - ac.Init.Ge)
		nrn.Gi -= decay * (nrn.Gi - ac.Init.Gi)
		nrn.GiSelf -= decay * nrn.GiSelf
		nrn.Gk -= decay * nrn.Gk
		nrn.Vm -= decay * (nrn.Vm - ac.Init.Vm)
		nrn.VmDend -= decay * (nrn.VmDend - ac.Init.Vm)
		nrn.GiSyn -= decay * nrn.GiSyn
		nrn.Gnmda -= decay * nrn.Gnmda
		nrn.NMDA -= decay * nrn.NMDA
		nrn.NMDASyn -= decay * nrn.NMDASyn
		nrn.GgabaB -= decay * nrn.GgabaB
		nrn.GABAB -= decay * nrn.GABAB
		nrn.GABABx -= decay * nrn.GABABx
	}
	nrn.ActDel = 0
	nrn.Inet = 0
}

// InitActs initializes activation state in neuron -- called during InitWts but otherwise not
// automatically called (DecayState is used instead)
func (ac *ActParams) InitActs(nrn *Neuron) {
	nrn.Spike = 0
	nrn.ISI = -1
	nrn.ISIAvg = -1
	nrn.Act = ac.Init.Act
	nrn.ActLrn = ac.Init.Act
	nrn.Ge = ac.Init.Ge
	nrn.Gi = ac.Init.Gi
	nrn.Gk = 0
	nrn.GknaFast = 0
	nrn.GknaMed = 0
	nrn.GknaSlow = 0
	nrn.GiSelf = 0
	nrn.GiSyn = 0
	nrn.Inet = 0
	nrn.Vm = ac.Init.Vm
	nrn.VmDend = ac.Init.Vm
	nrn.Targ = 0
	nrn.Ext = 0
	nrn.ActDel = 0
	nrn.Gnmda = 0
	nrn.NMDA = 0
	nrn.NMDASyn = 0
	nrn.GgabaB = 0
	nrn.GABAB = 0
	nrn.GABABx = 0

	ac.InitActQs(nrn)
}

// InitActQs initializes quarter-based activation states in neuron (ActQ0-2, ActM, ActP, ActDif)
// Called from InitActs, which is called from InitWts, but otherwise not automatically called
// (DecayState is used instead)
func (ac *ActParams) InitActQs(nrn *Neuron) {
	nrn.ActQ0 = 0
	nrn.ActQ1 = 0
	nrn.ActQ2 = 0
	nrn.ActM = 0
	nrn.ActP = 0
	nrn.ActDif = 0
}

///////////////////////////////////////////////////////////////////////
//  Cycle

// GeFmRaw integrates Ge excitatory conductance from GeRaw value
// (can add other terms to geRaw prior to calling this)
func (ac *ActParams) GeFmRaw(nrn *Neuron, geRaw float32) {
	if !ac.Clamp.Hard && nrn.HasFlag(NeurHasExt) {
		if ac.Clamp.Avg {
			geRaw = ac.Clamp.AvgGe(nrn.Ext, geRaw)
		} else {
			geRaw += nrn.Ext * ac.Clamp.Gain
		}
	}

	ac.Dt.GeFmRaw(geRaw, &nrn.Ge, ac.Init.Ge)
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
	nrn.GiSyn = math32.Max(nrn.GiSyn, 0) // negative inhib G doesn't make any sense
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
				math32.Exp((vmEff-ac.Spike.Thr)/ac.Spike.ExpSlope)
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
	if ac.HasHardClamp(nrn) {
		ac.HardClamp(nrn) // todo: spiking..
		return
	}
	var thr float32
	if ac.Spike.Exp {
		thr = ac.Spike.ExpThr
	} else {
		thr = ac.Spike.Thr
	}
	if nrn.Vm > thr {
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

// HasHardClamp returns true if this neuron has external input that should be hard clamped
func (ac *ActParams) HasHardClamp(nrn *Neuron) bool {
	return ac.Clamp.Hard && nrn.HasFlag(NeurHasExt)
}

// HardClamp drives Poisson rate spiking according to external input.
// Also adds any Noise *if* noise is set to ActNoise.
func (ac *ActParams) HardClamp(nrn *Neuron) {
	ext := nrn.Ext
	if ac.Noise.Type == ActNoise {
		ext += nrn.Noise
	}
	if nrn.ISI > 1 {
		nrn.ISI -= 1
		nrn.Spike = 0
	} else {
		if ext <= 0 {
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
	VmR      float32 `def:"0.3,0,0.15" desc:"post-spiking membrane potential to reset to, produces refractory effect if lower than VmInit -- 0.30 is apropriate biologically-based value for AdEx (Brette & Gurstner, 2005) parameters"`
	Tr       int     `def:"3" desc:"post-spiking explicit refractory period, in cycles -- prevents Vm updating for this number of cycles post firing"`
	MaxHz    float32 `def:"180" min:"1" desc:"for translating spiking interval (rate) into rate-code activation equivalent (and vice-versa, for clamped layers), what is the maximum firing rate associated with a maximum activation value (max act is typically 1.0 -- depends on act_range)"`
	RateTau  float32 `def:"5" min:"1" desc:"constant for integrating the spiking interval in estimating spiking rate"`
	RateDt   float32 `view:"-" desc:"rate = 1 / tau"`
	Exp      bool    `def:"true" desc:"if true, turn on exponential excitatory current that drives Vm rapidly upward for spiking as it gets past its nominal firing threshold (Thr) -- nicely captures the Hodgkin Huxley dynamics of Na and K channels -- uses Brette & Gurstner 2005 AdEx formulation"`
	ExpSlope float32 `viewif:"Exp" def:"0.02" desc:"slope in Vm (2 mV = .02 in normalized units) for extra exponential excitatory current that drives Vm rapidly upward for spiking as it gets past its nominal firing threshold (Thr) -- nicely captures the Hodgkin Huxley dynamics of Na and K channels -- uses Brette & Gurstner 2005 AdEx formulation"`
	ExpThr   float32 `viewif:"Exp" def:"1" desc:"membrane potential threshold for actually triggering a spike when using the exponential mechanism"`
}

func (sk *SpikeParams) Defaults() {
	sk.Thr = 0.5
	sk.VmR = 0.3
	sk.Tr = 3
	sk.MaxHz = 180
	sk.RateTau = 5
	sk.Exp = true
	sk.ExpSlope = 0.02
	sk.ExpThr = 1.0
	sk.Update()
}

func (sk *SpikeParams) Update() {
	sk.RateDt = 1 / sk.RateTau
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
		*avg += sk.RateDt * (isi - *avg) // running avg updt
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  ActInitParams

// ActInitParams are initial values for key network state variables.
// Initialized at start of trial with Init_Acts or DecayState.
type ActInitParams struct {
	Decay float32 `def:"0,1" max:"1" min:"0" desc:"proportion to decay activation state toward initial values at start of every trial"`
	Vm    float32 `def:"0.3" desc:"initial membrane potential -- see e_rev.l for the resting potential (typically .3)"`
	Act   float32 `def:"0" desc:"initial activation value -- typically 0"`
	Ge    float32 `def:"0" desc:"baseline level of excitatory conductance (net input) -- Ge is initialized to this value, and it is added in as a constant background level of excitatory input -- captures all the other inputs not represented in the model, and intrinsic excitability, etc"`
	Gi    float32 `def:"0" desc:"baseline level of inhibitory conductance (net input) -- Gi is initialized to this value, and it is added in as a constant background level of inhibitory input -- captures all the other inputs not represented in the model"`
}

func (ai *ActInitParams) Update() {
}

func (ai *ActInitParams) Defaults() {
	ai.Decay = 0
	ai.Vm = 0.3
	ai.Act = 0
	ai.Ge = 0
	ai.Gi = 0
}

//////////////////////////////////////////////////////////////////////////////////////
//  DtParams

// DtParams are time and rate constants for temporal derivatives in Axon (Vm, G)
type DtParams struct {
	Integ     float32 `def:"1,0.5" min:"0" desc:"overall rate constant for numerical integration, for all equations at the unit level -- all time constants are specified in millisecond units, with one cycle = 1 msec -- if you instead want to make one cycle = 2 msec, you can do this globally by setting this integ value to 2 (etc).  However, stability issues will likely arise if you go too high.  For improved numerical stability, you may even need to reduce this value to 0.5 or possibly even lower (typically however this is not necessary).  MUST also coordinate this with network.time_inc variable to ensure that global network.time reflects simulated time accurately"`
	VmTau     float32 `def:"2.81" min:"1" desc:"membrane potential time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) -- reflects the capacitance of the neuron in principle -- biological default for AdEx spiking model C = 281 pF = 2.81 normalized"`
	VmDendTau float32 `def:"5" min:"1" desc:"dendritic membrane potential integration time constant"`
	GeTau     float32 `def:"5" min:"1" desc:"time constant for decay of excitatory AMPA receptor conductance."`
	GiTau     float32 `def:"7" min:"1" desc:"time constant for decay of inhibitory GABAa receptor conductance."`
	AvgTau    float32 `def:"200" desc:"for integrating activation average (ActAvg), time constant in trials (roughly, how long it takes for value to change significantly) -- used mostly for visualization and tracking *hog* units"`

	VmDt     float32 `view:"-" json:"-" xml:"-" desc:"nominal rate = Integ / tau"`
	VmDendDt float32 `view:"-" json:"-" xml:"-" desc:"nominal rate = Integ / tau"`
	GeDt     float32 `view:"-" json:"-" xml:"-" desc:"rate = Integ / tau"`
	GiDt     float32 `view:"-" json:"-" xml:"-" desc:"rate = Integ / tau"`
	AvgDt    float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (dp *DtParams) Update() {
	dp.VmDt = dp.Integ / dp.VmTau
	dp.VmDendDt = dp.Integ / dp.VmDendTau
	dp.GeDt = dp.Integ / dp.GeTau
	dp.GiDt = dp.Integ / dp.GiTau
	dp.AvgDt = 1 / dp.AvgTau
}

func (dp *DtParams) Defaults() {
	dp.Integ = 1
	dp.VmTau = 2.81
	dp.VmDendTau = 5
	dp.GeTau = 5
	dp.GiTau = 7
	dp.AvgTau = 200
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

//////////////////////////////////////////////////////////////////////////////////////
//  Noise

// ActNoiseType are different types / locations of random noise for activations
type ActNoiseType int

//go:generate stringer -type=ActNoiseType

var KiT_ActNoiseType = kit.Enums.AddEnum(ActNoiseTypeN, kit.NotBitFlag, nil)

func (ev ActNoiseType) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *ActNoiseType) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The activation noise types
const (
	// NoNoise means no noise added
	NoNoise ActNoiseType = iota

	// VmNoise means noise is added to the membrane potential.
	VmNoise

	// GeNoise means noise is added to the excitatory conductance (Ge).
	GeNoise

	// ActNoise means noise is added to the final rate code activation
	ActNoise

	// GeMultNoise means that noise is multiplicative on the Ge excitatory conductance values
	GeMultNoise

	ActNoiseTypeN
)

// ActNoiseParams contains parameters for activation-level noise
type ActNoiseParams struct {
	erand.RndParams
	Type  ActNoiseType `desc:"where and how to add processing noise"`
	Fixed bool         `desc:"keep the same noise value over the entire alpha cycle -- prevents noise from being washed out and produces a stable effect that can be better used for learning -- this is strongly recommended for most learning situations"`
}

func (an *ActNoiseParams) Update() {
}

func (an *ActNoiseParams) Defaults() {
	an.Fixed = false
}

//////////////////////////////////////////////////////////////////////////////////////
//  ClampParams

// ClampParams are for specifying how external inputs are clamped onto network activation values
type ClampParams struct {
	Hard    bool    `def:"true" desc:"whether to hard clamp inputs where spiking rate is set to Poisson noise with external input * Rate factor"`
	Rate    float32 `desc:"maximum spiking rate in Hz for Poisson spike generator (multiplies clamped input value to get rate)"`
	Gain    float32 `viewif:"!Hard" def:"0.02:0.5" desc:"soft clamp gain factor (Ge += Gain * Ext)"`
	Avg     bool    `viewif:"!Hard" desc:"compute soft clamp as the average of current and target netins, not the sum -- prevents some of the main effect problems associated with adding external inputs"`
	AvgGain float32 `viewif:"!Hard && Avg" def:"0.2" desc:"gain factor for averaging the Ge -- clamp value Ext contributes with AvgGain and current Ge as (1-AvgGain)"`
}

func (cp *ClampParams) Update() {
}

func (cp *ClampParams) Defaults() {
	cp.Hard = true
	cp.Rate = 100
	cp.Gain = 0.2
	cp.Avg = false
	cp.AvgGain = 0.2
}

// AvgGe computes Avg-based Ge clamping value if using that option.
func (cp *ClampParams) AvgGe(ext, ge float32) float32 {
	return cp.AvgGain*cp.Gain*ext + (1-cp.AvgGain)*ge
}

//////////////////////////////////////////////////////////////////////////////////////
//  SynComParams

/// SynComParams are synaptic communication parameters: delay and probability of failure
type SynComParams struct {
	Delay      int     `desc:"synaptic delay for inputs arriving at this projection -- IMPORTANT: if you change this, you must rebuild network!"`
	PFail      float32 `desc:"probability of synaptic transmission failure -- if > 0, then weights are turned off at random as a function of PFail * (1-Min(Wt/Max, 1))^2"`
	PFailWtMax float32 `desc:"maximum weight value that experiences no synaptic failure -- weights at or above this level never fail to communicate, while probability of failure increases parabolically below this level"`
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

//////////////////////////////////////////////////////////////////////////////////////
//  WtInitParams

// WtInitParams are weight initialization parameters -- basically the
// random distribution parameters but also Symmetry flag
type WtInitParams struct {
	erand.RndParams
	Sym bool `desc:"symmetrize the weight values with those in reciprocal projection -- typically true for bidirectional excitatory connections"`
}

func (wp *WtInitParams) Defaults() {
	wp.Mean = 0.5
	wp.Var = 0.25
	wp.Dist = erand.Uniform
	wp.Sym = true
}

//////////////////////////////////////////////////////////////////////////////////////
//  WtScaleParams

/// WtScaleParams are weight scaling parameters: modulates overall strength of projection,
// using both absolute and relative factors
type WtScaleParams struct {
	Abs float32 `def:"1" min:"0" desc:"absolute scaling, which is not subject to normalization: directly multiplies weight values"`
	Rel float32 `min:"0" desc:"[Default: 1] relative scaling that shifts balance between different projections -- this is subject to normalization across all other projections into unit"`
}

func (ws *WtScaleParams) Defaults() {
	ws.Abs = 1
	ws.Rel = 1
}

func (ws *WtScaleParams) Update() {
}

// SLayActScale computes scaling factor based on sending layer activity level (savg), number of units
// in sending layer (snu), and number of recv connections (ncon).
// Uses a fixed sem_extra standard-error-of-the-mean (SEM) extra value of 2
// to add to the average expected number of active connections to receive,
// for purposes of computing scaling factors with partial connectivity
// For 25% layer activity, binomial SEM = sqrt(p(1-p)) = .43, so 3x = 1.3 so 2 is a reasonable default.
func (ws *WtScaleParams) SLayActScale(savg, snu, ncon float32) float32 {
	ncon = math32.Max(ncon, 1) // prjn Avg can be < 1 in some cases
	semExtra := 2
	slayActN := int(mat32.Round(savg * snu)) // sending layer actual # active
	slayActN = ints.MaxInt(slayActN, 1)
	var sc float32
	if ncon == snu {
		sc = 1 / float32(slayActN)
	} else {
		maxActN := int(math32.Min(ncon, float32(slayActN))) // max number we could get
		avgActN := int(mat32.Round(savg * ncon))            // recv average actual # active if uniform
		avgActN = ints.MaxInt(avgActN, 1)
		expActN := avgActN + semExtra // expected
		expActN = ints.MinInt(expActN, maxActN)
		sc = 1 / float32(expActN)
	}
	return sc
}

// FullScale returns full scaling factor, which is product of Abs * Rel * SLayActScale
func (ws *WtScaleParams) FullScale(savg, snu, ncon float32) float32 {
	return ws.Abs * ws.Rel * ws.SLayActScale(savg, snu, ncon)
}
