// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"math/rand"

	"github.com/emer/axon/chans"
	"github.com/emer/axon/knadapt"
	"github.com/emer/emergent/erand"
	"github.com/emer/etable/minmax"
	"github.com/goki/ki/ints"
	"github.com/goki/mat32"
)

///////////////////////////////////////////////////////////////////////
//  act.go contains the activation params and functions for axon

// axon.ActParams contains all the activation computation params and functions
// for basic Axon, at the neuron level .
// This is included in axon.Layer to drive the computation.
type ActParams struct {
	Spike   SpikeParams       `view:"inline" desc:"Spiking function parameters"`
	Dend    DendParams        `view:"inline" desc:"dendrite-specific parameters"`
	Init    ActInitParams     `view:"inline" desc:"initial values for key network state variables -- initialized in InitActs called by InitWts, and provides target values for DecayState"`
	Decay   DecayParams       `view:"inline" desc:"amount to decay between AlphaCycles, simulating passage of time and effects of saccades etc, especially important for environments with random temporal structure (e.g., most standard neural net training corpora) "`
	Dt      DtParams          `view:"inline" desc:"time and rate constants for temporal derivatives / updating of activation state"`
	Gbar    chans.Chans       `view:"inline" desc:"[Defaults: 1, .2, 1, 1] maximal conductances levels for channels"`
	Erev    chans.Chans       `view:"inline" desc:"[Defaults: 1, .3, .25, .1] reversal potentials for each channel"`
	GTarg   GTargParams       `view:"inline" desc:"target conductance levels for excitation and inhibition, driving adaptation of GScale.Scale conductance scaling"`
	Clamp   ClampParams       `view:"inline" desc:"how external inputs drive neural activations"`
	Noise   SpikeNoiseParams  `view:"inline" desc:"how, where, when, and how much noise to add"`
	VmRange minmax.F32        `view:"inline" desc:"range for Vm membrane potential -- [0.1, 1.0] -- important to keep just at extreme range of reversal potentials to prevent numerical instability"`
	KNa     knadapt.Params    `view:"no-inline" desc:"sodium-gated potassium channel adaptation parameters -- activates an inhibitory leak-like current as a function of neural activity (firing = Na influx) at three different time-scales (M-type = fast, Slick = medium, Slack = slow)"`
	NMDA    chans.NMDAParams  `view:"inline" desc:"NMDA channel parameters plus more general params"`
	GABAB   chans.GABABParams `view:"inline" desc:"GABA-B / GIRK channel parameters"`
	Attn    AttnParams        `view:"inline" desc:"Attentional modulation parameters: how Attn modulates Ge"`
}

func (ac *ActParams) Defaults() {
	ac.Spike.Defaults()
	ac.Dend.Defaults()
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
	ac.NMDA.Gbar = 0.15 // .15 now -- was 0.3 best.
	ac.GABAB.Defaults()
	ac.Attn.Defaults()
	ac.Update()
}

// Update must be called after any changes to parameters
func (ac *ActParams) Update() {
	ac.Spike.Update()
	ac.Dend.Update()
	ac.Init.Update()
	ac.Decay.Update()
	ac.Dt.Update()
	ac.GTarg.Update()
	ac.Clamp.Update()
	ac.Noise.Update()
	ac.KNa.Update()
	ac.NMDA.Update()
	ac.GABAB.Update()
	ac.Attn.Update()
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
	nrn.ActInt = ac.Init.Act // start fresh

	if decay > 0 { // no-op for most, but not all..
		nrn.Spike = 0
		nrn.Act -= decay * (nrn.Act - ac.Init.Act)
		nrn.ActInt -= decay * (nrn.ActInt - ac.Init.Act)
		nrn.GeSyn -= decay * (nrn.GeSyn - ac.Init.Ge)
		nrn.Ge -= decay * (nrn.Ge - ac.Init.Ge)
		nrn.Gi -= decay * (nrn.Gi - ac.Init.Gi)
		nrn.Gk -= decay * nrn.Gk

		nrn.Vm -= decay * (nrn.Vm - ac.Init.Vm)

		nrn.GeNoise -= decay * nrn.GeNoise
		nrn.GiNoise -= decay * nrn.GiNoise

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
	nrn.GeSyn = ac.Init.Ge
	nrn.Ge = ac.Init.Ge
	nrn.Gi = ac.Init.Gi
	nrn.Gk = 0
	nrn.Inet = 0
	nrn.Vm = ac.Init.Vm
	nrn.VmDend = ac.Init.Vm
	nrn.Targ = 0
	nrn.Ext = 0

	nrn.ActDel = 0
	nrn.RLrate = 1

	nrn.GeNoiseP = 1
	nrn.GeNoise = 0
	nrn.GiNoiseP = 1
	nrn.GiNoise = 0

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
	nrn.Attn = 1

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

// GeFmRaw integrates Ge excitatory conductance from GeRaw value into GeSyn
// geExt is extra conductance to add to the final Ge value
func (ac *ActParams) GeFmRaw(nrn *Neuron, geRaw, geExt float32, cyc int, actm float32) {
	if ac.Clamp.Add && nrn.HasFlag(NeurHasExt) {
		geRaw += nrn.Ext * ac.Clamp.Ge
	}
	geRaw = ac.Attn.ModVal(geRaw, nrn.Attn)

	if !ac.Clamp.Add && nrn.HasFlag(NeurHasExt) {
		nrn.GeSyn = nrn.Ext * ac.Clamp.Ge
	} else {
		ac.Dt.GeSynFmRaw(geRaw, &nrn.GeSyn, ac.Init.Ge)
	}

	nrn.Ge = nrn.GeSyn + geExt

	if ac.Noise.On && ac.Noise.Ge > 0 {
		ge := ac.Noise.PGe(&nrn.GeNoiseP)
		ac.Dt.GeSynFmRaw(ge, &nrn.GeNoise, 0)
		nrn.Ge += nrn.GeNoise
	}
}

// GiFmRaw integrates GiSyn inhibitory synaptic conductance from GiRaw value
// (can add other terms to geRaw prior to calling this)
func (ac *ActParams) GiFmRaw(nrn *Neuron, giRaw float32) {
	ac.Dt.GiSynFmRaw(giRaw, &nrn.GiSyn, ac.Init.Gi)
	if ac.Noise.On && ac.Noise.Gi > 0 {
		gi := ac.Noise.PGi(&nrn.GiNoiseP)
		ac.Dt.GiSynFmRaw(gi, &nrn.GiNoise, 0)
	}
	if nrn.GiSyn < 0 { // negative inhib G doesn't make any sense
		nrn.GiSyn = 0
	}
}

// InetFmG computes net current from conductances and Vm
func (ac *ActParams) InetFmG(vm, ge, gl, gi, gk float32) float32 {
	inet := ge*(ac.Erev.E-vm) + gl*ac.Gbar.L*(ac.Erev.L-vm) + gi*(ac.Erev.I-vm) + gk*(ac.Erev.K-vm)
	if inet > ac.Dt.VmTau {
		inet = ac.Dt.VmTau
	} else if inet < -ac.Dt.VmTau {
		inet = -ac.Dt.VmTau
	}
	return inet
}

// VmFmInet computes new Vm value from inet, clamping range
func (ac *ActParams) VmFmInet(vm, dt, inet float32) float32 {
	return ac.VmRange.ClipVal(vm + dt*inet)
}

// VmInteg integrates Vm over VmSteps to obtain a more stable value
// Returns the new Vm and inet values.
func (ac *ActParams) VmInteg(vm, dt, ge, gl, gi, gk float32) (float32, float32) {
	dt *= ac.Dt.DtStep
	nvm := vm
	var inet float32
	for i := 0; i < ac.Dt.VmSteps; i++ {
		inet = ac.InetFmG(nvm, ge, gl, gi, gk)
		nvm = ac.VmFmInet(nvm, dt, inet)
	}
	return nvm, inet
}

// VmFmG computes membrane potential Vm from conductances Ge, Gi, and Gk.
func (ac *ActParams) VmFmG(nrn *Neuron) {
	updtVm := true
	// note: nrn.ISI has NOT yet been updated at this point: 0 right after spike, etc
	// so it takes a full 3 time steps after spiking for Tr period
	if ac.Spike.Tr > 0 && nrn.ISI >= 0 && nrn.ISI < float32(ac.Spike.Tr) {
		updtVm = false // don't update the spiking vm during refract
	}

	ge := nrn.Ge * ac.Gbar.E
	gi := nrn.Gi * ac.Gbar.I
	gk := nrn.Gk * ac.Gbar.K
	var expi float32
	if updtVm {
		nvm, inet := ac.VmInteg(nrn.Vm, ac.Dt.VmDt, ge, 1, gi, gk)
		if updtVm && ac.Spike.Exp { // add spike current if relevant
			exVm := 0.5 * (nvm + nrn.Vm) // midpoint for this
			expi = ac.Gbar.L * ac.Spike.ExpSlope *
				mat32.FastExp((exVm-ac.Spike.Thr)/ac.Spike.ExpSlope)
			if expi > ac.Dt.VmTau {
				expi = ac.Dt.VmTau
			}
			inet += expi
			nvm = ac.VmFmInet(nvm, ac.Dt.VmDt, expi)
		}
		nrn.Vm = nvm
		nrn.Inet = inet
	} else { // decay back to VmR
		var dvm float32
		if int(nrn.ISI) == ac.Spike.Tr-1 {
			dvm = (ac.Spike.VmR - nrn.Vm)
		} else {
			dvm = ac.Spike.RDt * (ac.Spike.VmR - nrn.Vm)
		}
		nrn.Vm = nrn.Vm + dvm
		nrn.Inet = dvm * ac.Dt.VmTau
	}

	{ // always update VmDend
		glEff := float32(1)
		if !updtVm {
			glEff += ac.Dend.GbarR
		}
		nvm, _ := ac.VmInteg(nrn.VmDend, ac.Dt.VmDendDt, ge, glEff, gi, gk)
		if updtVm {
			nvm = ac.VmFmInet(nvm, ac.Dt.VmDendDt, ac.Dend.GbarExp*expi)
		}
		nrn.VmDend = nvm
	}
}

// ActFmG computes Spike from Vm and ISI-based activation
func (ac *ActParams) ActFmG(nrn *Neuron) {
	var thr float32
	if ac.Spike.Exp {
		thr = ac.Spike.ExpThr
	} else {
		thr = ac.Spike.Thr
	}
	if nrn.Vm >= thr {
		nrn.Spike = 1
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

//////////////////////////////////////////////////////////////////////////////////////
//  SpikeParams

// SpikeParams contains spiking activation function params.
// Implements a basic thresholded Vm model, and optionally
// the AdEx adaptive exponential function (adapt is KNaAdapt)
type SpikeParams struct {
	Thr      float32 `def:"0.5" desc:"threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization"`
	VmR      float32 `def:"0.3" desc:"post-spiking membrane potential to reset to, produces refractory effect if lower than VmInit -- 0.3 is apropriate biologically-based value for AdEx (Brette & Gurstner, 2005) parameters.  See also RTau"`
	Tr       int     `min:"1" def:"3" desc:"post-spiking explicit refractory period, in cycles -- prevents Vm updating for this number of cycles post firing -- Vm is reduced in exponential steps over this period according to RTau, being fixed at Tr to VmR exactly"`
	RTau     float32 `def:"1.6667" desc:"time constant for decaying Vm down to VmR -- at end of Tr it is set to VmR exactly -- this provides a more realistic shape of the post-spiking Vm which is only relevant for more realistic channels that key off of Vm -- does not otherwise affect standard computation"`
	Exp      bool    `def:"true" desc:"if true, turn on exponential excitatory current that drives Vm rapidly upward for spiking as it gets past its nominal firing threshold (Thr) -- nicely captures the Hodgkin Huxley dynamics of Na and K channels -- uses Brette & Gurstner 2005 AdEx formulation"`
	ExpSlope float32 `viewif:"Exp" def:"0.02" desc:"slope in Vm (2 mV = .02 in normalized units) for extra exponential excitatory current that drives Vm rapidly upward for spiking as it gets past its nominal firing threshold (Thr) -- nicely captures the Hodgkin Huxley dynamics of Na and K channels -- uses Brette & Gurstner 2005 AdEx formulation"`
	ExpThr   float32 `viewif:"Exp" def:"0.8" desc:"membrane potential threshold for actually triggering a spike when using the exponential mechanism"`
	MaxHz    float32 `def:"180" min:"1" desc:"for translating spiking interval (rate) into rate-code activation equivalent, what is the maximum firing rate associated with a maximum activation value of 1"`
	ISITau   float32 `def:"5" min:"1" desc:"constant for integrating the spiking interval in estimating spiking rate"`
	ISIDt    float32 `view:"-" desc:"rate = 1 / tau"`
	RDt      float32 `view:"-" desc:"rate = 1 / tau"`
}

func (sk *SpikeParams) Defaults() {
	sk.Thr = 0.5
	sk.VmR = 0.3
	sk.Tr = 3
	sk.RTau = 1.6667
	sk.Exp = true
	sk.ExpSlope = 0.02
	sk.ExpThr = 0.8
	sk.MaxHz = 180
	sk.ISITau = 5
	sk.Update()
}

func (sk *SpikeParams) Update() {
	if sk.Tr <= 0 {
		sk.Tr = 1 // hard min
	}
	sk.ISIDt = 1 / sk.ISITau
	sk.RDt = 1 / sk.RTau
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
//  DendParams

// DendParams are the parameters for updating dendrite-specific dynamics
type DendParams struct {
	GbarExp float32 `def:"0.2" desc:"dendrite-specific strength multiplier of the exponential spiking drive on Vm -- e.g., .5 makes it half as strong as at the soma (which uses Gbar.L as a strength multiplier per the AdEx standard model)"`
	GbarR   float32 `def:"3" desc:"dendrite-specific conductance of Kdr delayed rectifier currents, used to reset membrane potential for dendrite -- applied for Tr msec"`
}

func (dp *DendParams) Defaults() {
	dp.GbarExp = 0.2
	dp.GbarR = 3
}

func (dp *DendParams) Update() {
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
	VmDendTau  float32 `def:"5" min:"1" desc:"dendritic membrane potential time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life) -- reflects the capacitance of the neuron in principle -- biological default for AdEx spiking model C = 281 pF = 2.81 normalized"`
	VmSteps    int     `def:"2" min:"1" desc:"number of integration steps to take in computing new Vm value -- this is the one computation that can be most numerically unstable so taking multiple steps with proportionally smaller dt is beneficial"`
	GeTau      float32 `def:"5" min:"1" desc:"time constant for decay of excitatory AMPA receptor conductance."`
	GiTau      float32 `def:"7" min:"1" desc:"time constant for decay of inhibitory GABAa receptor conductance."`
	IntTau     float32 `def:"40" min:"1" desc:"time constant for integrating values over timescale of an individual input state (e.g., roughly 200 msec -- theta cycle), used in computing ActInt, and for GeM from Ge -- this is used for scoring performance, not for learning, in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life), "`
	LongAvgTau float32 `def:"20" min:"1" desc:"time constant for integrating slower long-time-scale averages, such as nrn.ActAvg, ly.ActAvg.AvgMaxGeM, Pool.ActsMAvg, ActsPAvg -- computed in NewState when a new input state is present (i.e., not msec but in units of a theta cycle) (tau is roughly how long it takes for value to change significantly) -- set lower for smaller models"`

	VmDt      float32 `view:"-" json:"-" xml:"-" desc:"nominal rate = Integ / tau"`
	VmDendDt  float32 `view:"-" json:"-" xml:"-" desc:"nominal rate = Integ / tau"`
	DtStep    float32 `view:"-" json:"-" xml:"-" desc:"1 / VmSteps"`
	GeDt      float32 `view:"-" json:"-" xml:"-" desc:"rate = Integ / tau"`
	GiDt      float32 `view:"-" json:"-" xml:"-" desc:"rate = Integ / tau"`
	IntDt     float32 `view:"-" json:"-" xml:"-" desc:"rate = Integ / tau"`
	LongAvgDt float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (dp *DtParams) Update() {
	if dp.VmSteps < 1 {
		dp.VmSteps = 1
	}
	dp.VmDt = dp.Integ / dp.VmTau
	dp.VmDendDt = dp.Integ / dp.VmDendTau
	dp.DtStep = 1 / float32(dp.VmSteps)
	dp.GeDt = dp.Integ / dp.GeTau
	dp.GiDt = dp.Integ / dp.GiTau
	dp.IntDt = dp.Integ / dp.IntTau
	dp.LongAvgDt = 1 / dp.LongAvgTau
}

func (dp *DtParams) Defaults() {
	dp.Integ = 1
	dp.VmTau = 2.81
	dp.VmDendTau = 5
	dp.VmSteps = 2
	dp.GeTau = 5
	dp.GiTau = 7
	dp.IntTau = 40
	dp.LongAvgTau = 20
	dp.Update()
}

// GeSynFmRaw updates GeSyn from raw input, decaying with time constant,
// back to min baseline value
func (dp *DtParams) GeSynFmRaw(geRaw float32, geSyn *float32, min float32) {
	*geSyn += geRaw - dp.GeDt*(*geSyn-min)
}

// GiSynFmRaw updates GiSyn from raw input, decaying with time constant,
// back to min baseline value
func (dp *DtParams) GiSynFmRaw(giRaw float32, giSyn *float32, min float32) {
	*giSyn += giRaw - dp.GiDt*(*giSyn-min)
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
	GeMax float32 `def:"1.2" min:"0" desc:"target maximum excitatory conductance in the minus phase: GeM"`
	GiMax float32 `def:"1.2" min:"0" desc:"target maximum inhibitory conductance in the minus phase: GiM -- for actual synaptic inhibitory neuron inputs (GiSyn) not FFFB computed inhibition"`
}

func (gt *GTargParams) Update() {
}

func (gt *GTargParams) Defaults() {
	gt.GeMax = 1.2
	gt.GiMax = 1.2
	gt.Update()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Noise

// SpikeNoiseParams parameterizes background spiking activity impinging on the neuron,
// simulated using a poisson spiking process.
type SpikeNoiseParams struct {
	On   bool    `desc:"add noise simulating background spiking levels"`
	GeHz float32 `def:"100" desc:"mean frequency of excitatory spikes -- typically 50Hz but multiple inputs increase rate -- poisson lambda parameter, also the variance"`
	Ge   float32 `min:"0" desc:"excitatory conductance per spike -- .001 has minimal impact, .01 can be strong, and .15 is needed to influence timing of clamped inputs"`
	GiHz float32 `def:"200" desc:"mean frequency of inhibitory spikes -- typically 100Hz fast spiking but multiple inputs increase rate -- poisson lambda parameter, also the variance"`
	Gi   float32 `min:"0" desc:"excitatory conductance per spike -- .001 has minimal impact, .01 can be strong, and .15 is needed to influence timing of clamped inputs"`

	GeExpInt float32 `view:"-" json:"-" xml:"-" desc:"Exp(-Interval) which is the threshold for GeNoiseP as it is updated"`
	GiExpInt float32 `view:"-" json:"-" xml:"-" desc:"Exp(-Interval) which is the threshold for GiNoiseP as it is updated"`
}

func (an *SpikeNoiseParams) Update() {
	an.GeExpInt = mat32.Exp(-1000.0 / an.GeHz)
	an.GiExpInt = mat32.Exp(-1000.0 / an.GiHz)
}

func (an *SpikeNoiseParams) Defaults() {
	an.GeHz = 100
	an.Ge = 0.001
	an.GiHz = 200
	an.Gi = 0.001
	an.Update()
}

// PGe updates the GeNoiseP probability, multiplying a uniform random number [0-1]
// and returns Ge from spiking if a spike is triggered
func (an *SpikeNoiseParams) PGe(p *float32) float32 {
	*p *= rand.Float32()
	if *p <= an.GeExpInt {
		*p = 1
		return an.Ge
	}
	return 0
}

// PGi updates the GiNoiseP probability, multiplying a uniform random number [0-1]
// and returns Gi from spiking if a spike is triggered
func (an *SpikeNoiseParams) PGi(p *float32) float32 {
	*p *= rand.Float32()
	if *p <= an.GiExpInt {
		*p = 1
		return an.Gi
	}
	return 0
}

//////////////////////////////////////////////////////////////////////////////////////
//  ClampParams

// ClampParams specify how external inputs drive excitatory conductances
// (like a current clamp) -- either adds or overwrites existing conductances.
// Noise is added in either case.
type ClampParams struct {
	Ge     float32 `def:"0.6,1" desc:"amount of Ge driven for clamping -- generally use 0.6 for Target layers, 1.0 for Input layers"`
	Add    bool    `def:"false" view:"add external conductance on top of any existing -- generally this is not a good idea for target layers (creates a main effect that learning can never match), but may be ok for input layers"`
	ErrThr float32 `def:"0.5" desc:"threshold on neuron Act activity to count as active for computing error relative to target in PctErr method"`
}

func (cp *ClampParams) Update() {
}

func (cp *ClampParams) Defaults() {
	cp.Ge = 0.6
	cp.ErrThr = 0.5
}

//////////////////////////////////////////////////////////////////////////////////////
//  AttnParams

// AttnParams determine how the Attn modulates Ge
type AttnParams struct {
	On  bool    `desc:"is attentional modulation active?"`
	Min float32 `desc:"minimum act multiplier if attention is 0"`
}

func (at *AttnParams) Defaults() {
	at.On = true
	at.Min = 0.8
}

func (at *AttnParams) Update() {
}

// ModVal returns the attn-modulated value -- attn must be between 1-0
func (at *AttnParams) ModVal(val float32, attn float32) float32 {
	if !at.On {
		return val
	}
	return val * (at.Min + (1-at.Min)*attn)
}

//////////////////////////////////////////////////////////////////////////////////////
//  SynComParams

/// SynComParams are synaptic communication parameters: delay and probability of failure
type SynComParams struct {
	Delay    int     `min:"0" def:"2" desc:"additional synaptic delay for inputs arriving at this projection -- IMPORTANT: if you change this, you must call InitWts() on Network!  Delay = 0 means a spike reaches receivers in the next Cycle, which is the minimum time.  Biologically, subtract 1 from synaptic delay values to set corresponding Delay value."`
	PFail    float32 `desc:"probability of synaptic transmission failure -- if > 0, then weights are turned off at random as a function of PFail (times 1-SWt if PFailSwt)"`
	PFailSWt bool    `desc:"if true, then probability of failure is inversely proportional to SWt structural / slow weight value (i.e., multiply PFail * (1-SWt)))"`
}

func (sc *SynComParams) Defaults() {
	sc.Delay = 2
	sc.PFail = 0 // 0.5 works?
	sc.PFailSWt = false
}

func (sc *SynComParams) Update() {
}

// WtFailP returns probability of weight (synapse) failure given current SWt value
func (sc *SynComParams) WtFailP(swt float32) float32 {
	if !sc.PFailSWt {
		return sc.PFail
	}
	return sc.PFail * (1 - swt)
}

// WtFail returns true if synapse should fail, as function of SWt value (optionally)
func (sc *SynComParams) WtFail(swt float32) bool {
	fp := sc.WtFailP(swt)
	if fp == 0 {
		return false
	}
	return erand.BoolP(fp)
}

// Fail updates failure status of given weight, given SWt value
func (sc *SynComParams) Fail(wt *float32, swt float32) {
	if sc.PFail > 0 {
		if sc.WtFail(swt) {
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
	Abs        float32 `def:"1" min:"0" desc:"absolute multiplier adjustment factor for the prjn scaling -- can be used to adjust for idiosyncrasies not accommodated by the standard scaling based on initial target activation level and relative scaling factors -- any adaptation operates by directly adjusting scaling factor from the initially computed value"`
	Adapt      bool    `def:"false" desc:"Adapt the 'GScale' scaling value so the ActAvg.AvgMaxGeM / GiM running-average value for this projections remains in the target range, specified in Acts.GTarg -- sometimes this is essential but often it is better to tune the Abs values manually, as the adaptation-based adjustments can disrupt things during learning"`
	ScaleLrate float32 `viewif:"Adapt" def:"0.5" desc:"learning rate for adapting the GScale value, as function of target value -- lrate is also multiplied by the GScale.Orig to compensate for significant differences in overall scale of these scaling factors -- fastest value with some smoothing at .5 works well."`
	HiTol      float32 `def:"0" viewif:"Adapt" desc:"tolerance for higher than target AvgMaxGeM / GiM as a proportion of that target value (0 = exactly the target, 0.2 = 20% higher than target) -- only once activations move outside this tolerance are scale values adapted"`
	LoTol      float32 `def:"0.8" viewif:"Adapt" desc:"tolerance for lower than target AvgMaxGeM / GiM as a proportion of that target value (0 = exactly the target, 0.8 = 80% lower than target) -- only once activations move outside this tolerance are scale values adapted"`
	AvgTau     float32 `def:"500" desc:"time constant for integrating projection-level averages for this scaling process: Prjn.GScale.AvgAvg, AvgMax (tau is roughly how long it takes for value to change significantly) -- these are updated at the cycle level and thus require a much slower rate constant compared to other such variables integrated at the AlphaCycle level."`

	AvgDt float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (ws *PrjnScaleParams) Defaults() {
	ws.Rel = 1
	ws.Abs = 1
	ws.Adapt = false
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
	if ncon < 1 { // prjn Avg can be < 1 in some cases
		ncon = 1
	}
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

// FullScale returns full scaling factor, which is product of Abs * Rel * SLayActScale
func (ws *PrjnScaleParams) FullScale(savg, snu, ncon float32) float32 {
	return ws.Abs * ws.Rel * ws.SLayActScale(savg, snu, ncon)
}
