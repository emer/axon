// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/math32"
	"github.com/emer/axon/v2/chans"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/etable/v2/minmax"
	"github.com/emer/gosl/v2/slbool"
)

///////////////////////////////////////////////////////////////////////
//  act.go contains the activation params and functions for axon

//gosl: hlsl act
// #include "chans.hlsl"
// #include "minmax.hlsl"
// #include "neuron.hlsl"
//gosl: end act

//gosl: start act

//////////////////////////////////////////////////////////////////////////////////////
//  SpikeParams

// SpikeParams contains spiking activation function params.
// Implements a basic thresholded Vm model, and optionally
// the AdEx adaptive exponential function (adapt is KNaAdapt)
type SpikeParams struct {

	// threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization
	Thr float32 `default:"0.5"`

	// post-spiking membrane potential to reset to, produces refractory effect if lower than VmInit -- 0.3 is apropriate biologically-based value for AdEx (Brette & Gurstner, 2005) parameters.  See also RTau
	VmR float32 `default:"0.3"`

	// post-spiking explicit refractory period, in cycles -- prevents Vm updating for this number of cycles post firing -- Vm is reduced in exponential steps over this period according to RTau, being fixed at Tr to VmR exactly
	Tr int32 `min:"1" default:"3"`

	// time constant for decaying Vm down to VmR -- at end of Tr it is set to VmR exactly -- this provides a more realistic shape of the post-spiking Vm which is only relevant for more realistic channels that key off of Vm -- does not otherwise affect standard computation
	RTau float32 `default:"1.6667"`

	// if true, turn on exponential excitatory current that drives Vm rapidly upward for spiking as it gets past its nominal firing threshold (Thr) -- nicely captures the Hodgkin Huxley dynamics of Na and K channels -- uses Brette & Gurstner 2005 AdEx formulation
	Exp slbool.Bool `default:"true"`

	// slope in Vm (2 mV = .02 in normalized units) for extra exponential excitatory current that drives Vm rapidly upward for spiking as it gets past its nominal firing threshold (Thr) -- nicely captures the Hodgkin Huxley dynamics of Na and K channels -- uses Brette & Gurstner 2005 AdEx formulation
	ExpSlope float32 `default:"0.02"`

	// membrane potential threshold for actually triggering a spike when using the exponential mechanism
	ExpThr float32 `default:"0.9"`

	// for translating spiking interval (rate) into rate-code activation equivalent, what is the maximum firing rate associated with a maximum activation value of 1
	MaxHz float32 `default:"180" min:"1"`

	// constant for integrating the spiking interval in estimating spiking rate
	ISITau float32 `default:"5" min:"1"`

	// rate = 1 / tau
	ISIDt float32 `view:"-"`

	// rate = 1 / tau
	RDt float32 `view:"-"`

	pad int32
}

func (sk *SpikeParams) Defaults() {
	sk.Thr = 0.5
	sk.VmR = 0.3
	sk.Tr = 3
	sk.RTau = 1.6667
	sk.Exp.SetBool(true)
	sk.ExpSlope = 0.02
	sk.ExpThr = 0.9
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

func (sk *SpikeParams) ShouldShow(field string) bool {
	switch field {
	case "ExpSlope", "ExpThr":
		return sk.Exp.IsTrue()
	default:
		return true
	}
}

// ActToISI compute spiking interval from a given rate-coded activation,
// based on time increment (.001 = 1msec default), Act.Dt.Integ
func (sk *SpikeParams) ActToISI(act, timeInc, integ float32) float32 {
	if act == 0 {
		return 0
	}
	return (1 / (timeInc * integ * act * sk.MaxHz))
}

// ActFromISI computes rate-code activation from estimated spiking interval
func (sk *SpikeParams) ActFromISI(isi, timeInc, integ float32) float32 {
	if isi <= 0 {
		return 0
	}
	maxInt := 1.0 / (timeInc * integ * sk.MaxHz) // interval at max hz..
	return maxInt / isi                          // normalized
}

// AvgFromISI returns updated spiking ISI from current isi interval value
func (sk *SpikeParams) AvgFromISI(avg float32, isi float32) float32 {
	if avg <= 0 {
		avg = isi
	} else if isi < 0.8*avg {
		avg = isi // if significantly less than we take that
	} else { // integrate on slower
		avg += sk.ISIDt * (isi - avg) // running avg updt
	}
	return avg
}

//////////////////////////////////////////////////////////////////////////////////////
//  DendParams

// DendParams are the parameters for updating dendrite-specific dynamics
type DendParams struct {

	// dendrite-specific strength multiplier of the exponential spiking drive on Vm -- e.g., .5 makes it half as strong as at the soma (which uses Gbar.L as a strength multiplier per the AdEx standard model)
	GbarExp float32 `default:"0.2,0.5"`

	// dendrite-specific conductance of Kdr delayed rectifier currents, used to reset membrane potential for dendrite -- applied for Tr msec
	GbarR float32 `default:"3,6"`

	// SST+ somatostatin positive slow spiking inhibition level specifically affecting dendritic Vm (VmDend) -- this is important for countering a positive feedback loop from NMDA getting stronger over the course of learning -- also typically requires SubMean = 1 for TrgAvgAct and learning to fully counter this feedback loop.
	SSGi float32 `default:"0,2"`

	// set automatically based on whether this layer has any recv projections that have a GType conductance type of Modulatory -- if so, then multiply GeSyn etc by GModSyn
	HasMod slbool.Bool `edit:"-"`

	// multiplicative gain factor on the total modulatory input -- this can also be controlled by the PrjnScale.Abs factor on ModulatoryG inputs, but it is convenient to be able to control on the layer as well.
	ModGain float32

	// if true, modulatory signal also includes ACh multiplicative factor
	ModACh slbool.Bool

	// baseline modulatory level for modulatory effects -- net modulation is ModBase + ModGain * GModSyn
	ModBase float32

	pad int32
}

func (dp *DendParams) Defaults() {
	dp.SSGi = 2
	dp.GbarExp = 0.2
	dp.GbarR = 3
	dp.ModGain = 1
	dp.ModBase = 0
}

func (dp *DendParams) Update() {
}

func (dp *DendParams) ShouldShow(field string) bool {
	switch field {
	case "ModGain", "ModACh", "ModBase":
		return dp.HasMod.IsTrue()
	default:
		return true
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  ActInitParams

// ActInitParams are initial values for key network state variables.
// Initialized in InitActs called by InitWts, and provides target values for DecayState.
type ActInitParams struct {

	// initial membrane potential -- see Erev.L for the resting potential (typically .3)
	Vm float32 `default:"0.3"`

	// initial activation value -- typically 0
	Act float32 `default:"0"`

	// baseline level of excitatory conductance (net input) -- Ge is initialized to this value, and it is added in as a constant background level of excitatory input -- captures all the other inputs not represented in the model, and intrinsic excitability, etc
	GeBase float32 `default:"0"`

	// baseline level of inhibitory conductance (net input) -- Gi is initialized to this value, and it is added in as a constant background level of inhibitory input -- captures all the other inputs not represented in the model
	GiBase float32 `default:"0"`

	// variance (sigma) of gaussian distribution around baseline Ge values, per unit, to establish variability in intrinsic excitability.  value never goes < 0
	GeVar float32 `default:"0"`

	// variance (sigma) of gaussian distribution around baseline Gi values, per unit, to establish variability in intrinsic excitability.  value never goes < 0
	GiVar float32 `default:"0"`

	pad, pad1 int32
}

func (ai *ActInitParams) Update() {
}

func (ai *ActInitParams) Defaults() {
	ai.Vm = 0.3
	ai.Act = 0
	ai.GeBase = 0
	ai.GiBase = 0
	ai.GeVar = 0
	ai.GiVar = 0
}

//gosl: end act

// GeBase returns the baseline Ge value: Ge + rand(GeVar) > 0
func (ai *ActInitParams) GetGeBase(rnd erand.Rand) float32 {
	ge := ai.GeBase
	if ai.GeVar > 0 {
		ge += float32(float64(ai.GeVar) * rnd.NormFloat64(-1))
		if ge < 0 {
			ge = 0
		}
	}
	return ge
}

// GiBase returns the baseline Gi value: Gi + rand(GiVar) > 0
func (ai *ActInitParams) GetGiBase(rnd erand.Rand) float32 {
	gi := ai.GiBase
	if ai.GiVar > 0 {
		gi += float32(float64(ai.GiVar) * rnd.NormFloat64(-1))
		if gi < 0 {
			gi = 0
		}
	}
	return gi
}

//gosl: start act

//////////////////////////////////////////////////////////////////////////////////////
//  DecayParams

// DecayParams control the decay of activation state in the DecayState function
// called in NewState when a new state is to be processed.
type DecayParams struct {

	// proportion to decay most activation state variables toward initial values at start of every ThetaCycle (except those controlled separately below) -- if 1 it is effectively equivalent to full clear, resetting other derived values.  ISI is reset every AlphaCycle to get a fresh sample of activations (doesn't affect direct computation -- only readout).
	Act float32 `default:"0,0.2,0.5,1" max:"1" min:"0"`

	// proportion to decay long-lasting conductances, NMDA and GABA, and also the dendritic membrane potential -- when using random stimulus order, it is important to decay this significantly to allow a fresh start -- but set Act to 0 to enable ongoing activity to keep neurons in their sensitive regime.
	Glong float32 `default:"0,0.6" max:"1" min:"0"`

	// decay of afterhyperpolarization currents, including mAHP, sAHP, and KNa, Kir -- has a separate decay because often useful to have this not decay at all even if decay is on.
	AHP float32 `default:"0" max:"1" min:"0"`

	// decay of Ca variables driven by spiking activity used in learning: CaSpk* and Ca* variables. These are typically not decayed but may need to be in some situations.
	LearnCa float32 `default:"0" max:"1" min:"0"`

	// decay layer at end of ThetaCycle when there is a global reward -- true by default for PTPred, PTMaint and PFC Super layers
	OnRew slbool.Bool

	pad, pad1, pad2 float32
}

func (dp *DecayParams) Update() {
}

func (dp *DecayParams) Defaults() {
	dp.Act = 0.2
	dp.Glong = 0.6
	dp.AHP = 0
	dp.LearnCa = 0
}

//////////////////////////////////////////////////////////////////////////////////////
//  DtParams

// DtParams are time and rate constants for temporal derivatives in Axon (Vm, G)
type DtParams struct {

	// overall rate constant for numerical integration, for all equations at the unit level -- all time constants are specified in millisecond units, with one cycle = 1 msec -- if you instead want to make one cycle = 2 msec, you can do this globally by setting this integ value to 2 (etc).  However, stability issues will likely arise if you go too high.  For improved numerical stability, you may even need to reduce this value to 0.5 or possibly even lower (typically however this is not necessary).  MUST also coordinate this with network.time_inc variable to ensure that global network.time reflects simulated time accurately
	Integ float32 `default:"1,0.5" min:"0"`

	// membrane potential time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life) -- reflects the capacitance of the neuron in principle -- biological default for AdEx spiking model C = 281 pF = 2.81 normalized
	VmTau float32 `default:"2.81" min:"1"`

	// dendritic membrane potential time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life) -- reflects the capacitance of the neuron in principle -- biological default for AdEx spiking model C = 281 pF = 2.81 normalized
	VmDendTau float32 `default:"5" min:"1"`

	// number of integration steps to take in computing new Vm value -- this is the one computation that can be most numerically unstable so taking multiple steps with proportionally smaller dt is beneficial
	VmSteps int32 `default:"2" min:"1"`

	// time constant for decay of excitatory AMPA receptor conductance.
	GeTau float32 `default:"5" min:"1"`

	// time constant for decay of inhibitory GABAa receptor conductance.
	GiTau float32 `default:"7" min:"1"`

	// time constant for integrating values over timescale of an individual input state (e.g., roughly 200 msec -- theta cycle), used in computing ActInt, GeInt from Ge, and GiInt from GiSyn -- this is used for scoring performance, not for learning, in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life),
	IntTau float32 `default:"40" min:"1"`

	// time constant for integrating slower long-time-scale averages, such as nrn.ActAvg, Pool.ActsMAvg, ActsPAvg -- computed in NewState when a new input state is present (i.e., not msec but in units of a theta cycle) (tau is roughly how long it takes for value to change significantly) -- set lower for smaller models
	LongAvgTau float32 `default:"20" min:"1"`

	// cycle to start updating the SpkMaxCa, SpkMax values within a theta cycle -- early cycles often reflect prior state
	MaxCycStart int32 `default:"10" min:"0"`

	// nominal rate = Integ / tau
	VmDt float32 `view:"-" json:"-" xml:"-"`

	// nominal rate = Integ / tau
	VmDendDt float32 `view:"-" json:"-" xml:"-"`

	// 1 / VmSteps
	DtStep float32 `view:"-" json:"-" xml:"-"`

	// rate = Integ / tau
	GeDt float32 `view:"-" json:"-" xml:"-"`

	// rate = Integ / tau
	GiDt float32 `view:"-" json:"-" xml:"-"`

	// rate = Integ / tau
	IntDt float32 `view:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	LongAvgDt float32 `view:"-" json:"-" xml:"-"`
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
	dp.MaxCycStart = 10
	dp.Update()
}

// GeSynFromRaw integrates a synaptic conductance from raw spiking using GeTau
func (dp *DtParams) GeSynFromRaw(geSyn, geRaw float32) float32 {
	return geSyn + geRaw - dp.GeDt*geSyn
}

// GeSynFromRawSteady returns the steady-state GeSyn that would result from
// receiving a steady increment of GeRaw every time step = raw * GeTau.
// dSyn = Raw - dt*Syn; solve for dSyn = 0 to get steady state:
// dt*Syn = Raw; Syn = Raw / dt = Raw * Tau
func (dp *DtParams) GeSynFromRawSteady(geRaw float32) float32 {
	return geRaw * dp.GeTau
}

// GiSynFromRaw integrates a synaptic conductance from raw spiking using GiTau
func (dp *DtParams) GiSynFromRaw(giSyn, giRaw float32) float32 {
	return giSyn + giRaw - dp.GiDt*giSyn
}

// GiSynFromRawSteady returns the steady-state GiSyn that would result from
// receiving a steady increment of GiRaw every time step = raw * GiTau.
// dSyn = Raw - dt*Syn; solve for dSyn = 0 to get steady state:
// dt*Syn = Raw; Syn = Raw / dt = Raw * Tau
func (dp *DtParams) GiSynFromRawSteady(giRaw float32) float32 {
	return giRaw * dp.GiTau
}

// AvgVarUpdate updates the average and variance from current value, using LongAvgDt
func (dp *DtParams) AvgVarUpdate(avg, vr *float32, val float32) {
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
//  Noise

// SpikeNoiseParams parameterizes background spiking activity impinging on the neuron,
// simulated using a poisson spiking process.
type SpikeNoiseParams struct {

	// add noise simulating background spiking levels
	On slbool.Bool

	// mean frequency of excitatory spikes -- typically 50Hz but multiple inputs increase rate -- poisson lambda parameter, also the variance
	GeHz float32 `default:"100"`

	// excitatory conductance per spike -- .001 has minimal impact, .01 can be strong, and .15 is needed to influence timing of clamped inputs
	Ge float32 `min:"0"`

	// mean frequency of inhibitory spikes -- typically 100Hz fast spiking but multiple inputs increase rate -- poisson lambda parameter, also the variance
	GiHz float32 `default:"200"`

	// excitatory conductance per spike -- .001 has minimal impact, .01 can be strong, and .15 is needed to influence timing of clamped inputs
	Gi float32 `min:"0"`

	// add Ge noise to GeMaintRaw instead of standard Ge -- used for PTMaintLayer for example
	MaintGe slbool.Bool

	// Exp(-Interval) which is the threshold for GeNoiseP as it is updated
	GeExpInt float32 `view:"-" json:"-" xml:"-"`

	// Exp(-Interval) which is the threshold for GiNoiseP as it is updated
	GiExpInt float32 `view:"-" json:"-" xml:"-"`
}

func (an *SpikeNoiseParams) Update() {
	an.GeExpInt = math32.Exp(-1000.0 / an.GeHz)
	an.GiExpInt = math32.Exp(-1000.0 / an.GiHz)
}

func (an *SpikeNoiseParams) Defaults() {
	an.GeHz = 100
	an.Ge = 0.001
	an.GiHz = 200
	an.Gi = 0.001
	an.Update()
}

func (an *SpikeNoiseParams) ShouldShow(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return an.On.IsTrue()
	}
}

// PGe updates the GeNoiseP probability, multiplying a uniform random number [0-1]
// and returns Ge from spiking if a spike is triggered
func (an *SpikeNoiseParams) PGe(ctx *Context, p *float32, ni, di uint32) float32 {
	ndi := di*ctx.NetIndexes.NNeurons + ni
	*p *= GetRandomNumber(ndi, ctx.RandCtr, RandFunActPGe)
	if *p <= an.GeExpInt {
		*p = 1
		return an.Ge
	}
	return 0
}

// PGi updates the GiNoiseP probability, multiplying a uniform random number [0-1]
// and returns Gi from spiking if a spike is triggered
func (an *SpikeNoiseParams) PGi(ctx *Context, p *float32, ni, di uint32) float32 {
	ndi := di*ctx.NetIndexes.NNeurons + ni
	*p *= GetRandomNumber(ndi, ctx.RandCtr, RandFunActPGi)
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

	// is this a clamped input layer?  set automatically based on layer type at initialization
	IsInput slbool.Bool `edit:"-"`

	// is this a target layer?  set automatically based on layer type at initialization
	IsTarget slbool.Bool `edit:"-"`

	// amount of Ge driven for clamping -- generally use 0.8 for Target layers, 1.5 for Input layers
	Ge float32 `default:"0.8,1.5"`

	// add external conductance on top of any existing -- generally this is not a good idea for target layers (creates a main effect that learning can never match), but may be ok for input layers
	Add slbool.Bool `default:"false"`

	// threshold on neuron Act activity to count as active for computing error relative to target in PctErr method
	ErrThr float32 `default:"0.5"`

	pad, pad1, pad2 float32
}

func (cp *ClampParams) Update() {
}

func (cp *ClampParams) Defaults() {
	cp.Ge = 0.8
	cp.ErrThr = 0.5
}

//////////////////////////////////////////////////////////////////////////////////////
//  SMaintParams

// SMaintParams for self-maintenance simulating a population of
// NMDA-interconnected spiking neurons
type SMaintParams struct {

	// is self maintenance active?
	On slbool.Bool

	// number of neurons within the self-maintenance pool,
	// each of which is assumed to have the same probability of spiking
	NNeurons float32 `default:"10"`

	// conductance multiplier for self maintenance synapses
	Gbar float32 `default:"0.2"`

	// inhib controls how much of the extra maintenance conductance goes to the GeExt, which drives extra proportional inhibition
	Inhib float32 `default:"1"`

	// ISI (inter spike interval) range -- min is used as min ISIAvg for poisson spike rate expected from the population, and above max, no additional maintenance conductance is added
	ISI minmax.F32 `view:"inline"`
}

func (sm *SMaintParams) Defaults() {
	sm.NNeurons = 10
	sm.ISI.Set(1, 20)
	sm.Gbar = 0.2
	sm.Inhib = 1
}

func (sm *SMaintParams) Update() {
}

func (sm *SMaintParams) ShouldShow(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return sm.On.IsTrue()
	}
}

// ExpInt returns the exponential interval value for determining
// when the next excitatory spike will arrive, based on given ISI
// value for this neuron.
func (sm *SMaintParams) ExpInt(isi float32) float32 {
	if isi <= 0 {
		return 0
	}
	isi = max(isi, sm.ISI.Min)
	return math32.FastExp(-isi / sm.NNeurons)
}

//////////////////////////////////////////////////////////////////////////////////////
//  PopCodeParams

// PopCodeParams provides an encoding of scalar value using population code,
// where a single continuous (scalar) value is encoded as a gaussian bump
// across a population of neurons (1 dimensional).
// It can also modulate rate code and number of neurons active according to the value.
// This is for layers that represent values as in the Rubicon system (from Context.Rubicon).
// Both normalized activation values (1 max) and Ge conductance values can be generated.
type PopCodeParams struct {

	// use popcode encoding of variable(s) that this layer represents
	On slbool.Bool

	// Ge multiplier for driving excitatory conductance based on PopCode -- multiplies normalized activation values
	Ge float32 `default:"0.1"`

	// minimum value representable -- for GaussBump, typically include extra to allow mean with activity on either side to represent the lowest value you want to encode
	Min float32 `default:"-0.1"`

	// maximum value representable -- for GaussBump, typically include extra to allow mean with activity on either side to represent the lowest value you want to encode
	Max float32 `default:"1.1"`

	// activation multiplier for values at Min end of range, where values at Max end have an activation of 1 -- if this is &lt; 1, then there is a rate code proportional to the value in addition to the popcode pattern -- see also MinSigma, MaxSigma
	MinAct float32 `default:"1,0.5"`

	// sigma parameter of a gaussian specifying the tuning width of the coarse-coded units, in normalized 0-1 range -- for Min value -- if MinSigma &lt; MaxSigma then more units are activated for Max values vs. Min values, proportionally
	MinSigma float32 `default:"0.1,0.08"`

	// sigma parameter of a gaussian specifying the tuning width of the coarse-coded units, in normalized 0-1 range -- for Min value -- if MinSigma &lt; MaxSigma then more units are activated for Max values vs. Min values, proportionally
	MaxSigma float32 `default:"0.1,0.12"`

	// ensure that encoded and decoded value remains within specified range
	Clip slbool.Bool
}

func (pc *PopCodeParams) Defaults() {
	pc.Ge = 0.1
	pc.Min = -0.1
	pc.Max = 1.1
	pc.MinAct = 1
	pc.MinSigma = 0.1
	pc.MaxSigma = 0.1
	pc.Clip.SetBool(true)
}

func (pc *PopCodeParams) Update() {
}

func (pc *PopCodeParams) ShouldShow(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return pc.On.IsTrue()
	}
}

// SetRange sets the min, max and sigma values
func (pc *PopCodeParams) SetRange(min, max, minSigma, maxSigma float32) {
	pc.Min = min
	pc.Max = max
	pc.MinSigma = minSigma
	pc.MaxSigma = maxSigma
}

// ClipVal returns clipped (clamped) value in min / max range
func (pc *PopCodeParams) ClipValue(val float32) float32 {
	clipVal := val
	if clipVal < pc.Min {
		clipVal = pc.Min
	}
	if clipVal > pc.Max {
		clipVal = pc.Max
	}
	return clipVal
}

// ProjectParam projects given min / max param value onto val within range
func (pc *PopCodeParams) ProjectParam(minParam, maxParam, clipVal float32) float32 {
	normVal := (clipVal - pc.Min) / (pc.Max - pc.Min)
	return minParam + normVal*(maxParam-minParam)
}

// EncodeVal returns value for given value, for neuron index i
// out of n total neurons. n must be 2 or more.
func (pc *PopCodeParams) EncodeValue(i, n uint32, val float32) float32 {
	clipVal := pc.ClipValue(val)
	if pc.Clip.IsTrue() {
		val = clipVal
	}
	rng := pc.Max - pc.Min
	act := float32(1)
	if pc.MinAct < 1 {
		act = pc.ProjectParam(pc.MinAct, 1.0, clipVal)
	}
	sig := pc.MinSigma
	if pc.MaxSigma > pc.MinSigma {
		sig = pc.ProjectParam(pc.MinSigma, pc.MaxSigma, clipVal)
	}
	gnrm := 1.0 / (rng * sig)
	incr := rng / float32(n-1)
	trg := pc.Min + incr*float32(i)
	dist := gnrm * (trg - val)
	return act * math32.FastExp(-(dist * dist))
}

// EncodeGe returns Ge value for given value, for neuron index i
// out of n total neurons. n must be 2 or more.
func (pc *PopCodeParams) EncodeGe(i, n uint32, val float32) float32 {
	return pc.Ge * pc.EncodeValue(i, n, val)
}

//////////////////////////////////////////////////////////////////////////////////////
//  ActParams

// axon.ActParams contains all the activation computation params and functions
// for basic Axon, at the neuron level .
// This is included in axon.Layer to drive the computation.
type ActParams struct {

	// Spiking function parameters
	Spikes SpikeParams `view:"inline"`

	// dendrite-specific parameters
	Dend DendParams `view:"inline"`

	// initial values for key network state variables -- initialized in InitActs called by InitWts, and provides target values for DecayState
	Init ActInitParams `view:"inline"`

	// amount to decay between AlphaCycles, simulating passage of time and effects of saccades etc, especially important for environments with random temporal structure (e.g., most standard neural net training corpora)
	Decay DecayParams `view:"inline"`

	// time and rate constants for temporal derivatives / updating of activation state
	Dt DtParams `view:"inline"`

	// maximal conductances levels for channels
	Gbar chans.Chans `view:"inline"`

	// reversal potentials for each channel
	Erev chans.Chans `view:"inline"`

	// how external inputs drive neural activations
	Clamp ClampParams `view:"inline"`

	// how, where, when, and how much noise to add
	Noise SpikeNoiseParams `view:"inline"`

	// range for Vm membrane potential -- -- important to keep just at extreme range of reversal potentials to prevent numerical instability
	VmRange minmax.F32 `view:"inline"`

	// M-type medium time-scale afterhyperpolarization mAHP current -- this is the primary form of adaptation on the time scale of multiple sequences of spikes
	Mahp chans.MahpParams `view:"inline"`

	// slow time-scale afterhyperpolarization sAHP current -- integrates CaSpkD at theta cycle intervals and produces a hard cutoff on sustained activity for any neuron
	Sahp chans.SahpParams `view:"inline"`

	// sodium-gated potassium channel adaptation parameters -- activates a leak-like current as a function of neural activity (firing = Na influx) at two different time-scales (Slick = medium, Slack = slow)
	KNa chans.KNaMedSlow `view:"inline"`

	// potassium (K) inwardly rectifying (ir) current, which is similar to GABAB
	// (which is a GABA modulated Kir channel).  This channel is off by default
	// but plays a critical role in making medium spiny neurons (MSNs) relatively
	// quiet in the striatum.
	Kir chans.KirParams `view:"inline"`

	// NMDA channel parameters used in computing Gnmda conductance for bistability, and postsynaptic calcium flux used in learning.  Note that Learn.Snmda has distinct parameters used in computing sending NMDA parameters used in learning.
	NMDA chans.NMDAParams `view:"inline"`

	// NMDA channel parameters used in computing Gnmda conductance for bistability, and postsynaptic calcium flux used in learning.  Note that Learn.Snmda has distinct parameters used in computing sending NMDA parameters used in learning.
	MaintNMDA chans.NMDAParams `view:"inline"`

	// GABA-B / GIRK channel parameters
	GabaB chans.GABABParams `view:"inline"`

	// voltage gated calcium channels -- provide a key additional source of Ca for learning and positive-feedback loop upstate for active neurons
	VGCC chans.VGCCParams `view:"inline"`

	// A-type potassium (K) channel that is particularly important for limiting the runaway excitation from VGCC channels
	AK chans.AKsParams `view:"inline"`

	// small-conductance calcium-activated potassium channel produces the pausing function as a consequence of rapid bursting.
	SKCa chans.SKCaParams `view:"inline"`

	// for self-maintenance simulating a population of
	// NMDA-interconnected spiking neurons
	SMaint SMaintParams `view:"inline"`

	// provides encoding population codes, used to represent a single continuous (scalar) value, across a population of units / neurons (1 dimensional)
	PopCode PopCodeParams `view:"inline"`
}

func (ac *ActParams) Defaults() {
	ac.Spikes.Defaults()
	ac.Dend.Defaults()
	ac.Init.Defaults()
	ac.Decay.Defaults()
	ac.Dt.Defaults()
	ac.Gbar.SetAll(1.0, 0.2, 1.0, 1.0) // E, L, I, K: gbar l = 0.2 > 0.1
	ac.Erev.SetAll(1.0, 0.3, 0.1, 0.1) // E, L, I, K: K = hyperpolarized -90mv
	ac.Clamp.Defaults()
	ac.Noise.Defaults()
	ac.VmRange.Set(0.1, 1.0)
	ac.Mahp.Defaults()
	ac.Mahp.Gbar = 0.02
	ac.Sahp.Defaults()
	ac.Sahp.Gbar = 0.05
	ac.Sahp.CaTau = 5
	ac.KNa.Defaults()
	ac.KNa.On.SetBool(true)
	ac.Kir.Defaults()
	ac.Kir.Gbar = 0
	ac.NMDA.Defaults()
	ac.NMDA.Gbar = 0.006
	ac.MaintNMDA.Defaults()
	ac.MaintNMDA.Gbar = 0.007
	ac.MaintNMDA.Tau = 200
	ac.GabaB.Defaults()
	ac.VGCC.Defaults()
	ac.VGCC.Gbar = 0.02
	ac.VGCC.Ca = 25
	ac.AK.Defaults()
	ac.AK.Gbar = 0.1
	ac.SKCa.Defaults()
	ac.SKCa.Gbar = 0
	ac.SMaint.Defaults()
	ac.PopCode.Defaults()
	ac.Update()
}

// Update must be called after any changes to parameters
func (ac *ActParams) Update() {
	ac.Spikes.Update()
	ac.Dend.Update()
	ac.Init.Update()
	ac.Decay.Update()
	ac.Dt.Update()
	ac.Clamp.Update()
	ac.Noise.Update()
	ac.Mahp.Update()
	ac.Sahp.Update()
	ac.KNa.Update()
	ac.Kir.Update()
	ac.NMDA.Update()
	ac.MaintNMDA.Update()
	ac.GabaB.Update()
	ac.VGCC.Update()
	ac.AK.Update()
	ac.SKCa.Update()
	ac.SMaint.Update()
	ac.PopCode.Update()
}

///////////////////////////////////////////////////////////////////////
//  Init

// DecayLearnCa decays neuron-level calcium learning and spiking variables
// by given factor.  Note: this is generally NOT useful,
// causing variability in these learning factors as a function
// of the decay parameter that then has impacts on learning rates etc.
// see Act.Decay.LearnCa param controlling this
func (ac *ActParams) DecayLearnCa(ctx *Context, ni, di uint32, decay float32) {
	AddNrnV(ctx, ni, di, GnmdaLrn, -decay*NrnV(ctx, ni, di, GnmdaLrn))
	AddNrnV(ctx, ni, di, NmdaCa, -decay*NrnV(ctx, ni, di, NmdaCa))

	AddNrnV(ctx, ni, di, VgccCa, -decay*NrnV(ctx, ni, di, VgccCa))
	AddNrnV(ctx, ni, di, VgccCaInt, -decay*NrnV(ctx, ni, di, VgccCaInt))

	AddNrnV(ctx, ni, di, CaLrn, -decay*NrnV(ctx, ni, di, CaLrn))

	AddNrnV(ctx, ni, di, CaSyn, -decay*NrnV(ctx, ni, di, CaSyn))
	AddNrnV(ctx, ni, di, CaSpkM, -decay*NrnV(ctx, ni, di, CaSpkM))
	AddNrnV(ctx, ni, di, CaSpkP, -decay*NrnV(ctx, ni, di, CaSpkP))
	AddNrnV(ctx, ni, di, CaSpkD, -decay*NrnV(ctx, ni, di, CaSpkD))

	AddNrnV(ctx, ni, di, NrnCaM, -decay*NrnV(ctx, ni, di, NrnCaM))
	AddNrnV(ctx, ni, di, NrnCaP, -decay*NrnV(ctx, ni, di, NrnCaP))
	AddNrnV(ctx, ni, di, NrnCaD, -decay*NrnV(ctx, ni, di, NrnCaD))

	AddNrnV(ctx, ni, di, SKCaIn, decay*(1.0-NrnV(ctx, ni, di, SKCaIn))) // recovers
	AddNrnV(ctx, ni, di, SKCaR, -decay*NrnV(ctx, ni, di, SKCaR))
	AddNrnV(ctx, ni, di, SKCaM, -decay*NrnV(ctx, ni, di, SKCaM))
}

// DecayAHP decays after-hyperpolarization variables
// by given factor (typically Decay.AHP)
func (ac *ActParams) DecayAHP(ctx *Context, ni, di uint32, decay float32) {
	AddNrnV(ctx, ni, di, MahpN, -decay*NrnV(ctx, ni, di, MahpN))
	AddNrnV(ctx, ni, di, Gmahp, -decay*NrnV(ctx, ni, di, Gmahp))
	AddNrnV(ctx, ni, di, SahpCa, -decay*NrnV(ctx, ni, di, SahpCa))
	AddNrnV(ctx, ni, di, SahpN, -decay*NrnV(ctx, ni, di, SahpN))
	AddNrnV(ctx, ni, di, Gsahp, -decay*NrnV(ctx, ni, di, Gsahp))
	AddNrnV(ctx, ni, di, GknaMed, -decay*NrnV(ctx, ni, di, GknaMed))
	AddNrnV(ctx, ni, di, GknaSlow, -decay*NrnV(ctx, ni, di, GknaSlow))
	kirMrest := ac.Kir.Mrest
	AddNrnV(ctx, ni, di, KirM, decay*(kirMrest-NrnV(ctx, ni, di, KirM)))
	AddNrnV(ctx, ni, di, Gkir, -decay*NrnV(ctx, ni, di, Gkir))
}

// DecayState decays the activation state toward initial values
// in proportion to given decay parameter.  Special case values
// such as Glong and KNa are also decayed with their
// separately parameterized values.
// Called with ac.Decay.Act by Layer during NewState
func (ac *ActParams) DecayState(ctx *Context, ni, di uint32, decay, glong, ahp float32) {
	// always reset these -- otherwise get insanely large values that take forever to update
	SetNrnV(ctx, ni, di, ISIAvg, -1)
	SetNrnV(ctx, ni, di, ActInt, ac.Init.Act) // start fresh
	SetNrnV(ctx, ni, di, Spiked, 0)           // always fresh

	if decay > 0 { // no-op for most, but not all..
		SetNrnV(ctx, ni, di, Spike, 0)
		AddNrnV(ctx, ni, di, Act, -decay*(NrnV(ctx, ni, di, Act)-ac.Init.Act))
		AddNrnV(ctx, ni, di, ActInt, -decay*(NrnV(ctx, ni, di, ActInt)-ac.Init.Act))
		AddNrnV(ctx, ni, di, GeSyn, -decay*(NrnV(ctx, ni, di, GeSyn)-NrnAvgV(ctx, ni, GeBase)))
		AddNrnV(ctx, ni, di, Ge, -decay*(NrnV(ctx, ni, di, Ge)-NrnAvgV(ctx, ni, GeBase)))
		AddNrnV(ctx, ni, di, Gi, -decay*(NrnV(ctx, ni, di, Gi)-NrnAvgV(ctx, ni, GiBase)))
		AddNrnV(ctx, ni, di, Gk, -decay*NrnV(ctx, ni, di, Gk))

		AddNrnV(ctx, ni, di, Vm, -decay*(NrnV(ctx, ni, di, Vm)-ac.Init.Vm))

		AddNrnV(ctx, ni, di, GeNoise, -decay*NrnV(ctx, ni, di, GeNoise))
		AddNrnV(ctx, ni, di, GiNoise, -decay*NrnV(ctx, ni, di, GiNoise))

		AddNrnV(ctx, ni, di, GiSyn, -decay*NrnV(ctx, ni, di, GiSyn))

		AddNrnV(ctx, ni, di, GeInt, -decay*NrnV(ctx, ni, di, GeInt))
		AddNrnV(ctx, ni, di, GiInt, -decay*NrnV(ctx, ni, di, GiInt))
		AddNrnV(ctx, ni, di, GeIntNorm, -decay*NrnV(ctx, ni, di, GeIntNorm))
	}

	AddNrnV(ctx, ni, di, VmDend, -glong*(NrnV(ctx, ni, di, VmDend)-ac.Init.Vm))

	if ahp > 0 {
		ac.DecayAHP(ctx, ni, di, ahp)
	}
	AddNrnV(ctx, ni, di, GgabaB, -glong*NrnV(ctx, ni, di, GgabaB))
	AddNrnV(ctx, ni, di, GABAB, -glong*NrnV(ctx, ni, di, GABAB))
	AddNrnV(ctx, ni, di, GABABx, -glong*NrnV(ctx, ni, di, GABABx))

	AddNrnV(ctx, ni, di, GnmdaSyn, -glong*NrnV(ctx, ni, di, GnmdaSyn))
	AddNrnV(ctx, ni, di, Gnmda, -glong*NrnV(ctx, ni, di, Gnmda))
	AddNrnV(ctx, ni, di, GMaintSyn, -glong*NrnV(ctx, ni, di, GMaintSyn))
	AddNrnV(ctx, ni, di, GnmdaMaint, -glong*NrnV(ctx, ni, di, GnmdaMaint))

	AddNrnV(ctx, ni, di, Gvgcc, -glong*NrnV(ctx, ni, di, Gvgcc))
	AddNrnV(ctx, ni, di, VgccM, -glong*NrnV(ctx, ni, di, VgccM))
	AddNrnV(ctx, ni, di, VgccH, -glong*NrnV(ctx, ni, di, VgccH))
	AddNrnV(ctx, ni, di, Gak, -glong*NrnV(ctx, ni, di, Gak))

	// don't mess with SKCa -- longer time scale
	AddNrnV(ctx, ni, di, Gsk, -glong*NrnV(ctx, ni, di, Gsk))

	if ac.Decay.LearnCa > 0 { // learning-based Ca values -- not usual
		ac.DecayLearnCa(ctx, ni, di, ac.Decay.LearnCa)
	}

	SetNrnV(ctx, ni, di, Inet, 0)
	SetNrnV(ctx, ni, di, GeRaw, 0)
	SetNrnV(ctx, ni, di, GiRaw, 0)
	SetNrnV(ctx, ni, di, GModRaw, 0)
	SetNrnV(ctx, ni, di, GModSyn, 0)
	SetNrnV(ctx, ni, di, GMaintRaw, 0)
	SetNrnV(ctx, ni, di, SSGi, 0)
	SetNrnV(ctx, ni, di, SSGiDend, 0)
	SetNrnV(ctx, ni, di, GeExt, 0)

	AddNrnV(ctx, ni, di, CtxtGeOrig, -glong*NrnV(ctx, ni, di, CtxtGeOrig))
}

//gosl: end act

// InitActs initializes activation state in neuron -- called during InitWts but otherwise not
// automatically called (DecayState is used instead)
func (ac *ActParams) InitActs(ctx *Context, ni, di uint32) {
	SetNrnV(ctx, ni, di, Spike, 0)
	SetNrnV(ctx, ni, di, Spiked, 0)
	SetNrnV(ctx, ni, di, ISI, -1)
	SetNrnV(ctx, ni, di, ISIAvg, -1)
	SetNrnV(ctx, ni, di, Act, ac.Init.Act)
	SetNrnV(ctx, ni, di, ActInt, ac.Init.Act)
	SetNrnV(ctx, ni, di, GeSyn, NrnAvgV(ctx, ni, GeBase))
	SetNrnV(ctx, ni, di, Ge, NrnAvgV(ctx, ni, GeBase))
	SetNrnV(ctx, ni, di, Gi, NrnAvgV(ctx, ni, GiBase))
	SetNrnV(ctx, ni, di, Gk, 0)
	SetNrnV(ctx, ni, di, Inet, 0)
	SetNrnV(ctx, ni, di, Vm, ac.Init.Vm)
	SetNrnV(ctx, ni, di, VmDend, ac.Init.Vm)
	SetNrnV(ctx, ni, di, Target, 0)
	SetNrnV(ctx, ni, di, Ext, 0)

	SetNrnV(ctx, ni, di, SpkMaxCa, 0)
	SetNrnV(ctx, ni, di, SpkMax, 0)
	SetNrnV(ctx, ni, di, RLRate, 1)

	SetNrnV(ctx, ni, di, GeNoiseP, 1)
	SetNrnV(ctx, ni, di, GeNoise, 0)
	SetNrnV(ctx, ni, di, GiNoiseP, 1)
	SetNrnV(ctx, ni, di, GiNoise, 0)
	SetNrnV(ctx, ni, di, GiSyn, 0)
	SetNrnV(ctx, ni, di, SMaintP, 1)

	SetNrnV(ctx, ni, di, GeInt, 0)
	SetNrnV(ctx, ni, di, GeIntNorm, 0)
	SetNrnV(ctx, ni, di, GiInt, 0)

	SetNrnV(ctx, ni, di, MahpN, 0)
	SetNrnV(ctx, ni, di, Gmahp, 0)
	SetNrnV(ctx, ni, di, SahpCa, 0)
	SetNrnV(ctx, ni, di, SahpN, 0)
	SetNrnV(ctx, ni, di, Gsahp, 0)
	SetNrnV(ctx, ni, di, GknaMed, 0)
	SetNrnV(ctx, ni, di, GknaSlow, 0)
	SetNrnV(ctx, ni, di, KirM, ac.Kir.Mrest)
	SetNrnV(ctx, ni, di, Gkir, 0)

	SetNrnV(ctx, ni, di, GnmdaSyn, 0)
	SetNrnV(ctx, ni, di, Gnmda, 0)
	SetNrnV(ctx, ni, di, GnmdaMaint, 0)
	SetNrnV(ctx, ni, di, GnmdaLrn, 0)
	SetNrnV(ctx, ni, di, NmdaCa, 0)

	SetNrnV(ctx, ni, di, GgabaB, 0)
	SetNrnV(ctx, ni, di, GABAB, 0)
	SetNrnV(ctx, ni, di, GABABx, 0)

	SetNrnV(ctx, ni, di, Gvgcc, 0)
	SetNrnV(ctx, ni, di, VgccM, 0)
	SetNrnV(ctx, ni, di, VgccH, 0)
	SetNrnV(ctx, ni, di, Gak, 0)
	SetNrnV(ctx, ni, di, VgccCaInt, 0)

	SetNrnV(ctx, ni, di, SKCaIn, 1)
	SetNrnV(ctx, ni, di, SKCaR, 0)
	SetNrnV(ctx, ni, di, SKCaM, 0)
	SetNrnV(ctx, ni, di, Gsk, 0)

	SetNrnV(ctx, ni, di, GeExt, 0)
	SetNrnV(ctx, ni, di, GeRaw, 0)
	SetNrnV(ctx, ni, di, GiRaw, 0)
	SetNrnV(ctx, ni, di, GModRaw, 0)
	SetNrnV(ctx, ni, di, GModSyn, 0)
	SetNrnV(ctx, ni, di, GMaintRaw, 0)
	SetNrnV(ctx, ni, di, GMaintSyn, 0)

	SetNrnV(ctx, ni, di, SSGi, 0)
	SetNrnV(ctx, ni, di, SSGiDend, 0)

	SetNrnV(ctx, ni, di, Burst, 0)
	SetNrnV(ctx, ni, di, BurstPrv, 0)

	SetNrnV(ctx, ni, di, CtxtGe, 0)
	SetNrnV(ctx, ni, di, CtxtGeRaw, 0)
	SetNrnV(ctx, ni, di, CtxtGeOrig, 0)

	ac.InitLongActs(ctx, ni, di)
}

// InitLongActs initializes longer time-scale activation states in neuron
// (SpkPrv, SpkSt*, ActM, ActP)
// Called from InitActs, which is called from InitWts,
// but otherwise not automatically called
// (DecayState is used instead)
func (ac *ActParams) InitLongActs(ctx *Context, ni, di uint32) {
	SetNrnV(ctx, ni, di, SpkPrv, 0)
	SetNrnV(ctx, ni, di, SpkSt1, 0)
	SetNrnV(ctx, ni, di, SpkSt2, 0)
	SetNrnV(ctx, ni, di, ActM, 0)
	SetNrnV(ctx, ni, di, ActP, 0)
}

//gosl: start act

///////////////////////////////////////////////////////////////////////
//  Cycle

// NMDAFromRaw updates all the NMDA variables from
// total Ge (GeRaw + Ext) and current Vm, Spiking
func (ac *ActParams) NMDAFromRaw(ctx *Context, ni, di uint32, geTot float32) {
	if ac.NMDA.Gbar == 0 {
		return
	}
	if geTot < 0 {
		geTot = 0
	}
	SetNrnV(ctx, ni, di, GnmdaSyn, ac.NMDA.NMDASyn(NrnV(ctx, ni, di, GnmdaSyn), geTot))
	SetNrnV(ctx, ni, di, Gnmda, ac.NMDA.Gnmda(NrnV(ctx, ni, di, GnmdaSyn), NrnV(ctx, ni, di, VmDend)))
	// note: nrn.NmdaCa computed via Learn.LrnNMDA in learn.go, CaM method
}

// MaintNMDAFromRaw updates all the Maint NMDA variables from
// GModRaw and current Vm, Spiking
func (ac *ActParams) MaintNMDAFromRaw(ctx *Context, ni, di uint32) {
	if ac.MaintNMDA.Gbar == 0 {
		return
	}
	if ac.SMaint.On.IsTrue() {
		ac.SMaintFromISI(ctx, ni, di)
	}
	SetNrnV(ctx, ni, di, GMaintSyn, ac.MaintNMDA.NMDASyn(NrnV(ctx, ni, di, GMaintSyn), NrnV(ctx, ni, di, GMaintRaw)))
	SetNrnV(ctx, ni, di, GnmdaMaint, ac.MaintNMDA.Gnmda(NrnV(ctx, ni, di, GMaintSyn), NrnV(ctx, ni, di, VmDend)))
}

// SMaintFromISI updates the SMaint self-maintenance current into GMaintRaw
func (ac *ActParams) SMaintFromISI(ctx *Context, ni, di uint32) {
	isi := NrnV(ctx, ni, di, ISIAvg)
	if isi < ac.SMaint.ISI.Min || isi > ac.SMaint.ISI.Max {
		return
	}
	ndi := di*ctx.NetIndexes.NNeurons + ni
	smp := NrnV(ctx, ni, di, SMaintP)
	smp *= GetRandomNumber(ndi, ctx.RandCtr, RandFunActSMaintP)
	trg := ac.SMaint.ExpInt(isi)
	if smp <= trg {
		smp = 1
		AddNrnV(ctx, ni, di, GMaintRaw, ac.SMaint.Gbar)
	}
	SetNrnV(ctx, ni, di, SMaintP, smp)
}

// GvgccFromVm updates all the VGCC voltage-gated calcium channel variables
// from VmDend
func (ac *ActParams) GvgccFromVm(ctx *Context, ni, di uint32) {
	if ac.VGCC.Gbar == 0 {
		return
	}
	SetNrnV(ctx, ni, di, Gvgcc, ac.VGCC.Gvgcc(NrnV(ctx, ni, di, VmDend), NrnV(ctx, ni, di, VgccM), NrnV(ctx, ni, di, VgccH)))
	var dm, dh float32
	ac.VGCC.DMHFromV(NrnV(ctx, ni, di, VmDend), NrnV(ctx, ni, di, VgccM), NrnV(ctx, ni, di, VgccH), &dm, &dh)
	AddNrnV(ctx, ni, di, VgccM, dm)
	AddNrnV(ctx, ni, di, VgccH, dh)
	SetNrnV(ctx, ni, di, VgccCa, ac.VGCC.CaFromG(NrnV(ctx, ni, di, VmDend), NrnV(ctx, ni, di, Gvgcc), NrnV(ctx, ni, di, VgccCa))) // note: may be overwritten!
}

// GkFromVm updates all the Gk-based conductances: Mahp, KNa, Gak
func (ac *ActParams) GkFromVm(ctx *Context, ni, di uint32) {
	vm := NrnV(ctx, ni, di, Vm)
	vmd := NrnV(ctx, ni, di, VmDend)
	mahpN := NrnV(ctx, ni, di, MahpN)
	gmahp := ac.Mahp.GmAHP(vm, &mahpN)
	SetNrnV(ctx, ni, di, Gmahp, gmahp)
	SetNrnV(ctx, ni, di, MahpN, mahpN)

	gsahp := NrnV(ctx, ni, di, Gsahp)

	gak := ac.AK.Gak(vmd)
	SetNrnV(ctx, ni, di, Gak, gak)

	nrnKirM := NrnV(ctx, ni, di, KirM)
	gkir := ac.Kir.Gkir(vm, &nrnKirM)
	SetNrnV(ctx, ni, di, Gkir, gkir)
	SetNrnV(ctx, ni, di, KirM, nrnKirM)

	gktot := gmahp + gsahp + gak + gkir

	if ac.KNa.On.IsTrue() {
		gknaMed := NrnV(ctx, ni, di, GknaMed)
		gknaSlow := NrnV(ctx, ni, di, GknaSlow)
		ac.KNa.GcFromSpike(&gknaMed, &gknaSlow, NrnV(ctx, ni, di, Spike) > .5)
		SetNrnV(ctx, ni, di, GknaMed, gknaMed)
		SetNrnV(ctx, ni, di, GknaSlow, gknaSlow)
		gktot += gknaMed + gknaSlow
	}

	SetNrnV(ctx, ni, di, Gk, gktot)
}

// KNaNewState does TrialSlow version of KNa during NewState if option is set
func (ac *ActParams) KNaNewState(ctx *Context, ni, di uint32) {
	if ac.KNa.On.IsTrue() && ac.KNa.TrialSlow.IsTrue() {
		AddNrnV(ctx, ni, di, GknaSlow, ac.KNa.Slow.Max*NrnV(ctx, ni, di, SpkPrv))
	}
}

// GSkCaFromCa updates the SKCa channel if used
func (ac *ActParams) GSkCaFromCa(ctx *Context, ni, di uint32) {
	if ac.SKCa.Gbar == 0 {
		return
	}
	skcar := NrnV(ctx, ni, di, SKCaR)
	skcain := NrnV(ctx, ni, di, SKCaIn)
	SetNrnV(ctx, ni, di, SKCaM, ac.SKCa.MFromCa(skcar, NrnV(ctx, ni, di, SKCaM)))
	ac.SKCa.CaInRFromSpike(NrnV(ctx, ni, di, Spike), NrnV(ctx, ni, di, CaSpkD), &skcain, &skcar)
	SetNrnV(ctx, ni, di, SKCaR, skcar)
	SetNrnV(ctx, ni, di, SKCaIn, skcain)
	SetNrnV(ctx, ni, di, Gsk, ac.SKCa.Gbar*NrnV(ctx, ni, di, SKCaM))
	AddNrnV(ctx, ni, di, Gk, NrnV(ctx, ni, di, Gsk))
}

// GeFromSyn integrates Ge excitatory conductance from GeSyn.
// geExt is extra conductance to add to the final Ge value
func (ac *ActParams) GeFromSyn(ctx *Context, ni, di uint32, geSyn, geExt float32) {
	SetNrnV(ctx, ni, di, GeExt, 0)
	if ac.Clamp.Add.IsTrue() && NrnHasFlag(ctx, ni, di, NeuronHasExt) {
		SetNrnV(ctx, ni, di, GeExt, NrnV(ctx, ni, di, Ext)*ac.Clamp.Ge)
		geSyn += NrnV(ctx, ni, di, GeExt)
	}

	if ac.Clamp.Add.IsFalse() && NrnHasFlag(ctx, ni, di, NeuronHasExt) { // todo: this flag check is not working
		geSyn = NrnV(ctx, ni, di, Ext) * ac.Clamp.Ge
		SetNrnV(ctx, ni, di, GeExt, geSyn)
		geExt = 0 // no extra in this case
	}

	SetNrnV(ctx, ni, di, Ge, geSyn+geExt)
	if NrnV(ctx, ni, di, Ge) < 0.0 {
		SetNrnV(ctx, ni, di, Ge, 0)
	}
	ac.AddGeNoise(ctx, ni, di)
}

// AddGeNoise updates nrn.GeNoise if active
func (ac *ActParams) AddGeNoise(ctx *Context, ni, di uint32) {
	if ac.Noise.On.IsFalse() || ac.Noise.Ge == 0 {
		return
	}
	p := NrnV(ctx, ni, di, GeNoiseP)
	ge := ac.Noise.PGe(ctx, &p, ni, di)
	SetNrnV(ctx, ni, di, GeNoiseP, p)
	SetNrnV(ctx, ni, di, GeNoise, ac.Dt.GeSynFromRaw(NrnV(ctx, ni, di, GeNoise), ge))
	AddNrnV(ctx, ni, di, Ge, NrnV(ctx, ni, di, GeNoise))
}

// AddGiNoise updates nrn.GiNoise if active
func (ac *ActParams) AddGiNoise(ctx *Context, ni, di uint32) {
	if ac.Noise.On.IsFalse() || ac.Noise.Gi == 0 {
		return
	}
	p := NrnV(ctx, ni, di, GiNoiseP)
	gi := ac.Noise.PGi(ctx, &p, ni, di)
	SetNrnV(ctx, ni, di, GiNoiseP, p)
	SetNrnV(ctx, ni, di, GiNoise, ac.Dt.GiSynFromRaw(NrnV(ctx, ni, di, GiNoise), gi))
}

// GiFromSyn integrates GiSyn inhibitory synaptic conductance from GiRaw value
// (can add other terms to geRaw prior to calling this)
func (ac *ActParams) GiFromSyn(ctx *Context, ni, di uint32, giSyn float32) float32 {
	ac.AddGiNoise(ctx, ni, di)
	if giSyn < 0 { // negative inhib G doesn't make any sense
		giSyn = 0
	}
	return giSyn
}

// InetFromG computes net current from conductances and Vm
func (ac *ActParams) InetFromG(vm, ge, gl, gi, gk float32) float32 {
	inet := ge*(ac.Erev.E-vm) + gl*ac.Gbar.L*(ac.Erev.L-vm) + gi*(ac.Erev.I-vm) + gk*(ac.Erev.K-vm)
	if inet > ac.Dt.VmTau {
		inet = ac.Dt.VmTau
	} else if inet < -ac.Dt.VmTau {
		inet = -ac.Dt.VmTau
	}
	return inet
}

// VmFromInet computes new Vm value from inet, clamping range
func (ac *ActParams) VmFromInet(vm, dt, inet float32) float32 {
	return ac.VmRange.ClipValue(vm + dt*inet)
}

// VmInteg integrates Vm over VmSteps to obtain a more stable value
// Returns the new Vm and inet values.
func (ac *ActParams) VmInteg(vm, dt, ge, gl, gi, gk float32, nvm, inet *float32) {
	dt *= ac.Dt.DtStep
	*nvm = vm
	for i := int32(0); i < ac.Dt.VmSteps; i++ {
		*inet = ac.InetFromG(*nvm, ge, gl, gi, gk)
		*nvm = ac.VmFromInet(*nvm, dt, *inet)
	}
}

// VmFromG computes membrane potential Vm from conductances Ge, Gi, and Gk.
func (ac *ActParams) VmFromG(ctx *Context, ni, di uint32) {
	updtVm := true
	// note: nrn.ISI has NOT yet been updated at this point: 0 right after spike, etc
	// so it takes a full 3 time steps after spiking for Tr period
	isi := NrnV(ctx, ni, di, ISI)
	if ac.Spikes.Tr > 0 && isi >= 0 && isi < float32(ac.Spikes.Tr) {
		updtVm = false // don't update the spiking vm during refract
	}

	ge := NrnV(ctx, ni, di, Ge) * ac.Gbar.E
	gi := NrnV(ctx, ni, di, Gi) * ac.Gbar.I
	gk := NrnV(ctx, ni, di, Gk) * ac.Gbar.K
	var nvm, inet, expi float32
	if updtVm {
		ac.VmInteg(NrnV(ctx, ni, di, Vm), ac.Dt.VmDt, ge, 1, gi, gk, &nvm, &inet)
		if updtVm && ac.Spikes.Exp.IsTrue() { // add spike current if relevant
			var exVm float32
			exVm = 0.5 * (nvm + NrnV(ctx, ni, di, Vm)) // midpoint for this
			expi = ac.Gbar.L * ac.Spikes.ExpSlope *
				math32.FastExp((exVm-ac.Spikes.Thr)/ac.Spikes.ExpSlope)
			if expi > ac.Dt.VmTau {
				expi = ac.Dt.VmTau
			}
			inet += expi
			nvm = ac.VmFromInet(nvm, ac.Dt.VmDt, expi)
		}
		SetNrnV(ctx, ni, di, Vm, nvm)
		SetNrnV(ctx, ni, di, Inet, inet)
	} else { // decay back to VmR
		var dvm float32
		if int32(isi) == ac.Spikes.Tr-1 {
			dvm = ac.Spikes.VmR - NrnV(ctx, ni, di, Vm)
		} else {
			dvm = ac.Spikes.RDt * (ac.Spikes.VmR - NrnV(ctx, ni, di, Vm))
		}
		SetNrnV(ctx, ni, di, Vm, NrnV(ctx, ni, di, Vm)+dvm)
		SetNrnV(ctx, ni, di, Inet, dvm*ac.Dt.VmTau)
	}

	{ // always update VmDend
		glEff := float32(1)
		if !updtVm {
			glEff += ac.Dend.GbarR
		}
		var giEff float32
		giEff = gi + ac.Gbar.I*NrnV(ctx, ni, di, SSGiDend)
		ac.VmInteg(NrnV(ctx, ni, di, VmDend), ac.Dt.VmDendDt, ge, glEff, giEff, gk, &nvm, &inet)
		if updtVm {
			nvm = ac.VmFromInet(nvm, ac.Dt.VmDendDt, ac.Dend.GbarExp*expi)
		}
		SetNrnV(ctx, ni, di, VmDend, nvm)
	}
}

// SpikeFromVmVars computes Spike from Vm and ISI-based activation, using pointers to variables
func (ac *ActParams) SpikeFromVmVars(nrnISI, nrnISIAvg, nrnSpike, nrnSpiked, nrnAct *float32, nrnVm float32) {
	var thr float32
	if ac.Spikes.Exp.IsTrue() {
		thr = ac.Spikes.ExpThr
	} else {
		thr = ac.Spikes.Thr
	}
	if nrnVm >= thr {
		*nrnSpike = 1
		if *nrnISIAvg == -1 {
			*nrnISIAvg = -2
		} else if *nrnISI > 0 { // must have spiked to update
			*nrnISIAvg = ac.Spikes.AvgFromISI(*nrnISIAvg, *nrnISI+1)
		}
		*nrnISI = 0
	} else {
		*nrnSpike = 0
		if *nrnISI >= 0 {
			*nrnISI += 1
			if *nrnISI < 10 {
				*nrnSpiked = 1
			} else {
				*nrnSpiked = 0
			}
			if *nrnISI > 200 { // keep from growing infinitely large
				// used to do this arbitrarily in DecayState but that
				// caused issues with missing refractory periods
				*nrnISI = -1
			}
		} else {
			*nrnSpiked = 0
		}
		if *nrnISIAvg >= 0 && *nrnISI > 0 && *nrnISI > 1.2**nrnISIAvg {
			*nrnISIAvg = ac.Spikes.AvgFromISI(*nrnISIAvg, *nrnISI)
		}
	}

	nwAct := ac.Spikes.ActFromISI(*nrnISIAvg, .001, ac.Dt.Integ)
	if nwAct > 1 {
		nwAct = 1
	}
	nwAct = *nrnAct + ac.Dt.VmDt*(nwAct-*nrnAct)
	*nrnAct = nwAct
}

// SpikeFromVm computes Spike from Vm and ISI-based activation
func (ac *ActParams) SpikeFromVm(ctx *Context, ni, di uint32) {
	nrnISI := NrnV(ctx, ni, di, ISI)
	nrnISIAvg := NrnV(ctx, ni, di, ISIAvg)
	nrnSpike := NrnV(ctx, ni, di, Spike)
	nrnSpiked := NrnV(ctx, ni, di, Spiked)
	nrnAct := NrnV(ctx, ni, di, Act)
	nrnVm := NrnV(ctx, ni, di, Vm)
	ac.SpikeFromVmVars(&nrnISI, &nrnISIAvg, &nrnSpike, &nrnSpiked, &nrnAct, nrnVm)
	SetNrnV(ctx, ni, di, ISI, nrnISI)
	SetNrnV(ctx, ni, di, ISIAvg, nrnISIAvg)
	SetNrnV(ctx, ni, di, Spike, nrnSpike)
	SetNrnV(ctx, ni, di, Spiked, nrnSpiked)
	SetNrnV(ctx, ni, di, Act, nrnAct)
}

//gosl: end act
