// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/erand"
	"github.com/goki/ki/bools"
	"github.com/goki/mat32"
)

// DriveParams manages the drive parameters for computing and updating drive state.
// Most of the params are for optional case where drives are automatically
// updated based on US consumption (which satisfies drives) and time passing
// (which increases drives).
type DriveParams struct {

	// minimum effective drive value -- this is an automatic baseline ensuring that a positive US results in at least some minimal level of reward.  Unlike Base values, this is not reflected in the activity of the drive values -- applies at the time of reward calculation as a minimum baseline.
	DriveMin float32 `desc:"minimum effective drive value -- this is an automatic baseline ensuring that a positive US results in at least some minimal level of reward.  Unlike Base values, this is not reflected in the activity of the drive values -- applies at the time of reward calculation as a minimum baseline."`

	// baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range).
	Base []float32 `desc:"baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range)."`

	// time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update.
	Tau []float32 `desc:"time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update."`

	// decrement in drive value when US is consumed, thus partially satisfying the drive -- positive values are subtracted from current Drive value.
	Satisfaction []float32 `desc:"decrement in drive value when US is consumed, thus partially satisfying the drive -- positive values are subtracted from current Drive value."`

	// [view: -] 1/Tau
	Dt []float32 `view:"-" desc:"1/Tau"`
}

func (dp *DriveParams) Alloc(nDrives int) {
	if len(dp.Base) == nDrives {
		return
	}
	dp.Base = make([]float32, nDrives)
	dp.Tau = make([]float32, nDrives)
	dp.Dt = make([]float32, nDrives)
	dp.Satisfaction = make([]float32, nDrives)
}

func (dp *DriveParams) Defaults() {
	dp.DriveMin = 0.5
	for i := range dp.Satisfaction {
		dp.Satisfaction[i] = 0
	}
	dp.Update()
}

func (dp *DriveParams) Update() {
	for i, tau := range dp.Tau {
		if tau <= 0 {
			dp.Dt[i] = 0
		} else {
			dp.Dt[i] = 1 / tau
		}
	}
}

// VarToZero sets all values of given drive-sized variable to 0
func (dp *DriveParams) VarToZero(ctx *Context, di uint32, gvar GlobalVars) {
	for i := range dp.Base {
		SetGlbUSposV(ctx, di, gvar, uint32(i), 0)
	}
}

// ToZero sets all drives to 0
func (dp *DriveParams) ToZero(ctx *Context, di uint32) {
	dp.VarToZero(ctx, di, GvDrives)
}

// ToBaseline sets all drives to their baseline levels
func (dp *DriveParams) ToBaseline(ctx *Context, di uint32) {
	for i := range dp.Base {
		SetGlbUSposV(ctx, di, GvDrives, uint32(i), dp.Base[i])
	}
}

// AddTo increments drive by given amount, subject to 0-1 range clamping.
// Returns new val.
func (dp *DriveParams) AddTo(ctx *Context, di uint32, drv uint32, delta float32) float32 {
	dv := GlbUSposV(ctx, di, GvDrives, drv) + delta
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	SetGlbUSposV(ctx, di, GvDrives, drv, dv)
	return dv
}

// SoftAdd increments drive by given amount, using soft-bounding to 0-1 extremes.
// if delta is positive, multiply by 1-val, else val.  Returns new val.
func (dp *DriveParams) SoftAdd(ctx *Context, di uint32, drv uint32, delta float32) float32 {
	dv := GlbUSposV(ctx, di, GvDrives, drv)
	if delta > 0 {
		dv += (1 - dv) * delta
	} else {
		dv += dv * delta
	}
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	SetGlbUSposV(ctx, di, GvDrives, drv, dv)
	return dv
}

// ExpStep updates drive with an exponential step with given dt value
// toward given baseline value.
func (dp *DriveParams) ExpStep(ctx *Context, di uint32, drv uint32, dt, base float32) float32 {
	dv := GlbUSposV(ctx, di, GvDrives, drv)
	dv += dt * (base - dv)
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	SetGlbUSposV(ctx, di, GvDrives, drv, dv)
	return dv
}

// ExpStepAll updates given drives with an exponential step using dt values
// toward baseline values.
func (dp *DriveParams) ExpStepAll(ctx *Context, di uint32) {
	for i := range dp.Base {
		dp.ExpStep(ctx, di, uint32(i), dp.Dt[i], dp.Base[i])
	}
}

// EffectiveDrive returns the Max of Drives at given index and DriveMin.
// note that index 0 is the novelty / curiosity drive, which doesn't use DriveMin.
func (dp *DriveParams) EffectiveDrive(ctx *Context, di uint32, i uint32) float32 {
	if i == 0 {
		return GlbUSposV(ctx, di, GvDrives, uint32(0))
	}
	return mat32.Max(GlbUSposV(ctx, di, GvDrives, i), dp.DriveMin)
}

/////////////////////////////////////////////////////////
//  EffortParams

// EffortParams has parameters for giving up based on effort,
// which is the first negative US.
type EffortParams struct {

	// default maximum raw effort level, for deciding when to give up on goal pursuit, when MaxNovel and MaxPostDip don't apply.
	Max float32 `desc:"default maximum raw effort level, for deciding when to give up on goal pursuit, when MaxNovel and MaxPostDip don't apply."`

	// maximum raw effort level when novelty / curiosity drive is engaged -- typically shorter than default Max
	MaxNovel float32 `desc:"maximum raw effort level when novelty / curiosity drive is engaged -- typically shorter than default Max"`

	// if the LowThr amount of VSPatch expectation is triggered, as accumulated in LHb.DipSum, then CurMax is set to the current Raw effort plus this increment, which is generally low -- once an expectation has been activated, don't wait around forever..
	MaxPostDip float32 `desc:"if the LowThr amount of VSPatch expectation is triggered, as accumulated in LHb.DipSum, then CurMax is set to the current Raw effort plus this increment, which is generally low -- once an expectation has been activated, don't wait around forever.."`

	// variance in additional maximum effort level, applied whenever CurMax is updated
	MaxVar float32 `desc:"variance in additional maximum effort level, applied whenever CurMax is updated"`
}

func (ef *EffortParams) Defaults() {
	ef.Max = 100
	ef.MaxNovel = 8
	ef.MaxPostDip = 4
	ef.MaxVar = 2
}

func (ef *EffortParams) Update() {

}

// Reset resets the raw effort back to zero -- at start of new gating event
func (ef *EffortParams) Reset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvEffortRaw, 0)
	SetGlbV(ctx, di, GvEffortCurMax, ef.Max)
	SetGlbUSneg(ctx, di, GvUSnegRaw, 0, 0) // effort is neg 0
	SetGlbUSneg(ctx, di, GvUSneg, 0, 0)
}

// ReStart restarts restarts the raw effort back to zero
// and sets the Max with random additional variance.
func (ef *EffortParams) ReStart(ctx *Context, di uint32, rnd erand.Rand) {
	ef.Reset(ctx, di)
	SetGlbV(ctx, di, GvEffortCurMax, ef.PlusVar(rnd, ef.Max))
}

// SetPostDipMax sets the current Max to current raw effort plus
// MaxPostDip, with random additional variance.
func (ef *EffortParams) SetPostDipMax(ctx *Context, di uint32, rnd erand.Rand) {
	SetGlbV(ctx, di, GvEffortCurMax, ef.PlusVar(rnd, GlbV(ctx, di, GvEffortRaw)+ef.MaxPostDip))
}

// SetNovelMax sets the current Max to MaxNovel with random additional variance.
func (ef *EffortParams) SetNovelMax(ctx *Context, di uint32, rnd erand.Rand) {
	SetGlbV(ctx, di, GvEffortCurMax, ef.PlusVar(rnd, ef.MaxNovel))
}

// EffortNorm returns the current effort value as a normalized number,
// for external calculation purposes (this method not used for PVLV computations).
// This normalization is performed internally on _all_ negative USs
func EffortNorm(ctx *Context, di uint32, factor float32) float32 {
	return PVLVNormFun(factor * GlbV(ctx, di, GvEffortRaw))
}

// AddEffort adds an increment of effort and updates the Disc discount factor
func (ef *EffortParams) AddEffort(ctx *Context, di uint32, inc float32) {
	AddGlbV(ctx, di, GvEffortRaw, inc)
	eff := GlbV(ctx, di, GvEffortRaw)
	SetGlbUSneg(ctx, di, GvUSnegRaw, 0, eff) // effort is neg 0
}

// todo: on all not this:

// GiveUp returns true if maximum effort has been exceeded
func (ef *EffortParams) GiveUp(ctx *Context, di uint32) bool {
	raw := GlbV(ctx, di, GvEffortRaw)
	curMax := GlbV(ctx, di, GvEffortCurMax)
	if curMax > 0 && raw > curMax {
		return true
	}
	return false
}

// PlusVar returns value plus random variance
func (ef *EffortParams) PlusVar(rnd erand.Rand, max float32) float32 {
	if ef.MaxVar == 0 {
		return max
	}
	return max + ef.MaxVar*float32(rnd.NormFloat64(-1))
}

///////////////////////////////////////////////////////////////////////////////
//  UrgencyParams

// UrgencyParams has urgency (increasing pressure to do something)
// and parameters for updating it.
// Raw urgency integrates effort when _not_ goal engaged
// while effort (negative US 0) integrates when a goal _is_ engaged.
type UrgencyParams struct {

	// value of raw urgency where the urgency activation level is 50%
	U50 float32 `desc:"value of raw urgency where the urgency activation level is 50%"`

	// [def: 4] exponent on the urge factor -- valid numbers are 1,2,4,6
	Power int32 `def:"4" desc:"exponent on the urge factor -- valid numbers are 1,2,4,6"`

	// [def: 0.2] threshold for urge -- cuts off small baseline values
	Thr float32 `def:"0.2" desc:"threshold for urge -- cuts off small baseline values"`
}

func (ur *UrgencyParams) Defaults() {
	ur.U50 = 10
	ur.Power = 4
	ur.Thr = 0.2
}

func (ur *UrgencyParams) Update() {

}

// UrgeFun is the urgency function: urgency / (urgency + 1) where
// urgency = (Raw / U50)^Power
func (ur *UrgencyParams) UrgeFun(urgency float32) float32 {
	urgency /= ur.U50
	switch ur.Power {
	case 2:
		urgency *= urgency
	case 4:
		urgency *= urgency * urgency * urgency
	case 6:
		urgency *= urgency * urgency * urgency * urgency * urgency
	}
	return urgency / (1.0 + urgency)
}

// Reset resets the raw urgency back to zero -- at start of new gating event
func (ur *UrgencyParams) Reset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvUrgencyRaw, 0)
	SetGlbV(ctx, di, GvUrgency, 0)
}

// Urge computes normalized Urge value from Raw
func (ur *UrgencyParams) Urge(ctx *Context, di uint32) float32 {
	urge := ur.UrgeFun(GlbV(ctx, di, GvUrgencyRaw))
	if urge < ur.Thr {
		urge = 0
	}
	SetGlbV(ctx, di, GvUrgency, urge)
	return urge
}

// AddEffort adds an effort increment of urgency and updates the Urge factor
func (ur *UrgencyParams) AddEffort(ctx *Context, di uint32, inc float32) {
	AddGlbV(ctx, di, GvUrgencyRaw, inc)
	ur.Urge(ctx, di)
}

/////////////////////////////////////////////////////////
// USParams

// PVLVLNormFun is the normalizing function applied to the sum of all
// weighted raw values: 1 - (1 / (1 + usRaw.Sum()))
func PVLVNormFun(raw float32) float32 {
	return 1.0 - (1.0 / (1.0 + raw))
}

// USParams control how positive and negative USs are
// weighted and integrated to compute an overall PV primary value.
type USParams struct {

	// threshold for a negative US increment, _after_ multiplying by the NegGains factor for that US (to allow for normalized input magnitudes that may translate into different magnitude of effects), to drive a phasic ACh response and associated VSMatrix gating and dopamine firing -- i.e., a full negative US outcome event (global NegUSOutcome flag is set)
	NegUSOutcomeThr float32 `desc:"threshold for a negative US increment, _after_ multiplying by the NegGains factor for that US (to allow for normalized input magnitudes that may translate into different magnitude of effects), to drive a phasic ACh response and associated VSMatrix gating and dopamine firing -- i.e., a full negative US outcome event (global NegUSOutcome flag is set)"`

	// gain factor for sum of positive USs, multiplied prior to 1/(1+x) normalization in computing PVPos.
	PVPosGain float32 `desc:"gain factor for sum of positive USs, multiplied prior to 1/(1+x) normalization in computing PVPos."`

	// gain factor for sum of negative USs, multiplied prior to 1/(1+x) normalization in computing PVNeg.
	PVNegGain float32 `desc:"gain factor for sum of negative USs, multiplied prior to 1/(1+x) normalization in computing PVNeg."`

	// gain factor for each negative US, multiplied prior to 1/(1+x) normalization of each term for activating the NegUS value.
	NegGains []float32 `desc:"gain factor for each negative US, multiplied prior to 1/(1+x) normalization of each term for activating the NegUS value."`

	// weight factor for each positive US, multiplied prior to 1/(1+x) normalization of the sum.  Each pos US is also multiplied by its dynamic Drive factor as well
	PVPosWts []float32 `desc:"weight factor for each positive US, multiplied prior to 1/(1+x) normalization of the sum.  Each pos US is also multiplied by its dynamic Drive factor as well"`

	// weight factor for each negative US, multiplied prior to 1/(1+x) normalization of the sum.
	PVNegWts []float32 `desc:"weight factor for each negative US, multiplied prior to 1/(1+x) normalization of the sum."`
}

func (us *USParams) Alloc(nPos, nNeg int) {
	if len(us.PVPosWts) != nPos {
		us.PVPosWts = make([]float32, nPos)
	}
	if len(us.PVNegWts) != nNeg {
		us.NegGains = make([]float32, nNeg)
		us.PVNegWts = make([]float32, nNeg)
	}
}

func (us *USParams) Defaults() {
	us.NegUSOutcomeThr = 0.5
	us.PVPosGain = 5
	us.PVNegGain = 0.05
	for i := range us.PVPosWts {
		us.PVPosWts[i] = 1
	}
	for i := range us.NegGains {
		us.NegGains[i] = 0.05
		us.PVNegWts[i] = 1
	}
}

func (us *USParams) Update() {
}

// USnegFromRaw sets normalized NegUS values from Raw values
func (us *USParams) USnegFromRaw(ctx *Context, di uint32) {
	for i, ng := range us.NegGains {
		raw := GlbUSneg(ctx, di, GvUSnegRaw, uint32(i))
		norm := PVLVNormFun(ng * raw)
		SetGlbUSneg(ctx, di, GvUSneg, uint32(i), norm)
		// fmt.Printf("neg %d  raw: %g  norm: %g\n", i, raw, norm)
	}
}

// USnegToZero sets all values of USneg, USNegRaw to zero
func (us *USParams) USnegToZero(ctx *Context, di uint32) {
	for i := range us.NegGains {
		SetGlbUSneg(ctx, di, GvUSneg, uint32(i), 0)
		SetGlbUSneg(ctx, di, GvUSnegRaw, uint32(i), 0)
	}
}

// USposToZero sets all values of USpos to zero
func (us *USParams) USposToZero(ctx *Context, di uint32) {
	for i := range us.PVPosWts {
		SetGlbUSposV(ctx, di, GvUSpos, uint32(i), 0)
	}
}

// NegUSOutcome returns true if given magnitude of negative US increment
// is sufficient to drive a full-blown outcome event, clearing goals, driving DA etc.
// usIdx is actual index (0 = effort)
func (us *USParams) NegUSOutcome(ctx *Context, di uint32, usIdx int, mag float32) bool {
	gmag := us.NegGains[usIdx] * mag
	if gmag > us.NegUSOutcomeThr {
		SetGlbV(ctx, di, GvNegUSOutcome, 1)
		return true
	}
	return false
}

///////////////////////////////////////////////////////////////////
//  LHb & RMTg

// LHbParams has values for computing LHb & RMTg which drives dips / pauses in DA firing.
// LHb handles all US-related (PV = primary value) processing.
// Positive net LHb activity drives dips / pauses in VTA DA activity,
// e.g., when predicted pos > actual or actual neg > predicted.
// Negative net LHb activity drives bursts in VTA DA activity,
// e.g., when actual pos > predicted (redundant with LV / Amygdala)
// or "relief" burst when actual neg < predicted.
type LHbParams struct {

	// [def: 1] threshold factor that multiplies integrated pvNeg value to establish a threshold for whether the integrated pvPos value is good enough to drive overall net positive reward
	NegThr float32 `def:"1" desc:"threshold factor that multiplies integrated pvNeg value to establish a threshold for whether the integrated pvPos value is good enough to drive overall net positive reward"`

	// [def: 1] gain multiplier on PVpos for purposes of generating bursts (not for  discounting negative dips) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)
	PosGain float32 `def:"1" desc:"gain multiplier on PVpos for purposes of generating bursts (not for  discounting negative dips) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)"`

	// [def: 1] gain multiplier on PVneg for purposes of generating dips (not for  discounting positive bursts) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)
	NegGain float32 `def:"1" desc:"gain multiplier on PVneg for purposes of generating dips (not for  discounting positive bursts) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)"`

	// [def: 0.2] threshold on summed LHbDip over trials for triggering a reset of goal engaged state
	GiveUpThr float32 `def:"0.2" desc:"threshold on summed LHbDip over trials for triggering a reset of goal engaged state"`

	// [def: 0.05] low threshold on summed LHbDip, used for triggering switch to a faster effort max timeout -- Effort.MaxPostDip
	DipLowThr float32 `def:"0.05" desc:"low threshold on summed LHbDip, used for triggering switch to a faster effort max timeout -- Effort.MaxPostDip"`
}

func (lh *LHbParams) Defaults() {
	lh.NegThr = 1
	lh.PosGain = 1
	lh.NegGain = 1
	lh.GiveUpThr = 0.2
	lh.DipLowThr = 0.05
}

func (lh *LHbParams) Update() {
}

// Reset resets all LHb vars back to 0
func (lh *LHbParams) Reset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvLHbDip, 0)
	SetGlbV(ctx, di, GvLHbBurst, 0)
	SetGlbV(ctx, di, GvLHbPVDA, 0)
	SetGlbV(ctx, di, GvLHbDipSumCur, 0)
	SetGlbV(ctx, di, GvLHbDipSum, 0)
	SetGlbV(ctx, di, GvLHbGiveUp, 0)
}

// DAforUS computes the overall LHb Dip or Burst (one is always 0),
// and PVDA ~= Burst - Dip, for case when there is a primary
// positive reward value or a give-up state has triggered.
func (lh *LHbParams) DAforUS(ctx *Context, di uint32, pvPos, pvNeg, vsPatchPos float32) {
	thr := lh.NegThr * pvNeg
	pos := lh.PosGain * pvPos
	neg := lh.NegGain * pvNeg
	burst := float32(0)
	dip := float32(0)
	net := pvPos - thr // if > 0, net positive outcome; else net negative (not worth it)
	if net > 0 {       // worth it
		pr := pos * (1 - pvNeg) // positive reward value: pos with mult neg discount factor
		rpe := pr - vsPatchPos  // prediction error relative to pos reward value
		if rpe < 0 {
			dip = -rpe // positive dip = negative value
		} else {
			burst = rpe
		}
	} else { // not worth it: net negative but moderated (discounted) by strength of positive
		dip = neg * (1 - pvPos)
		// todo: vsPatchNeg needed
	}
	SetGlbV(ctx, di, GvLHbDip, dip)
	SetGlbV(ctx, di, GvLHbBurst, burst)
	SetGlbV(ctx, di, GvLHbPVDA, burst-dip)
}

// DAforNoUS computes the overall LHb Dip = vsPatchPos,
// and PVDA ~= -Dip, for case when there is _NOT_ a primary
// positive reward value or a give-up state.
// In this case, inhibition of VS via ACh is assumed to prevent activity of PVneg
// (and there is no PVpos), so only vsPatchPos is operative.
func (lh *LHbParams) DAforNoUS(ctx *Context, di uint32, vsPatchPos float32) {
	burst := float32(0)
	dip := vsPatchPos // dip is entirely mis-prediction of positive outcome
	SetGlbV(ctx, di, GvLHbDip, dip)
	SetGlbV(ctx, di, GvLHbBurst, burst)
	SetGlbV(ctx, di, GvLHbPVDA, burst-dip)
}

// todo: based on total negus too!

// ShouldGiveUp increments DipSum and checks if should give up if above threshold
func (lh *LHbParams) ShouldGiveUp(ctx *Context, di uint32) bool {
	dip := GlbV(ctx, di, GvLHbDip)
	AddGlbV(ctx, di, GvLHbDipSumCur, dip)
	cur := GlbV(ctx, di, GvLHbDipSumCur)
	SetGlbV(ctx, di, GvLHbDipSum, cur)
	SetGlbV(ctx, di, GvLHbGiveUp, 0)
	giveUp := false
	if cur > lh.GiveUpThr {
		giveUp = true
		SetGlbV(ctx, di, GvLHbGiveUp, 1)
		SetGlbV(ctx, di, GvLHbDipSumCur, 0)
	}
	return giveUp
}

///////////////////////////////////////////////////////////////////////////////
//  PVLV

// PVLV represents the core brainstem-level (hypothalamus) bodily drives
// and resulting dopamine from US (unconditioned stimulus) inputs,
// as computed by the PVLV model of primary value (PV)
// and learned value (LV), describing the functions of the Amygala,
// Ventral Striatum, VTA and associated midbrain nuclei (LDT, LHb, RMTg).
// Core LHb (lateral habenula) and VTA (ventral tegmental area) dopamine
// are computed in equations using inputs from specialized network layers
// (LDTLayer driven by BLA, CeM layers, VSPatchLayer).
// The Drives, Effort, US and resulting LHb PV dopamine computation all happens at the
// at the start of each trial (NewState, Step).  The LV / CS dopamine is computed
// cycle-by-cycle by the VTA layer using parameters set by the VTA layer.
// Renders USLayer, PVLayer, DrivesLayer representations based on state updated here.
type PVLV struct {

	// number of possible positive US states and corresponding drives -- the first is always reserved for novelty / curiosity.  Must be set programmatically via SetNUSs method, which allocates corresponding parameters.
	NPosUSs uint32 `inactive:"+" desc:"number of possible positive US states and corresponding drives -- the first is always reserved for novelty / curiosity.  Must be set programmatically via SetNUSs method, which allocates corresponding parameters."`

	// number of possible negative US states -- the first is always reserved for the accumulated effort cost (which drives dissapointment when an expected US is not achieved).  Must be set programmatically via SetNUSs method, which allocates corresponding parameters.
	NNegUSs uint32 `inactive:"+" desc:"number of possible negative US states -- the first is always reserved for the accumulated effort cost (which drives dissapointment when an expected US is not achieved).  Must be set programmatically via SetNUSs method, which allocates corresponding parameters."`

	// parameters and state for built-in drives that form the core motivations of agent, controlled by lateral hypothalamus and associated body state monitoring such as glucose levels and thirst.
	Drive DriveParams `desc:"parameters and state for built-in drives that form the core motivations of agent, controlled by lateral hypothalamus and associated body state monitoring such as glucose levels and thirst."`

	// [view: inline] effort parameters and state, tracking relative depletion of glucose levels and water levels as a function of time and exertion
	Effort EffortParams `view:"inline" desc:"effort parameters and state, tracking relative depletion of glucose levels and water levels as a function of time and exertion"`

	// [view: inline] urgency (increasing pressure to do something) and parameters for updating it. Raw urgency is incremented by same units as effort, but is only reset with a positive US.
	Urgency UrgencyParams `view:"inline" desc:"urgency (increasing pressure to do something) and parameters for updating it. Raw urgency is incremented by same units as effort, but is only reset with a positive US."`

	// controls how positive and negative USs are weighted and integrated to compute an overall PV primary value.
	USs USParams `desc:"controls how positive and negative USs are weighted and integrated to compute an overall PV primary value."`

	// [view: inline] lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing
	LHb LHbParams `view:"inline" desc:"lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing"`
}

func (pp *PVLV) Defaults() {
	pp.Drive.Defaults()
	pp.Effort.Defaults()
	pp.Urgency.Defaults()
	pp.USs.Defaults()
	pp.LHb.Defaults()
}

func (pp *PVLV) Update() {
	pp.Drive.Update()
	pp.Effort.Update()
	pp.Urgency.Update()
	pp.USs.Update()
	pp.LHb.Update()
}

// SetNUSs sets the number of positive and negative USs (primary value outcomes).
// This must be called _before_ network Build, which allocates global values
// that depend on these numbers.  Any change must also call network.BuildGlobals.
// Positive USs each have corresponding Drives, and drive 0 is novelty / curiosity
// (nPos must be >= 1).
// Negative US 0 is effort (implicitly represents time), so nNeg must be >= 1).
func (pp *PVLV) SetNUSs(ctx *Context, nPos, nNeg int) {
	if nPos < 1 {
		nPos = 1
	}
	if nNeg < 1 {
		nNeg = 1
	}
	pp.NPosUSs = uint32(nPos)
	pp.NNegUSs = uint32(nNeg)
	ctx.NetIdxs.PVLVNPosUSs = pp.NPosUSs
	ctx.NetIdxs.PVLVNNegUSs = pp.NNegUSs
	pp.Drive.Alloc(nPos)
	pp.USs.Alloc(nPos, nNeg)
}

// Reset resets all PVLV state
func (pp *PVLV) Reset(ctx *Context, di uint32) {
	pp.Drive.ToBaseline(ctx, di)
	pp.Effort.Reset(ctx, di)
	pp.Urgency.Reset(ctx, di)
	pp.InitUS(ctx, di)
	pp.LHb.Reset(ctx, di)
	pp.Drive.VarToZero(ctx, di, GvVSPatch)
	SetGlbV(ctx, di, GvVtaDA, 0)
	SetGlbV(ctx, di, GvVSMatrixJustGated, 0)
	SetGlbV(ctx, di, GvVSMatrixHasGated, 0)
	SetGlbV(ctx, di, GvHadRew, 0)
	// pp.HasPosUSPrev.SetBool(false) // key to not reset!!
}

// InitUS initializes all the USs to zero
func (pp *PVLV) InitUS(ctx *Context, di uint32) {
	pp.USs.USposToZero(ctx, di)
	pp.USs.USnegToZero(ctx, di)
	SetGlbV(ctx, di, GvHasRew, 0)
	SetGlbV(ctx, di, GvRew, 0)
}

// InitDrives initializes all the Drives to baseline values (default = 0)
func (pp *PVLV) InitDrives(ctx *Context, di uint32) {
	pp.Drive.ToBaseline(ctx, di)
}

// EffortUrgencyUpdt updates the Effort or Urgency based on
// given effort increment.
// Effort is incremented when VSMatrixHasGated (i.e., goal engaged)
// and Urgency updates otherwise (when not goal engaged)
// Call this at the start of the trial, in ApplyPVLV method,
// after NewState.
func (pp *PVLV) EffortUrgencyUpdt(ctx *Context, di uint32, effort float32) {
	if GlbV(ctx, di, GvVSMatrixHasGated) > 0 {
		pp.Effort.AddEffort(ctx, di, effort)
	} else {
		pp.Urgency.AddEffort(ctx, di, effort)
	}
}

// PVposFmDriveEffort returns the net primary value ("reward") based on
// given US value and drive for that value (typically in 0-1 range),
// and total effort, from which the effort discount factor is computed an applied:
// usValue * drive * Effort.DiscFun(effort).
// This is not called directly in the PVLV code -- can be used to compute
// what the PVLV code itself will compute -- see LHbPVDA
func (pp *PVLV) PVposFmDriveEffort(ctx *Context, usValue, drive, effort float32) float32 {
	return usValue * drive * (1 - PVLVNormFun(pp.USs.PVNegWts[0]*effort))
}

// PVLVSetDrive sets given Drive to given value
func (pp *PVLV) SetDrive(ctx *Context, di uint32, dr uint32, val float32) {
	SetGlbUSposV(ctx, di, GvDrives, dr, val)
}

// SetDrives is used when directly controlling drive levels externally.
// It resets all drives to baseline (default 0)
// and then sets given drive indexes (0 based) to given magnitude,
// and first curiosity drive to given level.
// Drive indexes are 0 based, but 0 is the curiosity drive,
// so 1 is added automatically when setting drives from indexes.
func (pp *PVLV) SetDrives(ctx *Context, di uint32, curiosity, magnitude float32, drives ...int) {
	pp.InitDrives(ctx, di)
	pp.SetDrive(ctx, di, 0, curiosity)
	for _, i := range drives {
		pp.SetDrive(ctx, di, uint32(1+i), magnitude)
	}
}

// DriveUpdt is used when auto-updating drive levels based on US consumption,
// which partially satisfies (decrements) corresponding drive,
// and on time passing, where drives adapt to their overall baseline levels.
func (pp *PVLV) DriveUpdt(ctx *Context, di uint32) {
	pp.Drive.ExpStepAll(ctx, di)
	nd := pp.NPosUSs
	for i := uint32(0); i < nd; i++ {
		us := GlbUSposV(ctx, di, GvUSpos, i)
		nwdrv := GlbUSposV(ctx, di, GvDrives, i) - us*pp.Drive.Satisfaction[i]
		if nwdrv < 0 {
			nwdrv = 0
		}
		SetGlbUSposV(ctx, di, GvDrives, i, nwdrv)
	}
}

// SetUS sets the given unconditioned stimulus (US) state for PVLV algorithm.
// This then drives activity of relevant PVLV-rendered inputs, and dopamine.
// The US index is automatically adjusted for the curiosity drive / US for
// positive US outcomes and effort for negative USs --
// i.e., pass in a value with 0 starting index.
// By default, negative USs only set the overall ctx.NeuroMod.HasRew flag,
// when they exceed the NegUSOutcomeThr, thus triggering a full-blown US learning event.
func (pp *PVLV) SetUS(ctx *Context, di uint32, valence ValenceTypes, usIdx int, magnitude float32) {
	if valence == Positive {
		SetGlbV(ctx, di, GvHasRew, 1)                              // only for positive USs
		SetGlbUSposV(ctx, di, GvUSpos, uint32(usIdx)+1, magnitude) // +1 for curiosity
	} else {
		AddGlbUSneg(ctx, di, GvUSnegRaw, uint32(usIdx)+1, magnitude) // +1 for effort
		if pp.USs.NegUSOutcome(ctx, di, usIdx+1, magnitude) {
			SetGlbV(ctx, di, GvHasRew, 1)
		}
	}
}

// NewState is called at very start of new state (trial) of processing.
// sets HadRew = HasRew from last trial -- used to then reset various things
// after reward.
func (pp *PVLV) NewState(ctx *Context, di uint32, rnd erand.Rand) {
	hadRewF := GlbV(ctx, di, GvHasRew)
	hadRew := bools.FromFloat32(hadRewF)
	SetGlbV(ctx, di, GvHadRew, hadRewF)
	SetGlbV(ctx, di, GvHadPosUS, GlbV(ctx, di, GvHasPosUS))
	SetGlbV(ctx, di, GvHadNegUSOutcome, GlbV(ctx, di, GvNegUSOutcome))
	SetGlbV(ctx, di, GvLHbGaveUp, GlbV(ctx, di, GvLHbGiveUp))

	SetGlbV(ctx, di, GvHasRew, 0)
	SetGlbV(ctx, di, GvNegUSOutcome, 0)

	vsPatchPos := pp.VSPatchMax(ctx, di)
	SetGlbV(ctx, di, GvLHbVSPatchPos, vsPatchPos)
	SetGlbV(ctx, di, GvRewPred, GlbV(ctx, di, GvLHbVSPatchPos))

	if hadRew {
		SetGlbV(ctx, di, GvVSMatrixHasGated, 0)
		pp.Effort.ReStart(ctx, di, rnd)
		pp.USs.USnegToZero(ctx, di) // all negs restart
		SetGlbV(ctx, di, GvLHbVSPatchPos, 0)
		SetGlbV(ctx, di, GvRewPred, 0)
	} else if GlbV(ctx, di, GvVSMatrixJustGated) > 0 {
		SetGlbV(ctx, di, GvVSMatrixHasGated, 1)
		pp.Urgency.Reset(ctx, di)
		pp.Effort.ReStart(ctx, di, rnd)
		pp.USs.USnegToZero(ctx, di) // all negs restart
		SetGlbV(ctx, di, GvLHbDipSumCur, 0)
		if GlbV(ctx, di, GvCuriosityPoolGated) > 0 {
			pp.Effort.SetNovelMax(ctx, di, rnd)
		}
	}
	SetGlbV(ctx, di, GvVSMatrixJustGated, 0)
	pp.USs.USposToZero(ctx, di) // pos USs must be set fresh every time
}

// Step does one step (trial) after applying USs, Drives,
// and updating Effort.  It should be the final call in ApplyPVLV.
// Updates USneg from USnegRaw, ShouldGiveUp, and PVDA,
// which computes the primary value DA.
func (pp *PVLV) Step(ctx *Context, di uint32, rnd erand.Rand) {
	pp.SetHasPosUS(ctx, di)
	pp.USs.USnegFromRaw(ctx, di)
	pp.ShouldGiveUp(ctx, di, rnd)
	pp.PVDA(ctx, di)
}

//////////////////////////////////////////////////////////////////////////////////////
//    methods below used in computing PVLV state, not generally called from sims

// PVpos returns the summed weighted positive value of
// current positive US state, where each US is multiplied by
// its current drive and weighting factor (usPosSum),
// and the normalized version of this sum (PVpos = overall positive PV)
// as 1 / (1 + (PVPosGain * usPosSum))
func (pp *PVLV) PVpos(ctx *Context, di uint32) (usPosSum, pvPos float32) {
	nd := pp.NPosUSs
	wts := pp.USs.PVPosWts
	for i := uint32(0); i < nd; i++ {
		usPosSum += wts[i] * GlbUSposV(ctx, di, GvUSpos, i) * pp.Drive.EffectiveDrive(ctx, di, i)
	}
	pvPos = PVLVNormFun(pp.USs.PVPosGain * usPosSum)
	return
}

// PVneg returns the summed weighted negative value
// of current negative US state, where each US
// is multiplied by a weighting factor and summed (usNegSum)
// and the normalized version of this sum (PVneg = overall negative PV)
// as 1 / (1 + (PVNegGain * usNegSum))
func (pp *PVLV) PVneg(ctx *Context, di uint32) (usNegSum, pvNeg float32) {
	nn := pp.NNegUSs
	wts := pp.USs.PVNegWts
	for i := uint32(0); i < nn; i++ {
		usNegSum += wts[i] * GlbUSneg(ctx, di, GvUSnegRaw, i)
	}
	pvNeg = PVLVNormFun(pp.USs.PVNegGain * usNegSum)
	return
}

// VSPatchMax returns the max VSPatch value across drives
func (pp *PVLV) VSPatchMax(ctx *Context, di uint32) float32 {
	max := float32(0)
	nd := pp.NPosUSs
	for i := uint32(0); i < nd; i++ {
		vs := GlbUSposV(ctx, di, GvVSPatch, i)
		if vs > max {
			max = vs
		}
	}
	return max
}

// HasPosUS returns true if there is at least one non-zero positive US
func (pp *PVLV) HasPosUS(ctx *Context, di uint32) bool {
	nd := pp.NPosUSs
	for i := uint32(0); i < nd; i++ {
		if GlbUSposV(ctx, di, GvUSpos, i) > 0 {
			return true
		}
	}
	return false
}

// HasNegUS returns true if there is at least one non-zero negative US
func (pp *PVLV) HasNegUS(ctx *Context, di uint32) bool {
	nd := pp.NNegUSs
	for i := uint32(0); i < nd; i++ {
		if GlbUSneg(ctx, di, GvUSnegRaw, i) > 0 {
			return true
		}
	}
	return false
}

// SetHasPosUS sets the HasPosUS global flag
func (pp *PVLV) SetHasPosUS(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvHasPosUS, bools.ToFloat32(pp.HasPosUS(ctx, di)))
}

// ShouldGiveUp tests whether it is time to give up on the current goal,
// based on sum of LHb Dip (missed expected rewards) and maximum effort.
func (pp *PVLV) ShouldGiveUp(ctx *Context, di uint32, rnd erand.Rand) bool {
	hasRew := GlbV(ctx, di, GvHasRew) > 0
	SetGlbV(ctx, di, GvLHbGiveUp, 0)
	if hasRew { // can't give up if got something now
		SetGlbV(ctx, di, GvLHbDipSumCur, 0)
		return false
	}
	prevSum := GlbV(ctx, di, GvLHbDipSumCur)
	giveUp := pp.LHb.ShouldGiveUp(ctx, di)
	if prevSum < pp.LHb.DipLowThr && GlbV(ctx, di, GvLHbDipSumCur) >= pp.LHb.DipLowThr {
		pp.Effort.SetPostDipMax(ctx, di, rnd)
	}
	if pp.Effort.GiveUp(ctx, di) {
		SetGlbV(ctx, di, GvLHbGiveUp, 1)
		giveUp = true
	}
	if giveUp {
		GlobalSetRew(ctx, di, 0, true) // sets HasRew -- drives maint reset, ACh
	}
	return giveUp
}

// PVDA computes the PV (primary value) based dopamine
// based on current state information, at the start of a trial.
// PV DA is computed by the VS (ventral striatum) and the LHb / RMTg,
// and the resulting values are stored in LHb global variables.
// Called after updating USs, Effort, Drives at start of trial step,
// in Step.  Returns the resulting LHbPVDA value.
func (pp *PVLV) PVDA(ctx *Context, di uint32) float32 {
	hasRew := (GlbV(ctx, di, GvHasRew) > 0)
	usPosSum, pvPos := pp.PVpos(ctx, di)
	usNegSum, pvNeg := pp.PVneg(ctx, di)
	SetGlbV(ctx, di, GvLHbPVposSum, usPosSum)
	SetGlbV(ctx, di, GvLHbPVnegSum, usNegSum)
	SetGlbV(ctx, di, GvLHbPVpos, pvPos)
	SetGlbV(ctx, di, GvLHbPVneg, pvNeg)

	vsPatchPos := GlbV(ctx, di, GvLHbVSPatchPos)

	if hasRew { // note: also true for giveup
		pp.LHb.DAforUS(ctx, di, pvPos, pvNeg, vsPatchPos) // only when actual pos rew
		SetGlbV(ctx, di, GvRew, pvPos-pvNeg)              // primary value diff
	} else {
		pp.LHb.DAforNoUS(ctx, di, vsPatchPos)
		SetGlbV(ctx, di, GvRew, 0)
	}
	return GlbV(ctx, di, GvLHbPVDA)
}
