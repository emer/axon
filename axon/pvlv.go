// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/erand"
	"goki.dev/glop/num"
	"goki.dev/mat32/v2"
)

// DriveParams manages the drive parameters for computing and updating drive state.
// Most of the params are for optional case where drives are automatically
// updated based on US consumption (which satisfies drives) and time passing
// (which increases drives).
type DriveParams struct {

	// minimum effective drive value -- this is an automatic baseline ensuring that a positive US results in at least some minimal level of reward.  Unlike Base values, this is not reflected in the activity of the drive values -- applies at the time of reward calculation as a minimum baseline.
	DriveMin float32

	// baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range).
	Base []float32

	// time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update.
	Tau []float32

	// decrement in drive value when US is consumed, thus partially satisfying the drive -- positive values are subtracted from current Drive value.
	Satisfaction []float32

	// 1/Tau
	Dt []float32 `view:"-"`
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

///////////////////////////////////////////////////////////////////////////////
//  UrgencyParams

// UrgencyParams has urgency (increasing pressure to do something)
// and parameters for updating it.
// Raw urgency integrates effort when _not_ goal engaged
// while effort (negative US 0) integrates when a goal _is_ engaged.
type UrgencyParams struct {

	// value of raw urgency where the urgency activation level is 50%
	U50 float32

	// exponent on the urge factor -- valid numbers are 1,2,4,6
	Power int32 `def:"4"`

	// threshold for urge -- cuts off small baseline values
	Thr float32 `def:"0.2"`
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

	// threshold for a negative US increment, _after_ multiplying by the USnegGains factor for that US (to allow for normalized input magnitudes that may translate into different magnitude of effects), to drive a phasic ACh response and associated VSMatrix gating and dopamine firing -- i.e., a full negative US outcome event (global NegUSOutcome flag is set)
	NegUSOutcomeThr float32 `def:"0.5"`

	// gain factor applied to sum of weighted, drive-scaled positive USs to compute PVpos primary value summary -- multiplied prior to 1/(1+x) normalization.  Use this to adjust the overall scaling of PVpos reward within 0-1 normalized range (see also PVnegGain).  Each USpos is assumed to be in 0-1 range, default 1.
	PVposGain float32 `def:"2"`

	// gain factor applied to sum of weighted negative USs to compute PVneg primary value summary -- multiplied prior to 1/(1+x) normalization.  Use this to adjust overall scaling of PVneg within 0-1 normalized range (see also PVposGain).
	PVnegGain float32 `def:"1"`

	// gain factor for each individual negative US, multiplied prior to 1/(1+x) normalization of each term for activating the OFCnegUS pools.  These gains are _not_ applied in computing summary PVneg value (see PVnegWts), and generally must be larger than the weights to leverage the dynamic range within each US pool.
	USnegGains []float32

	// weight factor applied to each separate positive US on the way to computing the overall PVpos summary value, to control the weighting of each US relative to the others. Each pos US is also multiplied by its dynamic Drive factor as well.  Use PVposGain to control the overall scaling of the PVpos value.
	PVposWts []float32

	// weight factor applied to each separate negative US on the way to computing the overall PVneg summary value, to control the weighting of each US relative to the others.  The first pool is Time, second is Effort, and these are typically weighted lower (.02) than salient simulation-specific USs (1).
	PVnegWts []float32

	// computed estimated US values, based on OFCposUSPT and VSMatrix gating, in PVposEst
	USposEst []float32 `inactive:"+"`
}

func (us *USParams) Alloc(nPos, nNeg int) {
	if len(us.PVposWts) != nPos {
		us.PVposWts = make([]float32, nPos)
		us.USposEst = make([]float32, nPos)
	}
	if len(us.PVnegWts) != nNeg {
		us.USnegGains = make([]float32, nNeg)
		us.PVnegWts = make([]float32, nNeg)
	}
}

func (us *USParams) Defaults() {
	us.NegUSOutcomeThr = 0.5
	us.PVposGain = 2
	us.PVnegGain = 1
	for i := range us.PVposWts {
		us.PVposWts[i] = 1
	}
	for i := range us.USnegGains {
		if i < 2 { // time, effort
			us.USnegGains[i] = 0.1
			us.PVnegWts[i] = 0.02
		} else { // other sim-specific USs
			us.USnegGains[i] = 2
			us.PVnegWts[i] = 1
		}
	}
}

func (us *USParams) Update() {
}

// USnegFromRaw sets normalized NegUS values from Raw values
func (us *USParams) USnegFromRaw(ctx *Context, di uint32) {
	for i, ng := range us.USnegGains {
		raw := GlbUSneg(ctx, di, GvUSnegRaw, uint32(i))
		norm := PVLVNormFun(ng * raw)
		SetGlbUSneg(ctx, di, GvUSneg, uint32(i), norm)
		// fmt.Printf("neg %d  raw: %g  norm: %g\n", i, raw, norm)
	}
}

// USnegToZero sets all values of USneg, USNegRaw to zero
func (us *USParams) USnegToZero(ctx *Context, di uint32) {
	for i := range us.USnegGains {
		SetGlbUSneg(ctx, di, GvUSneg, uint32(i), 0)
		SetGlbUSneg(ctx, di, GvUSnegRaw, uint32(i), 0)
	}
}

// USposToZero sets all values of USpos to zero
func (us *USParams) USposToZero(ctx *Context, di uint32) {
	for i := range us.PVposWts {
		SetGlbUSposV(ctx, di, GvUSpos, uint32(i), 0)
	}
}

// NegUSOutcome returns true if given magnitude of negative US increment
// is sufficient to drive a full-blown outcome event, clearing goals, driving DA etc.
// usIdx is actual index (0 = effort)
func (us *USParams) NegUSOutcome(ctx *Context, di uint32, usIdx int, mag float32) bool {
	gmag := us.USnegGains[usIdx] * mag
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

	// threshold factor that multiplies integrated pvNeg value to establish a threshold for whether the integrated pvPos value is good enough to drive overall net positive reward
	NegThr float32 `def:"1"`

	// gain multiplier on PVpos for purposes of generating bursts (not for  discounting negative dips) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)
	BurstGain float32 `def:"1"`

	// gain multiplier on PVneg for purposes of generating dips (not for  discounting positive bursts) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)
	DipGain float32 `def:"1"`
}

func (lh *LHbParams) Defaults() {
	lh.NegThr = 1
	lh.BurstGain = 1
	lh.DipGain = 1
}

func (lh *LHbParams) Update() {
}

// Reset resets all LHb vars back to 0
func (lh *LHbParams) Reset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvLHbDip, 0)
	SetGlbV(ctx, di, GvLHbBurst, 0)
	SetGlbV(ctx, di, GvLHbPVDA, 0)
}

// DAFmPVs computes the overall PV DA in terms of LHb burst and dip
// activity from given pvPos, pvNeg, and vsPatchPos values.
// Also returns the net "reward" value as the discounted PV value,
// separate from the vsPatchPos prediction error factor.
func (lh *LHbParams) DAFmPVs(pvPos, pvNeg, vsPatchPos float32) (burst, dip, da, rew float32) {
	thr := lh.NegThr * pvNeg
	net := pvPos - thr // if > 0, net positive outcome; else net negative (not worth it)
	if net > 0 {       // worth it
		rew = lh.BurstGain * pvPos * (1 - pvNeg) // positive reward value: pos with mult neg discount factor
		rpe := rew - vsPatchPos                  // prediction error relative to pos reward value
		if rpe < 0 {
			dip = -rpe // positive dip = negative value
		} else {
			burst = rpe
		}
	} else { // not worth it: net negative but moderated (discounted) by strength of positive
		rew = -lh.DipGain * pvNeg * (1 - pvPos)
		dip = -rew // magnitude
	}
	da = burst - dip
	return
}

// DAforUS computes the overall LHb Dip or Burst (one is always 0),
// and PVDA ~= Burst - Dip, for case when there is a primary
// positive reward value or a give-up state has triggered.
// Returns the overall net reward magnitude, prior to VSPatch discounting.
func (lh *LHbParams) DAforUS(ctx *Context, di uint32, pvPos, pvNeg, vsPatchPos float32) float32 {
	burst, dip, da, rew := lh.DAFmPVs(pvPos, pvNeg, vsPatchPos)
	SetGlbV(ctx, di, GvLHbDip, dip)
	SetGlbV(ctx, di, GvLHbBurst, burst)
	SetGlbV(ctx, di, GvLHbPVDA, da)
	return rew
}

// DAforNoUS computes the overall LHb Dip = vsPatchPos,
// and PVDA ~= -Dip, for case when there is _NOT_ a primary
// positive reward value or a give-up state.
// In this case, inhibition of VS via ACh is assumed to prevent activity of PVneg
// (and there is no PVpos), so only vsPatchPos is operative.
// Returns net dopamine which is -vsPatchPos.
func (lh *LHbParams) DAforNoUS(ctx *Context, di uint32, vsPatchPos float32) float32 {
	burst := float32(0)
	dip := vsPatchPos // dip is entirely mis-prediction of positive outcome
	SetGlbV(ctx, di, GvLHbDip, dip)
	SetGlbV(ctx, di, GvLHbBurst, burst)
	SetGlbV(ctx, di, GvLHbPVDA, burst-dip)
	return burst - dip
}

//////////////////////////////////////////////////////////
//  GiveUpParams

// GiveUpParams are parameters for computing when to give up
type GiveUpParams struct {

	// threshold factor that multiplies integrated pvNeg value to establish a threshold for whether the integrated pvPos value is good enough to drive overall net positive reward
	NegThr float32 `def:"1"`

	// multiplier on pos - neg for logistic probability function -- higher gain values produce more binary give up behavior and lower values produce more graded stochastic behavior around the threshold
	Gain float32 `def:"10"`

	// minimum estimated PVpos value -- deals with any errors in the estimation process to make sure that erroneous GiveUp doesn't happen.
	MinPVposEst float32
}

func (gp *GiveUpParams) Defaults() {
	gp.NegThr = 1
	gp.Gain = 10
	gp.MinPVposEst = 0.2
}

// LogisticFun is the sigmoid logistic function
func LogisticFun(v, gain float32) float32 {
	return (1.0 / (1.0 + mat32.Exp(-gain*v)))
}

func (gp *GiveUpParams) Prob(pvDiff float32, rnd erand.Rand) (float32, bool) {
	prob := LogisticFun(pvDiff, gp.Gain)
	giveUp := erand.BoolP32(prob, -1, rnd)
	return prob, giveUp
}

//////////////////////////////////////////////////////////
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
	NPosUSs uint32 `inactive:"+"`

	// number of possible negative US states -- is reserved for accumulated time, the accumulated effort cost.  Must be set programmatically via SetNUSs method, which allocates corresponding parameters.
	NNegUSs uint32 `inactive:"+"`

	// parameters and state for built-in drives that form the core motivations of agent, controlled by lateral hypothalamus and associated body state monitoring such as glucose levels and thirst.
	Drive DriveParams

	// urgency (increasing pressure to do something) and parameters for updating it. Raw urgency is incremented by same units as effort, but is only reset with a positive US.
	Urgency UrgencyParams `view:"inline"`

	// controls how positive and negative USs are weighted and integrated to compute an overall PV primary value.
	USs USParams

	// lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing
	LHb LHbParams `view:"inline"`

	// parameters for giving up based on PV pos - neg difference
	GiveUp GiveUpParams
}

func (pp *PVLV) Defaults() {
	pp.Drive.Defaults()
	pp.Urgency.Defaults()
	pp.USs.Defaults()
	pp.LHb.Defaults()
	pp.GiveUp.Defaults()
}

func (pp *PVLV) Update() {
	pp.Drive.Update()
	pp.Urgency.Update()
	pp.USs.Update()
	pp.LHb.Update()
}

// USposIdx adds 1 to the given _simulation specific_ positive US index
// to get the actual US / Drive index, where the first pool is reserved
// for curiosity / novelty.
func (pp *PVLV) USposIdx(simUsIdx int) int {
	return simUsIdx + 1
}

// USnegIdx adds 2 to the given _simulation specific_ negative US index
// to get the actual US index, where the first pool is reserved
// for time, and the second for effort
func (pp *PVLV) USnegIdx(simUsIdx int) int {
	return simUsIdx + 2
}

// SetNUSs sets the number of _additional_ simulation-specific
// positive and negative USs (primary value outcomes).
// This must be called _before_ network Build, which allocates global values
// that depend on these numbers.  Any change must also call network.BuildGlobals.
// 1 PosUS (curiosity / novelty) and 2 NegUSs (time, effort) are managed automatically
// by the PVLV code; any additional USs specified here need to be managed by the
// simulation via the SetUS method.
// Positive USs each have corresponding Drives.
func (pp *PVLV) SetNUSs(ctx *Context, nPos, nNeg int) {
	nPos = pp.USposIdx(nPos)
	nNeg = pp.USnegIdx(nNeg)
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
	pp.TimeEffortReset(ctx, di)
	pp.Urgency.Reset(ctx, di)
	pp.InitUS(ctx, di)
	pp.LHb.Reset(ctx, di)
	pp.Drive.VarToZero(ctx, di, GvVSPatch)
	pp.ResetGoalState(ctx, di)
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

// AddTimeEffort adds a unit of time and an increment of effort
func (pp *PVLV) AddTimeEffort(ctx *Context, di uint32, effort float32) {
	AddGlbV(ctx, di, GvTime, 1)
	tm := GlbV(ctx, di, GvTime)
	SetGlbUSneg(ctx, di, GvUSnegRaw, 0, tm) // time is neg 0

	AddGlbV(ctx, di, GvEffort, effort)
	eff := GlbV(ctx, di, GvEffort)
	SetGlbUSneg(ctx, di, GvUSnegRaw, 1, eff) // effort is neg 1
}

// EffortUrgencyUpdt updates the Effort or Urgency based on
// given effort increment.
// Effort is incremented when VSMatrixHasGated (i.e., goal engaged)
// and Urgency updates otherwise (when not goal engaged)
// Call this at the start of the trial, in ApplyPVLV method,
// after NewState.
func (pp *PVLV) EffortUrgencyUpdt(ctx *Context, di uint32, effort float32) {
	if GlbV(ctx, di, GvVSMatrixHasGated) > 0 {
		pp.AddTimeEffort(ctx, di, effort)
	} else {
		pp.Urgency.AddEffort(ctx, di, effort)
	}
}

// TimeEffortReset resets the raw time and effort back to zero,
// at start of new gating event
func (pp *PVLV) TimeEffortReset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvTime, 0)
	SetGlbV(ctx, di, GvEffort, 0)
	SetGlbUSneg(ctx, di, GvUSnegRaw, 0, 0) // effort is neg 0
	SetGlbUSneg(ctx, di, GvUSneg, 0, 0)
}

// PVposFmDriveEffort returns the net primary value ("reward") based on
// given US value and drive for that value (typically in 0-1 range),
// and total effort, from which the effort discount factor is computed an applied:
// usValue * drive * Effort.DiscFun(effort).
// This is not called directly in the PVLV code -- can be used to compute
// what the PVLV code itself will compute -- see LHbPVDA
// todo: this is not very meaningful anymore
// func (pp *PVLV) PVposFmDriveEffort(ctx *Context, usValue, drive, effort float32) float32 {
// 	return usValue * drive * (1 - PVLVNormFun(pp.USs.PVnegWts[0]*effort))
// }

// PVLVSetDrive sets given Drive to given value
func (pp *PVLV) SetDrive(ctx *Context, di uint32, dr uint32, val float32) {
	SetGlbUSposV(ctx, di, GvDrives, dr, val)
}

// SetDrives is used when directly controlling drive levels externally.
// curiosity sets the strength for the curiosity drive
// and drives are strengths of the remaining sim-specified drives, in order.
// any drives not so specified are at the InitDrives baseline level.
func (pp *PVLV) SetDrives(ctx *Context, di uint32, curiosity float32, drives ...float32) {
	pp.InitDrives(ctx, di)
	pp.SetDrive(ctx, di, 0, curiosity)
	for i, v := range drives {
		pp.SetDrive(ctx, di, uint32(1+i), v)
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

// SetUS sets the given _simulation specific_ unconditioned
// stimulus (US) state for PVLV algorithm.  usIdx = 0 is first additional US, etc.
// The US then drives activity of relevant PVLV-rendered inputs, and dopamine.
// By default, negative USs only set the overall ctx.NeuroMod.HasRew flag
// when they exceed the NegUSOutcomeThr, thus triggering a full-blown US learning event.
// Otherwise, they accumulate as in the case of effort and time, and can then
// trigger giving up as a function of the total accumulated negative valence.
func (pp *PVLV) SetUS(ctx *Context, di uint32, valence ValenceTypes, usIdx int, magnitude float32) {
	if valence == Positive {
		usIdx = pp.USposIdx(usIdx)
		SetGlbV(ctx, di, GvHasRew, 1) // all positive USs are final outcomes
		SetGlbUSposV(ctx, di, GvUSpos, uint32(usIdx), magnitude)
	} else {
		usIdx = pp.USnegIdx(usIdx)
		AddGlbUSneg(ctx, di, GvUSnegRaw, uint32(usIdx), magnitude)
		if pp.USs.NegUSOutcome(ctx, di, usIdx, magnitude) {
			SetGlbV(ctx, di, GvHasRew, 1)
		}
	}
}

// ResetGoalState resets all the goal-engaged global values.
// Critically, this is only called after goal accomplishment,
// not after goal gating -- prevents "shortcutting" by re-gating.
func (pp *PVLV) ResetGoalState(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvVSMatrixHasGated, 0)
	pp.Urgency.Reset(ctx, di)
	pp.TimeEffortReset(ctx, di)
	pp.USs.USnegToZero(ctx, di) // all negs restart
	pp.ResetGiveUp(ctx, di)
	SetGlbV(ctx, di, GvVSPatchPos, 0)
	SetGlbV(ctx, di, GvVSPatchPosPrev, 0)
	SetGlbV(ctx, di, GvVSPatchPosSum, 0)
	SetGlbV(ctx, di, GvRewPred, 0)
	nd := pp.NPosUSs
	for i := uint32(0); i < nd; i++ {
		SetGlbUSposV(ctx, di, GvOFCposUSPTMaint, i, 0)
		SetGlbUSposV(ctx, di, GvVSMatrixPoolGated, i, 0)
	}
}

// ResetGiveUp resets all the give-up related global values.
func (pp *PVLV) ResetGiveUp(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvPVposEst, 0)
	SetGlbV(ctx, di, GvPVposEstSum, 0)
	SetGlbV(ctx, di, GvPVposEstDisc, 0)
	SetGlbV(ctx, di, GvGiveUpDiff, 0)
	SetGlbV(ctx, di, GvGiveUpProb, 0)
	SetGlbV(ctx, di, GvGiveUp, 0)
}

// NewState is called at very start of new state (trial) of processing.
// sets HadRew = HasRew from last trial -- used to then reset various things
// after reward.
func (pp *PVLV) NewState(ctx *Context, di uint32, rnd erand.Rand) {
	hadRewF := GlbV(ctx, di, GvHasRew)
	hadRew := num.ToBool(hadRewF)
	SetGlbV(ctx, di, GvHadRew, hadRewF)
	SetGlbV(ctx, di, GvHadPosUS, GlbV(ctx, di, GvHasPosUS))
	SetGlbV(ctx, di, GvHadNegUSOutcome, GlbV(ctx, di, GvNegUSOutcome))
	SetGlbV(ctx, di, GvGaveUp, GlbV(ctx, di, GvGiveUp))

	SetGlbV(ctx, di, GvHasRew, 0)
	SetGlbV(ctx, di, GvNegUSOutcome, 0)

	pp.VSPatchNewState(ctx, di)

	if hadRew {
		pp.ResetGoalState(ctx, di)
	} else if GlbV(ctx, di, GvVSMatrixJustGated) > 0 {
		SetGlbV(ctx, di, GvVSMatrixHasGated, 1)
		pp.Urgency.Reset(ctx, di)
	}
	SetGlbV(ctx, di, GvVSMatrixJustGated, 0)
	pp.USs.USposToZero(ctx, di) // pos USs must be set fresh every time
}

// Step does one step (trial) after applying USs, Drives,
// and updating Effort.  It should be the final call in ApplyPVLV.
// Calls PVDA which does all US, PV, LHb, GiveUp updating.
func (pp *PVLV) Step(ctx *Context, di uint32, rnd erand.Rand) {
	pp.PVDA(ctx, di, rnd)
}

//////////////////////////////////////////////////////////////////////////////////////
//    methods below used in computing PVLV state, not generally called from sims

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

// PVpos returns the summed weighted positive value of
// current positive US state, where each US is multiplied by
// its current drive and weighting factor (pvPosSum),
// and the normalized version of this sum (PVpos = overall positive PV)
// as 1 / (1 + (PVposGain * pvPosSum))
func (pp *PVLV) PVpos(ctx *Context, di uint32) (pvPosSum, pvPos float32) {
	nd := pp.NPosUSs
	wts := pp.USs.PVposWts
	for i := uint32(0); i < nd; i++ {
		pvPosSum += wts[i] * GlbUSposV(ctx, di, GvUSpos, i) * pp.Drive.EffectiveDrive(ctx, di, i)
	}
	pvPos = PVLVNormFun(pp.USs.PVposGain * pvPosSum)
	return
}

// PVneg returns the summed weighted negative value
// of current negative US state, where each US
// is multiplied by a weighting factor and summed (usNegSum)
// and the normalized version of this sum (PVneg = overall negative PV)
// as 1 / (1 + (PVnegGain * PVnegSum))
func (pp *PVLV) PVneg(ctx *Context, di uint32) (pvNegSum, pvNeg float32) {
	nn := pp.NNegUSs
	wts := pp.USs.PVnegWts
	for i := uint32(0); i < nn; i++ {
		pvNegSum += wts[i] * GlbUSneg(ctx, di, GvUSnegRaw, i)
	}
	pvNeg = PVLVNormFun(pp.USs.PVnegGain * pvNegSum)
	return
}

// PVsFmUSs updates the current PV summed, weighted, normalized values
// from the underlying US values.
func (pp *PVLV) PVsFmUSs(ctx *Context, di uint32) {
	pvPosSum, pvPos := pp.PVpos(ctx, di)
	SetGlbV(ctx, di, GvPVposSum, pvPosSum)
	SetGlbV(ctx, di, GvPVpos, pvPos)
	SetGlbV(ctx, di, GvHasPosUS, num.FromBool[float32](pp.HasPosUS(ctx, di)))

	pvNegSum, pvNeg := pp.PVneg(ctx, di)
	SetGlbV(ctx, di, GvPVnegSum, pvNegSum)
	SetGlbV(ctx, di, GvPVneg, pvNeg)
}

// VSPatchNewState does VSPatch processing in NewState:
// saves to Prev, updates global VSPatchPos and VSPatchPosSum.
// uses max across recorded VSPatch activity levels.
func (pp *PVLV) VSPatchNewState(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvVSPatchPosPrev, GlbV(ctx, di, GvVSPatchPos))
	mx := float32(0)
	nd := pp.NPosUSs
	for i := uint32(0); i < nd; i++ {
		vs := GlbUSposV(ctx, di, GvVSPatch, i)
		SetGlbUSposV(ctx, di, GvVSPatchPrev, i, vs)
		if vs > mx {
			mx = vs
		}
	}
	SetGlbV(ctx, di, GvVSPatchPos, mx)
	AddGlbV(ctx, di, GvVSPatchPosSum, mx)
	SetGlbV(ctx, di, GvRewPred, GlbV(ctx, di, GvVSPatchPos))
}

// PVposEst returns the estimated positive PV value
// based on drives and OFCposUSPT maint and VSMatrix gating
func (pp *PVLV) PVposEst(ctx *Context, di uint32) (pvPosSum, pvPos float32) {
	nd := pp.NPosUSs
	for i := uint32(0); i < nd; i++ {
		maint := GlbUSposV(ctx, di, GvOFCposUSPTMaint, i)  // avg act
		gate := GlbUSposV(ctx, di, GvVSMatrixPoolGated, i) // bool
		est := float32(0)
		if maint > 0.2 || gate > 0 {
			est = 1 // don't have value
		}
		pp.USs.USposEst[i] = est
	}
	pvPosSum, pvPos = pp.PVposEstFmUSs(ctx, di, pp.USs.USposEst)
	if pvPos < pp.GiveUp.MinPVposEst {
		pvPos = pp.GiveUp.MinPVposEst
	}
	return
}

// PVposEstFmUSs returns the estimated positive PV value
// based on drives and given US values.  This can be used
// to compute estimates to compare network performance.
func (pp *PVLV) PVposEstFmUSs(ctx *Context, di uint32, uss []float32) (pvPosSum, pvPos float32) {
	nd := pp.NPosUSs
	if len(uss) < int(nd) {
		nd = uint32(len(uss))
	}
	wts := pp.USs.PVposWts
	for i := uint32(0); i < nd; i++ {
		pvPosSum += wts[i] * uss[i] * pp.Drive.EffectiveDrive(ctx, di, i)
	}
	pvPos = PVLVNormFun(pp.USs.PVposGain * pvPosSum)
	return
}

// PVposEstFmUSsDrives returns the estimated positive PV value
// based on given externally-provided drives and US values.
// This can be used to compute estimates to compare network performance.
func (pp *PVLV) PVposEstFmUSsDrives(uss, drives []float32) (pvPosSum, pvPos float32) {
	nd := pp.NPosUSs
	if len(uss) < int(nd) {
		nd = uint32(len(uss))
	}
	wts := pp.USs.PVposWts
	for i := uint32(0); i < nd; i++ {
		pvPosSum += wts[i] * uss[i] * drives[i]
	}
	pvPos = PVLVNormFun(pp.USs.PVposGain * pvPosSum)
	return
}

// PVnegEstFmUSs returns the estimated negative PV value
// based on given externally-provided US values.
// This can be used to compute estimates to compare network performance.
func (pp *PVLV) PVnegEstFmUSs(uss []float32) (pvNegSum, pvNeg float32) {
	nn := pp.NNegUSs
	wts := pp.USs.PVnegWts
	for i := uint32(0); i < nn; i++ {
		pvNegSum += wts[i] * uss[i]
	}
	pvNeg = PVLVNormFun(pp.USs.PVnegGain * pvNegSum)
	return
}

// DAFmPVs computes the overall PV DA in terms of LHb burst and dip
// activity from given pvPos, pvNeg, and vsPatchPos values.
// Also returns the net "reward" value as the discounted PV value,
// separate from the vsPatchPos prediction error factor.
func (pp *PVLV) DAFmPVs(pvPos, pvNeg, vsPatchPos float32) (burst, dip, da, rew float32) {
	return pp.LHb.DAFmPVs(pvPos, pvNeg, vsPatchPos)
}

// GiveUpFmPV determines whether to give up on current goal
// based on balance between estimated PVpos and accumulated PVneg.
// returns true if give up triggered.
func (pp *PVLV) GiveUpFmPV(ctx *Context, di uint32, pvNeg float32, rnd erand.Rand) bool {
	// now compute give-up
	posEstSum, posEst := pp.PVposEst(ctx, di)
	vsPatchSum := GlbV(ctx, di, GvVSPatchPosSum)
	posDisc := posEst - vsPatchSum
	// note: cannot do ratio here because discounting can get negative
	diff := posDisc - pvNeg
	prob, giveUp := pp.GiveUp.Prob(-diff, rnd)

	SetGlbV(ctx, di, GvPVposEst, posEst)
	SetGlbV(ctx, di, GvPVposEstSum, posEstSum)
	SetGlbV(ctx, di, GvPVposEstDisc, posDisc)
	SetGlbV(ctx, di, GvGiveUpDiff, diff)
	SetGlbV(ctx, di, GvGiveUpProb, prob)
	SetGlbV(ctx, di, GvGiveUp, num.FromBool[float32](giveUp))
	return giveUp
}

// PVDA computes the PV (primary value) based dopamine
// based on current state information, at the start of a trial.
// PV DA is computed by the VS (ventral striatum) and the LHb / RMTg,
// and the resulting values are stored in global variables.
// Called after updating USs, Effort, Drives at start of trial step,
// in Step.
func (pp *PVLV) PVDA(ctx *Context, di uint32, rnd erand.Rand) {
	pp.USs.USnegFromRaw(ctx, di)
	pp.PVsFmUSs(ctx, di)

	hasRew := (GlbV(ctx, di, GvHasRew) > 0)
	pvPos := GlbV(ctx, di, GvPVpos)
	pvNeg := GlbV(ctx, di, GvPVneg)
	vsPatchPos := GlbV(ctx, di, GvVSPatchPos)

	if hasRew {
		pp.ResetGiveUp(ctx, di)
		rew := pp.LHb.DAforUS(ctx, di, pvPos, pvNeg, vsPatchPos) // only when actual pos rew
		SetGlbV(ctx, di, GvRew, rew)
		return
	}

	if GlbV(ctx, di, GvVSMatrixHasGated) > 0 {
		giveUp := pp.GiveUpFmPV(ctx, di, pvNeg, rnd)
		if giveUp {
			SetGlbV(ctx, di, GvHasRew, 1)                            // key for triggering reset
			rew := pp.LHb.DAforUS(ctx, di, pvPos, pvNeg, vsPatchPos) // only when actual pos rew
			SetGlbV(ctx, di, GvRew, rew)
			return
		}
	}

	// no US regular case
	pp.LHb.DAforNoUS(ctx, di, vsPatchPos)
	SetGlbV(ctx, di, GvRew, 0)
}
