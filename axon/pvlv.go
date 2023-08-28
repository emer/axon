// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"

	"github.com/emer/emergent/erand"
	"github.com/goki/ki/bools"
	"github.com/goki/mat32"
)

// Drives manages the drive parameters for updating drive state,
// and drive state.
type Drives struct {

	// minimum effective drive value -- this is an automatic baseline ensuring that a positive US results in at least some minimal level of reward.  Unlike Base values, this is not reflected in the activity of the drive values -- applies at the time of reward calculation as a minimum baseline.
	DriveMin float32 `desc:"minimum effective drive value -- this is an automatic baseline ensuring that a positive US results in at least some minimal level of reward.  Unlike Base values, this is not reflected in the activity of the drive values -- applies at the time of reward calculation as a minimum baseline."`

	// baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range).
	Base []float32 `desc:"baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range)."`

	// time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update.
	Tau []float32 `desc:"time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update."`

	// decrement in drive value when Drive-US is consumed -- positive values are subtracted from current Drive value.
	USDec []float32 `desc:"decrement in drive value when Drive-US is consumed -- positive values are subtracted from current Drive value."`

	// [view: -] 1/Tau
	Dt []float32 `view:"-" desc:"1/Tau"`
}

func (dp *Drives) Alloc(nDrives int) {
	if len(dp.Base) == nDrives {
		return
	}
	dp.Base = make([]float32, nDrives)
	dp.Tau = make([]float32, nDrives)
	dp.Dt = make([]float32, nDrives)
	dp.USDec = make([]float32, nDrives)
}

func (dp *Drives) Defaults() {
	if dp.NActive <= 0 {
		dp.NActive = 1
	}
	dp.DriveMin = 0.5
	dp.Update()
	for i := range dp.USDec {
		dp.USDec[i] = 0
	}
}

func (dp *Drives) Update() {
	for i, tau := range dp.Tau {
		if tau <= 0 {
			dp.Dt[i] = 0
		} else {
			dp.Dt[i] = 1 / tau
		}
	}
}

// VarToZero sets all values of given drive-sized variable to 0
func (dp *Drives) VarToZero(ctx *Context, di uint32, gvar GlobalVars) {
	nd := dp.NActive
	for i := range dp.Base {
		SetGlbDrvV(ctx, di, uint32(i), gvar, 0)
	}
}

// ToZero sets all drives to 0
func (dp *Drives) ToZero(ctx *Context, di uint32) {
	dp.VarToZero(ctx, di, GvDrives)
}

// ToBaseline sets all drives to their baseline levels
func (dp *Drives) ToBaseline(ctx *Context, di uint32) {
	for i := uint32(0); i < nd; i++ {
		SetGlbDrvV(ctx, di, i, GvDrives, dp.Base[i])
	}
}

// AddTo increments drive by given amount, subject to 0-1 range clamping.
// Returns new val.
func (dp *Drives) AddTo(ctx *Context, di uint32, drv uint32, delta float32) float32 {
	dv := GlbDrvV(ctx, di, drv, GvDrives) + delta
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	SetGlbDrvV(ctx, di, drv, GvDrives, dv)
	return dv
}

// SoftAdd increments drive by given amount, using soft-bounding to 0-1 extremes.
// if delta is positive, multiply by 1-val, else val.  Returns new val.
func (dp *Drives) SoftAdd(ctx *Context, di uint32, drv uint32, delta float32) float32 {
	dv := GlbDrvV(ctx, di, drv, GvDrives)
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
	SetGlbDrvV(ctx, di, drv, GvDrives, dv)
	return dv
}

// ExpStep updates drive with an exponential step with given dt value
// toward given baseline value.
func (dp *Drives) ExpStep(ctx *Context, di uint32, drv uint32, dt, base float32) float32 {
	dv := GlbDrvV(ctx, di, drv, GvDrives)
	dv += dt * (base - dv)
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	SetGlbDrvV(ctx, di, drv, GvDrives, dv)
	return dv
}

// ExpStepAll updates given drives with an exponential step using dt values
// toward baseline values.
func (dp *Drives) ExpStepAll(ctx *Context, di uint32) {
	nd := dp.NActive
	for i := uint32(0); i < nd; i++ {
		dp.ExpStep(ctx, di, i, dp.Dt[i], dp.Base[i])
	}
}

// EffectiveDrive returns the Max of Drives at given index and DriveMin.
// note that index 0 is the novelty / curiosity drive.
func (dp *Drives) EffectiveDrive(ctx *Context, di uint32, i uint32) float32 {
	if i == 0 {
		return GlbDrvV(ctx, di, uint32(0), GvDrives)
	}
	return mat32.Max(GlbDrvV(ctx, di, i, GvDrives), dp.DriveMin)
}

/////////////////////////////////////////////////////////
//  Effort

// Effort has parameters for giving up based on effort,
// which is the first negative US.
type Effort struct {

	// default maximum raw effort level, for deciding when to give up on goal pursuit, when MaxNovel and MaxPostDip don't apply.
	Max float32 `desc:"default maximum raw effort level, for deciding when to give up on goal pursuit, when MaxNovel and MaxPostDip don't apply."`

	// maximum raw effort level when novelty / curiosity drive is engaged -- typically shorter than default Max
	MaxNovel float32 `desc:"maximum raw effort level when novelty / curiosity drive is engaged -- typically shorter than default Max"`

	// if the LowThr amount of VSPatch expectation is triggered, as accumulated in LHb.DipSum, then CurMax is set to the current Raw effort plus this increment, which is generally low -- once an expectation has been activated, don't wait around forever..
	MaxPostDip float32 `desc:"if the LowThr amount of VSPatch expectation is triggered, as accumulated in LHb.DipSum, then CurMax is set to the current Raw effort plus this increment, which is generally low -- once an expectation has been activated, don't wait around forever.."`

	// variance in additional maximum effort level, applied whenever CurMax is updated
	MaxVar float32 `desc:"variance in additional maximum effort level, applied whenever CurMax is updated"`
}

func (ef *Effort) Defaults() {
	ef.Max = 100
	ef.MaxNovel = 8
	ef.MaxPostDip = 4
	ef.MaxVar = 2
}

func (ef *Effort) Update() {

}

// Reset resets the raw effort back to zero -- at start of new gating event
func (ef *Effort) Reset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvEffortRaw, 0)
	SetGlbV(ctx, di, GvEffortCurMax, ef.Max)
	SetGlbUSneg(ctx, di, GvUSnegRaw, 0, 0) // effort is neg 0
	SetGlbUSneg(ctx, di, GvUSneg, 0, 0)
}

// EffortNorm returns the current effort value as a normalized number,
// for external calculation purposes (this method not used for PVLV computations).
// This normalization is performed internally on _all_ negative USs
func EffortNorm(ctx *Context, di uint32, factor float32) float32 {
	return PVLVNormFun(factor * GlbV(ctx, di, GvEffortRaw))
}

// AddEffort adds an increment of effort and updates the Disc discount factor
func (ef *Effort) AddEffort(ctx *Context, di uint32, inc float32) {
	AddGlbV(ctx, di, GvEffortRaw, inc)
	eff := GlbV(ctx, di, GvEffortRaw)
	SetGlbUSneg(ctx, di, GvUSnegRaw, 0, eff) // effort is neg 0
	fmt.Printf("add eff: %g\n", eff)
}

// GiveUp returns true if maximum effort has been exceeded
func (ef *Effort) GiveUp(ctx *Context, di uint32) bool {
	raw := GlbV(ctx, di, GvEffortRaw)
	curMax := GlbV(ctx, di, GvEffortCurMax)
	if curMax > 0 && raw > curMax {
		return true
	}
	return false
}

// PlusVar returns value plus random variance
func (ef *Effort) PlusVar(rnd erand.Rand, max float32) float32 {
	if ef.MaxVar == 0 {
		return max
	}
	return max + ef.MaxVar*float32(rnd.NormFloat64(-1))
}

// ReStart restarts restarts the raw effort back to zero
// and sets the Max with random additional variance.
func (ef *Effort) ReStart(ctx *Context, di uint32, rnd erand.Rand) {
	SetGlbV(ctx, di, GvEffortRaw, 0)
	SetGlbV(ctx, di, GvEffortCurMax, ef.PlusVar(rnd, ef.Max))
	SetGlbUSneg(ctx, di, GvUSnegRaw, 0, 0) // effort is neg 0
	SetGlbUSneg(ctx, di, GvUSneg, 0, 0)    // effort is neg 0
}

///////////////////////////////////////////////////////////////////////////////
//  Urgency

// Urgency has urgency (increasing pressure to do something) and parameters for updating it.
// Raw urgency is incremented by same units as effort, but is only reset with a positive US.
// Could also make it a function of drives and bodily state factors
// e.g., desperate thirst, hunger.  Drive activations probably have limited range
// and renormalization, so urgency can be another dimension with more impact by directly biasing Go.
type Urgency struct {

	// value of raw urgency where the urgency activation level is 50%
	U50 float32 `desc:"value of raw urgency where the urgency activation level is 50%"`

	// [def: 4] exponent on the urge factor -- valid numbers are 1,2,4,6
	Power int32 `def:"4" desc:"exponent on the urge factor -- valid numbers are 1,2,4,6"`

	// [def: 0.2] threshold for urge -- cuts off small baseline values
	Thr float32 `def:"0.2" desc:"threshold for urge -- cuts off small baseline values"`
}

func (ur *Urgency) Defaults() {
	ur.U50 = 20
	ur.Power = 4
	ur.Thr = 0.2
}

func (ur *Urgency) Update() {

}

// UrgeFun is the urgency function: urgency / (urgency + 1) where
// urgency = (Raw / U50)^Power
func (ur *Urgency) UrgeFun(urgency float32) float32 {
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
func (ur *Urgency) Reset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvUrgencyRaw, 0)
	SetGlbV(ctx, di, GvUrgency, 0)
}

// UrgeFmUrgency computes Urge from Raw
func (ur *Urgency) UrgeFmUrgency(ctx *Context, di uint32) float32 {
	urge := ur.UrgeFun(GlbV(ctx, di, GvUrgencyRaw))
	if urge < ur.Thr {
		urge = 0
	}
	SetGlbV(ctx, di, GvUrgency, urge)
	return urge
}

// AddEffort adds an effort increment of urgency and updates the Urge factor
func (ur *Urgency) AddEffort(ctx *Context, di uint32, inc float32) {
	AddGlbV(ctx, di, GvUrgencyRaw, inc)
	ur.UrgeFmUrgency(ctx, di)
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
	us.PVPosGain = 1
	us.PVNegGain = 0.05
	for i := range us.PVPosWts {
		us.PVPosWts[i] = 1
	}
	for i := range us.NegGains {
		us.NegGains[i] = 0.05
		us.PVNegWts[i] = 1
	}
}

// USnegFromRaw sets normalized NegUS values from Raw values
func (us *USParams) USnegFromRaw(ctx *Context, di uint32) {
	nn := us.NNegUSs
	for i := uint32(0); i < nn; i++ {
		raw := GlbUSneg(ctx, di, GvUSnegRaw, i)
		norm := PVLVNormFun(us.NegGains[i] * raw)
		SetGlbUSneg(ctx, di, GvUSneg, i, norm)
		// fmt.Printf("neg %d  raw: %g  norm: %g\n", i, raw, norm)
	}
}

// USnegToZero sets all values of USneg, USNegRaw to zero
func (us *USParams) USnegToZero(ctx *Context, di uint32) {
	nn := us.NNegUSs
	for i := uint32(0); i < nn; i++ {
		SetGlbUSneg(ctx, di, GvUSneg, i, 0)
		SetGlbUSneg(ctx, di, GvUSnegRaw, i, 0)
	}
}

// USposToZero sets all values of USpos, USPosRaw to zero
func (us *USParams) USposToZero(ctx *Context, di uint32) {
	nn := us.NPosUSs
	for i := uint32(0); i < nn; i++ {
		SetGlbDrvV(ctx, di, i, GvUSpos, 0)
	}
}

///////////////////////////////////////////////////////////////////
//  LHb & RMTg

// LHb has values for computing LHb & RMTg which drives dips / pauses in DA firing.
// LHb handles all US-related (PV = primary value) processing.
// Positive net LHb activity drives dips / pauses in VTA DA activity,
// e.g., when predicted pos > actual or actual neg > predicted.
// Negative net LHb activity drives bursts in VTA DA activity,
// e.g., when actual pos > predicted (redundant with LV / Amygdala)
// or "relief" burst when actual neg < predicted.
type LHb struct {

	// [def: 1] threshold factor that multiplies integrated pvNeg value to establish a threshold for whether the integrated pvPos value is good enough to drive overall net positive reward
	NegThr float32 `def:"1" desc:"threshold factor that multiplies integrated pvNeg value to establish a threshold for whether the integrated pvPos value is good enough to drive overall net positive reward"`

	// [def: 4] gain multiplier on PosPV for purposes of generating bursts (not for  discounting negative dips) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)
	PosGain float32 `def:"4" desc:"gain multiplier on PosPV for purposes of generating bursts (not for  discounting negative dips) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)"`

	// [def: 4] gain multiplier on NegPV for purposes of generating dips (not for  discounting positive bursts) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)
	NegGain float32 `def:"4" desc:"gain multiplier on NegPV for purposes of generating dips (not for  discounting positive bursts) -- 4 renormalizes for typical ~.5 values (.5 * .5 = .25)"`

	// [def: 0.2] threshold on summed LHbDip over trials for triggering a reset of goal engaged state
	GiveUpThr float32 `def:"0.2" desc:"threshold on summed LHbDip over trials for triggering a reset of goal engaged state"`

	// [def: 0.05] low threshold on summed LHbDip, used for triggering switch to a faster effort max timeout -- Effort.MaxPostDip
	DipLowThr float32 `def:"0.05" desc:"low threshold on summed LHbDip, used for triggering switch to a faster effort max timeout -- Effort.MaxPostDip"`
}

func (lh *LHb) Defaults() {
	lh.NegThr = 1
	lh.PosGain = 4
	lh.NegGain = 4
	lh.GiveUpThr = 0.2
	lh.DipLowThr = 0.05
}

func (lh *LHb) Update() {
}

/////////////////////////////////////////////////////////
// 	LHb

// Reset resets all LHb vars back to 0
func (lh *LHb) Reset(ctx *Context, di uint32) {
	SetGlbV(ctx, di, GvLHbDip, 0)
	SetGlbV(ctx, di, GvLHbBurst, 0)
	SetGlbV(ctx, di, GvLHbPVDA, 0)
	SetGlbV(ctx, di, GvLHbDipSumCur, 0)
	SetGlbV(ctx, di, GvLHbDipSum, 0)
	SetGlbV(ctx, di, GvLHbGiveUp, 0)
}

// LHbFmPVVS computes the overall LHbDip and LHbBurst, and LHbPVDA as their diff,
// from PV (primary value) and VSPatch inputs.
func (lh *LHb) LHbFmPVVS(ctx *Context, di uint32, pvPos, pvNeg, vsPatchPos float32) {
	thr := lh.NegThr * pvNeg

	pos := lh.PosGain * pvPos
	neg := lh.NegGain * pvNeg
	burst := float32(0)
	dip := float32(0)
	if pvPos > thr { // worth it, got reward
		burst = pos*(1-pvNeg) - vsPatchPos
	} else {
		dip = neg * (1 - pvPos) // todo: vsPatchNeg needed
	}
	SetGlbV(ctx, di, GvLHbDip, dip)
	SetGlbV(ctx, di, GvLHbBurst, burst)
	SetGlbV(ctx, di, GvLHbPVDA, burst-dip)
}

// ShouldGiveUp increments DipSum and checks if should give up if above threshold
func (lh *LHb) ShouldGiveUp(ctx *Context, di uint32) bool {
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

// VTA has parameters and values for computing VTA DA dopamine,
// as a function of:
//   - PV (primary value) driving inputs reflecting US inputs,
//     which are modulated by Drives and discounted by Effort for positive.
//   - LV / Amygdala which drives bursting for unexpected CSs or USs via CeM.
//   - Shunting expectations of reward from VSPatchPosD1 - D2.
//   - Dipping / pausing inhibitory inputs from lateral habenula (LHb) reflecting
//     predicted positive outcome > actual, or actual negative > predicted.
//   - ACh from LDT (laterodorsal tegmentum) reflecting sensory / reward salience,
//     which disinhibits VTA activity.
type VTA struct {

	// gain on CeM activity difference (CeMPos - CeMNeg) for generating LV CS-driven dopamine values
	CeMGain float32 `desc:"gain on CeM activity difference (CeMPos - CeMNeg) for generating LV CS-driven dopamine values"`

	// gain on computed LHb DA (Burst - Dip) -- for controlling DA levels
	LHbGain float32 `desc:"gain on computed LHb DA (Burst - Dip) -- for controlling DA levels"`
}

func (vt *VTA) Defaults() {
	vt.CeMGain = 2
	vt.LHbGain = 1
}

func (vt *VTA) Update() {
}

// see context.go for most VTA methods

///////////////////////////////////////////////////////////////////////////////
//  PVLV

// PVLV represents the core brainstem-level (hypothalamus) bodily drives
// and resulting dopamine from US (unconditioned stimulus) inputs,
// as computed by the PVLV model of primary value (PV)
// and learned value (LV), describing the functions of the Amygala,
// Ventral Striatum, VTA and associated midbrain nuclei (LDT, LHb, RMTg)
// Core LHb (lateral habenula) and VTA (ventral tegmental area) dopamine
// are computed in equations using inputs from specialized network layers
// (LDTLayer driven by BLA, CeM layers, VSPatchLayer).
// Renders USLayer, PVLayer, DrivesLayer representations based on state updated here.
type PVLV struct {

	// number of possible positive US states and corresponding drives -- the first is always reserved for novelty / curiosity.  Must be set programmatically via SetNUSs method, which allocates corresponding parameters.
	NPosUSs uint32 `inactive:"+" desc:"number of possible positive US states and corresponding drives -- the first is always reserved for novelty / curiosity.  Must be set programmatically via SetNUSs method, which allocates corresponding parameters."`

	// number of possible negative US states -- the first is always reserved for the accumulated effort cost (which drives dissapointment when an expected US is not achieved).  Must be set programmatically via SetNUSs method, which allocates corresponding parameters.
	NNegUSs uint32 `inactive:"+" desc:"number of possible negative US states -- the first is always reserved for the accumulated effort cost (which drives dissapointment when an expected US is not achieved).  Must be set programmatically via SetNUSs method, which allocates corresponding parameters."`

	// parameters and state for built-in drives that form the core motivations of agent, controlled by lateral hypothalamus and associated body state monitoring such as glucose levels and thirst.
	Drive Drives `desc:"parameters and state for built-in drives that form the core motivations of agent, controlled by lateral hypothalamus and associated body state monitoring such as glucose levels and thirst."`

	// control how positive and negative USs are weighted and integrated to compute an overall PV primary value.
	USs USParams `desc:"control how positive and negative USs are weighted and integrated to compute an overall PV primary value."`

	// [view: inline] effort parameters and state, tracking relative depletion of glucose levels and water levels as a function of time and exertion
	Effort Effort `view:"inline" desc:"effort parameters and state, tracking relative depletion of glucose levels and water levels as a function of time and exertion"`

	// [view: inline] urgency (increasing pressure to do something) and parameters for updating it. Raw urgency is incremented by same units as effort, but is only reset with a positive US.
	Urgency Urgency `view:"inline" desc:"urgency (increasing pressure to do something) and parameters for updating it. Raw urgency is incremented by same units as effort, but is only reset with a positive US."`

	// [view: inline] lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing
	LHb LHb `view:"inline" desc:"lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing"`

	// parameters and values for computing VTA dopamine, as a function of PV primary values (via Pos / Neg US), LV learned values (Amygdala bursting from unexpected CSs, USs), shunting VSPatchPos expectations, and dipping / pausing inputs from LHb
	VTA VTA `desc:"parameters and values for computing VTA dopamine, as a function of PV primary values (via Pos / Neg US), LV learned values (Amygdala bursting from unexpected CSs, USs), shunting VSPatchPos expectations, and dipping / pausing inputs from LHb"`
}

func (pp *PVLV) Defaults() {
	pp.Drive.Defaults()
	pp.Effort.Defaults()
	pp.Urgency.Defaults()
	pp.USs.Defaults()
	pp.LHb.Defaults()
	pp.VTA.Defaults()
}

func (pp *PVLV) Update() {
	pp.Drive.Update()
	pp.USs.NPosUSs = pp.NPosUSs
	pp.Effort.Update()
	pp.Urgency.Update()
	pp.USs.Update()
	pp.LHb.Update()
	pp.VTA.Update()
}

func (pp *PVLV) SetNUSs(ctx *Context, nPos, nNeg) {
	ctx.NetIdxs.PVLVNPosUSs = nPos
	ctx.NetIdxs.PVLVNNegUSs = nNeg
	pp.Drive.Alloc(nPos)
	
}



// PosPV returns the weighted positive reward
// for current positive US state, where each US is multiplied by
// its current drive and weighting factor and summed
func (pp *PVLV) PosPV(ctx *Context, di uint32) float32 {
	rew := float32(0)
	nd := pp.NPosUSs
	wts := pp.USs.PVPosWts
	for i := uint32(0); i < nd; i++ {
		rew += wts[i] * GlbDrvV(ctx, di, i, GvUSpos) * pp.Drive.EffectiveDrive(ctx, di, i)
	}
	return rew
}

// NegPV returns the weighted negative value
// associated with current negative US state, where each US
// is multiplied by a weighting factor and summed
func (pp *PVLV) NegPV(ctx *Context, di uint32) float32 {
	rew := float32(0)
	nn := pp.USs.NNegUSs
	wts := pp.USs.PVNegWts
	for i := uint32(0); i < nn; i++ {
		rew += wts[i] * GlbUSneg(ctx, di, GvUSnegRaw, i)
	}
	return rew
}

// VSPatchMax returns the max VSPatch value across drives
func (pp *PVLV) VSPatchMax(ctx *Context, di uint32) float32 {
	max := float32(0)
	nd := pp.NPosUSs
	for i := uint32(0); i < nd; i++ {
		vs := GlbDrvV(ctx, di, i, GvVSPatch)
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
		if GlbDrvV(ctx, di, i, GvUSpos) > 0 {
			return true
		}
	}
	return false
}

// HasNegUS returns true if there is at least one non-zero negative US
func (pp *PVLV) HasNegUS(ctx *Context, di uint32) bool {
	nd := pp.USs.NNegUSs
	for i := uint32(0); i < nd; i++ {
		if GlbUSneg(ctx, di, GvUSnegRaw, i) > 0 {
			return true
		}
	}
	return false
}

// NetPV returns PVpos - PVneg as an overall signed net external reward
func (pp *PVLV) NetPV(ctx *Context, di uint32) float32 {
	return GlbV(ctx, di, GvLHbPVpos) - GlbV(ctx, di, GvLHbPVneg)
}

// VSGated updates JustGated and HasGated as function of VS
// (ventral striatum / ventral pallidum) gating at end of the plus phase.
// Also resets effort and LHb.DipSumCur counters -- starting fresh at start
// of a new goal engaged state.
func (pp *PVLV) VSGated(ctx *Context, di uint32, rnd erand.Rand, gated, hasRew bool, poolIdx int) {
	hasGated := GlbV(ctx, di, GvVSMatrixHasGated) > 0
	if !hasRew && gated && !hasGated {
		pp.Urgency.Reset(ctx, di)
		pp.Effort.ReStart(ctx, di, rnd)
		SetGlbV(ctx, di, GvLHbDipSumCur, 0)
		if poolIdx == 0 { // novelty / curiosity pool
			SetGlbV(ctx, di, GvEffortCurMax, pp.Effort.MaxNovel)
		}
	}
	SetGlbV(ctx, di, GvVSMatrixJustGated, bools.ToFloat32(gated))
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
		SetGlbV(ctx, di, GvEffortCurMax, pp.Effort.PlusVar(rnd, GlbV(ctx, di, GvEffortRaw)+pp.Effort.MaxPostDip))
	}
	if pp.Effort.GiveUp(ctx, di) {
		SetGlbV(ctx, di, GvLHbGiveUp, 1)
		giveUp = true
	}
	if giveUp {
		NeuroModSetRew(ctx, di, 0, true) // sets HasRew -- drives maint reset, ACh
	}
	return giveUp
}

// EffortUpdt updates the effort based on given effort increment,
// resetting instead if HasRewPrev flag is true.
// Call this at the start of the trial, in ApplyPVLV method.
func (pp *PVLV) EffortUpdt(ctx *Context, di uint32, rnd erand.Rand, effort float32) {
	if GlbV(ctx, di, GvHasRewPrev) > 0 {
		pp.Effort.ReStart(ctx, di, rnd)
	} else {
		pp.Effort.AddEffort(ctx, di, effort)
	}
}

// EffortUrgencyUpdt updates the Effort & Urgency based on
// given effort increment, resetting instead if HasRewPrev flag is true.
// Call this at the start of the trial, in ApplyPVLV method.
func (pp *PVLV) EffortUrgencyUpdt(ctx *Context, di uint32, rnd erand.Rand, effort float32) {
	pp.EffortUpdt(ctx, di, rnd, effort)
	pp.UrgencyUpdt(ctx, di, effort)
}

// InitUS initializes all the USs to zero
func (pp *PVLV) InitUS(ctx *Context, di uint32) {
	pp.Drive.VarToZero(ctx, di, GvUSpos)
	pp.USs.USnegToZero(ctx, di)
	SetGlbV(ctx, di, GvHasRew, 0)
	SetGlbV(ctx, di, GvRew, 0)
}

// InitDrives initializes all the Drives to zero
func (pp *PVLV) InitDrives(ctx *Context, di uint32) {
	pp.Drive.ToZero(ctx, di)
}

// Reset resets all PVLV state
func (pp *PVLV) Reset(ctx *Context, di uint32) {
	pp.Drive.ToZero(ctx, di)
	pp.Effort.Reset(ctx, di)
	pp.Urgency.Reset(ctx, di)
	pp.LHb.Reset(ctx, di)
	VTAReset(ctx, di)
	pp.InitUS(ctx, di)
	pp.Drive.VarToZero(ctx, di, GvVSPatch)
	SetGlbV(ctx, di, GvVSMatrixJustGated, 0)
	SetGlbV(ctx, di, GvVSMatrixHasGated, 0)
	SetGlbV(ctx, di, GvHasRewPrev, 0)
	// pp.HasPosUSPrev.SetBool(false) // key to not reset!!
}

// PosPVFmDriveEffort returns the net primary value ("reward") based on
// given US value and drive for that value (typically in 0-1 range),
// and total effort, from which the effort discount factor is computed an applied:
// usValue * drive * Effort.DiscFun(effort).
// This is not called directly in the PVLV code -- can be used to compute
// what the PVLV code itself will compute -- see LHbPVDA
func (pp *PVLV) PosPVFmDriveEffort(ctx *Context, usValue, drive, effort float32) float32 {
	return usValue * drive * (1 - PVLVNormFun(pp.USs.PVNegWts[0]*effort))
}

// PVLVSetDrive sets given Drive to given value
func (pp *PVLV) SetDrive(ctx *Context, di uint32, dr uint32, val float32) {
	SetGlbDrvV(ctx, di, dr, GvDrives, val)
}

// DriveUpdt updates the drives based on the current USs,
// subtracting USDec * US from current Drive,
// and calling ExpStep with the Dt and Base params.
func (pp *PVLV) DriveUpdt(ctx *Context, di uint32) {
	pp.Drive.ExpStepAll(ctx, di)
	nd := pp.NPosUSs
	for i := uint32(0); i < nd; i++ {
		us := GlbDrvV(ctx, di, i, GvUSpos)
		nwdrv := GlbDrvV(ctx, di, i, GvDrives) - us*pp.Drive.USDec[i]
		if nwdrv < 0 {
			nwdrv = 0
		}
		SetGlbDrvV(ctx, di, i, GvDrives, nwdrv)
	}
}

// UrgencyUpdt updates the urgency and urgency based on given effort increment,
// resetting instead if HasRewPrev and HasPosUSPrev is true indicating receipt
// of an actual positive US.
// Call this at the start of the trial, in ApplyPVLV method.
func (pp *PVLV) UrgencyUpdt(ctx *Context, di uint32, effort float32) {
	if (GlbV(ctx, di, GvHasRewPrev) > 0) && (GlbV(ctx, di, GvHasPosUSPrev) > 0) {
		pp.Urgency.Reset(ctx, di)
	} else {
		pp.Urgency.AddEffort(ctx, di, effort)
	}
}

// SetUS sets the given unconditioned stimulus (US) state for PVLV algorithm.
// Call PVLVInitUS before calling this, and only call this when a US has been received,
// at the start of a Trial typically.
// This then drives activity of relevant PVLV-rendered inputs, and dopamine.
// The US index is automatically adjusted for the curiosity drive / US for
// positive US outcomes -- i.e., pass in a value with 0 starting index.
// By default, negative USs do not set the overall ctx.NeuroMod.HasRew flag,
// which is the trigger for a full-blown US learning event. Set this yourself
// if the negative US is more of a discrete outcome vs. something that happens
// in the course of goal engaged approach.
func (pp *PVLV) SetUS(ctx *Context, di uint32, valence ValenceTypes, usIdx int, magnitude float32) {
	if valence == Positive {
		SetGlbV(ctx, di, GvHasRew, 1)                            // only for positive USs
		SetGlbDrvV(ctx, di, uint32(usIdx)+1, GvUSpos, magnitude) // +1 for curiosity
	} else {
		SetGlbUSneg(ctx, di, GvUSnegRaw, uint32(usIdx)+1, magnitude) // +1 for effort
	}
}

// SetDrives sets current PVLV drives to given magnitude,
// and sets the first curiosity drive to given level.
// Drive indexes are 0 based, so 1 is added automatically to accommodate
// the first curiosity drive.
func (pp *PVLV) SetDrives(ctx *Context, di uint32, curiosity, magnitude float32, drives ...int) {
	pp.InitDrives(ctx, di)
	pp.SetDrive(ctx, di, 0, curiosity)
	for _, i := range drives {
		pp.SetDrive(ctx, di, uint32(1+i), magnitude)
	}
}

// PVDA computes the PV (primary value) based dopamine
// based on current state information, at the start of a trial.
// PV DA is computed by the VS (ventral striatum) and the LHb / RMTg,
// and the resulting values are stored in LHb global variables.
// Called after updating USs, Effort, Drives at start of trial step,
// in PVLVStepStart.  Returns the resulting LHbPVDA value.
func (pp *PVLV) PVDA(ctx *Context, di uint32) float32 {
	hasRew := (GlbV(ctx, di, GvHasRew) > 0)
	pvPos := pp.PosPV(ctx, di)
	pvNeg := pp.NegPV(ctx, di)
	SetGlbV(ctx, di, GvLHbUSpos, pvPos)
	SetGlbV(ctx, di, GvLHbUSneg, pvNeg)

	pvPosNorm := PVLVNormFun(pp.USs.PVPosGain * pvPos)
	pvNegNorm := PVLVNormFun(pp.USs.PVNegGain * pvNeg)
	SetGlbV(ctx, di, GvLHbPVpos, pvPosNorm)
	SetGlbV(ctx, di, GvLHbPVneg, pvNegNorm)

	vsPatchPos := pp.VSPatchMax(ctx, di)
	SetGlbV(ctx, di, GvLHbVSPatchPos, vsPatchPos)
	SetGlbV(ctx, di, GvRewPred, GlbV(ctx, di, GvLHbVSPatchPos))

	if hasRew { // note: also true for giveup
		pp.LHb.LHbFmPVVS(ctx, di, pvPosNorm, pvNegNorm, vsPatchPos) // only when actual pos rew
		SetGlbV(ctx, di, GvRew, pp.NetPV(ctx, di))                  // primary value diff
	} else {
		SetGlbV(ctx, di, GvLHbDip, 0)
		SetGlbV(ctx, di, GvLHbBurst, 0)
		SetGlbV(ctx, di, GvLHbPVDA, 0)
		SetGlbV(ctx, di, GvRew, 0)
	}
	return GlbV(ctx, di, GvLHbPVDA)
}

// StepStart must be called at start of a new iteration (trial)
// of behavior when using the PVLV framework, after applying USs,
// Drives, and updating Effort (e.g., as last step in ApplyPVLV method).
// Calls PVLVGiveUp and LHbPVDA, which computes the primary value DA.
func (pp *PVLV) StepStart(ctx *Context, di uint32, rnd erand.Rand) {
	pp.USs.USnegFromRaw(ctx, di)
	pp.ShouldGiveUp(ctx, di, rnd)
	pp.PVDA(ctx, di)
}

//	PVLVNewState(ctx, di, bools.FromFloat32(GlbV(ctx, di, GvHasRew)))

// NewState is called at start of new state (trial) of processing.
// hadRew indicates if there was a reward state the previous trial.
// It calls LHGiveUpFmSum to trigger a "give up" state on this trial
// if previous expectation of reward exceeds critical sum.
func (pp *PVLV) NewState(ctx *Context, di uint32, hadRew bool) {
	SetGlbV(ctx, di, GvHasRewPrev, bools.ToFloat32(hadRew))
	SetGlbV(ctx, di, GvHasPosUSPrev, bools.ToFloat32(pp.HasPosUS(ctx, di)))

	if hadRew {
		SetGlbV(ctx, di, GvVSMatrixHasGated, 0)
	} else if GlbV(ctx, di, GvVSMatrixJustGated) > 0 {
		SetGlbV(ctx, di, GvVSMatrixHasGated, 1)
	}
	SetGlbV(ctx, di, GvVSMatrixJustGated, 0)
}
