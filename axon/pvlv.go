// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/erand"
	"github.com/goki/ki/bools"
)

//gosl: start pvlv

// PVec is a PVLV Primary Value Vector
// used for representing a variable number of
// values associated with USs, Drives, etc
// using a fixed memory allocation that is
// GPU-compatible.  The number of values can be
// increased if in increments of 4 if needed.
type PVec struct {
	V0 float32
	V1 float32
	V2 float32
	V3 float32
	V4 float32
	V5 float32
	V6 float32
	V7 float32
}

func (pv *PVec) SetAll(val float32) {
	pv.V0 = val
	pv.V1 = val
	pv.V2 = val
	pv.V3 = val
	pv.V4 = val
	pv.V5 = val
	pv.V6 = val
	pv.V7 = val
}

func (pv *PVec) Zero() {
	pv.SetAll(0)
}

func (pv *PVec) Set(idx uint32, val float32) {
	switch idx {
	case 0:
		pv.V0 = val
	case 1:
		pv.V1 = val
	case 2:
		pv.V2 = val
	case 3:
		pv.V3 = val
	case 4:
		pv.V4 = val
	case 5:
		pv.V5 = val
	case 6:
		pv.V6 = val
	case 7:
		pv.V7 = val
	}
}

func (pv *PVec) Get(idx uint32) float32 {
	val := float32(0)
	switch idx {
	case 0:
		val = pv.V0
	case 1:
		val = pv.V1
	case 2:
		val = pv.V2
	case 3:
		val = pv.V3
	case 4:
		val = pv.V4
	case 5:
		val = pv.V5
	case 6:
		val = pv.V6
	case 7:
		val = pv.V7
	}
	return val
}

// Drives manages the drive parameters for updating drive state,
// and drive state.
type Drives struct {

	// [max: 8] number of active drives -- first drive is novelty / curiosity drive -- total must be &lt;= 8
	NActive uint32 `max:"8" desc:"number of active drives -- first drive is novelty / curiosity drive -- total must be &lt;= 8"`

	// [min: 1] [max: 8] number of active negative US states recognized -- the first is always reserved for the accumulated effort cost / dissapointment when an expected US is not achieved
	NNegUSs uint32 `min:"1" max:"8" desc:"number of active negative US states recognized -- the first is always reserved for the accumulated effort cost / dissapointment when an expected US is not achieved"`

	// minimum effective drive value -- this is an automatic baseline ensuring that a positive US results in at least some minimal level of reward.  Unlike Base values, this is not reflected in the activity of the drive values -- applies at the time of reward calculation as a minimum baseline.
	DriveMin float32 `desc:"minimum effective drive value -- this is an automatic baseline ensuring that a positive US results in at least some minimal level of reward.  Unlike Base values, this is not reflected in the activity of the drive values -- applies at the time of reward calculation as a minimum baseline."`

	pad int32

	// [view: inline] baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range).
	Base PVec `view:"inline" desc:"baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range)."`

	// [view: inline] time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update.
	Tau PVec `view:"inline" desc:"time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update."`

	// [view: inline] decrement in drive value when Drive-US is consumed -- positive values are subtracted from current Drive value.
	USDec PVec `view:"inline" desc:"decrement in drive value when Drive-US is consumed -- positive values are subtracted from current Drive value."`

	// [view: -] 1/Tau
	Dt PVec `view:"-" desc:"1/Tau"`
}

func (dp *Drives) Defaults() {
	if dp.NActive <= 0 {
		dp.NActive = 1
	}
	if dp.NNegUSs <= 0 {
		dp.NNegUSs = 1
	}
	dp.DriveMin = 0.5
	dp.Update()
	dp.USDec.SetAll(1)
}

func (dp *Drives) Update() {
	for i := uint32(0); i < 8; i++ {
		tau := dp.Tau.Get(i)
		if tau <= 0 {
			dp.Dt.Set(i, 0)
		} else {
			dp.Dt.Set(i, 1.0/tau)
		}
	}
}

// see context.go for most Drives methods

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

	// weight factor for each positive US, multiplied prior to 1/(1+x) normalization of the sum.  Each pos US is also multiplied by its dynamic Drive factor as well
	PosWts PVec `desc:"weight factor for each positive US, multiplied prior to 1/(1+x) normalization of the sum.  Each pos US is also multiplied by its dynamic Drive factor as well"`

	// weight factor for each negative US, multiplied prior to 1/(1+x) normalization of the sum.
	NegWts PVec `desc:"weight factor for each negative US, multiplied prior to 1/(1+x) normalization of the sum."`
}

func (us *USParams) Defaults() {
	us.PosWts.SetAll(1)
	us.NegWts.SetAll(1)
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

// see context.go for most Effort methods

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

	pad float32
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

// see context.go for most Urgency methods

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

	pad, pad1, pad2 float32
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

// see context.go for most LHb methods

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

	pad, pad1 float32
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
	pp.VTA.Defaults()
	pp.LHb.Defaults()
}

func (pp *PVLV) Update() {
	pp.Drive.Update()
	pp.Effort.Update()
	pp.Urgency.Update()
	pp.VTA.Update()
	pp.LHb.Update()
}

//gosl: end pvlv

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

// VSGated updates JustGated and HasGated as function of VS
// (ventral striatum / ventral pallidum) gating at end of the plus phase.
// Also resets effort and LHb.DipSumCur counters -- starting fresh at start
// of a new goal engaged state.
func (pp *PVLV) VSGated(ctx *Context, di uint32, rnd erand.Rand, gated, hasRew bool, poolIdx int) {
	hasGated := GlbV(ctx, di, GvVSMatrixHasGated) > 0
	if !hasRew && gated && !hasGated {
		UrgencyReset(ctx, di)
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
func (pp *PVLV) ShouldGiveUp(ctx *Context, di uint32, rnd erand.Rand, hasRew bool) bool {
	SetGlbV(ctx, di, GvLHbGiveUp, 0)
	if hasRew { // can't give up if got something now
		SetGlbV(ctx, di, GvLHbDipSumCur, 0)
		return false
	}
	prevSum := GlbV(ctx, di, GvLHbDipSumCur)
	giveUp := LHbShouldGiveUp(ctx, di)
	if prevSum < pp.LHb.DipLowThr && GlbV(ctx, di, GvLHbDipSumCur) >= pp.LHb.DipLowThr {
		SetGlbV(ctx, di, GvEffortCurMax, pp.Effort.PlusVar(rnd, GlbV(ctx, di, GvEffortRaw)+pp.Effort.MaxPostDip))
	}
	if EffortGiveUp(ctx, di) {
		SetGlbV(ctx, di, GvLHbGiveUp, 1)
		giveUp = true
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
		EffortAddEffort(ctx, di, effort)
	}
}

// EffortUrgencyUpdt updates the Effort & Urgency based on
// given effort increment, resetting instead if HasRewPrev flag is true.
// Call this at the start of the trial, in ApplyPVLV method.
func (pp *PVLV) EffortUrgencyUpdt(ctx *Context, di uint32, rnd erand.Rand, effort float32) {
	pp.EffortUpdt(ctx, di, rnd, effort)
	PVLVUrgencyUpdt(ctx, di, effort)
}
