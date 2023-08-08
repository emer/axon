// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/erand"
	"github.com/goki/ki/bools"
)

//gosl: start pvlv

// DriveVals represents different internal drives,
// such as hunger, thirst, etc.  The first drive is
// typically reserved for novelty / curiosity.
// labels can be provided by specific environments.
type DriveVals struct {
	D0 float32
	D1 float32
	D2 float32
	D3 float32
	D4 float32
	D5 float32
	D6 float32
	D7 float32
}

func (ds *DriveVals) SetAll(val float32) {
	ds.D0 = val
	ds.D1 = val
	ds.D2 = val
	ds.D3 = val
	ds.D4 = val
	ds.D5 = val
	ds.D6 = val
	ds.D7 = val
}

func (ds *DriveVals) Zero() {
	ds.SetAll(0)
}

func (ds *DriveVals) Set(drv uint32, val float32) {
	switch drv {
	case 0:
		ds.D0 = val
	case 1:
		ds.D1 = val
	case 2:
		ds.D2 = val
	case 3:
		ds.D3 = val
	case 4:
		ds.D4 = val
	case 5:
		ds.D5 = val
	case 6:
		ds.D6 = val
	case 7:
		ds.D7 = val
	}
}

func (ds *DriveVals) Get(drv uint32) float32 {
	val := float32(0)
	switch drv {
	case 0:
		val = ds.D0
	case 1:
		val = ds.D1
	case 2:
		val = ds.D2
	case 3:
		val = ds.D3
	case 4:
		val = ds.D4
	case 5:
		val = ds.D5
	case 6:
		val = ds.D6
	case 7:
		val = ds.D7
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
	Base DriveVals `view:"inline" desc:"baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range)."`

	// [view: inline] time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update.
	Tau DriveVals `view:"inline" desc:"time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update."`

	// [view: inline] decrement in drive value when Drive-US is consumed -- positive values are subtracted from current Drive value.
	USDec DriveVals `view:"inline" desc:"decrement in drive value when Drive-US is consumed -- positive values are subtracted from current Drive value."`

	// [view: -] 1/Tau
	Dt DriveVals `view:"-" desc:"1/Tau"`
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

///////////////////////////////////////////////////////////////////////////////
//  Effort

// Effort has effort and parameters for updating it
type Effort struct {

	// gain factor for computing effort discount factor -- larger = quicker discounting
	Gain float32 `desc:"gain factor for computing effort discount factor -- larger = quicker discounting"`

	// default maximum raw effort level, when MaxNovel and MaxPostDip don't apply.
	Max float32 `desc:"default maximum raw effort level, when MaxNovel and MaxPostDip don't apply."`

	// maximum raw effort level when novelty / curiosity drive is engaged -- typically shorter than default Max
	MaxNovel float32 `desc:"maximum raw effort level when novelty / curiosity drive is engaged -- typically shorter than default Max"`

	// if the LowThr amount of VSPatch expectation is triggered, as accumulated in LHb.DipSum, then CurMax is set to the current Raw effort plus this increment, which is generally low -- once an expectation has been activated, don't wait around forever..
	MaxPostDip float32 `desc:"if the LowThr amount of VSPatch expectation is triggered, as accumulated in LHb.DipSum, then CurMax is set to the current Raw effort plus this increment, which is generally low -- once an expectation has been activated, don't wait around forever.."`

	// variance in additional maximum effort level, applied whenever CurMax is updated
	MaxVar float32 `desc:"variance in additional maximum effort level, applied whenever CurMax is updated"`

	pad, pad1, pad2 float32
}

func (ef *Effort) Defaults() {
	ef.Gain = 0.1
	ef.Max = 100
	ef.MaxNovel = 8
	ef.MaxPostDip = 4
	ef.MaxVar = 2
}

func (ef *Effort) Update() {

}

// DiscFun is the effort discount function: 1 / (1 + ef.Gain * effort)
func (ef *Effort) DiscFun(effort float32) float32 {
	return 1.0 / (1.0 + ef.Gain*effort)
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
// Positive net LHb activity drives dips / pauses in VTA DA activity,
// e.g., when predicted pos > actual or actual neg > predicted.
// Negative net LHb activity drives bursts in VTA DA activity,
// e.g., when actual pos > predicted (redundant with LV / Amygdala)
// or "relief" burst when actual neg < predicted.
type LHb struct {

	// [def: 1] gain multiplier on overall VSPatchPos - PosPV component
	PosGain float32 `def:"1" desc:"gain multiplier on overall VSPatchPos - PosPV component"`

	// [def: 1] gain multiplier on overall PVneg component
	NegGain float32 `def:"1" desc:"gain multiplier on overall PVneg component"`

	// [def: 0.2] threshold on summed LHbDip over trials for triggering a reset of goal engaged state
	GiveUpThr float32 `def:"0.2" desc:"threshold on summed LHbDip over trials for triggering a reset of goal engaged state"`

	// [def: 0.05] low threshold on summed LHbDip, used for triggering switch to a faster effort max timeout -- Effort.MaxPostDip
	DipLowThr float32 `def:"0.05" desc:"low threshold on summed LHbDip, used for triggering switch to a faster effort max timeout -- Effort.MaxPostDip"`
}

func (lh *LHb) Defaults() {
	lh.PosGain = 1
	lh.NegGain = 1
	lh.GiveUpThr = 0.2
	lh.DipLowThr = 0.05
}

func (lh *LHb) Update() {
}

// see context.go for most LHb methods

///////////////////////////////////////////////////////////////////////////////
//  VTA

// VTAVals has values for all the inputs to the VTA.
// Used as gain factors and computed values.
type VTAVals struct {

	// overall dopamine value reflecting all of the different inputs
	DA float32 `desc:"overall dopamine value reflecting all of the different inputs"`

	// total positive valence primary value = sum of USpos * Drive without effort discounting
	USpos float32 `desc:"total positive valence primary value = sum of USpos * Drive without effort discounting"`

	// total positive valence primary value = sum of USpos * Drive * (1-Effort.Disc) -- what actually drives DA bursting from actual USs received
	PVpos float32 `desc:"total positive valence primary value = sum of USpos * Drive * (1-Effort.Disc) -- what actually drives DA bursting from actual USs received"`

	// total negative valence primary value = sum of USneg inputs
	PVneg float32 `desc:"total negative valence primary value = sum of USneg inputs"`

	// positive valence central nucleus of the amygdala (CeM) LV (learned value) activity, reflecting |BLAPosAcqD1 - BLAPosExtD2|_+ positively rectified.  CeM sets Raw directly.  Note that a positive US onset even with no active Drive will be reflected here, enabling learning about unexpected outcomes.
	CeMpos float32 `desc:"positive valence central nucleus of the amygdala (CeM) LV (learned value) activity, reflecting |BLAPosAcqD1 - BLAPosExtD2|_+ positively rectified.  CeM sets Raw directly.  Note that a positive US onset even with no active Drive will be reflected here, enabling learning about unexpected outcomes."`

	// negative valence central nucleus of the amygdala (CeM) LV (learned value) activity, reflecting |BLANegAcqD2 - BLANegExtD1|_+ positively rectified.  CeM sets Raw directly.
	CeMneg float32 `desc:"negative valence central nucleus of the amygdala (CeM) LV (learned value) activity, reflecting |BLANegAcqD2 - BLANegExtD1|_+ positively rectified.  CeM sets Raw directly."`

	// dip from LHb / RMTg -- net inhibitory drive on VTA DA firing = dips
	LHbDip float32 `desc:"dip from LHb / RMTg -- net inhibitory drive on VTA DA firing = dips"`

	// burst from LHb / RMTg -- net excitatory drive on VTA DA firing = bursts
	LHbBurst float32 `desc:"burst from LHb / RMTg -- net excitatory drive on VTA DA firing = bursts"`

	// net shunting input from VSPatch (PosD1 -- PVi in original PVLV)
	VSPatchPos float32 `desc:"net shunting input from VSPatch (PosD1 -- PVi in original PVLV)"`

	pad, pad1, pad2 float32
}

func (vt *VTAVals) Set(usPos, pvPos, pvNeg, lhbDip, lhbBurst, vsPatchPos float32) {
	vt.USpos = usPos
	vt.PVpos = pvPos
	vt.PVneg = pvNeg
	vt.LHbDip = lhbDip
	vt.LHbBurst = lhbBurst
	vt.VSPatchPos = vsPatchPos
}

func (vt *VTAVals) SetAll(val float32) {
	vt.DA = val
	vt.USpos = val
	vt.PVpos = val
	vt.PVneg = val
	vt.CeMpos = val
	vt.CeMneg = val
	vt.LHbDip = val
	vt.LHbBurst = val
	vt.VSPatchPos = val
}

func (vt *VTAVals) Zero() {
	vt.SetAll(0)
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

	// threshold for activity of PVpos or VSPatchPos to determine if a PV event (actual PV or omission thereof) is present
	PVThr float32 `desc:"threshold for activity of PVpos or VSPatchPos to determine if a PV event (actual PV or omission thereof) is present"`

	pad, pad1, pad2 float32

	// [view: inline] gain multipliers on inputs from each input
	Gain VTAVals `view:"inline" desc:"gain multipliers on inputs from each input"`
}

func (vt *VTA) Defaults() {
	vt.PVThr = 0.05
	vt.Gain.SetAll(1)
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

	// [view: inline] effort parameters and state, tracking relative depletion of glucose levels and water levels as a function of time and exertion
	Effort Effort `view:"inline" desc:"effort parameters and state, tracking relative depletion of glucose levels and water levels as a function of time and exertion"`

	// [view: inline] urgency (increasing pressure to do something) and parameters for updating it. Raw urgency is incremented by same units as effort, but is only reset with a positive US.
	Urgency Urgency `view:"inline" desc:"urgency (increasing pressure to do something) and parameters for updating it. Raw urgency is incremented by same units as effort, but is only reset with a positive US."`

	// parameters and values for computing VTA dopamine, as a function of PV primary values (via Pos / Neg US), LV learned values (Amygdala bursting from unexpected CSs, USs), shunting VSPatchPos expectations, and dipping / pausing inputs from LHb
	VTA VTA `desc:"parameters and values for computing VTA dopamine, as a function of PV primary values (via Pos / Neg US), LV learned values (Amygdala bursting from unexpected CSs, USs), shunting VSPatchPos expectations, and dipping / pausing inputs from LHb"`

	// [view: inline] lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing
	LHb LHb `view:"inline" desc:"lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing"`
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
	SetGlbV(ctx, di, GvEffortDisc, 1)
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
