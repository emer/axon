// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/goki/gosl/slbool"
	"github.com/goki/mat32"
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

func (ds *DriveVals) Set(drv int32, val float32) {
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

func (ds *DriveVals) Get(drv int32) float32 {
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

// Add increments drive by given amount, subject to 0-1 range clamping.
// Returns new val.
func (ds *DriveVals) Add(drv int32, delta float32) float32 {
	dv := ds.Get(drv) + delta
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	ds.Set(drv, dv)
	return dv
}

// SoftAdd increments drive by given amount, using soft-bounding to 0-1 extremes.
// if delta is positive, multiply by 1-val, else val.  Returns new val.
func (ds *DriveVals) SoftAdd(drv int32, delta float32) float32 {
	dv := ds.Get(drv)
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
	ds.Set(drv, dv)
	return dv
}

// ExpStep updates drive with an exponential step with given dt value
// toward given baseline value.
func (ds *DriveVals) ExpStep(drv int32, dt, base float32) float32 {
	dv := ds.Get(drv)
	dv += dt * (base - dv)
	if dv > 1 {
		dv = 1
	} else if dv < 0 {
		dv = 0
	}
	ds.Set(drv, dv)
	return dv
}

// Drives manages the drive parameters for updating drive state,
// and drive state.
type Drives struct {
	NActive  int32   `max:"8" desc:"number of active drives -- first drive is novelty / curiosity drive -- total must be &lt;= 8"`
	NNegUSs  int32   `min:"1" max:"8" desc:"number of active negative US states recognized -- the first is always reserved for the accumulated effort cost / dissapointment when an expected US is not achieved"`
	DriveMin float32 `desc:"minimum effective drive value -- this is an automatic baseline ensuring that a positive US results in at least some minimal level of reward.  Unlike Base values, this is not reflected in the activity of the drive values -- applies at the time of reward calculation as a minimum baseline."`

	pad int32

	Base  DriveVals `view:"inline" desc:"baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range)."`
	Tau   DriveVals `view:"inline" desc:"time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update."`
	USDec DriveVals `view:"inline" desc:"decrement in drive value when Drive-US is consumed -- positive values are subtracted from current Drive value."`

	Drives DriveVals `inactive:"+" view:"inline" desc:"current drive state -- updated with optional homeostatic exponential return to baseline values"`

	Dt DriveVals `view:"-" desc:"1/Tau"`
}

func (dp *Drives) Defaults() {
	dp.NNegUSs = 1
	dp.DriveMin = 0.5
	dp.Update()
	dp.USDec.SetAll(1)
}

// ToBaseline sets all drives to their baseline levels
func (dp *Drives) ToBaseline() {
	dp.Drives = dp.Base
}

// ToZero sets all drives to 0
func (dp *Drives) ToZero() {
	dp.Drives = dp.Base
}

func (dp *Drives) Update() {
	for i := int32(0); i < 8; i++ {
		tau := dp.Tau.Get(i)
		if tau <= 0 {
			dp.Dt.Set(i, 0)
		} else {
			dp.Dt.Set(i, 1.0/tau)
		}
	}
}

// ExpStep updates given drives with an exponential step using dt values
// toward baseline values.
func (dp *Drives) ExpStep() {
	for i := int32(0); i < 8; i++ {
		dp.Drives.ExpStep(i, dp.Dt.Get(i), dp.Base.Get(i))
	}
}

///////////////////////////////////////////////////////////////////////////////
//  Effort

// Effort has effort and parameters for updating it
type Effort struct {
	Gain       float32 `desc:"gain factor for computing effort discount factor -- larger = quicker discounting"`
	CurMax     float32 `inactive:"-" desc:"current maximum raw effort level -- above this point, any current goal will be terminated during the LHbDipResetFmSum function, which also looks for accumulated disappointment.  See Max, MaxNovel, MaxPostDip for values depending on how the goal was triggered."`
	Max        float32 `desc:"default maximum raw effort level, when MaxNovel and MaxPostDip don't apply."`
	MaxNovel   float32 `desc:"maximum raw effort level when novelty / curiosity drive is engaged -- typically shorter than default Max"`
	MaxPostDip float32 `desc:"if the LowThr amount of VSPatch expectation is triggered, as accumulated in LHb.DipSum, then CurMax is set to the current Raw effort plus this increment, which is generally low -- once an expectation has been activated, don't wait around forever.."`
	Raw        float32 `desc:"raw effort -- increments linearly upward for each additional effort step"`
	Disc       float32 `inactive:"-" desc:"effort discount factor = 1 / (1 + gain * EffortRaw) -- goes up toward 1 -- the effect of effort is (1 - EffortDisc) multiplier"`
	pad        float32
}

func (ef *Effort) Defaults() {
	ef.Gain = 0.1
	ef.CurMax = 100
	ef.Max = 100
	ef.MaxNovel = 8
	ef.MaxPostDip = 4
}

func (ef *Effort) Update() {

}

// Reset resets the raw effort back to zero -- at start of new gating event
func (ef *Effort) Reset() {
	ef.Raw = 0
	ef.CurMax = ef.Max
	ef.Disc = 1
}

// DiscFun is the effort discount function: 1 / (1 + ef.Gain * effort)
func (ef *Effort) DiscFun(effort float32) float32 {
	return 1.0 / (1.0 + ef.Gain*effort)
}

// DiscFmEffort computes Disc from Raw effort
func (ef *Effort) DiscFmEffort() float32 {
	ef.Disc = ef.DiscFun(ef.Raw)
	return ef.Disc
}

// AddEffort adds an increment of effort and updates the Disc discount factor
func (ef *Effort) AddEffort(inc float32) {
	ef.Raw += inc
	ef.DiscFmEffort()
}

///////////////////////////////////////////////////////////////////////////////
//  Urgency

// Urgency has urgency (increasing pressure to do something) and parameters for updating it.
// Raw urgency is incremented by same units as effort, but is only reset with a positive US.
// Could also make it a function of drives and bodily state factors
// e.g., desperate thirst, hunger.  Drive activations probably have limited range
// and renormalization, so urgency can be another dimension with more impact by directly biasing Go.
type Urgency struct {
	U50   float32 `desc:"value of raw urgency where the urgency activation level is 50%"`
	Power int32   `def:"4" desc:"exponent on the urgency factor -- valid numbers are 1,2,4,6"`
	Raw   float32 `desc:"raw effort for urgency -- increments linearly upward in effort units"`
	Urge  float32 `inactive:"-" desc:"urgency activity level"`
}

func (ur *Urgency) Defaults() {
	ur.U50 = 20
	ur.Power = 4
	ur.Raw = 0
	ur.Urge = 0
}

func (ur *Urgency) Update() {

}

// Reset resets the raw urgency back to zero -- at start of new gating event
func (ur *Urgency) Reset() {
	ur.Raw = 0
	ur.Urge = 0
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

// UrgeFmUrgency computes Urge from Raw
func (ur *Urgency) UrgeFmUrgency() float32 {
	ur.Urge = ur.UrgeFun(ur.Raw)
	return ur.Urge
}

// AddEffort adds an effort increment of urgency and updates the Urge factor
func (ur *Urgency) AddEffort(inc float32) {
	ur.Raw += inc
	ur.UrgeFmUrgency()
}

///////////////////////////////////////////////////////////////////////////////
//  LHb & RMTg

// LHb has values for computing LHb & RMTg which drives dips / pauses in DA firing.
// Positive net LHb activity drives dips / pauses in VTA DA activity,
// e.g., when predicted pos > actual or actual neg > predicted.
// Negative net LHb activity drives bursts in VTA DA activity,
// e.g., when actual pos > predicted (redundant with LV / Amygdala)
// or "relief" burst when actual neg < predicted.
type LHb struct {
	PosGain     float32 `def:"1" desc:"gain multiplier on overall VSPatchPos - PosPV component"`
	NegGain     float32 `def:"1" desc:"gain multiplier on overall PVneg component"`
	DipResetThr float32 `def:"0.2" desc:"threshold on summed LHbDip over trials for triggering a reset of goal engaged state"`
	DipLowThr   float32 `def:"0.05" desc:"low threshold on summed LHbDip, used for triggering switch to a faster effort max timeout -- Effort.MaxPostDip"`

	Dip      float32     `inactive:"+" desc:"computed LHb activity level that drives more dipping / pausing of DA firing, when VSPatch pos prediction > actual PV reward drive"`
	Burst    float32     `inactive:"+" desc:"computed LHb activity level that drives bursts of DA firing, when actual  PV reward drive > VSPatch pos prediction"`
	DipSum   float32     `inactive:"+" desc:"sum of LHbDip over trials, which is reset when there is a PV value, an above-threshold PPTg value, or when it triggers reset"`
	DipReset slbool.Bool `inactive:"+" desc:"true if a reset was triggered from LHbDipSum > Reset Thr"`

	Pos float32 `inactive:"+" desc:"computed PosGain * (VSPatchPos - PVpos)"`
	Neg float32 `inactive:"+" desc:"computed NegGain * PVneg"`

	pad, pad1 float32
}

func (lh *LHb) Defaults() {
	lh.PosGain = 1
	lh.NegGain = 1
	lh.DipResetThr = 0.2
	lh.DipLowThr = 0.05
	lh.Reset()
}

func (lh *LHb) Update() {
}

func (lh *LHb) Reset() {
	lh.Dip = 0
	lh.Burst = 0
	lh.DipSum = 0
	lh.DipReset.SetBool(false)
}

// LHbFmPVVS computes the overall LHbDip and LHbBurst values from PV (primary value)
// and VSPatch inputs.
func (lh *LHb) LHbFmPVVS(pvPos, pvNeg, vsPatchPos float32) {
	lh.Pos = lh.PosGain * (vsPatchPos - pvPos)
	lh.Neg = lh.NegGain * pvNeg
	netLHb := lh.Pos + lh.Neg

	if netLHb > 0 {
		lh.Dip = netLHb
		lh.Burst = 0
	} else {
		lh.Burst = -netLHb
		lh.Dip = 0
	}
}

// DipResetFmSum increments DipSum and checks if should flag a reset
// returns true if did reset.  resetSum resets the accumulating DipSum
// at salient events like US input or CS-driven gating
func (lh *LHb) DipResetFmSum(resetSum bool) bool {
	lh.DipSum += lh.Dip
	if resetSum {
		lh.DipSum = 0
	}
	lh.DipReset.SetBool(false)
	if lh.DipSum > lh.DipResetThr {
		lh.DipReset.SetBool(true)
		lh.DipSum = 0
	}
	return lh.DipReset.IsTrue()
}

///////////////////////////////////////////////////////////////////////////////
//  VTA

// VTAVals has values for all the inputs to the VTA.
// Used as gain factors and computed values.
type VTAVals struct {
	DA         float32 `desc:"overall dopamine value reflecting all of the different inputs"`
	USpos      float32 `desc:"total positive valence primary value = sum of USpos * Drive without effort discounting"`
	PVpos      float32 `desc:"total positive valence primary value = sum of USpos * Drive * (1-Effort.Disc) -- what actually drives DA bursting from actual USs received"`
	PVneg      float32 `desc:"total negative valence primary value = sum of USneg inputs"`
	CeMpos     float32 `desc:"positive valence central nucleus of the amygdala (CeM) LV (learned value) activity, reflecting |BLAPosAcqD1 - BLAPosExtD2|_+ positively rectified.  CeM sets Raw directly.  Note that a positive US onset even with no active Drive will be reflected here, enabling learning about unexpected outcomes."`
	CeMneg     float32 `desc:"negative valence central nucleus of the amygdala (CeM) LV (learned value) activity, reflecting |BLANegAcqD2 - BLANegExtD1|_+ positively rectified.  CeM sets Raw directly."`
	LHbDip     float32 `desc:"dip from LHb / RMTg -- net inhibitory drive on VTA DA firing = dips"`
	LHbBurst   float32 `desc:"burst from LHb / RMTg -- net excitatory drive on VTA DA firing = bursts"`
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
	PVThr float32 `desc:"threshold for activity of PVpos or VSPatchPos to determine if a PV event (actual PV or omission thereof) is present"`

	pad, pad1, pad2 float32

	Gain VTAVals `view:"inline" desc:"gain multipliers on inputs from each input"`
	Raw  VTAVals `view:"inline" inactive:"+" desc:"raw current values -- inputs to the computation"`
	Vals VTAVals `view:"inline" inactive:"+" desc:"computed current values"`
	Prev VTAVals `view:"inline" inactive:"+" desc:"previous computed  values -- to avoid a data race"`
}

func (vt *VTA) Defaults() {
	vt.PVThr = 0.05
	vt.Gain.SetAll(1)
}

func (vt *VTA) Update() {
}

func (vt *VTA) Reset() {
	vt.Raw.Zero()
	vt.Vals.Zero()
	vt.Prev.Zero()
}

// DAFmRaw computes the intermediate Vals and final DA value from
// Raw values that have been set prior to calling.
// ACh value from LDT is passed as a parameter.
func (vt *VTA) DAFmRaw(ach float32, hasRew bool) {
	vt.Vals.PVpos = vt.Gain.PVpos * vt.Raw.PVpos
	vt.Vals.PVneg = vt.Gain.PVneg * vt.Raw.PVneg
	vt.Vals.CeMpos = vt.Gain.CeMpos * vt.Raw.CeMpos
	vt.Vals.CeMneg = vt.Gain.CeMneg * vt.Raw.CeMneg
	vt.Vals.LHbDip = vt.Gain.LHbDip * vt.Raw.LHbDip
	vt.Vals.LHbBurst = vt.Gain.LHbBurst * vt.Raw.LHbBurst
	vt.Vals.VSPatchPos = vt.Gain.VSPatchPos * vt.Raw.VSPatchPos

	if vt.Vals.VSPatchPos < 0 {
		vt.Vals.VSPatchPos = 0
	}
	pvDA := vt.Vals.PVpos - vt.Vals.VSPatchPos - vt.Vals.PVneg
	csNet := vt.Vals.CeMpos - vt.Vals.CeMneg         // todo: is this sensible?  max next with 0 so positive..
	csDA := ach * mat32.Max(csNet, vt.Vals.LHbBurst) // - vt.Vals.LHbDip
	// note that ach is only on cs -- should be 1 for PV events anyway..
	netDA := float32(0)
	if hasRew {
		netDA = pvDA
	} else {
		netDA = csDA
	}
	vt.Vals.DA = vt.Gain.DA * netDA
}

///////////////////////////////////////////////////////////////////////////////
//  VSMatrix

// VSMatrix has parameters and values for computing VSMatrix gating status.
// VS = ventral striatum, aka VP = ventral pallidum = output part of VS
type VSMatrix struct {
	JustGated slbool.Bool `inactive:"+" desc:"VSMatrix just gated (to engage goal maintenance in PFC areas), set at end of plus phase -- this excludes any gating happening at time of US"`
	HasGated  slbool.Bool `inactive:"+" desc:"VSMatrix has gated since the last time HasRew was set (US outcome received or expected one failed to be received)"`

	pad, pad1 float32
}

func (vt *VSMatrix) Defaults() {
}

func (vt *VSMatrix) Update() {
}

func (vt *VSMatrix) Reset() {
	vt.JustGated.SetBool(false)
	vt.HasGated.SetBool(false)
}

// NewState is called at start of new trial
func (vt *VSMatrix) NewState(hasRew bool) {
	if hasRew {
		vt.HasGated.SetBool(false)
	} else if vt.JustGated.IsTrue() {
		vt.HasGated.SetBool(true)
	}
	vt.JustGated.SetBool(false)
}

// VSGated updates JustGated as function of VS gating
// at end of the plus phase.
func (vt *VSMatrix) VSGated(gated bool) {
	vt.JustGated.SetBool(gated)
}

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
	Drive    Drives    `desc:"parameters and state for built-in drives that form the core motivations of agent, controlled by lateral hypothalamus and associated body state monitoring such as glucose levels and thirst."`
	Effort   Effort    `view:"inline" desc:"effort parameters and state, tracking relative depletion of glucose levels and water levels as a function of time and exertion"`
	Urgency  Urgency   `view:"inline" desc:"urgency (increasing pressure to do something) and parameters for updating it. Raw urgency is incremented by same units as effort, but is only reset with a positive US."`
	VTA      VTA       `desc:"parameters and values for computing VTA dopamine, as a function of PV primary values (via Pos / Neg US), LV learned values (Amygdala bursting from unexpected CSs, USs), shunting VSPatchPos expectations, and dipping / pausing inputs from LHb"`
	LHb      LHb       `view:"inline" desc:"lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing"`
	USpos    DriveVals `inactive:"+" view:"inline" desc:"current positive-valence drive-satisfying input(s) (unconditioned stimuli = US)"`
	USneg    DriveVals `inactive:"+" view:"inline" desc:"current negative-valence (aversive), non-drive-satisfying input(s) (unconditioned stimuli = US) -- does not have corresponding drive but uses DriveVals.  Number of active ones is Drive.NNegUSs -- the first is always reserved for the accumulated effort cost / dissapointment when an expected US is not achieved"`
	VSPatch  DriveVals `inactive:"+" view:"inline" desc:"current positive-valence drive-satisfying reward predicting VSPatch (PosD1) values"`
	VSMatrix VSMatrix  `view:"inline" desc:"VSMatrix has parameters and values for computing VSMatrix gating status. VS = ventral striatum, aka VP = ventral pallidum = output part of VS"`

	HasRewPrev   slbool.Bool `inactive:"+" desc:"HasRew state from the previous trial -- copied from HasRew in NewState -- used for updating Effort, Urgency at start of new trial"`
	HasPosUSPrev slbool.Bool `inactive:"+" desc:"HasPosUS state from the previous trial -- copied from HasPosUS in NewState -- used for updating Effort, Urgency at start of new trial"`
	pad, pad1    int32
}

func (pp *PVLV) Defaults() {
	pp.Drive.Defaults()
	pp.Effort.Defaults()
	pp.Urgency.Defaults()
	pp.VTA.Defaults()
	pp.LHb.Defaults()
	pp.USpos.Zero()
	pp.USneg.Zero()
	pp.VSPatch.Zero()
	pp.VSMatrix.Reset()
}

func (pp *PVLV) Update() {
	pp.Drive.Update()
	pp.Effort.Update()
	pp.Urgency.Update()
	pp.VTA.Update()
	pp.LHb.Update()
	pp.VSMatrix.Update()
}

func (pp *PVLV) Reset() {
	pp.Drive.ToZero()
	pp.Effort.Reset()
	pp.Urgency.Reset()
	pp.LHb.Reset()
	pp.VTA.Reset()
	pp.USpos.Zero()
	pp.USneg.Zero()
	pp.VSPatch.Zero()
	pp.VSMatrix.Reset()
	pp.HasRewPrev.SetBool(false)
	pp.HasPosUSPrev.SetBool(false)
}

// NewState is called at start of new trial
func (pp *PVLV) NewState(hasRew bool) {
	pp.HasRewPrev.SetBool(hasRew)
	pp.HasPosUSPrev.SetBool(pp.HasPosUS())
	pp.VSMatrix.NewState(hasRew)
}

// InitUS initializes all the USs to zero
func (pp *PVLV) InitUS() {
	pp.USpos.Zero()
	pp.USneg.Zero()
}

// SetPosUS sets given positive US (associated with same-indexed Drive) to given value
func (pp *PVLV) SetPosUS(usn int32, val float32) {
	pp.USpos.Set(usn, val)
}

// SetNegUS sets given negative US to given value
func (pp *PVLV) SetNegUS(usn int32, val float32) {
	pp.USneg.Set(usn, val)
}

// InitDrives initializes all the Drives to zero
func (pp *PVLV) InitDrives() {
	pp.Drive.Drives.Zero()
}

// SetDrive sets given Drive to given value
func (pp *PVLV) SetDrive(dr int32, val float32) {
	pp.Drive.Drives.Set(dr, val)
}

// USStimVal returns stimulus value for US at given index
// and valence.  If US > 0.01, a full 1 US activation is returned.
func (pp *PVLV) USStimVal(usIdx int32, valence ValenceTypes) float32 {
	us := float32(0)
	if valence == Positive {
		us = pp.USpos.Get(usIdx)
	} else {
		us = pp.USneg.Get(usIdx)
	}
	if us > 0.01 { // threshold for presentation to net
		us = 1 // https://github.com/emer/axon/issues/194
	}
	return us
}

// PosPV returns the reward for current positive US state relative to current drives
func (pp *PVLV) PosPV() float32 {
	rew := float32(0)
	for i := int32(0); i < pp.Drive.NActive; i++ {
		rew += pp.USpos.Get(i) * mat32.Max(pp.Drive.Drives.Get(i), pp.Drive.DriveMin)
	}
	return rew
}

// NegPV returns the reward for current negative US state -- just a sum of USneg
func (pp *PVLV) NegPV() float32 {
	rew := float32(0)
	for i := int32(0); i < pp.Drive.NNegUSs; i++ {
		rew += pp.USneg.Get(i)
	}
	return rew
}

// VSPatchMax returns the max VSPatch value across drives
func (pp *PVLV) VSPatchMax() float32 {
	max := float32(0)
	for i := int32(0); i < pp.Drive.NActive; i++ {
		vs := pp.VSPatch.Get(i)
		if vs > max {
			max = vs
		}
	}
	return max
}

// HasPosUS returns true if there is at least one non-zero positive US
func (pp *PVLV) HasPosUS() bool {
	for i := int32(0); i < pp.Drive.NActive; i++ {
		if pp.USpos.Get(i) > 0 {
			return true
		}
	}
	return false
}

// HasNegUS returns true if there is at least one non-zero negative US
func (pp *PVLV) HasNegUS() bool {
	for i := int32(0); i < pp.Drive.NActive; i++ {
		if pp.USpos.Get(i) > 0 {
			return true
		}
	}
	return false
}

// NetPV returns VTA.Vals.PVpos - VTA.Vals.PVneg
func (pp *PVLV) NetPV() float32 {
	return pp.VTA.Vals.PVpos - pp.VTA.Vals.PVneg
}

// PosPVFmDriveEffort returns the net primary value ("reward") based on
// given US value and drive for that value (typically in 0-1 range),
// and total effort, from which the effort discount factor is computed an applied:
// usValue * drive * Effort.DiscFun(effort)
func (pp *PVLV) PosPVFmDriveEffort(usValue, drive, effort float32) float32 {
	return usValue * drive * pp.Effort.DiscFun(effort)
}

// DA computes the updated dopamine from all the current state,
// including ACh from LDT via Context.
// Call after setting USs, Effort, Drives, VSPatch vals etc.
// Resulting DA is in VTA.Vals.DA, and is returned
// (to be set to Context.NeuroMod.DA)
func (pp *PVLV) DA(ach float32, hasRew bool) float32 {
	usPos := pp.PosPV()
	pvNeg := pp.NegPV()
	if pp.LHb.DipReset.IsTrue() {
		pvNeg += 1.0 - pp.Effort.Disc // pay effort cost here..
	}
	pvPos := usPos * pp.Effort.Disc
	vsPatchPos := pp.VSPatchMax()
	pp.LHb.LHbFmPVVS(pvPos, pvNeg, vsPatchPos)
	pp.VTA.Raw.Set(usPos, pvPos, pvNeg, pp.LHb.Dip, pp.LHb.Burst, vsPatchPos)
	pp.VTA.DAFmRaw(ach, hasRew)
	return pp.VTA.Vals.DA
}

// LHbDipResetFmSum increments DipSum and checks if should flag a reset.
// computed at end of minus phase -- so there is time for reset to have effects.
// most other updates happen in plus phase -- some variables could be stale here.
// The reset bool is used by Context version of this function to set HasRew flag
// with Rew = 0 if true.
func (pp *PVLV) LHbDipResetFmSum(ach float32) bool {
	resetSum := false                     // reset sum at salient events: actual US, CS-gating
	if pp.VTA.Vals.USpos > pp.VTA.PVThr { // if actual PV, reset dip sum -- can't rely on HasRew at this point
		resetSum = true
	}
	// note: VSGated resets DipSum at end of plus phase when VS gating happens
	prevSum := pp.LHb.DipSum
	dipReset := pp.LHb.DipResetFmSum(resetSum)
	if prevSum < pp.LHb.DipLowThr && pp.LHb.DipSum >= pp.LHb.DipLowThr {
		pp.Effort.CurMax = pp.Effort.Raw + pp.Effort.MaxPostDip
	}
	if pp.Effort.CurMax > 0 && pp.Effort.Raw > pp.Effort.CurMax {
		pp.LHb.DipReset.SetBool(true)
		dipReset = true
	}
	return dipReset
}

// DriveUpdt updates the drives based on the current USs,
// subtracting USDec * US from current Drive,
// and calling ExpStep with the Dt and Base params.
func (pp *PVLV) DriveUpdt() {
	pp.Drive.ExpStep()
	for i := int32(0); i < pp.Drive.NActive; i++ {
		us := pp.USpos.Get(i)
		pp.Drive.Drives.Add(i, -us*pp.Drive.USDec.Get(i))
	}
}

// VSGated updates JustGated and HasGated as function of VS
// (ventral striatum / ventral pallidum) gating at end of the plus phase.
// Also resets effort and LHb.DipSum counters -- starting fresh at start
// of a new goal engaged state.
func (pp *PVLV) VSGated(gated, hasRew bool, poolIdx int) {
	if !hasRew && gated {
		pp.Effort.Reset()
		pp.LHb.DipSum = 0
		if poolIdx == 0 { // novelty / curiosity pool
			pp.Effort.CurMax = pp.Effort.MaxNovel
		}
	}
	pp.VSMatrix.VSGated(gated)
}

// EffortUpdt updates the effort based on given effort increment,
// resetting instead if HasRewPrev flag is true.
// Call this at the start of the trial, in ApplyPVLV method.
func (pp *PVLV) EffortUpdt(effort float32) {
	if pp.HasRewPrev.IsTrue() {
		pp.Effort.Reset()
	} else {
		pp.Effort.AddEffort(effort)
	}
}

// UrgencyUpdt updates the urgency and urgency based on given effort increment,
// resetting instead if HasRewPrev and HasPosUSPrev is true indicating receipt
// of an actual positive US.
// Call this at the start of the trial, in ApplyPVLV method.
func (pp *PVLV) UrgencyUpdt(effort float32) {
	if pp.HasRewPrev.IsTrue() && pp.HasPosUSPrev.IsTrue() {
		pp.Urgency.Reset()
	} else {
		pp.Urgency.AddEffort(effort)
	}
}

// EffortUrgencyUpdt updates the Effort & Urgency based on
// given effort increment, resetting instead if HasRewPrev flag is true.
// Call this at the start of the trial, in ApplyPVLV method.
func (pp *PVLV) EffortUrgencyUpdt(effort float32) {
	pp.EffortUpdt(effort)
	pp.UrgencyUpdt(effort)
}

//gosl: end pvlv
