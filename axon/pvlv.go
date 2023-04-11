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
	NActive  int32   `max:"8" desc:"number of active drives -- must be <= 8"`
	NNegUSs  int32   `min:"1" max:"8" desc:"number of active negative US states recognized -- the first is always reserved for the accumulated effort cost / dissapointment when an expected US is not achieved"`
	DriveMin float32 `desc:"minimum effective drive value"`

	pad int32

	Base  DriveVals `view:"inline" desc:"baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range)."`
	Tau   DriveVals `view:"inline" desc:"time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update."`
	USDec DriveVals `view:"inline" desc:"decrement in drive value when Drive-US is consumed -- positive values are subtracted from current Drive value."`

	Drives DriveVals `inactive:"+" view:"inline" desc:"current drive state -- updated with optional homeostatic exponential return to baseline values"`

	Dt DriveVals `view:"-" desc:"1/Tau"`
}

func (dp *Drives) Defaults() {
	dp.NNegUSs = 1
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
	Gain float32 `desc:"gain factor for computing effort discount factor -- larger = quicker discounting"`
	Raw  float32 `desc:"raw effort -- increments linearly upward for each additional effort step"`
	Disc float32 `inactive:"-" desc:"effort discount factor = 1 / (1 + gain * EffortRaw) -- goes up toward 1 -- the effect of effort is (1 - EffortDisc) multiplier"`
	pad  float32
}

func (ef *Effort) Defaults() {
	ef.Gain = 0.1
}

func (ef *Effort) Update() {

}

// Reset resets the raw effort back to zero -- at start of new gating event
func (ef *Effort) Reset() {
	ef.Raw = 0
	ef.Disc = 1
}

// DiscFun is the effort discount function: 1 / (1 + ef.Gain * effort)
func (ef *Effort) DiscFun(effort float32) float32 {
	return 1.0 / (1.0 + ef.Gain*effort)
}

// DiscFmEffort computes Effort.Disc from EffortRaw
func (ef *Effort) DiscFmEffort() float32 {
	ef.Disc = ef.DiscFun(ef.Raw)
	return ef.Disc
}

// AddEffort adds an increment of effort and updates the Disc discount factor Disc
func (ef *Effort) AddEffort(inc float32) {
	ef.Raw += inc
	ef.DiscFmEffort()
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

	Dip      float32     `inactive:"+" desc:"computed LHb activity level that drives more dipping / pausing of DA firing, when VSPatch pos prediction > actual PV reward drive"`
	Burst    float32     `inactive:"+" desc:"computed LHb activity level that drives bursts of DA firing, when actual  PV reward drive > VSPatch pos prediction"`
	DipSum   float32     `inactive:"+" desc:"sum of LHbDip over trials, which is reset when there is a PV value, an above-threshold PPTg value, or when it triggers reset"`
	DipReset slbool.Bool `inactive:"+" desc:"true if a reset was triggered from LHbDipSum > Reset Thr"`

	Pos float32 `inactive:"+" desc:"computed PosGain * (VSPatchPos - PVpos)"`
	Neg float32 `inactive:"+" desc:"computed NegGain * PVneg"`

	pad, pad1, pad2 float32
}

func (lh *LHb) Defaults() {
	lh.PosGain = 1
	lh.NegGain = 1
	lh.DipResetThr = 0.2
}

func (lh *LHb) Update() {
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

// DAFmRaw computes the intermediate Vals and final DA value from
// Raw values that have been set prior to calling.
// ACh value from LDT is passed as a parameter.
func (vt *VTA) DAFmRaw(ach float32) {
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
	pvDA := vt.Vals.PVpos - vt.Vals.VSPatchPos
	csNet := vt.Vals.CeMpos - vt.Vals.CeMneg         // todo: is this sensible?  max next with 0 so positive..
	csDA := ach * mat32.Max(csNet, vt.Vals.LHbBurst) // - vt.Vals.LHbDip
	// note that ach is only on cs -- should be 1 for PV events anyway..
	netDA := pvDA
	if vt.Vals.PVpos < vt.PVThr && vt.Vals.VSPatchPos < vt.PVThr { // if not actual PV, add cs
		netDA += csDA
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

// VSGated updates JustGated and HasGated as function of VS gating (VP).
// at end of the plus phase.
func (vt *VSMatrix) VSGated(gated, hasRew bool) {
	vt.JustGated.SetBool(gated)
	if hasRew {
		vt.HasGated.SetBool(false)
	} else {
		if gated {
			vt.HasGated.SetBool(true)
		}
	}
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
	VTA      VTA       `desc:"parameters and values for computing VTA dopamine, as a function of PV primary values (via Pos / Neg US), LV learned values (Amygdala bursting from unexpected CSs, USs), shunting VSPatchPos expectations, and dipping / pausing inputs from LHb"`
	LHb      LHb       `view:"inline" desc:"lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing"`
	USpos    DriveVals `inactive:"+" view:"inline" desc:"current positive-valence drive-satisfying input(s) (unconditioned stimuli = US)"`
	USneg    DriveVals `inactive:"+" view:"inline" desc:"current negative-valence (aversive), non-drive-satisfying input(s) (unconditioned stimuli = US) -- does not have corresponding drive but uses DriveVals.  Number of active ones is Drive.NNegUSs -- the first is always reserved for the accumulated effort cost / dissapointment when an expected US is not achieved"`
	VSPatch  DriveVals `inactive:"+" view:"inline" desc:"current positive-valence drive-satisfying reward predicting VSPatch (PosD1) values"`
	VSMatrix VSMatrix  `view:"inline" desc:"VSMatrix has parameters and values for computing VSMatrix gating status. VS = ventral striatum, aka VP = ventral pallidum = output part of VS"`
}

func (pp *PVLV) Defaults() {
	pp.Drive.Defaults()
	pp.Effort.Defaults()
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
	pp.VTA.Update()
	pp.LHb.Update()
	pp.VSMatrix.Update()
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
func (pp *PVLV) DA(ach float32) float32 {
	usPos := pp.PosPV()
	pvNeg := pp.NegPV()
	if pp.LHb.DipReset.IsTrue() {
		pvNeg += 1.0 - pp.Effort.Disc // pay effort cost here..
	}
	pvPos := usPos * pp.Effort.Disc
	vsPatchPos := pp.VSPatchMax()
	pp.LHb.LHbFmPVVS(pvPos, pvNeg, vsPatchPos)
	pp.VTA.Raw.Set(usPos, pvPos, pvNeg, pp.LHb.Dip, pp.LHb.Burst, vsPatchPos)
	pp.VTA.DAFmRaw(ach)
	return pp.VTA.Vals.DA
}

// LHbDipResetFmSum increments DipSum and checks if should flag a reset.
// computed at end of minus phase -- so there is time for reset to have effects.
// most other updates in plus phase -- some variables could be stale here.
func (pp *PVLV) LHbDipResetFmSum(ach float32) bool {
	resetSum := false                     // reset sum at salient events: actual US, CS-gating
	if pp.VTA.Vals.PVpos > pp.VTA.PVThr { // if actual PV, reset dip sum
		resetSum = true
		// } else if pp.VSMatrix.JustGated.IsTrue() { // note: is leftover from prior trial
		// this would prevent extinction on trial just after CS gating..
		// using raw ACh is too sensitive.
		// 	resetSum = true
	}
	dipReset := pp.LHb.DipResetFmSum(resetSum)
	return dipReset
}

// DriveUpdt updates the drives based on the current USs,
// subtracting USDec * US from current Drive,
// and calling ExpStep with the Dt and Base params.
// if resetUs is true, USpos values are reset after update
// so they can be set on occurrence without having to reset.
func (pp *PVLV) DriveUpdt(resetUs bool) {
	pp.Drive.ExpStep()
	for i := int32(0); i < pp.Drive.NActive; i++ {
		us := pp.USpos.Get(i)
		pp.Drive.Drives.Add(i, -us*pp.Drive.USDec.Get(i))
		if resetUs {
			pp.USpos.Set(i, 0)
		}
	}
}

// TODO: need the gating flag to reset at start of gating

// EffortUpdt updates the effort based on given effort increment,
// resetting first if hasRew flag is true indicating receipt of a
// US based on Context.NeuroMod.HasRew flag.
func (pp *PVLV) EffortUpdt(effort float32, hasRew bool) {
	if hasRew {
		pp.Effort.Reset()
	}
	pp.Effort.AddEffort(effort)
}

// DriveEffortUpdt updates the Drives and Effort based on
// given effort increment, resetting first if hasRew flag is true
// indicating receipt of a US based on Context.NeuroMod.HasRew flag.
// if resetUs is true, USpos values are reset after update
// so they can be set on occurrence without having to reset.
func (pp *PVLV) DriveEffortUpdt(effort float32, hasRew, resetUs bool) {
	pp.DriveUpdt(resetUs)
	pp.EffortUpdt(effort, hasRew)
}

//gosl: end pvlv
