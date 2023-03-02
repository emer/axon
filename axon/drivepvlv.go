// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/goki/mat32"

//gosl: start drivepvlv

// DriveVals represents different internal drives, such as hunger, thirst, etc.
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
	DriveMin float32 `desc:"minimum effective drive value"`

	pad, pad1 int32

	Base  DriveVals `view:"inline" desc:"baseline levels for each drive -- what they naturally trend toward in the absence of any input.  Set inactive drives to 0 baseline, active ones typically elevated baseline (0-1 range)."`
	Tau   DriveVals `view:"inline" desc:"time constants in ThetaCycle (trial) units for natural update toward Base values -- 0 values means no natural update."`
	USDec DriveVals `view:"inline" desc:"decrement in drive value when Drive-US is consumed -- positive values are subtracted from current Drive value."`

	Drives DriveVals `inactive:"+" view:"inline" desc:"current drive state -- updated with optional homeostatic exponential return to baseline values"`

	Dt DriveVals `view:"-" desc:"1/Tau"`
}

func (dp *Drives) Defaults() {
	dp.Update()
	dp.USDec.SetAll(1)
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

	pad float32
}

func (ef *Effort) Defaults() {
	ef.Gain = 0.1
}

func (ef *Effort) Update() {

}

// AddEffort adds an increment of effort
func (ef *Effort) AddEffort(inc float32) {
	ef.Raw += inc
}

// Reset resets the raw effort back to zero -- at start of new gating event
func (ef *Effort) Reset() {
	ef.Raw = 0
}

// DiscFmEffort computes EffortDisc from EffortRaw
func (ef *Effort) DiscFmEffort() float32 {
	ef.Disc = 1.0 / (1.0 + ef.Gain*ef.Raw)
	return ef.Disc
}

///////////////////////////////////////////////////////////////////////////////
//  VSPatch collects VSPatch values

// VSVals are VSPatch / VSMatrix values
type VSVals struct {
	Pos float32 `desc:"VS D1 - D2 positive valence pathway -- patch opposes positive US (PV) outcomes"`
	Neg float32 `desc:"VS D2 - D1 negative valence pathway -- patch opposes negative US (PV) outcomes"`

	pad, pad1 float32

	PosD1 DriveVals `inactive:"+" view:"inline" desc:"VSPatchPosD1 pool activations, recorded by layers"`
	PosD2 DriveVals `inactive:"+" view:"inline" desc:"VSPatchPosD2 pool activations, recorded by layers"`
	NegD2 DriveVals `inactive:"+" view:"inline" desc:"VSPatchNegD2 pool activations, recorded by layers"`
	NegD1 DriveVals `inactive:"+" view:"inline" desc:"VSPatchNegD1 pool activations, recorded by layers"`
}

func (vs *VSVals) Zero() {
	vs.Pos = 0
	vs.Neg = 0
}

// SetVal sets a Pos/Neg D1/D2 value for given value index based on pool index
func (vs *VSVals) SetVal(val float32, poolIdx int32, valence ValenceTypes, dar DAModTypes) {
	if valence == Positive && dar == D1Mod {
		vs.PosD1.Set(poolIdx, val)
	} else if valence == Positive && dar == D2Mod {
		vs.PosD2.Set(poolIdx, val)
	} else if valence == Negative && dar == D2Mod {
		vs.NegD2.Set(poolIdx, val)
	} else if valence == Negative && dar == D1Mod {
		vs.NegD1.Set(poolIdx, val)
	}
}

// PosNegFmVals computes positive and negative summary values from detailed values
func (vs *VSVals) PosNegFmVals() {
	pos := float32(0)
	neg := float32(0)
	for i := int32(0); i < 8; i++ {
		pos += vs.PosD1.Get(i) - vs.PosD2.Get(i)
		neg += vs.NegD2.Get(i) - vs.NegD1.Get(i)
	}
	vs.Pos = pos
	vs.Neg = neg
}

///////////////////////////////////////////////////////////////////////////////
//  LHb & RMTg

// LHb has values for computing LHb & RMTg which drives dips / pauses in DA firing.
// Positive net LHb activity drives dips / pauses in VTA DA activity, e.g., when predicted pos > actual
// or actual neg > predicted.
// Negative net LHb activity drives bursts in VTA DA activity, e.g., when actual pos > predicted
// (redundant with LV / Amygdala via PPTg) or "relief" burst when actual neg < predicted.
type LHb struct {
	PosGain     float32 `def:"1" desc:"gain multiplier on overall VSPatchPos - PVpos component"`
	NegGain     float32 `def:"1" desc:"gain multiplier on overall PVneg - VSPatchNeg component"`
	PosOmitPred float32 `def:"0.2" desc:"additional gain factor for VSPatchPos when D2 > D1, which is a *predicted* omission of positive outcome (not actual) -- this can explain away actual ommission -- higher values here allow more such explaining away"`
	NegOmitPred float32 `def:"0.8" desc:"additional gain factor for VSPatchNeg when D1 > D2, which is a *predicted* omission of negative outcome (not actual) -- this can explain away actual ommission -- higher values here allow more such explaining away"`

	LHbDip   float32 `inactive:"+" desc:"computed LHb activity level that drives more dipping / pausing of DA firing, when VSPatch pos prediction > actual PV reward drive"`
	LHbBurst float32 `inactive:"+" desc:"computed LHb activity level that drives bursts of DA firing, when actual  PV reward drive > VSPatch pos prediction"`

	Pos float32 `inactive:"+" desc:"computed net VSPatchPos - PVpos"`
	Neg float32 `inactive:"+" desc:"computed net PVneg - VSPatchNeg"`
}

func (lh *LHb) Defaults() {
	lh.PosGain = 1
	lh.NegGain = 1
	lh.PosOmitPred = 0.2
	lh.NegOmitPred = 0.8
}

// LHbFmPVVS computes the overall LHbDip and LHbBurst values from PV (primary value)
// and VSPatch inputs.
func (lh *LHb) LHbFmPVVS(pvPos, pvNeg, vsPatchPos, vsPatchNeg float32) {
	if vsPatchPos < 0 {
		vsPatchPos *= lh.PosOmitPred
	}
	if vsPatchNeg < 0 {
		vsPatchNeg *= lh.NegOmitPred
	}

	// note: not doing matrix yet

	lh.Pos = lh.PosGain * (vsPatchPos - pvPos)
	lh.Neg = lh.NegGain * (pvNeg - vsPatchNeg)
	netLHb := lh.Pos + lh.Neg

	if netLHb > 0 {
		lh.LHbDip = netLHb
		lh.LHbBurst = 0
	} else {
		lh.LHbBurst = -netLHb
		lh.LHbDip = 0
	}
}

///////////////////////////////////////////////////////////////////////////////
//  VTA

// VTAVals has values for all the inputs to the VTA.
// Used as gain factors and computed values.
type VTAVals struct {
	DA         float32 `desc:"overall dopamine value reflecting all of the different inputs"`
	PVpos      float32 `desc:"PVposUS with effort discount factor applied (1-Effort.Disc) -- what actually drives DA bursting from actual USs received"`
	PVneg      float32 `desc:"negative valence primary value -- total negative US receipt"`
	PPTg       float32 `desc:"positive valence trial-level deltas, positive only rectified, from the PPTg, driven by Amygdala LV (learned value) system.  Reflects reward salience -- the onset of unexpected CSs and USs.  Note that positive US onset even with no active Drive will be reflected here, enabling learning about unexpected outcomes."`
	LHbDip     float32 `desc:"dip from LHb / RMTg -- net inhibitory drive on VTA DA firing = dips"`
	LHbBurst   float32 `desc:"burst from LHb / RMTg -- net excitatory drive on VTA DA firing = bursts"`
	VSPatchPos float32 `desc:"net shunting input from VSPatchPosD1 - D2.  No shunting from VSPatchNeg is currently supported."`

	pad float32
}

func (vt *VTAVals) Set(pvPos, pvNeg, pptg, lhbDip, lhbBurst, vsPatchPos float32) {
	vt.PVpos = pvPos
	vt.PVneg = pvNeg
	vt.PPTg = pptg
	vt.LHbDip = lhbDip
	vt.LHbBurst = lhbBurst
	vt.VSPatchPos = vsPatchPos
}

func (vt *VTAVals) SetAll(val float32) {
	vt.DA = val
	vt.PVpos = val
	vt.PVneg = val
	vt.PPTg = val
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
//   - LV / Amygdala which drives bursting for unexpected CSs or USs via PPTg.
//   - Shunting expectations of reward from VSPatchPosD1 - D2.
//   - Dipping / pausing inhibitory inputs from lateral habenula (LHb) reflecting
//     predicted positive outcome > actual, or actual negative > predicted.
type VTA struct {
	Gain VTAVals `view:"inline" desc:"gain multipliers on inputs from each input"`
	Raw  VTAVals `view:"inline" inactive:"+" desc:"raw current values -- inputs to the computation"`
	Vals VTAVals `view:"inline" inactive:"+" desc:"computed current values"`
}

func (vt *VTA) Defaults() {
	vt.Gain.SetAll(1)
	// todo: VSPatchPos may need to be 2
}

// DAFmRaw computes the intermediate Vals and final DA value from
// Raw values that have been set prior to calling.
func (vt *VTA) DAFmRaw() {
	vt.Vals.PVpos = vt.Gain.PVpos * vt.Raw.PVpos
	vt.Vals.PVneg = vt.Gain.PVneg * vt.Raw.PVneg
	vt.Vals.PPTg = vt.Gain.PPTg * vt.Raw.PPTg
	vt.Vals.LHbDip = vt.Gain.LHbDip * vt.Raw.LHbDip
	vt.Vals.LHbBurst = vt.Gain.LHbBurst * vt.Raw.LHbBurst
	vt.Vals.VSPatchPos = vt.Gain.VSPatchPos * vt.Raw.VSPatchPos

	if vt.Vals.VSPatchPos < 0 {
		vt.Vals.VSPatchPos = 0
	}

	burstDA := mat32.Max(vt.Vals.PVpos, vt.Vals.PPTg)
	burstDA = mat32.Max(burstDA, vt.Vals.LHbBurst)
	netBurstDA := burstDA - vt.Vals.VSPatchPos // shunting

	dipDA := vt.Vals.LHbDip // *not* shunted by VSPatchNeg -- controversial
	vt.Vals.DA = vt.Gain.DA * (netBurstDA - dipDA)
}

///////////////////////////////////////////////////////////////////////////////
//  DrivePVLV

// DrivePVLV represents the core brainstem-level (hypothalamus) bodily drives
// and resulting dopamine as computed by the PVLV model of primary value
// and learned value (PVLV), describing the functions of the Amygala,
// Ventral Striatum, VTA and associated midbrain nuclei (PPTg, LHb, RMTg)
type DrivePVLV struct {
	Drive       Drives    `desc:"parameters and state for built-in drives that form the core motivations of agent, controlled by lateral hypothalamus and associated body state monitoring such as glucose levels and thirst."`
	Effort      Effort    `view:"inline" desc:"effort parameters and state, tracking relative depletion of glucose levels and water levels as a function of time and exertion"`
	VTA         VTA       `desc:"parameters and values for computing VTA dopamine, as a function of PV primary values (via Pos / Neg US), LV learned values (Amygdala bursting from unexpected CSs, USs), shunting VSPatchPos expectations, and dipping / pausing inputs from LHb"`
	LHb         LHb       `view:"inline" desc:"lateral habenula (LHb) parameters and state, which drives dipping / pausing in dopamine when the predicted positive outcome > actual, or actual negative outcome > predicted.  Can also drive bursting for the converse, and via matrix phasic firing"`
	VSPatchVals VSVals    `desc:"VSPatch values"`
	PosUSs      DriveVals `inactive:"+" view:"inline" desc:"current positive-valence drive-satisfying input(s) (unconditioned stimuli)"`
	NegUSs      DriveVals `inactive:"+" view:"inline" desc:"current negative-valence (aversive), non-drive-satisfying input(s) (unconditioned stimuli) -- does not have corresponding drive but uses DriveVals"`
}

func (dp *DrivePVLV) Defaults() {
	dp.Drive.Defaults()
	dp.Effort.Defaults()
	dp.VTA.Defaults()
	dp.LHb.Defaults()
	dp.VSPatchVals.Zero()
	dp.PosUSs.Zero()
	dp.NegUSs.Zero()
}

// InitUS initializes all the USs to zero
func (dp *DrivePVLV) InitUS() {
	dp.PosUSs.Zero()
	dp.NegUSs.Zero()
}

// SetPosUS sets given positive US (associated with same-indexed Drive) to given value
func (dp *DrivePVLV) SetPosUS(usn int32, val float32) {
	dp.PosUSs.Set(usn, val)
}

// SetNegUS sets given negative US to given value
func (dp *DrivePVLV) SetNegUS(usn int32, val float32) {
	dp.NegUSs.Set(usn, val)
}

// InitDrives initializes all the Drives to zero
func (dp *DrivePVLV) InitDrives() {
	dp.Drive.Drives.Zero()
}

// SetDrive sets given Drive to given value
func (dp *DrivePVLV) SetDrive(dr int32, val float32) {
	dp.Drive.Drives.Set(dr, val)
}

// PosPV returns the reward for current positive US state relative to current drives
func (dp *DrivePVLV) PosPV() float32 {
	rew := float32(0)
	for i := int32(0); i < dp.Drive.NActive; i++ {
		rew += dp.PosUSs.Get(i) * mat32.Max(dp.Drive.Drives.Get(i), dp.Drive.DriveMin)
	}
	return rew
}

// NegPV returns the reward for current negative US state -- just a sum of NegUSs
func (dp *DrivePVLV) NegPV() float32 {
	rew := float32(0)
	for i := int32(0); i < dp.Drive.NActive; i++ {
		rew += dp.PosUSs.Get(i) * mat32.Max(dp.Drive.Drives.Get(i), dp.Drive.DriveMin)
	}
	return rew
}

// DA computes the updated dopamine from all the current state,
// including pptg passed in as an arg.
// Call after setting USs, VSPatchVals, Effort, Drives, etc.
// Resulting DA is in VTA.Vals.DA and is returned
func (dp *DrivePVLV) DA(pptg float32) float32 {
	pvPosRaw := dp.PosPV()
	pvNeg := dp.NegPV()
	pvPos := pvPosRaw * dp.Effort.DiscFmEffort()
	dp.VSPatchVals.PosNegFmVals()
	dp.LHb.LHbFmPVVS(pvPos, pvNeg, dp.VSPatchVals.Pos, dp.VSPatchVals.Neg)
	dp.VTA.Raw.Set(pvPos, pvNeg, pptg, dp.LHb.LHbDip, dp.LHb.LHbBurst, dp.VSPatchVals.Pos)
	dp.VTA.DAFmRaw()
	return dp.VTA.Vals.DA
}

// DriveUpdt updates the drives based on the current USs,
// subtracting USDec * US from current Drive,
// and calling ExpStep with the Dt and Base params.
// and optionally resets the USs back to zero.
func (dp *DrivePVLV) DriveUpdt(resetUs bool) {
	dp.Drive.ExpStep()
	for i := int32(0); i < dp.Drive.NActive; i++ {
		us := dp.PosUSs.Get(i)
		dp.Drive.Drives.Add(i, -us*dp.Drive.USDec.Get(i))
		if resetUs {
			dp.PosUSs.Set(i, 0)
		}
	}
}

//gosl: end drivepvlv
