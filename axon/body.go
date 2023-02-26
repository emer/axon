// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "github.com/goki/mat32"

//gosl: start body

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

func (ds *DriveVals) Set(drv int, val float32) {
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

func (ds *DriveVals) Get(drv int) float32 {
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
func (ds *DriveVals) Add(drv int, delta float32) float32 {
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
func (ds *DriveVals) SoftAdd(drv int, delta float32) float32 {
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
func (ds *DriveVals) ExpStep(drv int, dt, base float32) float32 {
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

// ExpStepN updates given number of drives with an exponential step
// with given dt values toward given baseline values.
func (ds *DriveVals) ExpStepN(n int, dt, base *DriveVals) {
	for i := 0; i < 8; i++ {
		ds.ExpStep(i, dt.Get(i), base.Get(i))
	}
}

// Drives manages the drive parameters for updating drive state,
// and drive state.
type Drives struct {
	NActive int32 `max:"8" desc:"number of active drives -- must be <= 8"`

	pad, pad1, pad2 int32

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
	for i := 0; i < 8; i++ {
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
	dp.Drives.ExpStepN(dp.NActive, &dp.Dt, &dp.Base)
}

///////////////////////////////////////////////////////////////////////////////
//  Effort

// Effort has effort and parameters for updating it
type Effort struct {
	Gain float32 `desc:"gain factor for computing effort discount factor"`
	Raw  float32 `desc:"raw effort, increments linearly updward"`
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
func (ef *Effort) DiscFmEffort() {
	ef.Disc = 1.0 / (1.0 + ef.Gain*ef.Raw)
}

///////////////////////////////////////////////////////////////////////////////
//  LHb & RMTg

// LHbVals are LHb & RMTg values
type LHbVals struct {
	LHb           float32 `desc:"final overall gain on everything"`
	VSPatchPosD1  float32 `desc:"patch D1 APPETITIVE pathway - versus pos PV outcomes"`
	VSPatchPosD2  float32 `desc:"patch D2 APPETITIVE pathway versus vspatch_pos_D1"`
	VSMatrixPosD1 float32 `desc:"gain on VS matrix D1 APPETITIVE guys"`
	VSMatrixPosD2 float32 `desc:"VS matrix D2 APPETITIVE"`
	VSPatchNegD1  float32 `desc:"VS patch D1 pathway versus neg PV outcomes"`
	VSPatchNegD2  float32 `desc:"VS patch D2 pathway versus vspatch_neg_D1"`
	VSMatrixNegD1 float32 `desc:"VS matrix D1 AVERSIVE"`
	VSMatrixNegD2 float32 `desc:"VS matrix D2 AVERSIVE"`

	pos, pos1 float32
}

func (lh *LHbVals) SetAll(val float32) {
	lh.LHb = val
	lh.VSPatchPosD1 = val
	lh.VSPatchPosD2 = val
	lh.VSMatrixPosD1 = val
	lh.VSMatrixPosD2 = val
	lh.VSPatchNegD1 = val
	lh.VSPatchNegD2 = val
	lh.VSMatrixNegD1 = val
	lh.VSMatrixNegD2 = val
}

func (lh *LHbVals) Zero() {
	lh.SetAll(0)
}

// LHbDA has values for computing LHb & RMTg dipping
type LHbDA struct {
	Gain LHbVals `view:"inline" desc:"gain multipliers on inputs from each input"`
	Raw  LHbVals `view:"inline" inactive:"+" desc:"raw current values"`
	Vals LHbVals `view:"inline" inactive:"+" desc:"computed current values"`

	VSPatchPosDisinhib float32 `desc:"proportion of positive reward prediction error (RPE) to use if RPE results from a predicted omission of positive"`
	PVNegDiscount      float32 `desc:"reduction in effective PVNeg net value (when positive) so that negative outcomes can never be completely predicted away -- still allows for positive da for less-bad outcomes"`

	pad, pad1 float32
}

func (lh *LHbDA) Defaults() {
	lh.Gain.SetAll(1)
	lh.VSPatchPosDisinhib = 0.2
	lh.PVNegDiscount = 0.8
}

func (lh *LHbDA) ValsFmRaw() {
	lh.Vals.VSPatchPosD1 = lh.Gain.VSPatchPosD1 * lh.Raw.VSPatchPosD1
	lh.Vals.VSPatchPosD2 = lh.Gain.VSPatchPosD2 * lh.Raw.VSPatchPosD2
	lh.Vals.VSPatchPosDis = lh.Gain.VSPatchPosDis * lh.Raw.VSPatchPosDis
	lh.Vals.VSMatrixPosD1 = lh.Gain.VSMatrixPosD1 * lh.Raw.VSMatrixPosD1
	lh.Vals.VSMatrixPosD2 = lh.Gain.VSMatrixPosD2 * lh.Raw.VSMatrixPosD2
	lh.Vals.VSPatchNegD1 = lh.Gain.VSPatchNegD1 * lh.Raw.VSPatchNegD1
	lh.Vals.VSPatchNegD2 = lh.Gain.VSPatchNegD2 * lh.Raw.VSPatchNegD2
	lh.Vals.VSMatrixNegD1 = lh.Gain.VSMatrixNegD1 * lh.Raw.VSMatrixNegD1
	lh.Vals.VSMatrixNegD2 = lh.Gain.VSMatrixNegD2 * lh.Raw.VSMatrixNegD2

	vsPatchPosNet := lh.Vals.VSPatchPosD1 - lh.Vals.VSPatchPosD2
	if vsPatchPosNet < 0 {
		vsPatchPosNet *= lh.VSPatchPosDisinhib
	}

	vsPatchNegNet := lh.Vals.VSPatchNegD2 - lh.Vals.VSPatchNegD1
	if vsPatchNegNet < 0 {
		vsPatchNegNet *= lh.PVNegDiscount
	}

	vsMatrixPosNet := lh.Vals.VSMatrixPosD1 - lh.Vals.VSMatrixPosD2
	vsMatrixNegNet := lh.Vals.VSMatrixNegD2 - lh.Vals.VSMatrixNegD1

	netPos := vsMatrixPosNet
	if pvPos != 0 {
		netPos = mat32.Max(pvPos, vsMatrixPosNet) // todo: need pvPos
	}
	netNeg := vsMatrixNegNet
	if pvNeg != 0 {
		if vsMatrixPosNet < 0 {
			netNeg = mat32.Max(netNeg, mat32.Abs(vsMatrixPosNet))
			netPos = 0
		}
		netNeg = mat32.Max(pvNeg, netNeg)
	}

	netLHb := netNeg - netPos + vsPatchPosNet - vsPatchNegNet
	netLHb *= lh.Gain.LHb
	lh.Vals.LHb = netLHb
}

///////////////////////////////////////////////////////////////////////////////
//  VTA

// VTAVals has values for all the inputs to the VTA
type VTAVals struct {
	DA           float32 `desc:"overall dopamine"`
	PV           float32 `desc:"primary value -- total reward from US receipt * drives"`
	PPTg         float32 `desc:"bursts from PPTg"`
	LHb          float32 `desc:"dips/bursts from LHbRMTg"`
	VSPatchPosD1 float32 `desc:"VSPatchPosD1 projection that shunts DA bursting"`
	VSPatchPosD2 float32 `desc:"VSPatchPosD2 projection that opposes shunting of bursting in VTA"`
	VSPatchNegD2 float32 `desc:"VSPatchNegD2 projection that shunts dipping of VTA"`
	VSPatchNegD1 float32 `desc:"VSPatchNegD1 projection that opposes the shunting of dipping in VTA"`
}

func (vt *VTAVals) SetAll(val float32) {
	vt.DA = val
	vt.PV = val
	vt.PPTg = val
	vt.LHb = val
	vt.VSPatchPosD1 = val
	vt.VSPatchPosD2 = val
	vt.VSPatchNegD2 = val
	vt.VSPatchNegD1 = val
}

func (vt *VTAVals) Zero() {
	vt.SetAll(0)
}

// VTADA has values for computing VTA DA dopamine
type VTADA struct {
	Gain VTAVals `view:"inline" desc:"gain multipliers on inputs from each input"`
	Raw  VTAVals `view:"inline" inactive:"+" desc:"raw current values"`
	Vals VTAVals `view:"inline" inactive:"+" desc:"computed current values"`
}

func (vt *VTADA) Defaults() {
	vt.Gain.SetAll(1)
}

func (vt *VTADA) ValsFmRaw() {
	vt.Vals.PV = vt.Gain.PV * vt.Raw.PV
	vt.Vals.PPTg = vt.Gain.PPTg * vt.Raw.PPTg
	vt.Vals.LHb = vt.Gain.LHb * vt.Raw.LHg
	vt.Vals.VSPatchPosD1 = vt.Gain.VSPatchPosD1 * vt.Raw.VSPatchPosD1
	vt.Vals.VSPatchPosD2 = vt.Gain.VSPatchPosD2 * vt.Raw.VSPatchPosD2
	vt.Vals.VSPatchNegD2 = vt.Gain.VSPatchNegD2 * vt.Raw.VSPatchNegD2
	vt.Vals.VSPatchNegD1 = vt.Gain.VSPatchNegD1 * vt.Raw.VSPatchNegD1

	vsPosPVI = vt.Vals.VSPatchPosD1 - vt.Vals.VSPatchPosD2
	if vsPosPVI < 0 {
		vsPosPVI = 0
	}

	vsNegPVI = vt.Vals.VSPatchNegD2 - vt.Vals.VSPatchNegD1
	if vsNegPVI < 0 {
		vsNegPVI = 0
	}

	burstDA := mat32.Max(vt.Vals.PV, vt.Vals.PPTg)
	netBurstDA := burstDA - vsPosPVI
}

///////////////////////////////////////////////////////////////////////////////
//  CoreBodyState

// CoreBodyState represents the current core brainstem-level (hypothalamus)
// body state.
type CoreBodyState struct {
	Drive  Drives    `desc:"parameters and state for drive updates"`
	Effort Effort    `view:"inline" desc:"effort parameters and state"`
	USs    DriveVals `view:"inline" desc:"current drive-satisfying input(s)"`
}

// InitUS initializes all the USs to zero
func (cb *CoreBodyState) InitUS() {
	cb.USs.Zero()
}

// SetUS sets given US to given value
func (cb *CoreBodyState) SetUS(usn int, val float32) {
	cb.USs.Set(usn, val)
}

// USRew returns the reward for given current US state relative to current drives
func (cb *CoreBodyState) USRew() float32 {
	rew := float32(0)
	for i := 0; i < cb.Params.NActive; i++ {
		rew += cb.USs.Get(i) * cb.Drives.Get(i)
	}
	return rew
}

// DriveUpdt updates the drives based on the current USs,
// subtracting USDec * US from current Drive,
// and calling ExpStep with the Dt and Base params.
// and optionally resets the USs back to zero.
func (cb *CoreBodyState) DriveUpdt(resetUs bool) {
	cb.Params.ExpStep(&cb.Drives)
	for i := 0; i < cb.Params.NActive; i++ {
		us := cb.USs.Get(i)
		cb.Drives.Add(i, -us*cb.Params.USDec.Get(i))
		if resetUs {
			cb.USs.Set(i, 0)
		}
	}
}
