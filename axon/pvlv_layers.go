// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"
	"strings"

	"cogentcore.org/core/mat32"
	"github.com/emer/gosl/v2/slbool"
)

//gosl: start pvlv_layers

// LDTParams compute reward salience as ACh global neuromodulatory signal
// as a function of the MAX activation of its inputs from salience detecting
// layers (e.g., the superior colliculus: SC), and whenever there is an external
// US outcome input (signalled by the global GvHasRew flag).
// ACh from salience inputs is discounted by GoalMaint activity,
// reducing distraction when pursuing a goal, but US ACh activity is not so reduced.
// ACh modulates excitability of goal-gating layers.
type LDTParams struct {

	// threshold per input source, on absolute value (magnitude), to count as a significant reward event, which then drives maximal ACh -- set to 0 to disable this nonlinear behavior
	SrcThr float32 `default:"0.05"`

	// use the global Context.NeuroMod.HasRew flag -- if there is some kind of external reward being given, then ACh goes to 1, else 0 for this component
	Rew slbool.Bool `default:"true"`

	// extent to which active goal maintenance (via Global GoalMaint)
	// inhibits ACh signals: when goal engaged, distractability is lower.
	MaintInhib float32 `default:"0.8" max:"1" min:"0"`

	// idx of Layer to get max activity from -- set during Build from BuildConfig SrcLay1Name if present -- -1 if not used
	SrcLay1Idx int32 `edit:"-"`

	// idx of Layer to get max activity from -- set during Build from BuildConfig SrcLay2Name if present -- -1 if not used
	SrcLay2Idx int32 `edit:"-"`

	// idx of Layer to get max activity from -- set during Build from BuildConfig SrcLay3Name if present -- -1 if not used
	SrcLay3Idx int32 `edit:"-"`

	// idx of Layer to get max activity from -- set during Build from BuildConfig SrcLay4Name if present -- -1 if not used
	SrcLay4Idx int32 `edit:"-"`

	pad float32
}

func (lp *LDTParams) Defaults() {
	lp.SrcThr = 0.05
	lp.Rew.SetBool(true)
	lp.MaintInhib = 0.8
}

func (lp *LDTParams) Update() {
}

// Thr applies SrcThr threshold to given value
func (lp *LDTParams) Thr(val float32) float32 {
	val = mat32.Abs(val) // only abs makes sense -- typically positive anyway
	if lp.SrcThr <= 0 {
		return val
	}
	if val < lp.SrcThr {
		return 0
	}
	return 1
}

// MaxSrcAct returns the updated maxSrcAct value from given
// source layer activity value.
func (lp *LDTParams) MaxSrcAct(maxSrcAct, srcLayAct float32) float32 {
	act := lp.Thr(srcLayAct)
	if act > maxSrcAct {
		maxSrcAct = act
	}
	return maxSrcAct
}

// ACh returns the computed ACh salience value based on given
// source layer activations and key values from the ctx Context.
func (lp *LDTParams) ACh(ctx *Context, di uint32, srcLay1Act, srcLay2Act, srcLay3Act, srcLay4Act float32) float32 {
	maxSrcAct := float32(0)
	maxSrcAct = lp.MaxSrcAct(maxSrcAct, srcLay1Act)
	maxSrcAct = lp.MaxSrcAct(maxSrcAct, srcLay2Act)
	maxSrcAct = lp.MaxSrcAct(maxSrcAct, srcLay3Act)
	maxSrcAct = lp.MaxSrcAct(maxSrcAct, srcLay4Act)

	maintInh := lp.MaintInhib * GlbV(ctx, di, GvGoalMaint)
	maintInh = min(1, maintInh)
	maxSrcAct *= (1.0 - maintInh)

	ach := maxSrcAct

	if GlbV(ctx, di, GvHasRew) > 0 {
		ach = 1
	} else {
		ach = mat32.Max(ach, GlbV(ctx, di, GvUrgency))
	}
	return ach
}

// VTAParams are for computing overall VTA DA based on LHb PVDA
// (primary value -- at US time, computed at start of each trial
// and stored in LHbPVDA global value)
// and Amygdala (CeM) CS / learned value (LV) activations, which update
// every cycle.
type VTAParams struct {

	// gain on CeM activity difference (CeMPos - CeMNeg) for generating LV CS-driven dopamine values
	CeMGain float32 `default:"0.75"`

	// gain on computed LHb DA (Burst - Dip) -- for controlling DA levels
	LHbGain float32 `default:"1.25"`

	// threshold on ACh level required to generate LV CS-driven dopamine burst
	AChThr float32 `default:"0.5"`

	pad float32
}

func (vt *VTAParams) Defaults() {
	vt.CeMGain = 0.75
	vt.LHbGain = 1.25
	vt.AChThr = 0.5
}

func (vt *VTAParams) Update() {
}

// VTADA computes the final DA value from LHb values
// ACh value from LDT is passed as a parameter.
func (vt *VTAParams) VTADA(ctx *Context, di uint32, ach float32, hasRew bool) {
	pvDA := vt.LHbGain * GlbV(ctx, di, GvLHbPVDA)
	csNet := GlbV(ctx, di, GvCeMpos) - GlbV(ctx, di, GvCeMneg)
	achMod := float32(0)
	if ach >= vt.AChThr {
		achMod = ach
	}
	csDA := achMod*vt.CeMGain*csNet - GlbV(ctx, di, GvVSPatchPos)

	// note that ach is only on cs -- should be 1 for PV events anyway..
	netDA := float32(0)
	if hasRew {
		netDA = pvDA
	} else {
		netDA = csDA
	}
	SetGlbV(ctx, di, GvVtaDA, netDA) // note: keeping this separately just for semantics
	SetGlbV(ctx, di, GvDA, netDA)    // general neuromod DA
}

//gosl: end pvlv_layers

func (ly *Layer) BLADefaults() {
	isAcq := strings.Contains(ly.Nm, "Acq") || strings.Contains(ly.Nm, "Novel")

	lp := ly.Params
	lp.Acts.Decay.Act = 0.2
	lp.Acts.Decay.Glong = 0.6
	lp.Acts.Dend.SSGi = 0
	lp.Inhib.Layer.On.SetBool(true)
	if isAcq {
		lp.Inhib.Layer.Gi = 2.4 // acq has more input
	} else {
		lp.Inhib.Layer.Gi = 1.8
		lp.Acts.Gbar.L = 0.25 // needed to not be active at start
	}
	lp.Inhib.Pool.On.SetBool(true)
	lp.Inhib.Pool.Gi = 1
	lp.Inhib.ActAvg.Nominal = 0.025
	lp.Learn.RLRate.SigmoidMin = 1.0
	lp.Learn.TrgAvgAct.RescaleOn.SetBool(false)
	lp.Learn.RLRate.Diff.SetBool(true)
	lp.Learn.RLRate.DiffThr = 0.01
	lp.CT.DecayTau = 0
	lp.CT.GeGain = 0.1 // 0.1 has effect, can go a bit lower if need to

	if isAcq {
		lp.Learn.NeuroMod.DALRateMod = 0.5
		lp.Learn.NeuroMod.BurstGain = 0.2
		lp.Learn.NeuroMod.DipGain = 0
	} else {
		lp.Learn.NeuroMod.BurstGain = 1
		lp.Learn.NeuroMod.DipGain = 1
	}
	lp.Learn.NeuroMod.AChLRateMod = 1
	lp.Learn.NeuroMod.AChDisInhib = 0 // needs to be always active

	for _, pj := range ly.RcvPrjns {
		slay := pj.Send
		if slay.LayerType() == BLALayer && !strings.Contains(slay.Nm, "Novel") { // inhibition from Ext
			pj.Params.SetFixedWts()
			pj.Params.PrjnScale.Abs = 2
		}
	}
}

// PVLVPostBuild is used for BLA, VSPatch, and PVLayer types to set NeuroMod params
func (ly *Layer) PVLVPostBuild() {
	dm, err := ly.BuildConfigByName("DAMod")
	if err == nil {
		err = ly.Params.Learn.NeuroMod.DAMod.SetString(dm)
		if err != nil {
			log.Println(err)
		}
	}
	vl, err := ly.BuildConfigByName("Valence")
	if err == nil {
		err = ly.Params.Learn.NeuroMod.Valence.SetString(vl)
		if err != nil {
			log.Println(err)
		}
	}
}

func (ly *Layer) CeMDefaults() {
	lp := ly.Params
	lp.Acts.Decay.Act = 1
	lp.Acts.Decay.Glong = 1
	lp.Acts.Dend.SSGi = 0
	lp.Inhib.Layer.On.SetBool(true)
	lp.Inhib.Layer.Gi = 0.5
	lp.Inhib.Pool.On.SetBool(true)
	lp.Inhib.Pool.Gi = 0.3
	lp.Inhib.ActAvg.Nominal = 0.15
	lp.Learn.TrgAvgAct.RescaleOn.SetBool(false)
	lp.Learn.RLRate.SigmoidMin = 1.0 // doesn't matter -- doesn't learn..

	for _, pj := range ly.RcvPrjns {
		pj.Params.SetFixedWts()
		pj.Params.PrjnScale.Abs = 1
	}
}

func (ly *Layer) LDTDefaults() {
	lp := ly.Params
	lp.Inhib.ActAvg.Nominal = 0.1
	lp.Inhib.Layer.On.SetBool(true)
	lp.Inhib.Layer.Gi = 1 // todo: explore
	lp.Inhib.Pool.On.SetBool(false)
	lp.Acts.Decay.Act = 1
	lp.Acts.Decay.Glong = 1
	lp.Acts.Decay.LearnCa = 1 // uses CaSpkD as a readout!
	lp.Learn.TrgAvgAct.RescaleOn.SetBool(false)
	// lp.PVLV.Thr = 0.2
	// lp.PVLV.Gain = 2

	for _, pj := range ly.RcvPrjns {
		pj.Params.SetFixedWts()
		pj.Params.PrjnScale.Abs = 1
	}
}

func (ly *LayerParams) VSPatchDefaults() {
	ly.Acts.Decay.Act = 1
	ly.Acts.Decay.Glong = 1
	ly.Acts.Decay.LearnCa = 1 // uses CaSpkD as a readout!
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 0.5
	ly.Inhib.Layer.FB = 0
	ly.Inhib.Pool.FB = 0
	ly.Inhib.Pool.Gi = 0.5
	ly.Inhib.ActAvg.Nominal = 0.2
	ly.Learn.RLRate.Diff.SetBool(false)
	ly.Learn.RLRate.SigmoidMin = 0.01 // 0.01 > 0.05
	ly.Learn.TrgAvgAct.RescaleOn.SetBool(false)
	ly.Learn.TrgAvgAct.GiBaseInit = 0.5

	// ms.Learn.NeuroMod.DAMod needs to be set via BuildConfig
	ly.Learn.NeuroMod.DALRateSign.SetBool(true)
	ly.Learn.NeuroMod.AChLRateMod = 0.8 // ACh now active for extinction, so this is ok
	ly.Learn.NeuroMod.AChDisInhib = 0   // essential: has to fire when expected but not present!
	ly.Learn.NeuroMod.BurstGain = 1
	ly.Learn.NeuroMod.DipGain = 1 // now must be balanced -- otherwise overshoots
}

func (ly *LayerParams) DrivesDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.01
	ly.Inhib.Layer.On.SetBool(false)
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 0.5
	ly.Acts.PopCode.On.SetBool(true)
	ly.Acts.PopCode.MinAct = 0.2 // low activity for low drive -- also has special 0 case = nothing
	ly.Acts.PopCode.MinSigma = 0.08
	ly.Acts.PopCode.MaxSigma = 0.12
	ly.Acts.Decay.Act = 1
	ly.Acts.Decay.Glong = 1
	ly.Learn.TrgAvgAct.RescaleOn.SetBool(false)
}

func (ly *LayerParams) UrgencyDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.2
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 0.5
	ly.Inhib.Pool.On.SetBool(false)
	ly.Acts.PopCode.On.SetBool(true) // use only popcode
	ly.Acts.PopCode.MinAct = 0
	ly.Acts.Decay.Act = 1
	ly.Acts.Decay.Glong = 1
	ly.Learn.TrgAvgAct.RescaleOn.SetBool(false)
}

func (ly *LayerParams) USDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.05
	ly.Inhib.Layer.On.SetBool(false)
	ly.Inhib.Pool.On.SetBool(true)
	ly.Inhib.Pool.Gi = 0.5
	ly.Acts.PopCode.On.SetBool(true)
	ly.Acts.PopCode.MinAct = 0.2 // low activity for low val -- also has special 0 case = nothing
	ly.Acts.PopCode.MinSigma = 0.08
	ly.Acts.PopCode.MaxSigma = 0.12
	ly.Acts.Decay.Act = 1
	ly.Acts.Decay.Glong = 1
	ly.Learn.TrgAvgAct.RescaleOn.SetBool(false)
}

func (ly *LayerParams) PVDefaults() {
	ly.Inhib.ActAvg.Nominal = 0.2
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 0.5
	ly.Inhib.Pool.On.SetBool(false)
	ly.Acts.PopCode.On.SetBool(true)
	// note: may want to modulate rate code as well:
	// ly.Acts.PopCode.MinAct = 0.2
	// ly.Acts.PopCode.MinSigma = 0.08
	// ly.Acts.PopCode.MaxSigma = 0.12
	ly.Acts.Decay.Act = 1
	ly.Acts.Decay.Glong = 1
	ly.Learn.TrgAvgAct.RescaleOn.SetBool(false)
}
