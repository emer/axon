// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/etable/minmax"
	"github.com/goki/mat32"
)

///////////////////////////////////////////////////////////////////////
//  learn.go contains the learning params and functions for axon

// axon.LearnNeurParams manages learning-related parameters at the neuron-level.
// This is mainly the running average activations that drive learning
type LearnNeurParams struct {
	ActAvg   LrnActAvgParams `view:"inline" desc:"parameters for computing running average activations that drive learning"`
	CosDiff  CosDiffParams   `view:"inline" desc:"parameters for computing cosine diff between minus and plus phase"`
	SynScale SynScaleParams  `view:"inline" desc:"synaptic scaling parameters for regulating overall average activity compared to neuron's own target level"`
}

func (ln *LearnNeurParams) Update() {
	ln.ActAvg.Update()
	ln.CosDiff.Update()
	ln.SynScale.Update()
}

func (ln *LearnNeurParams) Defaults() {
	ln.ActAvg.Defaults()
	ln.CosDiff.Defaults()
	ln.SynScale.Defaults()
}

// InitActAvg initializes the running-average activation values that drive learning.
// Called by InitWts (at start of learning).
func (ln *LearnNeurParams) InitActAvg(nrn *Neuron) {
	nrn.AvgSS = ln.ActAvg.Init
	nrn.AvgS = ln.ActAvg.Init
	nrn.AvgM = ln.ActAvg.Init
	nrn.AvgSLrn = 0
}

// AvgsFmAct updates the running averages based on current learning activation.
// Computed after new activation for current cycle is updated.
func (ln *LearnNeurParams) AvgsFmAct(nrn *Neuron) {
	ln.ActAvg.AvgsFmAct(ln.ActAvg.SpikeG*nrn.Spike, &nrn.AvgSS, &nrn.AvgS, &nrn.AvgM, &nrn.AvgSLrn)
}

///////////////////////////////////////////////////////////////////////
//  LearnSynParams

// axon.LearnSynParams manages learning-related parameters at the synapse-level.
type LearnSynParams struct {
	Learn     bool        `desc:"enable learning for this projection"`
	Lrate     float32     `desc:"current effective learning rate (multiplies DWt values, determining rate of change of weights)"`
	LrateInit float32     `desc:"initial learning rate -- this is set from Lrate in UpdateParams, which is called when Params are updated, and used in LrateMult to compute a new learning rate for learning rate schedules."`
	XCal      XCalParams  `view:"inline" desc:"parameters for the XCal learning rule"`
	WtSig     WtSigParams `view:"inline" desc:"parameters for the sigmoidal contrast weight enhancement"`
}

func (ls *LearnSynParams) Update() {
	ls.XCal.Update()
	ls.WtSig.Update()
}

func (ls *LearnSynParams) Defaults() {
	ls.Learn = true
	ls.Lrate = 0.04
	ls.LrateInit = ls.Lrate
	ls.XCal.Defaults()
	ls.WtSig.Defaults()
}

// LWtFmWt updates the linear weight value based on the current effective Wt value.
// effective weight is sigmoidally contrast-enhanced relative to the linear weight.
func (ls *LearnSynParams) LWtFmWt(syn *Synapse) {
	syn.LWt = ls.WtSig.LinFmSigWt(syn.Wt / syn.Scale) // must factor out scale too!
}

// WtFmLWt updates the effective weight value based on the current linear Wt value.
// effective weight is sigmoidally contrast-enhanced relative to the linear weight.
func (ls *LearnSynParams) WtFmLWt(syn *Synapse) {
	syn.Wt = ls.WtSig.SigFmLinWt(syn.LWt)
	syn.Wt *= syn.Scale
}

// CHLdWt returns the error-driven weight change component for the
// temporally eXtended Contrastive Attractor Learning (XCAL), CHL version
func (ls *LearnSynParams) CHLdWt(suAvgSLrn, suAvgM, ruAvgSLrn, ruAvgM float32) float32 {
	srs := suAvgSLrn * ruAvgSLrn
	srm := suAvgM * ruAvgM
	return ls.XCal.DWt(srs, srm)
}

// WtFmDWt updates the synaptic weights from accumulated weight changes
// wbInc and wbDec are the weight balance factors, wt is the sigmoidal contrast-enhanced
// weight and lwt is the linear weight value
func (ls *LearnSynParams) WtFmDWt(dwt, wt, lwt *float32, scale float32) {
	if *dwt == 0 {
		if *wt == 0 { // restore failed wts
			*wt = scale * ls.WtSig.SigFmLinWt(*lwt)
		}
		return
	}
	// always doing softbound by default
	// if *dwt > 0 {
	// 	*dwt *= (1 - *lwt)
	// } else {
	// 	*dwt *= *lwt
	// }
	*lwt += *dwt
	if *lwt < 0 {
		*lwt = 0
	} else if *lwt > 1 {
		*lwt = 1
	}
	*wt = scale * ls.WtSig.SigFmLinWt(*lwt)
	*dwt = 0
}

// LrnActAvgParams has rate constants for averaging over activations
// at different time scales, to produce the running average activation
// values that then drive learning in the XCAL learning rules.
// Is driven directly by spikes that increment running-average at super-short
// timescale.  Time cycle of 50 msec quarters / theta window learning works
// Cyc:50, SS:35 S:8, M:40 (best)
// Cyc:25, SS:20, S:4, M:20
type LrnActAvgParams struct {
	SpikeG float32 `def:"8" desc:"gain multiplier on spike: how much spike drives AvgSS value"`
	SSTau  float32 `def:"40" min:"1" desc:"time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life), for continuously updating the super-short time-scale AvgSS value -- this is provides a pre-integration step before integrating into the AvgS short time scale -- it is particularly important for spiking -- in general 4 is the largest value without starting to impair learning, but a value of 7 can be combined with m_in_s = 0 with somewhat worse results"`
	STau   float32 `def:"10" min:"1" desc:"time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life), for continuously updating the short time-scale AvgS value from the super-short AvgSS value (cascade mode) -- AvgS represents the plus phase learning signal that reflects the most recent past information"`
	MTau   float32 `def:"40" min:"1" desc:"time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life), for continuously updating the medium time-scale AvgM value from the short AvgS value (cascade mode) -- AvgM represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome) -- the default value of 10 generally cannot be exceeded without impairing learning"`
	LrnM   float32 `def:"0.1,0" min:"0" max:"1" desc:"how much of the medium term average activation to mix in with the short (plus phase) to compute the Neuron AvgSLrn variable that is used for the unit's short-term average in learning. This is important to ensure that when unit turns off in plus phase (short time scale), enough medium-phase trace remains so that learning signal doesn't just go all the way to 0, at which point no learning would take place -- typically need faster time constant for updating S such that this trace of the M signal is lost -- can set SSTau=7 and set this to 0 but learning is generally somewhat worse"`
	Init   float32 `def:"0.15" min:"0" max:"1" desc:"initial value for average"`

	SSDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	SDt  float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	MDt  float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	LrnS float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"1-LrnM"`
}

// AvgsFmAct computes averages based on current act
func (aa *LrnActAvgParams) AvgsFmAct(act float32, avgSS, avgS, avgM, avgSLrn *float32) {
	*avgSS += aa.SSDt * (act - *avgSS)
	*avgS += aa.SDt * (*avgSS - *avgS)
	*avgM += aa.MDt * (*avgS - *avgM)

	*avgSLrn = aa.LrnS**avgS + aa.LrnM**avgM
}

func (aa *LrnActAvgParams) Update() {
	aa.SSDt = 1 / aa.SSTau
	aa.SDt = 1 / aa.STau
	aa.MDt = 1 / aa.MTau
	aa.LrnS = 1 - aa.LrnM
}

func (aa *LrnActAvgParams) Defaults() {
	aa.SpikeG = 8
	aa.SSTau = 40 // 20 for 25 cycle qtr
	aa.STau = 10
	aa.MTau = 40 // 20 for 25 cycle qtr
	aa.LrnM = 0.1
	aa.Init = 0.15
	aa.Update()

}

//////////////////////////////////////////////////////////////////////////////////////
//  CosDiffParams

// CosDiffParams specify how to integrate cosine of difference between plus and minus phase activations
// Used to modulate amount of hebbian learning, and overall learning rate.
type CosDiffParams struct {
	Tau float32 `def:"100" min:"1" desc:"time constant in alpha-cycles (roughly how long significant change takes, 1.4 x half-life) for computing running average CosDiff value for the layer, CosDiffAvg = cosine difference between ActM and ActP -- this is an important statistic for how much phase-based difference there is between phases in this layer -- it is used in standard X_COS_DIFF modulation of l_mix in AxonConSpec, and for modulating learning rate as a function of predictability in the DeepAxon predictive auto-encoder learning -- running average variance also computed with this: cos_diff_var"`

	Dt  float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate constant = 1 / Tau"`
	DtC float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"complement of rate constant = 1 - Dt"`
}

func (cd *CosDiffParams) Update() {
	cd.Dt = 1 / cd.Tau
	cd.DtC = 1 - cd.Dt
}

func (cd *CosDiffParams) Defaults() {
	cd.Tau = 100
	cd.Update()
}

// AvgVarFmCos updates the average and variance from current cosine diff value
func (cd *CosDiffParams) AvgVarFmCos(avg, vr *float32, cos float32) {
	if *avg == 0 { // first time -- set
		*avg = cos
		*vr = 0
	} else {
		del := cos - *avg
		incr := cd.Dt * del
		*avg += incr
		// following is magic exponentially-weighted incremental variance formula
		// derived by Finch, 2009: Incremental calculation of weighted mean and variance
		if *vr == 0 {
			*vr = 2 * cd.DtC * del * incr
		} else {
			*vr = cd.DtC * (*vr + del*incr)
		}
	}
}

// LrateMod computes learning rate modulation based on cos diff vals
// func (cd *CosDiffParams) LrateMod(cos, avg, vr float32) float32 {
// 	if vr <= 0 {
// 		return 1
// 	}
// 	zval := (cos - avg) / mat32.Sqrt(vr) // stdev = sqrt of var
// 	// z-normal value is starting point for learning rate factor
// 	//    if zval < lrmod_z_thr {
// 	// 	return 0
// 	// }
// 	return 1
// }

//////////////////////////////////////////////////////////////////////////////////////
//  CosDiffStats

// CosDiffStats holds cosine-difference statistics at the layer level
type CosDiffStats struct {
	Cos float32 `inactive:"+" desc:"cosine (normalized dot product) activation difference between ActP and ActM on this alpha-cycle for this layer -- computed by CosDiffFmActs at end of QuarterFinal for quarter = 3"`
	Avg float32 `inactive:"+" desc:"running average of cosine (normalized dot product) difference between ActP and ActM -- computed with CosDiff.Tau time constant in QuarterFinal"`
	Var float32 `inactive:"+" desc:"running variance of cosine (normalized dot product) difference between ActP and ActM -- computed with CosDiff.Tau time constant in QuarterFinal, used for modulating overall learning rate"`
}

func (cd *CosDiffStats) Init() {
	cd.Cos = 0
	cd.Avg = 0
	cd.Var = 0
}

//////////////////////////////////////////////////////////////////////////////////////
//  SynScaleParams

// SynScaleParams govern the synaptic scaling to maintain target level of overall long-term
// average activity in neurons.
// Weights are rescaled in proportion to avg diff -- larger weights affected in proportion.
type SynScaleParams struct {
	ErrLrate float32    `def:"0.02" desc:"learning rate for adjustments to Trg value based on unit-level error signal.  Population TrgAvg values are renormalized to fixed overall average in TrgRange."`
	TrgRange minmax.F32 `desc:"default 0.5-2 -- range of target normalized average activations -- individual neurons are assigned values within this range to TrgAvg, and clamped within this range."`
	Permute  bool       `def:"true" desc:"permute the order of TrgAvg values within layer -- otherwise they are just assigned in order from highest to lowest for easy visualization -- generally must be true if any topographic weights are being used"`
	Rate     float32    `def:"0.005" desc:"learning rate parameter for how much to scale weights in proportion to the AvgDif between target and actual proportion activity -- set higher for smaller models"`
}

func (ss *SynScaleParams) Update() {
}

func (ss *SynScaleParams) Defaults() {
	ss.ErrLrate = 0.02
	ss.TrgRange.Set(0.5, 2)
	ss.Permute = true
	ss.Rate = 0.005
	ss.Update()
}

//////////////////////////////////////////////////////////////////////////////////////
//  XCalParams

// XCalParams are parameters for temporally eXtended Contrastive Attractor Learning function (XCAL)
// which is the standard learning equation for axon .
type XCalParams struct {
	SubMean float32 `def:"1" desc:"amount of the mean dWt to subtract -- 1.0 = full zero-sum dWt -- only on non-zero DWts (see DWtThr)"`
	DWtThr  float32 `def:"0.0001" desc:"threshold on DWt to be included in SubMean process -- this is *prior* to lrate multiplier"`
	DRev    float32 `def:"0.1" min:"0" max:"0.99" desc:"proportional point within LTD range where magnitude reverses to go back down to zero at zero -- err-driven svm component does better with smaller values"`
	DThr    float32 `def:"0.0001,0.01" min:"0" desc:"minimum LTD threshold value below which no weight change occurs -- this is now *relative* to the threshold"`
	LrnThr  float32 `def:"0.01" desc:"xcal learning threshold -- don't learn when sending unit activation is below this value in both phases -- due to the nature of the learning function being 0 when the sr coproduct is 0, it should not affect learning in any substantial way -- nonstandard learning algorithms that have different properties should ignore it"`

	DRevRatio float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"-(1-DRev)/DRev -- multiplication factor in learning rule -- builds in the minus sign!"`
}

func (xc *XCalParams) Update() {
	if xc.DRev > 0 {
		xc.DRevRatio = -(1 - xc.DRev) / xc.DRev
	} else {
		xc.DRevRatio = -1
	}
}

func (xc *XCalParams) Defaults() {
	xc.SubMean = 1
	xc.DWtThr = 0.0001
	xc.DRev = 0.1
	xc.DThr = 0.0001
	xc.LrnThr = 0.01
	xc.Update()
}

// DWt is the XCAL function for weight change -- the "check mark" function -- no DGain, no ThrPMin
func (xc *XCalParams) DWt(srval, thrP float32) float32 {
	var dwt float32
	if srval < xc.DThr {
		dwt = 0
	} else if srval > thrP*xc.DRev {
		dwt = (srval - thrP)
	} else {
		dwt = srval * xc.DRevRatio
	}
	return dwt
}

//////////////////////////////////////////////////////////////////////////////////////
//  WtSigParams

// WtSigParams are sigmoidal weight contrast enhancement function parameters
type WtSigParams struct {
	Gain float32 `def:"1,6" min:"0" desc:"gain (contrast, sharpness) of the weight contrast function (1 = linear)"`
	Off  float32 `def:"1" min:"0" desc:"offset of the function (1=centered at .5, >1=higher, <1=lower) -- 1 is standard for XCAL"`
}

func (ws *WtSigParams) Update() {
}

func (ws *WtSigParams) Defaults() {
	ws.Gain = 6
	ws.Off = 1
}

// SigFun is the sigmoid function for value w in 0-1 range, with gain and offset params
func SigFun(w, gain, off float32) float32 {
	if w <= 0 {
		return 0
	}
	if w >= 1 {
		return 1
	}
	return (1 / (1 + mat32.Pow((off*(1-w))/w, gain)))
}

// SigFun61 is the sigmoid function for value w in 0-1 range, with default gain = 6, offset = 1 params
func SigFun61(w float32) float32 {
	if w <= 0 {
		return 0
	}
	if w >= 1 {
		return 1
	}
	pw := (1 - w) / w
	return (1 / (1 + pw*pw*pw*pw*pw*pw))
}

// SigInvFun is the inverse of the sigmoid function
func SigInvFun(w, gain, off float32) float32 {
	if w <= 0 {
		return 0
	}
	if w >= 1 {
		return 1
	}
	return 1.0 / (1.0 + mat32.Pow((1.0-w)/w, 1/gain)/off)
}

// SigInvFun61 is the inverse of the sigmoid function, with default gain = 6, offset = 1 params
func SigInvFun61(w float32) float32 {
	if w <= 0 {
		return 0
	}
	if w >= 1 {
		return 1
	}
	rval := 1.0 / (1.0 + mat32.Pow((1.0-w)/w, 1.0/6.0))
	return rval
}

// SigFmLinWt returns sigmoidal contrast-enhanced weight from linear weight
func (ws *WtSigParams) SigFmLinWt(lw float32) float32 {
	var wt float32
	switch {
	case ws.Gain == 1 && ws.Off == 1:
		wt = lw
	case ws.Gain == 6 && ws.Off == 1:
		wt = SigFun61(lw)
	default:
		wt = SigFun(lw, ws.Gain, ws.Off)
	}
	return wt
}

// LinFmSigWt returns linear weight from sigmoidal contrast-enhanced weight
func (ws *WtSigParams) LinFmSigWt(sw float32) float32 {
	if ws.Gain == 1 && ws.Off == 1 {
		return sw
	}
	if ws.Gain == 6 && ws.Off == 1 {
		return SigInvFun61(sw)
	}
	return SigInvFun(sw, ws.Gain, ws.Off)
}
