// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/gosl/slbool"
)

//gosl:start

// BurstParams determine how the 5IB Burst activation is computed from
// CaP integrated spiking values in Super layers -- thresholded.
type BurstParams struct {

	// Relative component of threshold on superficial activation value,
	// below which it does not drive Burst (and above which, Burst = CaP).
	// This is the distance between the average and maximum activation values
	// within layer (e.g., 0 = average, 1 = max).  Overall effective threshold
	// is MAX of relative and absolute thresholds.
	ThrRel float32 `max:"1" default:"0.1"`

	// Absolute component of threshold on superficial activation value,
	// below which it does not drive Burst (and above which, Burst = CaP).
	// Overall effective threshold is MAX of relative and absolute thresholds.
	ThrAbs float32 `min:"0" max:"1" default:"0.1"`

	pad, pad1 float32
}

func (bp *BurstParams) Update() {
}

func (bp *BurstParams) Defaults() {
	bp.ThrRel = 0.1
	bp.ThrAbs = 0.1
}

// ThrFromAvgMax returns threshold from average and maximum values
func (bp *BurstParams) ThrFromAvgMax(avg, mx float32) float32 {
	thr := avg + bp.ThrRel*(mx-avg)
	thr = math32.Max(thr, bp.ThrAbs)
	return thr
}

// CTParams control the CT corticothalamic neuron special behavior
type CTParams struct {

	// GeGain is the gain factor for context excitatory input, which is
	// constant as compared to the spiking input from other pathways, so it
	// must be downscaled accordingly. This can make a difference
	// and may need to be scaled up or down.
	GeGain float32 `default:"0.05,0.1,1,2"`

	// DecayTau is the decay time constant for context Ge input.
	// if > 0, decays over time so intrinsic circuit dynamics have to take over.
	// For single-step copy-based cases, set to 0, while longer-time-scale
	// dynamics should use ~50 or more.
	DecayTau float32 `default:"0,50,70"`

	// OFCposPT is set for the OFCposPT PTMaintLayer, which sets the
	// GvOFCposPTMaint global variable.
	OFCposPT slbool.Bool

	// 1 / tau
	DecayDt float32 `display:"-" json:"-" xml:"-"`
}

func (cp *CTParams) Update() {
	if cp.DecayTau > 0 {
		cp.DecayDt = 1 / cp.DecayTau
	} else {
		cp.DecayDt = 0
	}
}

func (cp *CTParams) Defaults() {
	cp.GeGain = 1
	cp.DecayTau = 50
	cp.Update()
}

// PulvinarParams provides parameters for how the plus-phase (outcome)
// state of Pulvinar thalamic relay cell neurons is computed from
// the corresponding driver neuron Burst activation (or CaP if not Super)
type PulvinarParams struct {

	// DriveScale is the multiplier on driver input strength,
	// which multiplies CaP from driver layer to produce Ge excitatory
	// input to CNiPred unit.
	DriveScale float32 `default:"0.1" min:"0.0"`

	// FullDriveAct is the level of Max driver layer CaP at which the drivers
	// fully drive the burst phase activation. If there is weaker driver input,
	// then (Max/FullDriveAct) proportion of the non-driver inputs remain and
	// this critically prevents the network from learning to turn activation
	// off, which is difficult and severely degrades learning.
	FullDriveAct float32 `default:"0.6" min:"0.01"`

	// DriveLayIndex of layer that generates the driving activity into this one
	// set via SetBuildConfig(DriveLayName) setting
	DriveLayIndex int32 `edit:"-"`

	pad float32
}

func (tp *PulvinarParams) Update() {
}

func (tp *PulvinarParams) Defaults() {
	tp.DriveScale = 0.1
	tp.FullDriveAct = 0.6
}

// DriveGe returns effective excitatory conductance
// to use for given driver input Burst activation
func (tp *PulvinarParams) DriveGe(act float32) float32 {
	return tp.DriveScale * act
}

// NonDrivePct returns the multiplier proportion of the non-driver based Ge to
// keep around, based on FullDriveAct and the max activity in driver layer.
func (tp *PulvinarParams) NonDrivePct(drvMax float32) float32 {
	return 1.0 - math32.Min(1.0, drvMax/tp.FullDriveAct)
}

// PulvinarDriver gets the driver input excitation params for Pulvinar layer.
func (ly *LayerParams) PulvinarDriver(ctx *Context, lni, di uint32, drvGe, nonDrivePct *float32) {
	dli := uint32(ly.Pulvinar.DriveLayIndex)
	dly := GetLayers(dli)
	dpi := dly.PoolIndex(0)
	drvMax := PoolAvgMax(AMCaP, AMCycle, Max, dpi, di)
	*nonDrivePct = ly.Pulvinar.NonDrivePct(drvMax) // how much non-driver to keep
	burst := Neurons.Value(int(dly.Indexes.NeurSt+lni), int(di), int(Burst))
	*drvGe = ly.Pulvinar.DriveGe(burst)
}

//gosl:end

// note: Defaults not called on GPU

func (ly *LayerParams) CTDefaults() {
	ly.Acts.Decay.Act = 0 // deep doesn't decay!
	ly.Acts.Decay.Glong = 0
	ly.Acts.Decay.AHP = 0
	ly.Acts.Dend.SSGi = 2   // 2 > 0 for sure
	ly.Inhib.Layer.Gi = 2.2 // higher inhib for more NMDA, recurrents.
	ly.Inhib.Pool.Gi = 2.2
	// these are for longer temporal integration:
	// ly.Acts.NMDA.Ge = 0.003
	// ly.Acts.NMDA.Tau = 300
	// ly.Acts.GABAB.Gbar = 0.008
}

func (cp *CTParams) DecayForNCycles(ncycles int) {
	cp.DecayTau = 50 * (float32(ncycles) / float32(200))
	cp.Update()
}

// CTDefaultParamsFast sets fast time-integration parameters for CTLayer.
// This is what works best in the deep_move 1 trial history case,
// vs Medium and Long
func (ly *Layer) CTDefaultParamsFast() {
	ly.AddDefaultParams(func(ly *LayerParams) {
		ly.CT.GeGain = 1
		ly.CT.DecayTau = 0
		ly.Inhib.Layer.Gi = 2.0
		ly.Inhib.Pool.Gi = 2.0
		ly.Acts.GabaB.Gk = 0.006
		ly.Acts.NMDA.Ge = 0.004
		ly.Acts.NMDA.Tau = 100
		ly.Acts.Decay.Act = 0.0
		ly.Acts.Decay.Glong = 0.0
		ly.Acts.Sahp.Gk = 1.0
	})
}

// CTDefaultParamsMedium sets medium time-integration parameters for CTLayer.
// This is what works best in the FSA test case, compared to Fast (deep_move)
// and Long (deep_music) time integration.
func (ly *Layer) CTDefaultParamsMedium() {
	ly.AddDefaultParams(func(ly *LayerParams) {
		ly.CT.GeGain = 2
		ly.Inhib.Layer.Gi = 2.2
		ly.Inhib.Pool.Gi = 2.2
		ly.Acts.GabaB.Gk = 0.009
		ly.Acts.NMDA.Ge = 0.008
		ly.Acts.NMDA.Tau = 200
		ly.Acts.Decay.Act = 0.0
		ly.Acts.Decay.Glong = 0.0
		ly.Acts.Sahp.Gk = 1.0
	})
}

// CTDefaultParamsLong sets long time-integration parameters for CTLayer.
// This is what works best in the deep_music test case integrating over
// long time windows, compared to Medium and Fast.
func (ly *Layer) CTDefaultParamsLong() {
	ly.AddDefaultParams(func(ly *LayerParams) {
		ly.CT.GeGain = 1.0
		ly.Inhib.Layer.Gi = 2.8
		ly.Inhib.Pool.Gi = 2.8
		ly.Acts.GabaB.Gk = 0.01
		ly.Acts.NMDA.Ge = 0.01
		ly.Acts.NMDA.Tau = 300
		ly.Acts.Decay.Act = 0.0
		ly.Acts.Decay.Glong = 0.0
		ly.Acts.Dend.SSGi = 2 // 2 > 0 for sure
		ly.Acts.Sahp.Gk = 1.0
	})
}

func (lly *Layer) PTMaintDefaults() {
	ly := lly.Params
	ly.Acts.Decay.Act = 0 // deep doesn't decay!
	ly.Acts.Decay.Glong = 0
	ly.Acts.Decay.AHP = 0
	ly.Acts.Decay.OnRew.SetBool(true)
	ly.Acts.Sahp.Gk = 0.01  // not much pressure -- long maint
	ly.Acts.GabaB.Gk = 0.01 // needed for cons, good for smaint
	ly.Acts.Dend.ModGain = 1.5
	// ly.Inhib.ActAvg.Nominal = 0.1 // normal
	if lly.Is4D() {
		ly.Inhib.ActAvg.Nominal = 0.02
	}
	ly.Inhib.Layer.Gi = 2.4
	ly.Inhib.Pool.Gi = 2.4
	ly.Learn.TrgAvgAct.RescaleOn.SetBool(false)
	ly.Learn.NeuroMod.AChDisInhib = 0

	for _, pj := range lly.RecvPaths {
		slay := pj.Send
		if slay.Type == BGThalLayer {
			pj.Params.Com.GType = ModulatoryG
		}
	}
}

func (ly *LayerParams) PTPredDefaults() {
	ly.Acts.Decay.Act = 0.12 // keep it dynamically changing
	ly.Acts.Decay.Glong = 0.6
	ly.Acts.Decay.AHP = 0
	ly.Acts.Decay.OnRew.SetBool(true)
	ly.Acts.Sahp.Gk = 0.1     // more
	ly.Acts.KNa.Slow.Gk = 0.2 // todo: more?
	ly.Inhib.Layer.Gi = 0.8
	ly.Inhib.Pool.Gi = 0.8
	ly.CT.GeGain = 0.05
	ly.CT.DecayTau = 50

	// regular:
	// ly.Acts.GabaB.Gk = 0.006
	// ly.Acts.NMDA.Ge = 0.004
	// ly.Acts.NMDA.Tau = 100
}

// called in Defaults for Pulvinar layer type
func (ly *LayerParams) PulvinarDefaults() {
	ly.Acts.Decay.Act = 0
	ly.Acts.Decay.Glong = 0
	ly.Acts.Decay.AHP = 0
	ly.Learn.RLRate.SigmoidMin = 1.0 // 1.0 generally better but worth trying 0.05 too
}

// PulvinarPostBuild does post-Build config of Pulvinar based on BuildConfig options
func (ly *Layer) PulvinarPostBuild() {
	ly.Params.Pulvinar.DriveLayIndex = ly.BuildConfigFindLayer("DriveLayName", true)
}
