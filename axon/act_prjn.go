// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/erand"
	"github.com/goki/gosl/slbool"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

//go:generate stringer -type=PrjnGTypes

var KiT_PrjnGTypes = kit.Enums.AddEnum(PrjnGTypesN, kit.NotBitFlag, nil)

func (ev PrjnGTypes) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *PrjnGTypes) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

//gosl: start act_prjn

// PrjnGTypes represents the conductance (G) effects of a given projection,
// including excitatory, inhibitory, and modulatory.
type PrjnGTypes int32

// The projection conductance types
const (
	// Excitatory projections drive Ge conductance on receiving neurons,
	// which send to GiRaw and GiSyn neuron variables.
	ExcitatoryG PrjnGTypes = iota

	// Inhibitory projections drive Gi inhibitory conductance,
	// which send to GiRaw and GiSyn neuron variables.
	InhibitoryG

	// Modulatory projections have a multiplicative effect on other inputs,
	// which send to GModRaw and GModSyn neuron variables.
	ModulatoryG

	// Context projections are for inputs to CT layers, which update
	// only at the end of the plus phase, and send to CtxtGe.
	ContextG

	PrjnGTypesN
)

//////////////////////////////////////////////////////////////////////////////////////
//  SynComParams

// SynComParams are synaptic communication parameters:
// used in the Prjn parameters.  Includes delay and
// probability of failure, and Inhib for inhibitory connections,
// and modulatory projections that have multiplicative-like effects.
type SynComParams struct {
	GType         PrjnGTypes  `desc:"type of conductance (G) communicated by this projection"`
	Delay         uint32      `min:"0" def:"2" desc:"additional synaptic delay in msec for inputs arriving at this projection.  Must be <= MaxDelay which is set during network building based on MaxDelay of any existing Prjn in the network.  Delay = 0 means a spike reaches receivers in the next Cycle, which is the minimum time (1 msec).  Biologically, subtract 1 from biological synaptic delay values to set corresponding Delay value."`
	MaxDelay      uint32      `inactive:"+" desc:"maximum value of Delay -- based on MaxDelay values when the BuildGBuf function was called when the network was built -- cannot set it longer than this, except by calling BuildGBuf on network after changing MaxDelay to a larger value in any projection in the network."`
	PFail         float32     `desc:"probability of synaptic transmission failure -- if > 0, then weights are turned off at random as a function of PFail (times 1-SWt if PFailSwt)"`
	PFailSWt      slbool.Bool `desc:"if true, then probability of failure is inversely proportional to SWt structural / slow weight value (i.e., multiply PFail * (1-SWt)))"`
	CPURecvSpikes slbool.Bool `view:"-" desc:"copied from Network for local access: if true, use the RecvSpikes receiver-based spiking function -- on the CPU -- this is more than 35x slower than the default SendSpike function -- it is only an option for testing in comparison to the GPU mode, which always uses RecvSpikes because the sender mode is not possible."`

	DelLen uint32 `view:"-" desc:"delay length = actual length of the GBuf buffer per neuron = Delay+1 -- just for speed"`

	pad float32
}

func (sc *SynComParams) Defaults() {
	sc.Delay = 2
	sc.MaxDelay = 2
	sc.PFail = 0 // 0.5 works?
	sc.PFailSWt.SetBool(false)
	sc.Update()
}

func (sc *SynComParams) Update() {
	if sc.Delay > sc.MaxDelay {
		sc.Delay = sc.MaxDelay
	}
	sc.DelLen = sc.Delay + 1
}

// RingIdx returns the wrap-around ring index for given raw index.
// For writing and reading spikes to GBuf buffer, based on
// Context.CycleTot counter.
// RN: 0     1     2         <- recv neuron indexes
// DI: 0 1 2 0 1 2 0 1 2     <- delay indexes
// C0: ^ v                   <- cycle 0, ring index: ^ = write, v = read
// C1:   ^ v                 <- cycle 1, shift over by 1 -- overwrite last read
// C2: v   ^                 <- cycle 2: read out value stored on C0 -- index wraps around
func (sc *SynComParams) RingIdx(i uint32) uint32 {
	if i >= sc.DelLen {
		i -= sc.DelLen
	}
	return i
}

// WriteOff returns offset for writing new spikes into the GBuf buffer,
// based on Context CycleTot counter which increments each cycle.
// This is logically the last position in the ring buffer.
func (sc *SynComParams) WriteOff(cycTot int32) uint32 {
	return sc.RingIdx(uint32(cycTot)%sc.DelLen + sc.DelLen)
}

// WriteIdx returns actual index for writing new spikes into the GBuf buffer,
// based on the layer-based recv neuron index and the
// WriteOff offset computed from the CycleTot.
func (sc *SynComParams) WriteIdx(rnIdx uint32, cycTot int32, nRecvNeurs uint32) uint32 {
	return sc.WriteIdxOff(rnIdx, sc.WriteOff(cycTot), nRecvNeurs)
}

// WriteIdxOff returns actual index for writing new spikes into the GBuf buffer,
// based on the layer-based recv neuron index and the given WriteOff offset.
func (sc *SynComParams) WriteIdxOff(rnIdx, wrOff uint32, nRecvNeurs uint32) uint32 {
	// return rnIdx*sc.DelLen + wrOff
	return wrOff*nRecvNeurs + rnIdx
}

// ReadOff returns offset for reading existing spikes from the GBuf buffer,
// based on Context CycleTot counter which increments each cycle.
// This is logically the zero position in the ring buffer.
func (sc *SynComParams) ReadOff(cycTot int32) uint32 {
	return sc.RingIdx(uint32(cycTot) % sc.DelLen)
}

// ReadIdx returns index for reading existing spikes from the GBuf buffer,
// based on the layer-based recv neuron index and the
// ReadOff offset from the CycleTot.
func (sc *SynComParams) ReadIdx(rnIdx uint32, cycTot int32, nRecvNeurs uint32) uint32 {
	// return rnIdx*sc.DelLen + sc.ReadOff(cycTot)
	return sc.ReadOff(cycTot)*nRecvNeurs + rnIdx // delay is outer, neurs are inner -- should be faster?
}

// FloatToIntFactor returns the factor used for converting float32
// to int32 in GBuf encoding.  Because total G is constrained via
// scaling factors to be around ~1, it is safe to use a factor that
// uses most of the available bits, leaving enough room to prevent
// overflow when adding together the different vals.
// For encoding, bake this into scale factor in SendSpike, and
// cast the result to int32.
func (sc *SynComParams) FloatToIntFactor() float32 {
	return float32(1 << 24) // leaves 7 bits = 128 to cover any extreme cases
	// this is sufficient to pass existing tests at std tolerances.
}

// FloatToGBuf converts the given floating point value
// to a large int32 for accumulating in GBuf.
// Note: more efficient to bake factor into scale factor per prjn.
func (sc *SynComParams) FloatToGBuf(val float32) int32 {
	return int32(val * sc.FloatToIntFactor())
}

// FloatFromGBuf converts the given int32 value produced
// via FloatToGBuf back into a float32 (divides by factor).
// If the value is negative, a panic is triggered indicating
// there was numerical overflow in the aggregation.
// If this occurs, the FloatToIntFactor needs to be decreased.
func (sc *SynComParams) FloatFromGBuf(ival int32) float32 {
	//gosl: end act_prjn
	if ival < 0 {
		panic("axon.SynComParams: FloatFromGBuf is negative, there was an overflow error")
	}
	//gosl: start act_prjn
	return float32(ival) / sc.FloatToIntFactor()
}

// WtFailP returns probability of weight (synapse) failure given current SWt value
func (sc *SynComParams) WtFailP(swt float32) float32 {
	if sc.PFailSWt.IsFalse() {
		return sc.PFail
	}
	return sc.PFail * (1 - swt)
}

//gosl: end act_prjn

// WtFail returns true if synapse should fail, as function of SWt value (optionally)
func (sc *SynComParams) WtFail(swt float32) bool {
	fp := sc.WtFailP(swt)
	if fp == 0 {
		return false
	}
	return erand.BoolP(fp)
}

// Fail updates failure status of given weight, given SWt value
func (sc *SynComParams) Fail(wt *float32, swt float32) {
	if sc.PFail > 0 {
		if sc.WtFail(swt) {
			*wt = 0
		}
	}
}

//gosl: start act_prjn

//////////////////////////////////////////////////////////////////////////////////////
//  PrjnScaleParams

// PrjnScaleParams are projection scaling parameters: modulates overall strength of projection,
// using both absolute and relative factors.
type PrjnScaleParams struct {
	Rel    float32 `min:"0" desc:"[Defaults: Forward=1, Back=0.2] relative scaling that shifts balance between different projections -- this is subject to normalization across all other projections into receiving neuron, and determines the GScale.Target for adapting scaling"`
	Abs    float32 `def:"1" min:"0" desc:"absolute multiplier adjustment factor for the prjn scaling -- can be used to adjust for idiosyncrasies not accommodated by the standard scaling based on initial target activation level and relative scaling factors -- any adaptation operates by directly adjusting scaling factor from the initially computed value"`
	AvgTau float32 `def:"500" desc:"time constant for integrating projection-level averages to track G scale: Prjn.GScale.AvgAvg, AvgMax (tau is roughly how long it takes for value to change significantly) -- these are updated at the cycle level and thus require a much slower rate constant compared to other such variables integrated at the AlphaCycle level."`

	AvgDt float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (ws *PrjnScaleParams) Defaults() {
	ws.Rel = 1
	ws.Abs = 1
	ws.AvgTau = 500
	ws.Update()
}

func (ws *PrjnScaleParams) Update() {
	ws.AvgDt = 1 / ws.AvgTau
}

// SLayActScale computes scaling factor based on sending layer activity level (savg), number of units
// in sending layer (snu), and number of recv connections (ncon).
// Uses a fixed sem_extra standard-error-of-the-mean (SEM) extra value of 2
// to add to the average expected number of active connections to receive,
// for purposes of computing scaling factors with partial connectivity
// For 25% layer activity, binomial SEM = sqrt(p(1-p)) = .43, so 3x = 1.3 so 2 is a reasonable default.
func (ws *PrjnScaleParams) SLayActScale(savg, snu, ncon float32) float32 {
	if ncon < 1 { // prjn Avg can be < 1 in some cases
		ncon = 1
	}
	semExtra := 2
	slayActN := int(mat32.Round(savg * snu)) // sending layer actual # active
	slayActN = ints.MaxInt(slayActN, 1)
	var sc float32
	if ncon == snu {
		sc = 1 / float32(slayActN)
	} else {
		maxActN := int(mat32.Min(ncon, float32(slayActN))) // max number we could get
		avgActN := int(mat32.Round(savg * ncon))           // recv average actual # active if uniform
		avgActN = ints.MaxInt(avgActN, 1)
		expActN := avgActN + semExtra // expected
		expActN = ints.MinInt(expActN, maxActN)
		sc = 1 / float32(expActN)
	}
	return sc
}

// FullScale returns full scaling factor, which is product of Abs * Rel * SLayActScale
func (ws *PrjnScaleParams) FullScale(savg, snu, ncon float32) float32 {
	return ws.Abs * ws.Rel * ws.SLayActScale(savg, snu, ncon)
}

//gosl: end act_prjn
