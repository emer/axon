// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"sync/atomic"
)

// path_compute.go has the core computational methods, for the CPU.
// On GPU, this same functionality is implemented in corresponding gpu_*.hlsl
// files, which correspond to different shaders for each different function.

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// SendSpike sends a spike from the sending neuron at index sendIndex
// into the GBuf buffer on the receiver side. The buffer on the receiver side
// is a ring buffer, which is used for modelling the time delay between
// sending and receiving spikes.
func (pj *Path) SendSpike(ctx *Context, ni, di, maxData uint32) {
	scale := pj.Params.GScale.Scale * pj.Params.Com.FloatToIntFactor() // pre-bake in conversion to uint factor
	if pj.PathType() == CTCtxtPath {
		if ctx.Cycle != ctx.ThetaCycles-1-int32(pj.Params.Com.DelLen) {
			return
		}
		scale *= NrnV(ctx, ni, di, Burst) // Burst is regular CaSpkP for all non-SuperLayer neurons
	} else {
		if NrnV(ctx, ni, di, Spike) == 0 {
			return
		}
	}
	pjcom := &pj.Params.Com
	wrOff := pjcom.WriteOff(ctx.CyclesTotal)
	scon := pj.SendCon[ni-pj.Send.NeurStIndex]
	for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
		syni := pj.SynStIndex + syi
		recvIndex := pj.Params.SynRecvLayIndex(ctx, syni) // note: layer-specific is ok here
		sv := int32(scale * SynV(ctx, syni, Wt))
		bi := pjcom.WriteIndexOff(recvIndex, di, wrOff, pj.Params.Indexes.RecvNeurN, maxData)
		atomic.AddInt32(&pj.GBuf[bi], sv)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning), based on
// synaptically integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *Path) DWt(ctx *Context, si uint32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	scon := pj.SendCon[si-pj.Send.NeurStIndex]
	rlay := pj.Recv
	isTarget := rlay.Params.Acts.Clamp.IsTarget.IsTrue()
	for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
		syni := pj.SynStIndex + syi
		ri := SynI(ctx, syni, SynRecvIndex)
		dwt := float32(0)
		for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
			layPool := rlay.Pool(0, di)
			subPool := rlay.SubPool(ctx, ri, di)
			pj.Params.DWtSyn(ctx, syni, si, ri, di, layPool, subPool, isTarget)
			dwt += SynCaV(ctx, syni, di, DiDWt)
		}
		// note: on GPU, this must be a separate kernel, but can be combined here
		AddSynV(ctx, syni, DWt, dwt)
	}
}

// DWtSubMean subtracts the mean from any pathways that have SubMean > 0.
// This is called on *receiving* pathways, prior to WtFromDwt.
func (pj *Path) DWtSubMean(ctx *Context, ri uint32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	sm := pj.Params.Learn.Trace.SubMean
	if sm == 0 { // note default is now 0, so don't exclude Target layers, which should be 0
		return
	}
	syIndexes := pj.RecvSynIndexes(ri - pj.Recv.NeurStIndex)
	if len(syIndexes) < 1 {
		return
	}
	sumDWt := float32(0)
	nnz := 0 // non-zero
	for _, syi := range syIndexes {
		syni := pj.SynStIndex + syi
		dw := SynV(ctx, syni, DWt)
		if dw != 0 {
			sumDWt += dw
			nnz++
		}
	}
	if nnz <= 1 {
		return
	}
	sumDWt /= float32(nnz)
	for _, syi := range syIndexes {
		syni := pj.SynStIndex + syi
		if SynV(ctx, syni, DWt) != 0 {
			AddSynV(ctx, syni, DWt, -sm*sumDWt)
		}
	}
}

// WtFromDWt computes the weight change (learning), based on
// synaptically integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *Path) WtFromDWt(ctx *Context, ni uint32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	scon := pj.SendCon[ni-pj.Send.NeurStIndex]
	for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
		syni := pj.SynStIndex + syi
		pj.Params.WtFromDWtSyn(ctx, syni)
	}
}

// SlowAdapt does the slow adaptation: SWt learning and SynScale
func (pj *Path) SlowAdapt(ctx *Context) {
	pj.SWtFromWt(ctx)
	pj.SynScale(ctx)
}

// SWtFromWt updates structural, slowly adapting SWt value based on
// accumulated DSWt values, which are zero-summed with additional soft bounding
// relative to SWt limits.
func (pj *Path) SWtFromWt(ctx *Context) {
	if pj.Params.Learn.Learn.IsFalse() || pj.Params.SWts.Adapt.On.IsFalse() {
		return
	}
	rlay := pj.Recv
	if rlay.Params.IsTarget() {
		return
	}
	mx := pj.Params.SWts.Limit.Max
	mn := pj.Params.SWts.Limit.Min
	lr := pj.Params.SWts.Adapt.LRate
	for lni := uint32(0); lni < rlay.NNeurons; lni++ {
		syIndexes := pj.RecvSynIndexes(lni)
		nCons := len(syIndexes)
		if nCons < 1 {
			continue
		}
		avgDWt := float32(0)
		for _, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			swt := SynV(ctx, syni, SWt)
			if SynV(ctx, syni, DSWt) >= 0 { // softbound for SWt
				MulSynV(ctx, syni, DSWt, (mx - swt))
			} else {
				MulSynV(ctx, syni, DSWt, (swt - mn))
			}
			avgDWt += SynV(ctx, syni, DSWt)
		}
		avgDWt /= float32(nCons)
		avgDWt *= pj.Params.SWts.Adapt.SubMean
		for _, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			AddSynV(ctx, syni, SWt, lr*(SynV(ctx, syni, DSWt)-avgDWt))
			swt := SynV(ctx, syni, SWt)
			SetSynV(ctx, syni, DSWt, 0)
			if SynV(ctx, syni, Wt) == 0 { // restore failed wts
				wt := pj.Params.SWts.WtValue(swt, SynV(ctx, syni, LWt))
				SetSynV(ctx, syni, Wt, wt)
			}
			SetSynV(ctx, syni, LWt, pj.Params.SWts.LWtFromWts(SynV(ctx, syni, Wt), swt)) // + pj.Params.SWts.Adapt.RandVar()
			SetSynV(ctx, syni, Wt, pj.Params.SWts.WtValue(swt, SynV(ctx, syni, LWt)))
		}
	}
}

// SynScale performs synaptic scaling based on running average activation vs. targets.
// Layer-level AvgDifFromTrgAvg function must be called first.
func (pj *Path) SynScale(ctx *Context) {
	if pj.Params.Learn.Learn.IsFalse() || pj.Params.IsInhib() {
		return
	}
	rlay := pj.Recv
	if !rlay.Params.IsLearnTrgAvg() {
		return
	}
	tp := &rlay.Params.Learn.TrgAvgAct
	lr := tp.SynScaleRate
	for lni := uint32(0); lni < rlay.NNeurons; lni++ {
		ri := rlay.NeurStIndex + lni
		if NrnIsOff(ctx, ri) {
			continue
		}
		adif := -lr * NrnAvgV(ctx, ri, AvgDif)
		syIndexes := pj.RecvSynIndexes(lni)
		for _, syi := range syIndexes {
			syni := pj.SynStIndex + syi
			lwt := SynV(ctx, syni, LWt)
			swt := SynV(ctx, syni, SWt)
			if adif >= 0 { // key to have soft bounding on lwt here!
				AddSynV(ctx, syni, LWt, (1-lwt)*adif*swt)
			} else {
				AddSynV(ctx, syni, LWt, lwt*adif*swt)
			}
			SetSynV(ctx, syni, Wt, pj.Params.SWts.WtValue(swt, SynV(ctx, syni, LWt)))
		}
	}
}

// SynFail updates synaptic weight failure only -- normally done as part of DWt
// and WtFromDWt, but this call can be used during testing to update failing synapses.
func (pj *Path) SynFail(ctx *Context) {
	slay := pj.Send
	for lni := uint32(0); lni < slay.NNeurons; lni++ {
		scon := pj.SendCon[lni]
		for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
			syni := pj.SynStIndex + syi
			swt := SynV(ctx, syni, SWt)
			if SynV(ctx, syni, Wt) == 0 { // restore failed wts
				SetSynV(ctx, syni, Wt, pj.Params.SWts.WtValue(swt, SynV(ctx, syni, LWt)))
			}
			pj.Params.Com.Fail(ctx, syni, swt)
		}
	}
}

// LRateMod sets the LRate modulation parameter for Paths, which is
// for dynamic modulation of learning rate (see also LRateSched).
// Updates the effective learning rate factor accordingly.
func (pj *Path) LRateMod(mod float32) {
	pj.Params.Learn.LRate.Mod = mod
	pj.Params.Learn.LRate.Update()
}

// LRateSched sets the schedule-based learning rate multiplier.
// See also LRateMod.
// Updates the effective learning rate factor accordingly.
func (pj *Path) LRateSched(sched float32) {
	pj.Params.Learn.LRate.Sched = sched
	pj.Params.Learn.LRate.Update()
}
