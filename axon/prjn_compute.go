// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "sync/atomic"

// prjn_compute.go has the core computational methods, for the CPU.
// On GPU, this same functionality is implemented in corresponding gpu_*.hlsl
// files, which correspond to different shaders for each different function.

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// SendSpike sends a spike from the sending neuron at index sendIdx
// into the GBuf buffer on the receiver side. The buffer on the receiver side
// is a ring buffer, which is used for modelling the time delay between
// sending and receiving spikes.
func (pj *Prjn) SendSpike(ctx *Context, ni, di, maxData uint32) {
	scale := pj.Params.GScale.Scale * pj.Params.Com.FloatToIntFactor() // pre-bake in conversion to uint factor
	if pj.PrjnType() == CTCtxtPrjn {
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
	wrOff := pjcom.WriteOff(ctx.CyclesTotal) // todo: these require di offset!
	scon := pj.SendCon[ni-pj.Send.NeurStIdx]
	for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
		syni := pj.SynStIdx + syi
		recvIdx := pj.Params.SynRecvLayIdx(ctx, syni) // note: layer-specific is ok here
		sv := int32(scale * SynV(ctx, syni, Wt))
		bi := pjcom.WriteIdxOff(recvIdx, di, wrOff, pj.Params.Idxs.RecvNeurN, maxData)
		atomic.AddInt32(&pj.GBuf[bi], sv)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  SynCa methods

// note: important to exclude doing SynCa for types that don't use it!

// SynCaSend updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking.
// This pass goes through in sending order, filtering on sending spike.
// Sender will update even if recv neuron spiked -- recv will skip sender spike cases.
func (pj *Prjn) SynCaSend(ctx *Context, ni, di uint32, updtThr float32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	if !pj.Params.DoSynCa() {
		return
	}
	snCaSyn := pj.Params.Learn.KinaseCa.SpikeG * NrnV(ctx, ni, di, CaSyn)
	scon := pj.SendCon[ni-pj.Send.NeurStIdx]
	for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
		syni := pj.SynStIdx + syi
		ri := SynI(ctx, syni, SynRecvIdx)
		pj.Params.SynCaSyn(ctx, syni, ri, di, snCaSyn, updtThr)
	}
}

// SynCaRecv updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking.
// This pass goes through in recv order, filtering on recv spike,
// and skips when sender spiked, as those were already done in Send version.
func (pj *Prjn) SynCaRecv(ctx *Context, ni, di uint32, updtThr float32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	if !pj.Params.DoSynCa() {
		return
	}
	rnCaSyn := pj.Params.Learn.KinaseCa.SpikeG * NrnV(ctx, ni, di, CaSyn)
	syIdxs := pj.RecvSynIdxs(ni - pj.Recv.NeurStIdx)
	for _, syi := range syIdxs {
		syni := pj.SynStIdx + syi
		si := SynI(ctx, syni, SynSendIdx)
		if NrnV(ctx, si, di, Spike) > 0 { // already handled in send version
			continue
		}
		pj.Params.SynCaSyn(ctx, syni, si, di, rnCaSyn, updtThr)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning), based on
// synaptically-integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *Prjn) DWt(ctx *Context, si, di uint32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	rlay := pj.Recv
	layPool := &rlay.Pools[0]
	isTarget := rlay.Params.Act.Clamp.IsTarget.IsTrue()
	scon := pj.SendCon[si-pj.Send.NeurStIdx]
	for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
		syni := pj.SynStIdx + syi
		ri := SynI(ctx, syni, SynRecvIdx)
		subPool := rlay.SubPool(ctx, ri, di)
		pj.Params.DWtSyn(ctx, syni, si, ri, di, layPool, subPool, isTarget)
	}
}

// DWtSubMean subtracts the mean from any projections that have SubMean > 0.
// This is called on *receiving* projections, prior to WtFmDwt.
func (pj *Prjn) DWtSubMean(ctx *Context, ri uint32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	sm := pj.Params.Learn.Trace.SubMean
	if sm == 0 { // note default is now 0, so don't exclude Target layers, which should be 0
		return
	}
	syIdxs := pj.RecvSynIdxs(ri - pj.Recv.NeurStIdx)
	if len(syIdxs) < 1 {
		return
	}
	sumDWt := float32(0)
	nnz := 0 // non-zero
	for _, syi := range syIdxs {
		syni := pj.SynStIdx + syi
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
	for _, syi := range syIdxs {
		syni := pj.SynStIdx + syi
		if SynV(ctx, syni, DWt) != 0 {
			AddSynV(ctx, syni, DWt, -sm*sumDWt)
		}
	}
}

// WtFmDWt computes the weight change (learning), based on
// synaptically-integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *Prjn) WtFmDWt(ctx *Context, ni uint32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	scon := pj.SendCon[ni-pj.Send.NeurStIdx]
	for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
		syni := pj.SynStIdx + syi
		pj.Params.WtFmDWtSyn(ctx, syni)
	}
}

// SlowAdapt does the slow adaptation: SWt learning and SynScale
func (pj *Prjn) SlowAdapt(ctx *Context) {
	pj.SWtFmWt(ctx)
	pj.SynScale(ctx)
}

// SWtFmWt updates structural, slowly-adapting SWt value based on
// accumulated DSWt values, which are zero-summed with additional soft bounding
// relative to SWt limits.
func (pj *Prjn) SWtFmWt(ctx *Context) {
	if pj.Params.Learn.Learn.IsFalse() || pj.Params.SWt.Adapt.On.IsFalse() {
		return
	}
	rlay := pj.Recv
	if rlay.Params.IsTarget() {
		return
	}
	max := pj.Params.SWt.Limit.Max
	min := pj.Params.SWt.Limit.Min
	lr := pj.Params.SWt.Adapt.LRate
	for lni := uint32(0); lni < rlay.NNeurons; lni++ {
		syIdxs := pj.RecvSynIdxs(lni)
		nCons := len(syIdxs)
		if nCons < 1 {
			continue
		}
		avgDWt := float32(0)
		for _, syi := range syIdxs {
			syni := pj.SynStIdx + syi
			swt := SynV(ctx, syni, SWt)
			if SynV(ctx, syni, DSWt) >= 0 { // softbound for SWt
				MulSynV(ctx, syni, DSWt, (max - swt))
			} else {
				MulSynV(ctx, syni, DSWt, (swt - min))
			}
			avgDWt += SynV(ctx, syni, DSWt)
		}
		avgDWt /= float32(nCons)
		avgDWt *= pj.Params.SWt.Adapt.SubMean
		for _, syi := range syIdxs {
			syni := pj.SynStIdx + syi
			AddSynV(ctx, syni, SWt, lr*(SynV(ctx, syni, DSWt)-avgDWt))
			swt := SynV(ctx, syni, SWt)
			SetSynV(ctx, syni, DSWt, 0)
			if SynV(ctx, syni, Wt) == 0 { // restore failed wts
				wt := pj.Params.SWt.WtVal(swt, SynV(ctx, syni, LWt))
				SetSynV(ctx, syni, Wt, wt)
			}
			SetSynV(ctx, syni, LWt, pj.Params.SWt.LWtFmWts(SynV(ctx, syni, Wt), swt)) // + pj.Params.SWt.Adapt.RndVar()
			SetSynV(ctx, syni, Wt, pj.Params.SWt.WtVal(swt, SynV(ctx, syni, LWt)))
		}
	}
}

// SynScale performs synaptic scaling based on running average activation vs. targets.
// Layer-level AvgDifFmTrgAvg function must be called first.
func (pj *Prjn) SynScale(ctx *Context) {
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
		ri := rlay.NeurStIdx + lni
		if NrnIsOff(ctx, ri) {
			continue
		}
		adif := -lr * NrnAvgV(ctx, ri, AvgDif)
		syIdxs := pj.RecvSynIdxs(lni)
		for _, syi := range syIdxs {
			syni := pj.SynStIdx + syi
			lwt := SynV(ctx, syni, LWt)
			swt := SynV(ctx, syni, SWt)
			if adif >= 0 { // key to have soft bounding on lwt here!
				AddSynV(ctx, syni, LWt, (1-lwt)*adif*swt)
			} else {
				AddSynV(ctx, syni, LWt, lwt*adif*swt)
			}
			SetSynV(ctx, syni, Wt, pj.Params.SWt.WtVal(swt, SynV(ctx, syni, LWt)))
		}
	}
}

// SynFail updates synaptic weight failure only -- normally done as part of DWt
// and WtFmDWt, but this call can be used during testing to update failing synapses.
func (pj *Prjn) SynFail(ctx *Context) {
	slay := pj.Send
	for lni := uint32(0); lni < slay.NNeurons; lni++ {
		scon := pj.SendCon[lni]
		for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
			syni := pj.SynStIdx + syi
			swt := SynV(ctx, syni, SWt)
			if SynV(ctx, syni, Wt) == 0 { // restore failed wts
				SetSynV(ctx, syni, Wt, pj.Params.SWt.WtVal(swt, SynV(ctx, syni, LWt)))
			}
			pj.Params.Com.Fail(ctx, syni, swt)
		}
	}
}

// LRateMod sets the LRate modulation parameter for Prjns, which is
// for dynamic modulation of learning rate (see also LRateSched).
// Updates the effective learning rate factor accordingly.
func (pj *Prjn) LRateMod(mod float32) {
	pj.Params.Learn.LRate.Mod = mod
	pj.Params.Learn.LRate.Update()
}

// LRateSched sets the schedule-based learning rate multiplier.
// See also LRateMod.
// Updates the effective learning rate factor accordingly.
func (pj *Prjn) LRateSched(sched float32) {
	pj.Params.Learn.LRate.Sched = sched
	pj.Params.Learn.LRate.Update()
}
