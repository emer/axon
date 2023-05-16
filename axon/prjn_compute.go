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
func (pj *Prjn) SendSpike(ctx *Context, sendIdx uint32, nrn *Neuron) {
	scale := pj.Params.GScale.Scale * pj.Params.Com.FloatToIntFactor() // pre-bake in conversion to uint factor
	if pj.PrjnType() == CTCtxtPrjn {
		if ctx.Cycle != ctx.ThetaCycles-1-int32(pj.Params.Com.DelLen) {
			return
		}
		scale *= nrn.Burst // Burst is regular CaSpkP for all non-SuperLayer neurons
	} else {
		if nrn.Spike == 0 {
			return
		}
	}
	pjcom := &pj.Params.Com
	wrOff := pjcom.WriteOff(ctx.CyclesTotal)
	syns := pj.SendSyns(int(sendIdx))
	for syi := range syns {
		sy := &syns[syi]
		recvIdx := pj.Params.SynRecvLayIdx(sy)
		sv := int32(scale * sy.Wt)
		bi := pjcom.WriteIdxOff(recvIdx, wrOff, pj.Params.Idxs.RecvNeurN)
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
func (pj *Prjn) SynCaSend(ctx *Context, si uint32, sn *Neuron, updtThr float32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	if !pj.Params.DoSynCa() {
		return
	}
	rlay := pj.Recv
	snCaSyn := pj.Params.Learn.KinaseCa.SpikeG * sn.CaSyn
	syns := pj.SendSyns(int(si))
	for syi := range syns {
		sy := &syns[syi]
		ri := pj.Params.SynRecvLayIdx(sy)
		rn := &rlay.Neurons[ri]
		pj.Params.SynCaSyn(ctx, sy, rn, snCaSyn, updtThr)
	}
}

// SynCaRecv updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking.
// This pass goes through in recv order, filtering on recv spike,
// and skips when sender spiked, as those were already done in Send version.
func (pj *Prjn) SynCaRecv(ctx *Context, ri uint32, rn *Neuron, updtThr float32) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	if !pj.Params.DoSynCa() {
		return
	}
	slay := pj.Send
	rnCaSyn := pj.Params.Learn.KinaseCa.SpikeG * rn.CaSyn
	syIdxs := pj.RecvSynIdxs(int(ri))
	for _, syi := range syIdxs {
		sy := &pj.Syns[syi]
		si := pj.Params.SynSendLayIdx(sy)
		sn := &slay.Neurons[si]
		if sn.Spike > 0 {
			continue
		}
		pj.Params.SynCaSyn(ctx, sy, sn, rnCaSyn, updtThr)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning), based on
// synaptically-integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *Prjn) DWt(ctx *Context, si uint32, sn *Neuron) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	rlay := pj.Recv
	layPool := &rlay.Pools[0]
	isTarget := rlay.Params.Act.Clamp.IsTarget.IsTrue()
	syns := pj.SendSyns(int(si))
	for syi := range syns {
		sy := &syns[syi]
		ri := pj.Params.SynRecvLayIdx(sy)
		rn := &rlay.Neurons[ri]
		subPool := &rlay.Pools[rn.SubPool]
		pj.Params.DWtSyn(ctx, sy, sn, rn, layPool, subPool, isTarget)
	}
}

// DWtSubMean subtracts the mean from any projections that have SubMean > 0.
// This is called on *receiving* projections, prior to WtFmDwt.
func (pj *Prjn) DWtSubMean(ctx *Context, ri uint32, rn *Neuron) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	sm := pj.Params.Learn.Trace.SubMean
	if sm == 0 { // note default is now 0, so don't exclude Target layers, which should be 0
		return
	}
	syIdxs := pj.RecvSynIdxs(int(ri))
	if len(syIdxs) < 1 {
		return
	}
	sumDWt := float32(0)
	nnz := 0 // non-zero
	for _, syi := range syIdxs {
		sy := &pj.Syns[syi]
		dw := sy.DWt
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
		sy := &pj.Syns[syi]
		if sy.DWt != 0 {
			sy.DWt -= sm * sumDWt
		}
	}
}

// WtFmDWt computes the weight change (learning), based on
// synaptically-integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *Prjn) WtFmDWt(ctx *Context, si uint32, sn *Neuron) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	syns := pj.SendSyns(int(si))
	for syi := range syns {
		sy := &syns[syi]
		pj.Params.WtFmDWtSyn(ctx, sy)
	}
}

// SlowAdapt does the slow adaptation: SWt learning and SynScale
func (pj *Prjn) SlowAdapt(ctx *Context) {
	pj.SWtFmWt()
	pj.SynScale()
}

// SWtFmWt updates structural, slowly-adapting SWt value based on
// accumulated DSWt values, which are zero-summed with additional soft bounding
// relative to SWt limits.
func (pj *Prjn) SWtFmWt() {
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
	dvar := pj.Params.SWt.Adapt.DreamVar
	for ri := range rlay.Neurons {
		syIdxs := pj.RecvSynIdxs(ri)
		nCons := len(syIdxs)
		if nCons < 1 {
			continue
		}
		avgDWt := float32(0)
		for _, syi := range syIdxs {
			sy := &pj.Syns[syi]
			if sy.DSWt >= 0 { // softbound for SWt
				sy.DSWt *= (max - sy.SWt)
			} else {
				sy.DSWt *= (sy.SWt - min)
			}
			avgDWt += sy.DSWt
		}
		avgDWt /= float32(nCons)
		avgDWt *= pj.Params.SWt.Adapt.SubMean
		if dvar > 0 {
			for _, syi := range syIdxs {
				sy := &pj.Syns[syi]
				sy.SWt += lr * (sy.DSWt - avgDWt)
				sy.DSWt = 0
				if sy.Wt == 0 { // restore failed wts
					sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
				}
				sy.LWt = pj.Params.SWt.LWtFmWts(sy.Wt, sy.SWt) // + pj.Params.SWt.Adapt.RndVar()
				sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
			}
		} else {
			for _, syi := range syIdxs {
				sy := &pj.Syns[syi]
				sy.SWt += lr * (sy.DSWt - avgDWt)
				sy.DSWt = 0
				if sy.Wt == 0 { // restore failed wts
					sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
				}
				sy.LWt = pj.Params.SWt.LWtFmWts(sy.Wt, sy.SWt)
				sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
			}
		}
	}
}

// SynScale performs synaptic scaling based on running average activation vs. targets.
// Layer-level AvgDifFmTrgAvg function must be called first.
func (pj *Prjn) SynScale() {
	if pj.Params.Learn.Learn.IsFalse() || pj.Params.IsInhib() {
		return
	}
	rlay := pj.Recv
	if !rlay.Params.IsLearnTrgAvg() {
		return
	}
	tp := &rlay.Params.Learn.TrgAvgAct
	lr := tp.SynScaleRate
	for ri := range rlay.Neurons {
		nrn := &rlay.Neurons[ri]
		if nrn.IsOff() {
			continue
		}
		adif := -lr * nrn.AvgDif
		syIdxs := pj.RecvSynIdxs(ri)
		for _, syi := range syIdxs {
			sy := &pj.Syns[syi]
			if adif >= 0 { // key to have soft bounding on lwt here!
				sy.LWt += (1 - sy.LWt) * adif * sy.SWt
			} else {
				sy.LWt += sy.LWt * adif * sy.SWt
			}
			sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
		}
	}
}

// SynFail updates synaptic weight failure only -- normally done as part of DWt
// and WtFmDWt, but this call can be used during testing to update failing synapses.
func (pj *Prjn) SynFail(ctx *Context) {
	slay := pj.Send
	for si := range slay.Neurons {
		syns := pj.SendSyns(si)
		for syi := range syns {
			sy := &syns[syi]
			if sy.Wt == 0 { // restore failed wts
				sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
			}
			pj.Params.Com.Fail(ctx, &sy.Wt, sy.SWt)
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
