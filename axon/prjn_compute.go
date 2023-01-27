// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

// prjn_compute.go has the core computational methods, for the CPU.
// On GPU, this same functionality is implemented in corresponding gpu_*.hlsl
// files, which correspond to different shaders for each different function.

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// RecvSpikes receives spikes from the sending neurons at index sendIdx
// into the GBuf buffer on the receiver side. The buffer on the receiver side
// is a ring buffer, which is used for modelling the time delay between
// sending and receiving spikes.
func (pj *Prjn) RecvSpikes(ctx *Context, recvIdx int) {
	if PrjnTypes(pj.Typ) == CTCtxtPrjn { // skip regular
		return
	}
	slay := pj.Send.(AxonLayer).AsAxon()
	scale := pj.Params.GScale.Scale
	pjcom := &pj.Params.Com
	wrOff := pjcom.WriteOff(ctx.CycleTot - 1) // note: -1 because this is logically done on prior timestep
	syns := pj.RecvSyns(recvIdx)
	for ci := range syns {
		sy := &syns[ci]
		sendIdx := pj.Params.SynSendLayIdx(sy)
		sn := &slay.Neurons[sendIdx]
		sv := sn.Spike * scale * sy.Wt
		pj.GBuf[pjcom.WriteIdx(uint32(recvIdx), wrOff)] += sv
	}
}

// SendSpike sends a spike from the sending neuron at index sendIdx
// into the GBuf buffer on the receiver side. The buffer on the receiver side
// is a ring buffer, which is used for modelling the time delay between
// sending and receiving spikes.
func (pj *Prjn) SendSpike(ctx *Context, sendIdx int) {
	if PrjnTypes(pj.Typ) == CTCtxtPrjn { // skip regular
		return
	}
	scale := pj.Params.GScale.Scale
	pjcom := &pj.Params.Com
	wrOff := pjcom.WriteOff(ctx.CycleTot)
	sidxs := pj.SendSynIdxs(sendIdx)
	for _, ssi := range sidxs {
		sy := &pj.Syns[ssi]
		recvIdx := pj.Params.SynRecvLayIdx(sy)
		sv := scale * sy.Wt
		bi := pjcom.WriteIdx(recvIdx, wrOff)
		pj.GBuf[bi] += sv
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  SynCa methods

// SendSynCa updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking.
// This pass goes through in sending order, filtering on sending spike.
// Threading: Can be called concurrently for all prjns, since it updates synapses
// (which are local to a single prjn).
func (pj *Prjn) SendSynCa(ctx *Context) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	kp := &pj.Params.Learn.KinaseCa
	updtThr := kp.UpdtThr
	slay := pj.Send.(AxonLayer).AsAxon()
	rlay := pj.Recv.(AxonLayer).AsAxon()
	ssg := kp.SpikeG * slay.Params.Learn.CaSpk.SynSpkG
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		if sn.Spike == 0 {
			continue
		}
		if sn.CaSpkP < updtThr && sn.CaSpkD < updtThr {
			continue
		}
		snCaSyn := ssg * sn.CaSyn
		sidxs := pj.SendSynIdxs(si)
		for _, ssi := range sidxs {
			sy := &pj.Syns[ssi]
			ri := pj.Params.SynRecvLayIdx(sy)
			rn := &rlay.Neurons[ri]
			pj.Params.SendSynCaSyn(ctx, sy, rn, snCaSyn, updtThr)
		}
	}
}

// RecvSynCa updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking.
// This pass goes through in recv order, filtering on recv spike.
// Threading: Can be called concurrently for all prjns, since it updates synapses
// (which are local to a single prjn).
func (pj *Prjn) RecvSynCa(ctx *Context) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	kp := &pj.Params.Learn.KinaseCa
	updtThr := kp.UpdtThr
	slay := pj.Send.(AxonLayer).AsAxon()
	rlay := pj.Recv.(AxonLayer).AsAxon()
	ssg := kp.SpikeG * slay.Params.Learn.CaSpk.SynSpkG
	for ri := range rlay.Neurons {
		rn := &rlay.Neurons[ri]
		if rn.Spike == 0 {
			continue
		}
		if rn.CaSpkP < updtThr && rn.CaSpkD < updtThr {
			continue
		}
		rnCaSyn := ssg * rn.CaSyn
		syns := pj.RecvSyns(ri)
		for ci := range syns {
			sy := &syns[ci]
			si := pj.Params.SynSendLayIdx(sy)
			sn := &slay.Neurons[si]
			pj.Params.RecvSynCaSyn(ctx, sy, sn, rnCaSyn, updtThr)
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning), based on
// synaptically-integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *Prjn) DWt(ctx *Context) {
	if pj.Params.Learn.Learn.IsFalse() {
		return
	}
	slay := pj.Send.(AxonLayer).AsAxon()
	rlay := pj.Recv.(AxonLayer).AsAxon()
	layPool := &rlay.Pools[0]
	isTarget := rlay.Params.Act.Clamp.IsTarget.IsTrue()
	for ri := range rlay.Neurons {
		rn := &rlay.Neurons[ri]
		// note: UpdtThr doesn't make sense here b/c Tr needs to be updated
		syns := pj.RecvSyns(ri)
		for ci := range syns {
			sy := &syns[ci]
			si := pj.Params.SynSendLayIdx(sy)
			sn := &slay.Neurons[si]
			subPool := &rlay.Pools[rn.SubPool]
			pj.Params.DWtSyn(ctx, sy, sn, rn, layPool, subPool, isTarget)
		}
	}
}

// DWtSubMean subtracts the mean from any projections that have SubMean > 0.
// This is called on *receiving* projections, prior to WtFmDwt.
func (pj *Prjn) DWtSubMean(ctx *Context) {
	rlay := pj.Recv.(AxonLayer).AsAxon()
	sm := pj.Params.Learn.Trace.SubMean
	if sm == 0 { // || rlay.AxonLay.IsTarget() { // sm default is now 0, so don't exclude
		return
	}
	for ri := range rlay.Neurons {
		syns := pj.RecvSyns(ri)
		if len(syns) < 1 {
			continue
		}
		sumDWt := float32(0)
		nnz := 0 // non-zero
		for ci := range syns {
			sy := &syns[ci]
			dw := sy.DWt
			if dw != 0 {
				sumDWt += dw
				nnz++
			}
		}
		if nnz <= 1 {
			continue
		}
		sumDWt /= float32(nnz)
		for ci := range syns {
			sy := &syns[ci]
			if sy.DWt != 0 {
				sy.DWt -= sm * sumDWt
			}
		}
	}
}

// WtFmDWt updates the synaptic weight values from delta-weight changes.
// called on the *receiving* projections.
func (pj *Prjn) WtFmDWt(ctx *Context) {
	rlay := pj.Recv.(AxonLayer).AsAxon()
	for ri := range rlay.Neurons {
		syns := pj.RecvSyns(ri)
		for ci := range syns {
			sy := &syns[ci]
			pj.Params.WtFmDWtSyn(ctx, sy)
		}
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
	rlay := pj.Recv.(AxonLayer).AsAxon()
	if rlay.AxonLay.IsTarget() {
		return
	}
	max := pj.Params.SWt.Limit.Max
	min := pj.Params.SWt.Limit.Min
	lr := pj.Params.SWt.Adapt.LRate
	dvar := pj.Params.SWt.Adapt.DreamVar
	for ri := range rlay.Neurons {
		syns := pj.RecvSyns(ri)
		nCons := len(syns)
		if nCons < 1 {
			continue
		}
		avgDWt := float32(0)
		for ci := range syns {
			sy := &syns[ci]
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
			for ci := range syns {
				sy := &syns[ci]
				sy.SWt += lr * (sy.DSWt - avgDWt)
				sy.DSWt = 0
				if sy.Wt == 0 { // restore failed wts
					sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
				}
				sy.LWt = pj.Params.SWt.LWtFmWts(sy.Wt, sy.SWt) + pj.Params.SWt.Adapt.RndVar()
				sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
			}
		} else {
			for ci := range syns {
				sy := &syns[ci]
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
	rlay := pj.Recv.(AxonLayer).AsAxon()
	if !rlay.IsLearnTrgAvg() {
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
		syns := pj.RecvSyns(ri)
		for ci := range syns {
			sy := &syns[ci]
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
	rlay := pj.Recv.(AxonLayer).AsAxon()
	for ri := range rlay.Neurons {
		syns := pj.RecvSyns(ri)
		for ci := range syns {
			sy := &syns[ci]
			if sy.Wt == 0 { // restore failed wts
				sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
			}
			pj.Params.Com.Fail(&sy.Wt, sy.SWt)
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
