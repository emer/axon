// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

// prjn_compute.go has the core computational methods, which are also called by GPU

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// RecvSpikes increments synaptic conductances from Spikes,
// using a receiver-based computation that can be parallelized across receivers
// including pooled aggregation of spikes into Pools for FS-FFFB inhib.
func (pj *Prjn) RecvSpikes(ctx *Context, ri uint32, rn *Neuron) {
	/*
		if PrjnTypes(pj.Typ) == CTCtxtPrjn { // skip regular
			return
		}
		rlay := pj.Recv.(AxonLayer).AsAxon()
		scale := pj.Params.GScale.Scale
		maxDelay := pj.Params.Com.Delay
		delLen := pj.Params.Com.DelLen
		wrIdx := pj.Params.Com.WriteIdx(ctx.CycleTot)
		numCons := pj.RecvConN[ri]
		startIdx := pj.SendConIdxStart[sendIdx]
		syns := pj.Syns[startIdx : startIdx+numCons] // Get slice of synapses for current neuron
		synConIdxs := pj.SendConIdx[startIdx : startIdx+numCons]
		excite := pj.Params.IsExcitatory()
		zi := pj.Vals.Gidx.Zi
		for i := range syns {
			recvIdx := synConIdxs[i]
			rnPool := rlay.Neurons[recvIdx].SubPool
			sv := scale * syns[i].Wt
			pj.GBuf[recvIdx*delayBufSize+currDelayIdx] += sv
			if excite { // only excitatory drives inhibition
				pj.PIBuf[rnPool*delayBufSize+currDelayIdx] += sv
			}
		}

		if pj.Params.IsInhib() {
			for ri := range pj.GVals {
				gv := &pj.GVals[ri]
				bi := uint32(ri)*sz + zi
				gv.GRaw = pj.GBuf[bi]
				pj.GBuf[bi] = 0
				gv.GSyn = rlay.Params.Act.Dt.GiSynFmRaw(gv.GSyn, gv.GRaw)
			}
			pj.Vals.Gidx.Shift(1) // rotate buffer
			return
		}
		// TODO: Race condition if one layer has multiple incoming prjns (common)
		lpl := &rlay.Pools[0]
		if len(rlay.Pools) == 1 {
			lpl.Inhib.FFsRaw += pj.PIBuf[zi]
			pj.PIBuf[zi] = 0
		} else {
			for pi := range rlay.Pools {
				pl := &rlay.Pools[pi]
				bi := uint32(pi)*sz + zi
				sv := pj.PIBuf[bi]
				pl.Inhib.FFsRaw += sv
				lpl.Inhib.FFsRaw += sv
				pj.PIBuf[bi] = 0
			}
		}
		for ri := range pj.GVals {
			gv := &pj.GVals[ri]
			bi := uint32(ri)*sz + zi
			gv.GRaw = pj.GBuf[bi]
			pj.GBuf[bi] = 0
			gv.GSyn = rlay.Params.Act.Dt.GeSynFmRaw(gv.GSyn, gv.GRaw)
		}
		pj.Vals.Gidx.Shift(1) // rotate buffer
	*/
}

// SendSpike sends a spike from the sending neuron at index sendIdx
// into the GBuf buffer on the receiver side. The buffer on the receiver side
// is a ring buffer, which is used for modelling the time delay between
// sending and receiving spikes.
func (pj *Prjn) SendSpike(ctx *Context, sendIdx int) {
	if PrjnTypes(pj.Typ) == CTCtxtPrjn { // skip regular
		return
	}
	rlay := pj.Recv.(AxonLayer).AsAxon()
	scale := pj.Params.GScale.Scale
	del := pj.Params.Com.Delay
	delLen := pj.Params.Com.DelLen
	wrIdx := pj.Params.Com.WriteIdx(ctx.CycleTot)
	numCons := pj.SendConN[sendIdx]
	startIdx := pj.SendConIdxStart[sendIdx]
	ssidxs := pj.SendSynIdx[startIdx : startIdx+numCons]
	sconIdxs := pj.SendConIdx[startIdx : startIdx+numCons]
	excite := pj.Params.IsExcitatory()
	for i, ssi := range ssidxs {
		sy := &pj.Syns[ssi]
		recvIdx := sconIdxs[i]
		rnPool := rlay.Neurons[recvIdx].SubPool
		sv := scale * sy.Wt
		// TODO: race condition, multiple threads will write into the same recv neuron buffer
		// and spikes will get lost. Could use atomic, but atomics are expensive and scale poorly
		// better to re-write as matmul, or to re-write from recv neuron side.

		// Note: from the recv side, you can't leverage the sparse sending activity:
		// most neurons are not spiking at any given moment,
		// but it can be much more parallel, so maybe it is worth it on the GPU.

		pj.GBuf[recvIdx*delLen+wrIdx] += sv
		if excite { // only excitatory drives inhibition
			pj.PIBuf[rnPool*delLen+wrIdx] += sv
		}
	}
}

// InhibGatherSpikes increments synaptic conductances from Spikes
// for pooled aggregation of spikes into Pools for FS-FFFB inhib.
func (pj *Prjn) InhibGatherSpikes(ctx *Context) {
	lpl := &rlay.Pools[0]
	if len(rlay.Pools) == 1 {
		lpl.Inhib.FFsRaw += pj.PIBuf[zi]
		pj.PIBuf[zi] = 0
	} else {
		for pi := range rlay.Pools {
			pl := &rlay.Pools[pi]
			bi := uint32(pi)*sz + zi
			sv := pj.PIBuf[bi]
			pl.Inhib.FFsRaw += sv
			lpl.Inhib.FFsRaw += sv
			pj.PIBuf[bi] = 0
		}
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
		nc := int(pj.SendConN[si])
		st := int(pj.SendConIdxStart[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SendConIdx[st : st+nc]
		for ci := range syns {
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			sy := &syns[ci]
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
		nc := int(pj.RecvConN[ri])
		st := int(pj.RecvConIdxStart[ri])
		rsidxs := pj.RecvSynIdx[st : st+nc]
		rcons := pj.RecvConIdx[st : st+nc]
		for ci, rsi := range rsidxs {
			si := rcons[ci]
			sn := &slay.Neurons[si]
			sy := &pj.Syns[rsi]
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
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		// note: UpdtThr doesn't make sense here b/c Tr needs to be updated
		nc := int(pj.SendConN[si])
		st := int(pj.SendConIdxStart[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SendConIdx[st : st+nc]
		for ci := range syns {
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			subPool := &rlay.Pools[rn.SubPool]
			sy := &syns[ci]
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
		nc := int(pj.RecvConN[ri])
		if nc < 1 {
			continue
		}
		st := int(pj.RecvConIdxStart[ri])
		rsidxs := pj.RecvSynIdx[st : st+nc]
		sumDWt := float32(0)
		nnz := 0 // non-zero
		for _, rsi := range rsidxs {
			dw := pj.Syns[rsi].DWt
			if dw != 0 {
				sumDWt += dw
				nnz++
			}
		}
		if nnz <= 1 {
			continue
		}
		sumDWt /= float32(nnz)
		for _, rsi := range rsidxs {
			sy := &pj.Syns[rsi]
			if sy.DWt != 0 {
				sy.DWt -= sm * sumDWt
			}
		}
	}
}

// WtFmDWt updates the synaptic weight values from delta-weight changes.
// called on the *sending* projections.
func (pj *Prjn) WtFmDWt(ctx *Context) {
	slay := pj.Send.(AxonLayer).AsAxon()
	for si := range slay.Neurons {
		nc := int(pj.SendConN[si])
		st := int(pj.SendConIdxStart[si])
		syns := pj.Syns[st : st+nc]
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
		nc := int(pj.RecvConN[ri])
		if nc < 1 {
			continue
		}
		st := int(pj.RecvConIdxStart[ri])
		rsidxs := pj.RecvSynIdx[st : st+nc]
		avgDWt := float32(0)
		for _, rsi := range rsidxs {
			sy := &pj.Syns[rsi]
			if sy.DSWt >= 0 { // softbound for SWt
				sy.DSWt *= (max - sy.SWt)
			} else {
				sy.DSWt *= (sy.SWt - min)
			}
			avgDWt += sy.DSWt
		}
		avgDWt /= float32(nc)
		avgDWt *= pj.Params.SWt.Adapt.SubMean
		if dvar > 0 {
			for _, rsi := range rsidxs {
				sy := &pj.Syns[rsi]
				sy.SWt += lr * (sy.DSWt - avgDWt)
				sy.DSWt = 0
				if sy.Wt == 0 { // restore failed wts
					sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
				}
				sy.LWt = pj.Params.SWt.LWtFmWts(sy.Wt, sy.SWt) + pj.Params.SWt.Adapt.RndVar()
				sy.Wt = pj.Params.SWt.WtVal(sy.SWt, sy.LWt)
			}
		} else {
			for _, rsi := range rsidxs {
				sy := &pj.Syns[rsi]
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
		nc := int(pj.RecvConN[ri])
		st := int(pj.RecvConIdxStart[ri])
		rsidxs := pj.RecvSynIdx[st : st+nc]
		for _, rsi := range rsidxs {
			sy := &pj.Syns[rsi]
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
	slay := pj.Send.(AxonLayer).AsAxon()
	for si := range slay.Neurons {
		nc := int(pj.SendConN[si])
		st := int(pj.SendConIdxStart[si])
		syns := pj.Syns[st : st+nc]
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
