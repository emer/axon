// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"

	"github.com/emer/etable/minmax"
	"github.com/goki/mat32"
)

// layer_compute.go has the core computational methods, for the CPU.
// On GPU, this same functionality is implemented in corresponding gpu_*.hlsl
// files, which correspond to different shaders for each different function.

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GatherSpikes integrates G*Raw and G*Syn values for given recv neuron
// while integrating the Recv Prjn-level GSyn integrated values.
// ni is layer-specific index of neuron within its layer.
func (ly *Layer) GatherSpikes(ctx *Context, ni uint32, nrn *Neuron) {
	ly.Params.GatherSpikesInit(nrn)
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.AsAxon()

		if pj.Params.Com.CPURecvSpikes.IsTrue() { // about 35x slower!
			pj.RecvSpikes(ctx, int(ni)) // Note: iterates over all senders for given recv
		}
		bi := pj.Params.Com.ReadIdx(ni, ctx.CycleTot, pj.Params.Idxs.RecvNeurN)
		gRaw := pj.Params.Com.FloatFromGBuf(pj.GBuf[bi])
		pj.GBuf[bi] = 0
		pj.Params.GatherSpikes(ctx, ly.Params, ni, nrn, gRaw, &pj.GSyns[ni])
	}
}

// GiFmSpikes gets the Spike, GeRaw and GeExt from neurons in the pools
// where Spike drives FBsRaw = raw feedback signal,
// GeRaw drives FFsRaw = aggregate feedforward excitatory spiking input.
// GeExt represents extra excitatory input from other sources.
// Then integrates new inhibitory conductances therefrom,
// at the layer and pool level.
// Called separately by Network.CycleImpl on all Layers
// Also updates all AvgMax values at the Cycle level.
func (ly *Layer) GiFmSpikes(ctx *Context) {
	lpl := &ly.Pools[0]
	np := len(ly.Pools)
	subPools := (np > 1)
	for ni := range ly.Neurons { // note: layer-level iterating across neurons
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pl := &ly.Pools[nrn.SubPool]
		pl.Inhib.RawIncr(nrn.Spike, nrn.GeRaw, nrn.GeExt)
		pl.AvgMax.UpdateVals(nrn)
		if subPools { // update layer too -- otherwise pl == lpl
			lpl.Inhib.RawIncr(nrn.Spike, nrn.GeRaw, nrn.GeExt)
			lpl.AvgMax.UpdateVals(nrn)
		}
	}
	for pi := 0; pi < np; pi++ {
		pl := &ly.Pools[pi]
		pl.AvgMax.Calc()
	}
	ly.Params.LayPoolGiFmSpikes(ctx, lpl, ly.Vals)
	// ly.PoolGiFmSpikes(ctx) // note: this is now called as a second pass
	// so that we can do between-layer inhibition
}

// PoolGiFmSpikes computes inhibition Gi from Spikes within sub-pools.
// and also between different layers based on LayInhib* indexes
// must happen after LayPoolGiFmSpikes has been called.
func (ly *Layer) PoolGiFmSpikes(ctx *Context) {
	ly.BetweenLayerGi(ctx)
	np := len(ly.Pools)
	if np == 1 {
		return
	}
	lpl := &ly.Pools[0]
	lyInhib := ly.Params.Inhib.Layer.On.IsTrue()
	for pi := 1; pi < np; pi++ {
		pl := &ly.Pools[pi]
		ly.Params.SubPoolGiFmSpikes(ctx, pl, lpl, lyInhib, ly.Vals.ActAvg.GiMult)
	}
}

// BetweenLayerGi computes inhibition Gi between layers
func (ly *Layer) BetweenLayerGi(ctx *Context) {
	lpl := &ly.Pools[0]
	maxGi := lpl.Inhib.Gi
	net := ly.Network.(AxonNetwork).AsAxon()
	maxGi = ly.BetweenLayerGiMax(maxGi, net, ly.Params.LayInhib.Idx1)
	maxGi = ly.BetweenLayerGiMax(maxGi, net, ly.Params.LayInhib.Idx2)
	maxGi = ly.BetweenLayerGiMax(maxGi, net, ly.Params.LayInhib.Idx3)
	maxGi = ly.BetweenLayerGiMax(maxGi, net, ly.Params.LayInhib.Idx4)
	lpl.Inhib.Gi = maxGi // our inhib is max of us and everyone in the layer pool
}

// BetweenLayerGiMax returns max gi value for input maxGi vs
// the given layIdx layer
func (ly *Layer) BetweenLayerGiMax(maxGi float32, net *Network, layIdx int32) float32 {
	if layIdx < 0 {
		return maxGi
	}
	lay := net.Layers[layIdx].(AxonLayer).AsAxon()
	lpl := &lay.Pools[0]
	if lpl.Inhib.Gi > maxGi {
		maxGi = lpl.Inhib.Gi
	}
	return maxGi
}

func (ly *Layer) PulvinarDriver(ni uint32) (drvGe, nonDrvPct float32) {
	dly := ly.Network.Layer(int(ly.Params.Pulv.DriveLayIdx)).(AxonLayer).AsAxon()
	drvMax := dly.Pools[0].AvgMax.CaSpkP.Cycle.Max
	nonDrvPct = ly.Params.Pulv.NonDrivePct(drvMax) // how much non-driver to keep
	burst := dly.Neurons[ni].Burst
	drvGe = ly.Params.Pulv.DriveGe(burst)
	return
}

// GInteg integrates conductances G over time (Ge, NMDA, etc).
// calls SpecialGFmRawSyn, GiInteg
func (ly *Layer) GInteg(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, vals *LayerVals) {
	drvGe := float32(0)
	nonDrvPct := float32(0)
	if ly.LayerType() == PulvinarLayer {
		drvGe, nonDrvPct = ly.PulvinarDriver(ni)
	}

	saveVal := ly.Params.SpecialPreGs(ctx, ni, nrn, pl, vals, drvGe, nonDrvPct)

	ly.Params.GFmRawSyn(ctx, ni, nrn)
	ly.Params.GiInteg(ctx, ni, nrn, pl, vals)
	ly.Params.GNeuroMod(ctx, ni, nrn, vals)

	ly.Params.SpecialPostGs(ctx, ni, nrn, saveVal)
}

// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
func (ly *Layer) SpikeFmG(ctx *Context, ni uint32, nrn *Neuron) {
	ly.Params.SpikeFmG(ctx, ni, nrn)
}

// CycleNeuron does one cycle (msec) of updating at the neuron level
func (ly *Layer) CycleNeuron(ctx *Context, ni uint32, nrn *Neuron) {
	ly.AxonLay.GInteg(ctx, ni, nrn, &ly.Pools[nrn.SubPool], ly.Vals)
	ly.AxonLay.SpikeFmG(ctx, ni, nrn)
}

// PostSpike does updates at neuron level after spiking has been computed.
// This is where special layer types add extra code.
// It also updates the CaSpkPCyc stats.
func (ly *Layer) PostSpike(ctx *Context, ni uint32, nrn *Neuron) {
	ly.Params.PostSpikeSpecial(ctx, ni, nrn, &ly.Pools[nrn.SubPool], &ly.Pools[0], ly.Vals)
	ly.Params.PostSpike(ctx, ni, nrn, &ly.Pools[nrn.SubPool], ly.Vals)
}

// SendSpike sends spike to receivers for all neurons that spiked
// last step in Cycle, integrated the next time around.
func (ly *Layer) SendSpike(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.AxonLay.PostSpike(ctx, uint32(ni), nrn)
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			sp.SendSpike(ctx, ni, nrn)
		}
	}
}

// SynCaSend updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking, threaded over neurons.
// This pass updates sending projections -- all sending synapses are
// unique to a given sending neuron, so this is threadsafe.
// Cannot do both send and recv in same pass without potential for
// race conditions.
func (ly *Layer) SynCaSend(ctx *Context, ni uint32, sn *Neuron) {
	if sn.Spike == 0 {
		return
	}
	updtThr := ly.Params.Learn.CaLrn.UpdtThr
	if sn.CaSpkP < updtThr && sn.CaSpkD < updtThr {
		return
	}
	for _, sp := range ly.SndPrjns {
		if sp.IsOff() {
			continue
		}
		sp.SynCaSend(ctx, ni, sn, updtThr)
	}
}

// SynCaRecv updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking, threaded over neurons.
// This pass updates recv projections -- all recv synapses are
// unique to a given recv neuron, so this is threadsafe.
// Cannot do both send and recv in same pass without potential for
// race conditions.
func (ly *Layer) SynCaRecv(ctx *Context, ni uint32, rn *Neuron) {
	if rn.Spike == 0 {
		return
	}
	updtThr := ly.Params.Learn.CaLrn.UpdtThr
	if rn.CaSpkP < updtThr && rn.CaSpkD < updtThr {
		return
	}
	for _, rp := range ly.RcvPrjns {
		if rp.IsOff() {
			continue
		}
		rp.SynCaRecv(ctx, ni, rn, updtThr)
	}
}

// RSalAChLayMaxAct returns the lpl.AvgMax.Act.Cycle.Max for given layIdx
func (ly *Layer) RSalAChLayMaxAct(net *Network, layIdx int32) float32 {
	if layIdx < 0 {
		return 0
	}
	lay := net.Layers[layIdx].(AxonLayer).AsAxon()
	lpl := &lay.Pools[0]
	return lpl.AvgMax.Act.Cycle.Max
}

// CyclePost is called after the standard Cycle update, as a separate
// network layer loop.
// This is reserved for any kind of special ad-hoc types that
// need to do something special after Spiking is finally computed and Sent.
// It ONLY runs on the CPU, not the GPU -- should update global values
// in the Context state which are re-sync'd back to GPU,
// and values in other layers MUST come from LayerVals because
// this is the only data that is sync'd back from the GPU each cycle.
// For example, updating a neuromodulatory signal such as dopamine.
func (ly *Layer) CyclePost(ctx *Context) {
	switch ly.LayerType() {
	case RSalienceAChLayer:
		net := ly.Network.(AxonNetwork).AsAxon()
		lay1MaxAct := ly.RSalAChLayMaxAct(net, ly.Params.RSalACh.SrcLay1Idx)
		lay2MaxAct := ly.RSalAChLayMaxAct(net, ly.Params.RSalACh.SrcLay2Idx)
		lay3MaxAct := ly.RSalAChLayMaxAct(net, ly.Params.RSalACh.SrcLay3Idx)
		lay4MaxAct := ly.RSalAChLayMaxAct(net, ly.Params.RSalACh.SrcLay4Idx)
		ly.Params.CyclePostRSalAChLayer(ctx, ly.Vals, lay1MaxAct, lay2MaxAct, lay3MaxAct, lay4MaxAct)
	case RWDaLayer:
		net := ly.Network.(AxonNetwork).AsAxon()
		pvals := &net.LayVals[ly.Params.RWDa.RWPredLayIdx]
		ly.Params.CyclePostRWDaLayer(ctx, ly.Vals, pvals)
	case TDPredLayer:
		ly.Params.CyclePostTDPredLayer(ctx, ly.Vals)
	case TDIntegLayer:
		net := ly.Network.(AxonNetwork).AsAxon()
		pvals := &net.LayVals[ly.Params.TDInteg.TDPredLayIdx]
		ly.Params.CyclePostTDIntegLayer(ctx, ly.Vals, pvals)
	case TDDaLayer:
		net := ly.Network.(AxonNetwork).AsAxon()
		ivals := &net.LayVals[ly.Params.TDDa.TDIntegLayIdx]
		ly.Params.CyclePostTDDaLayer(ctx, ly.Vals, ivals)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Phase-level

// NewState handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
// Does NOT call InitGScale()
func (ly *Layer) NewState(ctx *Context) {
	lpl := &ly.Pools[0]
	ly.Params.NewStateLayer(ctx, lpl, ly.Vals)

	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		ly.Params.NewStatePool(ctx, pl) // also calls DecayState on pool
	}

	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		// note: this calls the basic neuron-level DecayState
		ly.Params.NewStateNeuron(ctx, uint32(ni), nrn, ly.Vals)
	}
	// if ly.Params.Act.Decay.Glong != 0 { // clear pipeline of incoming spikes, assuming time has passed
	// always safer to do this rather than not -- sometimes layer has specifically cleared
	ly.InitPrjnGBuffs()
	// }
}

// DecayState decays activation state by given proportion
// (default decay values are ly.Params.Act.Decay.Act, Glong)
func (ly *Layer) DecayState(ctx *Context, decay, glong float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.Act.DecayState(nrn, decay, glong)
		// ly.Params.Learn.DecayCaLrnSpk(nrn, glong) // NOT called by default
		// Note: synapse-level Ca decay happens in DWt
	}
	ly.DecayStateLayer(ctx, decay, glong)
}

// DecayStateLayer does layer-level decay, but not neuron level
func (ly *Layer) DecayStateLayer(ctx *Context, decay, glong float32) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Decay(decay)
	}
	if glong != 0 { // clear pipeline of incoming spikes, assuming time has passed
		ly.InitPrjnGBuffs()
	}
}

// DecayCaLrnSpk decays neuron-level calcium learning and spiking variables
// by given factor, which is typically ly.Params.Act.Decay.Glong.
// Note: this is NOT called by default and is generally
// not useful, causing variability in these learning factors as a function
// of the decay parameter that then has impacts on learning rates etc.
// It is only here for reference or optional testing.
func (ly *Layer) DecayCaLrnSpk(decay float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.Learn.DecayCaLrnSpk(nrn, decay)
	}
}

// DecayStatePool decays activation state by given proportion in given sub-pool index (0 based)
func (ly *Layer) DecayStatePool(pool int, decay, glong float32) {
	pi := int32(pool + 1) // 1 based
	pl := &ly.Pools[pi]
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.Act.DecayState(nrn, decay, glong)
	}
	pl.Inhib.Decay(decay)
}

// AvgMaxVarByPool returns the average and maximum value of given variable
// for given pool index (0 = entire layer, 1.. are subpools for 4D only).
// Uses fast index-based variable access.
func (ly *Layer) AvgMaxVarByPool(varNm string, poolIdx int) minmax.AvgMax32 {
	var am minmax.AvgMax32
	vidx, err := ly.AxonLay.UnitVarIdx(varNm)
	if err != nil {
		log.Printf("axon.Layer.AvgMaxVar: %s\n", err)
		return am
	}
	pl := &ly.Pools[poolIdx]
	am.Init()
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		vl := ly.UnitVal1D(vidx, int(ni))
		am.UpdateVal(vl, int32(ni))
	}
	am.CalcAvg()
	return am
}

// MinusPhase does updating at end of the minus phase
func (ly *Layer) MinusPhase(ctx *Context) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		ly.Params.MinusPhasePool(ctx, pl)
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.MinusPhaseNeuron(ctx, uint32(ni), nrn, &ly.Pools[nrn.SubPool], &ly.Pools[0], ly.Vals)
	}
	ly.Params.AvgGeM(ctx, &ly.Pools[0], ly.Vals)
}

// MinusPhasePost does special algorithm processing at end of minus
func (ly *Layer) MinusPhasePost(ctx *Context) {
	switch ly.LayerType() {
	case MatrixLayer:
		ly.MatrixGated(ctx) // need gated state for decisions about action processing, so do in minus too
	}
}

// PlusPhaseStart does updating at the start of the plus phase:
// applies Target inputs as External inputs.
func (ly *Layer) PlusPhaseStart(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.PlusPhaseStartNeuron(ctx, uint32(ni), nrn, &ly.Pools[nrn.SubPool], &ly.Pools[0], ly.Vals)
	}
}

// PlusPhase does updating at end of the plus phase
func (ly *Layer) PlusPhase(ctx *Context) {
	// todo: see if it is faster to just grab pool info now, then do everything below on CPU
	for pi := range ly.Pools { // gpu_cycletoplus
		pl := &ly.Pools[pi]
		ly.Params.PlusPhasePool(ctx, pl)
	}
	for ni := range ly.Neurons { // gpu_plusphase
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pl := &ly.Pools[nrn.SubPool]
		lpl := &ly.Pools[0]
		ly.Params.PlusPhaseNeuron(ctx, uint32(ni), nrn, pl, lpl, ly.Vals)
	}
}

// PlusPhasePost does special algorithm processing at end of plus
func (ly *Layer) PlusPhasePost(ctx *Context) {
	ly.TrgAvgFmD()
	ly.AxonLay.CorSimFmActs() // GPU syncs down the state
	if ly.Params.Act.Decay.OnRew.IsTrue() {
		if ctx.NeuroMod.HasRew.IsTrue() || ctx.DrivePVLV.LHb.DipReset.IsTrue() {
			ly.DecayState(ctx, 1, 1) // note: GPU will get, and GBuf are auto-cleared in NewState
		}
	}
	switch ly.LayerType() {
	case MatrixLayer:
		ly.MatrixGated(ctx)
	}
}

// TargToExt sets external input Ext from target values Target
// This is done at end of MinusPhase to allow targets to drive activity in plus phase.
// This can be called separately to simulate alpha cycles within theta cycles, for example.
func (ly *Layer) TargToExt() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.HasFlag(NeuronHasTarg) { // will be clamped in plus phase
			nrn.Ext = nrn.Target
			nrn.SetFlag(NeuronHasExt)
			nrn.ISI = -1 // get fresh update on plus phase output acts
			nrn.ISIAvg = -1
		}
	}
}

// ClearTargExt clears external inputs Ext that were set from target values Target.
// This can be called to simulate alpha cycles within theta cycles, for example.
func (ly *Layer) ClearTargExt() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.HasFlag(NeuronHasTarg) { // will be clamped in plus phase
			nrn.Ext = 0
			nrn.ClearFlag(NeuronHasExt)
			nrn.ISI = -1 // get fresh update on plus phase output acts
			nrn.ISIAvg = -1
		}
	}
}

// SpkSt1 saves current activation state in SpkSt1 variables (using CaP)
func (ly *Layer) SpkSt1(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.SpkSt1 = nrn.CaSpkP
	}
}

// SpkSt2 saves current activation state in SpkSt2 variables (using CaP)
func (ly *Layer) SpkSt2(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.SpkSt2 = nrn.CaSpkP
	}
}

// CorSimFmActs computes the correlation similarity
// (centered cosine aka normalized dot product)
// in activation state between minus and plus phases.
func (ly *Layer) CorSimFmActs() {
	lpl := &ly.Pools[0]
	avgM := lpl.AvgMax.Act.Minus.Avg
	avgP := lpl.AvgMax.Act.Plus.Avg
	cosv := float32(0)
	ssm := float32(0)
	ssp := float32(0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ap := nrn.ActP - avgP // zero mean = correl
		am := nrn.ActM - avgM
		cosv += ap * am
		ssm += am * am
		ssp += ap * ap
	}

	dist := mat32.Sqrt(ssm * ssp)
	if dist != 0 {
		cosv /= dist
	}
	ly.Vals.CorSim.Cor = cosv

	ly.Params.Act.Dt.AvgVarUpdt(&ly.Vals.CorSim.Avg, &ly.Vals.CorSim.Var, ly.Vals.CorSim.Cor)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learning

// DTrgSubMean subtracts the mean from DTrgAvg values
// Called by TrgAvgFmD
func (ly *Layer) DTrgSubMean() {
	submean := ly.Params.Learn.TrgAvgAct.SubMean
	if submean == 0 {
		return
	}
	if ly.HasPoolInhib() && ly.Params.Learn.TrgAvgAct.Pool.IsTrue() {
		np := len(ly.Pools)
		for pi := 1; pi < np; pi++ {
			pl := &ly.Pools[pi]
			nn := 0
			avg := float32(0)
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				if nrn.IsOff() {
					continue
				}
				avg += nrn.DTrgAvg
				nn++
			}
			if nn == 0 {
				continue
			}
			avg /= float32(nn)
			avg *= submean
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				if nrn.IsOff() {
					continue
				}
				nrn.DTrgAvg -= avg
			}
		}
	} else {
		nn := 0
		avg := float32(0)
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			avg += nrn.DTrgAvg
			nn++
		}
		if nn == 0 {
			return
		}
		avg /= float32(nn)
		avg *= submean
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			nrn.DTrgAvg -= avg
		}
	}
}

// TrgAvgFmD updates TrgAvg from DTrgAvg -- called in PlusPhasePost
func (ly *Layer) TrgAvgFmD() {
	lr := ly.Params.LearnTrgAvgErrLRate()
	if lr == 0 {
		return
	}
	ly.DTrgSubMean()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.TrgAvg = ly.Params.Learn.TrgAvgAct.TrgRange.ClipVal(nrn.TrgAvg + nrn.DTrgAvg)
		nrn.DTrgAvg = 0
	}
}

// WtFmDWtLayer does weight update at the layer level.
// does NOT call main projection-level WtFmDWt method.
// in base, only calls TrgAvgFmD
func (ly *Layer) WtFmDWtLayer(ctx *Context) {
	ly.TrgAvgFmD()
}

// SlowAdapt is the layer-level slow adaptation functions.
// Calls AdaptInhib and AvgDifFmTrgAvg for Synaptic Scaling.
// Does NOT call projection-level methods.
func (ly *Layer) SlowAdapt(ctx *Context) {
	ly.AdaptInhib(ctx)
	ly.AvgDifFmTrgAvg()
	// note: prjn level call happens at network level
}

// AdaptInhib adapts inhibition
func (ly *Layer) AdaptInhib(ctx *Context) {
	if ly.Params.Inhib.ActAvg.AdaptGi.IsFalse() || ly.Params.IsInput() {
		return
	}
	ly.Params.Inhib.ActAvg.Adapt(&ly.Vals.ActAvg.GiMult, ly.Vals.ActAvg.ActMAvg)
}

// AvgDifFmTrgAvg updates neuron-level AvgDif values from AvgPct - TrgAvg
// which is then used for synaptic scaling of LWt values in Prjn SynScale.
func (ly *Layer) AvgDifFmTrgAvg() {
	sp := 0
	if len(ly.Pools) > 1 {
		sp = 1
	}
	np := len(ly.Pools)
	for pi := sp; pi < np; pi++ {
		pl := &ly.Pools[pi]
		plavg := float32(0)
		nn := 0
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			plavg += nrn.ActAvg
			nn++
		}
		if nn == 0 {
			continue
		}
		plavg /= float32(nn)
		pl.AvgDif.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			nrn.AvgPct = nrn.ActAvg / plavg
			nrn.AvgDif = nrn.AvgPct - nrn.TrgAvg
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif))
		}
		pl.AvgDif.Calc()
	}
	if sp == 1 { // update stats
		pl := &ly.Pools[0]
		pl.AvgDif.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif))
		}
		pl.AvgDif.Calc()
	}
}

// SynFail updates synaptic weight failure only -- normally done as part of DWt
// and WtFmDWt, but this call can be used during testing to update failing synapses.
func (ly *Layer) SynFail(ctx *Context) {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(AxonPrjn).SynFail(ctx)
	}
}

// LRateMod sets the LRate modulation parameter for Prjns, which is
// for dynamic modulation of learning rate (see also LRateSched).
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LRateMod(mod float32) {
	for _, p := range ly.RcvPrjns {
		// if p.IsOff() { // keep all sync'd
		// 	continue
		// }
		p.(AxonPrjn).AsAxon().LRateMod(mod)
	}
}

// LRateSched sets the schedule-based learning rate multiplier.
// See also LRateMod.
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LRateSched(sched float32) {
	for _, p := range ly.RcvPrjns {
		// if p.IsOff() { // keep all sync'd
		// 	continue
		// }
		p.(AxonPrjn).AsAxon().LRateSched(sched)
	}
}

// SetSubMean sets the SubMean parameters in all the layers in the network
// trgAvg is for Learn.TrgAvgAct.SubMean
// prjn is for the prjns Learn.Trace.SubMean
// in both cases, it is generally best to have both parameters set to 0
// at the start of learning
func (ly *Layer) SetSubMean(trgAvg, prjn float32) {
	ly.Params.Learn.TrgAvgAct.SubMean = trgAvg
	for _, p := range ly.RcvPrjns {
		// if p.IsOff() { // keep all sync'd
		// 	continue
		// }
		p.(AxonPrjn).AsAxon().Params.Learn.Trace.SubMean = prjn
	}
}
