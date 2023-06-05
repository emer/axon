// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"log"

	"github.com/emer/etable/minmax"
	"github.com/goki/mat32"
)

// index naming:
// lni = layer-based neuron index (0 = first neuron in layer)
// ni  = absolute network-level neuron index

// layer_compute.go has the core computational methods, for the CPU.
// On GPU, this same functionality is implemented in corresponding gpu_*.hlsl
// files, which correspond to different shaders for each different function.

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GatherSpikes integrates G*Raw and G*Syn values for given recv neuron
// while integrating the Recv Prjn-level GSyn integrated values.
func (ly *Layer) GatherSpikes(ctx *Context, ni uint32) {
	lni := ni - ly.NeurStIdx
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		ly.Params.GatherSpikesInit(ctx, ni, di)
		for _, pj := range ly.RcvPrjns {
			if pj.IsOff() {
				continue
			}
			bi := pj.Params.Com.ReadIdx(lni, di, ctx.CyclesTotal, pj.Params.Idxs.RecvNeurN, ly.MaxData)
			gRaw := pj.Params.Com.FloatFromGBuf(pj.GBuf[bi])
			pj.GBuf[bi] = 0
			gsi := lni*ly.MaxData + di
			pj.Params.GatherSpikes(ctx, ly.Params, ni, di, gRaw, &pj.GSyns[gsi])
		}
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
	np := ly.NPools
	hasSubPools := (np > 1)
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		subPool := NrnI(ctx, ni, NrnSubPool)
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			pl := ly.Pool(subPool, di)
			pl.Inhib.RawIncr(NrnV(ctx, ni, di, Spike), NrnV(ctx, ni, di, GeRaw), NrnV(ctx, ni, di, GeExt))
			pl.AvgMaxUpdate(ctx, ni, di)
			if hasSubPools { // update layer too -- otherwise pl == lpl
				lpl := ly.Pool(0, di)
				lpl.Inhib.RawIncr(NrnV(ctx, ni, di, Spike), NrnV(ctx, ni, di, GeRaw), NrnV(ctx, ni, di, GeExt))
				lpl.AvgMaxUpdate(ctx, ni, di)
			}
		}
	}
	for pi := uint32(0); pi < ly.NPools; pi++ {
		for di := uint32(0); di < ly.MaxData; di++ {
			ppi := pi
			ddi := di
			SetAvgMaxFloatFromIntErr(func() {
				fmt.Printf("GiFmSpikes: Layer: %s  pool: %d  di: %d\n", ly.Nm, ppi, ddi)
			})
			pl := ly.Pool(pi, di)
			pl.AvgMax.Calc(int32(ly.Idx))
		}
	}
	for di := uint32(0); di < ly.MaxData; di++ {
		lpl := ly.Pool(0, di)
		ly.Params.LayPoolGiFmSpikes(ctx, lpl, ly.LayerVals(di))
	}
	// ly.PoolGiFmSpikes(ctx) // note: this is now called as a second pass
	// so that we can do between-layer inhibition
}

// PoolGiFmSpikes computes inhibition Gi from Spikes within sub-pools.
// and also between different layers based on LayInhib* indexes
// must happen after LayPoolGiFmSpikes has been called.
func (ly *Layer) PoolGiFmSpikes(ctx *Context) {
	ly.BetweenLayerGi(ctx)
	np := ly.NPools
	if np == 1 {
		return
	}
	lyInhib := ly.Params.Inhib.Layer.On.IsTrue()
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		lpl := ly.Pool(0, di)
		for pi := uint32(1); pi < np; pi++ {
			pl := ly.Pool(pi, di)
			ly.Params.SubPoolGiFmSpikes(ctx, di, pl, lpl, lyInhib, ly.Vals[di].ActAvg.GiMult)
		}
	}
}

// BetweenLayerGi computes inhibition Gi between layers
func (ly *Layer) BetweenLayerGi(ctx *Context) {
	net := ly.Network
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		lpl := ly.Pool(0, di)
		maxGi := lpl.Inhib.Gi
		maxGi = ly.BetweenLayerGiMax(net, di, maxGi, ly.Params.LayInhib.Idx1)
		maxGi = ly.BetweenLayerGiMax(net, di, maxGi, ly.Params.LayInhib.Idx2)
		maxGi = ly.BetweenLayerGiMax(net, di, maxGi, ly.Params.LayInhib.Idx3)
		maxGi = ly.BetweenLayerGiMax(net, di, maxGi, ly.Params.LayInhib.Idx4)
		lpl.Inhib.Gi = maxGi // our inhib is max of us and everyone in the layer pool
	}
}

// BetweenLayerGiMax returns max gi value for input maxGi vs
// the given layIdx layer
func (ly *Layer) BetweenLayerGiMax(net *Network, di uint32, maxGi float32, layIdx int32) float32 {
	if layIdx < 0 {
		return maxGi
	}
	lay := net.Layers[layIdx]
	lpl := lay.Pool(0, di)
	if lpl.Inhib.Gi > maxGi {
		maxGi = lpl.Inhib.Gi
	}
	return maxGi
}

func (ly *Layer) PulvinarDriver(ctx *Context, lni, di uint32) (drvGe, nonDrvPct float32) {
	dly := ly.Network.Layers[int(ly.Params.Pulv.DriveLayIdx)]
	drvMax := dly.Pool(0, di).AvgMax.CaSpkP.Cycle.Max
	nonDrvPct = ly.Params.Pulv.NonDrivePct(drvMax) // how much non-driver to keep
	burst := NrnV(ctx, uint32(dly.NeurStIdx)+lni, di, Burst)
	drvGe = ly.Params.Pulv.DriveGe(burst)
	return
}

// GInteg integrates conductances G over time (Ge, NMDA, etc).
// calls SpecialGFmRawSyn, GiInteg
func (ly *Layer) GInteg(ctx *Context, ni, di uint32, pl *Pool, vals *LayerVals) {
	drvGe := float32(0)
	nonDrvPct := float32(0)
	if ly.LayerType() == PulvinarLayer {
		drvGe, nonDrvPct = ly.PulvinarDriver(ctx, ni-ly.NeurStIdx, di)
	}

	saveVal := ly.Params.SpecialPreGs(ctx, ni, di, pl, vals, drvGe, nonDrvPct)

	ly.Params.GFmRawSyn(ctx, ni, di)
	ly.Params.GiInteg(ctx, ni, di, pl, vals)
	ly.Params.GNeuroMod(ctx, ni, di, vals)

	ly.Params.SpecialPostGs(ctx, ni, di, saveVal)
}

// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
func (ly *Layer) SpikeFmG(ctx *Context, ni, di uint32) {
	ly.Params.SpikeFmG(ctx, ni, di)
}

// CycleNeuron does one cycle (msec) of updating at the neuron level
// Called directly by Network, iterates over data.
func (ly *Layer) CycleNeuron(ctx *Context, ni uint32) {
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		pl := ly.SubPool(ctx, ni, di)
		ly.GInteg(ctx, ni, di, pl, ly.LayerVals(di))
		ly.SpikeFmG(ctx, ni, di)
	}
}

// PostSpike does updates at neuron level after spiking has been computed.
// This is where special layer types add extra code.
// It also updates the CaSpkPCyc stats.
// Called directly by Network, iterates over data.
func (ly *Layer) PostSpike(ctx *Context, ni uint32) {
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		lpl := ly.Pool(0, di)
		pl := ly.SubPool(ctx, ni, di)
		vals := ly.LayerVals(di)
		ly.Params.PostSpikeSpecial(ctx, ni, di, pl, lpl, vals)
		ly.Params.PostSpike(ctx, ni, di, pl, vals)
	}
}

// SendSpike sends spike to receivers for all neurons that spiked
// last step in Cycle, integrated the next time around.
// Called directly by Network, iterates over data.
func (ly *Layer) SendSpike(ctx *Context, ni uint32) {
	for _, sp := range ly.SndPrjns {
		if sp.IsOff() {
			continue
		}
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			sp.SendSpike(ctx, ni, di, ly.MaxData)
		}
	}
}

// SynCa updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking, threaded over neurons.
// Called directly by Network, iterates over data.
func (ly *Layer) SynCa(ctx *Context, ni uint32) {
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		if NrnV(ctx, ni, di, Spike) == 0 { // di has to be outer loop b/c of this test
			continue
		}
		updtThr := ly.Params.Learn.CaLearn.UpdtThr
		if NrnV(ctx, ni, di, CaSpkP) < updtThr && NrnV(ctx, ni, di, CaSpkD) < updtThr {
			continue
		}
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			sp.SynCaSend(ctx, ni, di, updtThr)
		}
		for _, rp := range ly.RcvPrjns {
			if rp.IsOff() {
				continue
			}
			rp.SynCaRecv(ctx, ni, di, updtThr)
		}
	}
}

// LDTSrcLayAct returns the overall activity level for given source layer
// for purposes of computing ACh salience value.
// Typically the input is a superior colliculus (SC) layer that rapidly
// accommodates after the onset of a stimulus.
// using lpl.AvgMax.CaSpkP.Cycle.Max for layer activity measure.
func (ly *Layer) LDTSrcLayAct(net *Network, layIdx int32, di uint32) float32 {
	if layIdx < 0 {
		return 0
	}
	lay := net.Layers[layIdx]
	lpl := lay.Pool(0, di)
	return lpl.AvgMax.CaSpkP.Cycle.Avg
}

// CyclePost is called after the standard Cycle update, as a separate
// network layer loop.
// This is reserved for any kind of special ad-hoc types that
// need to do something special after Spiking is finally computed and Sent.
// Typically used for updating global values in the Context state,
// such as updating a neuromodulatory signal such as dopamine.
// Any updates here must also be done in gpu_hlsl/gpu_cyclepost.hlsl
func (ly *Layer) CyclePost(ctx *Context) {
	net := ly.Network
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		vals := ly.LayerVals(di)
		switch ly.LayerType() {
		case PTNotMaintLayer:
			ly.Params.CyclePostPTNotMaintLayer(ctx, di, ly.Pool(0, di))
		case CeMLayer:
			ly.Params.CyclePostCeMLayer(ctx, di, ly.Pool(0, di))
		case VSPatchLayer:
			for pi := uint32(1); pi < ly.NPools; pi++ {
				pl := ly.Pool(pi, di)
				ly.Params.CyclePostVSPatchLayer(ctx, di, int32(pi), pl)
			}
		case LDTLayer:
			srcLay1Act := ly.LDTSrcLayAct(net, ly.Params.LDT.SrcLay1Idx, di)
			srcLay2Act := ly.LDTSrcLayAct(net, ly.Params.LDT.SrcLay2Idx, di)
			srcLay3Act := ly.LDTSrcLayAct(net, ly.Params.LDT.SrcLay3Idx, di)
			srcLay4Act := ly.LDTSrcLayAct(net, ly.Params.LDT.SrcLay4Idx, di)
			ly.Params.CyclePostLDTLayer(ctx, di, vals, srcLay1Act, srcLay2Act, srcLay3Act, srcLay4Act)
		case VTALayer:
			ly.Params.CyclePostVTALayer(ctx, di)
		case RWDaLayer:
			pvals := net.LayerVals(uint32(ly.Params.RWDa.RWPredLayIdx), di)
			ly.Params.CyclePostRWDaLayer(ctx, di, vals, pvals)
		case TDPredLayer:
			ly.Params.CyclePostTDPredLayer(ctx, di, vals)
		case TDIntegLayer:
			pvals := net.LayerVals(uint32(ly.Params.TDInteg.TDPredLayIdx), di)
			ly.Params.CyclePostTDIntegLayer(ctx, di, vals, pvals)
		case TDDaLayer:
			ivals := net.LayerVals(uint32(ly.Params.TDDa.TDIntegLayIdx), di)
			ly.Params.CyclePostTDDaLayer(ctx, di, vals, ivals)
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Phase-level

// NewState handles all initialization at start of new input pattern.
// Does NOT call InitGScale()
func (ly *Layer) NewState(ctx *Context) {
	nn := ly.NNeurons
	np := ly.NPools

	actMinusAvg := float32(0)
	actPlusAvg := float32(0)

	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		lpl := ly.Pool(0, di)
		vals := ly.LayerVals(di)

		actMinusAvg += lpl.AvgMax.Act.Minus.Avg
		actPlusAvg += lpl.AvgMax.Act.Plus.Avg

		ly.Params.NewStateLayer(ctx, lpl, vals)

		for pi := uint32(0); pi < np; pi++ {
			pl := ly.Pool(pi, di)
			ly.Params.NewStatePool(ctx, pl) // also calls DecayState on pool
		}

		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIdx + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			// note: this calls the basic neuron-level DecayState
			ly.Params.NewStateNeuron(ctx, ni, di, vals)
		}
	}

	// note: long-running averages must be based on aggregate data, drive adaptation
	// of Gi layer inhibition.
	davg := 1 / float32(ctx.NetIdxs.NData)
	actMinusAvg *= davg
	actPlusAvg *= davg
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		vals := ly.LayerVals(di)
		ly.Params.NewStateLayerActAvg(ctx, vals, actMinusAvg, actPlusAvg)
	}

	// note: would be somewhat more expensive to only clear the di specific subset
	// but all di are decayed every trial anyway so no big deal
	ly.InitPrjnGBuffs(ctx)
}

// NewStateNeurons only calls the neurons part of new state -- for misbehaving GPU
func (ly *Layer) NewStateNeurons(ctx *Context) {
	nn := ly.NNeurons
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		vals := ly.LayerVals(di)
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIdx + lni
			// note: this calls the basic neuron-level DecayState
			ly.Params.NewStateNeuron(ctx, ni, di, vals)
		}
	}
}

// DecayState decays activation state by given proportion
// (default decay values are ly.Params.Acts.Decay.Act, Glong)
func (ly *Layer) DecayState(ctx *Context, di uint32, decay, glong, ahp float32) {
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		ly.Params.Acts.DecayState(ctx, ni, di, decay, glong, ahp)
		// Note: synapse-level Ca decay happens in DWt
		if ahp == 1 {
			lt := ly.LayerType()
			if lt == PTNotMaintLayer || lt == PTMaintLayer {
				SetNrnV(ctx, ni, di, CtxtGe, 0)
				SetNrnV(ctx, ni, di, CtxtGeRaw, 0)
				SetNrnV(ctx, ni, di, CtxtGeOrig, 0)
			}
		}
	}
	ly.DecayStateLayer(ctx, di, decay, glong, ahp)
	// note: would be somewhat more expensive to only clear the di specific subset
	// but all di are decayed every trial anyway so no big deal
	ly.InitPrjnGBuffs(ctx)
}

// DecayStateLayer does layer-level decay, but not neuron level
func (ly *Layer) DecayStateLayer(ctx *Context, di uint32, decay, glong, ahp float32) {
	np := ly.NPools
	for pi := uint32(0); pi < np; pi++ {
		pl := ly.Pool(pi, di)
		pl.Inhib.Decay(decay)
	}
}

// DecayStatePool decays activation state by given proportion in given sub-pool index (0 based)
func (ly *Layer) DecayStatePool(ctx *Context, pool int, decay, glong, ahp float32) {
	pi := uint32(pool + 1) // 1 based
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		pl := ly.Pool(pi, di)
		for lni := pl.StIdx; lni < pl.EdIdx; lni++ {
			ni := ly.NeurStIdx + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			ly.Params.Acts.DecayState(ctx, ni, di, decay, glong, ahp)
		}
		pl.Inhib.Decay(decay)
	}
}

// AvgMaxVarByPool returns the average and maximum value of given variable
// for given pool index (0 = entire layer, 1.. are subpools for 4D only).
// Uses fast index-based variable access.
func (ly *Layer) AvgMaxVarByPool(ctx *Context, varNm string, poolIdx, di int) minmax.AvgMax32 {
	var am minmax.AvgMax32
	vidx, err := ly.UnitVarIdx(varNm)
	if err != nil {
		log.Printf("axon.Layer.AvgMaxVar: %s\n", err)
		return am
	}
	pl := ly.Pool(uint32(poolIdx), uint32(di))
	am.Init()
	for lni := pl.StIdx; lni < pl.EdIdx; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		vl := ly.UnitVal1D(vidx, int(ni), di)
		am.UpdateVal(vl, int32(ni))
	}
	am.CalcAvg()
	return am
}

// MinusPhase does updating at end of the minus phase
func (ly *Layer) MinusPhase(ctx *Context) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		ly.Params.MinusPhasePool(ctx, pl) // grabs AvgMax.Minus from Cycle
	}
	nn := ly.NNeurons
	geIntMinusMax := float32(0)
	giIntMinusMax := float32(0)
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		vals := ly.LayerVals(di)
		lpl := ly.Pool(0, di)
		geIntMinusMax = mat32.Max(geIntMinusMax, lpl.AvgMax.GeInt.Minus.Max)
		giIntMinusMax = mat32.Max(giIntMinusMax, lpl.AvgMax.GiInt.Minus.Max)
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIdx + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			pl := ly.SubPool(ctx, ni, di)
			ly.Params.MinusPhaseNeuron(ctx, ni, di, pl, lpl, vals)
		}
	}
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		vals := ly.LayerVals(di)
		ly.Params.AvgGeM(ctx, vals, geIntMinusMax, giIntMinusMax)
	}
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
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			lpl := ly.Pool(0, di)
			pl := ly.SubPool(ctx, ni, di)
			ly.Params.PlusPhaseStartNeuron(ctx, ni, di, pl, lpl, ly.LayerVals(di))
		}
	}
}

// PlusPhase does updating at end of the plus phase
func (ly *Layer) PlusPhase(ctx *Context) {
	// todo: see if it is faster to just grab pool info now, then do everything below on CPU
	for pi := range ly.Pools { // gpu_cycletoplus
		pl := &ly.Pools[pi]
		ly.Params.PlusPhasePool(ctx, pl)
	}
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			lpl := ly.Pool(0, di)
			pl := ly.SubPool(ctx, ni, di)
			ly.Params.PlusPhaseNeuron(ctx, ni, di, pl, lpl, ly.LayerVals(di))
		}
	}
}

// PlusPhasePost does special algorithm processing at end of plus
func (ly *Layer) PlusPhasePost(ctx *Context) {
	ly.PlusPhaseActAvg(ctx)
	ly.CorSimFmActs(ctx) // GPU syncs down the state before this
	if ly.Params.Acts.Decay.OnRew.IsTrue() {
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			hasRew := (GlbV(ctx, di, GvHasRew) > 0)
			giveUp := (GlbV(ctx, di, GvLHbGiveUp) > 0)
			if hasRew || giveUp {
				ly.DecayState(ctx, di, 1, 1, 1) // note: GPU will get, and GBuf are auto-cleared in NewState
			}
		}
	}
	switch ly.LayerType() {
	case MatrixLayer:
		ly.MatrixGated(ctx)
	}
}

// PlusPhaseActAvg updates ActAvg and DTrgAvg at the plus phase
// Note: could be done on GPU but not worth it at this point..
func (ly *Layer) PlusPhaseActAvg(ctx *Context) {
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		dTrgSum := float32(0)
		avgSum := float32(0)
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			dTrgSum += ly.Params.LearnTrgAvgErrLRate() * (NrnV(ctx, ni, di, CaSpkP) - NrnV(ctx, ni, di, CaSpkD))
			avgSum += ly.Params.Acts.Dt.LongAvgDt * (NrnV(ctx, ni, di, ActM) - NrnAvgV(ctx, ni, ActAvg))
		}
		AddNrnAvgV(ctx, ni, DTrgAvg, dTrgSum)
		AddNrnAvgV(ctx, ni, ActAvg, avgSum)
	}
}

// TargToExt sets external input Ext from target values Target
// This is done at end of MinusPhase to allow targets to drive activity in plus phase.
// This can be called separately to simulate alpha cycles within theta cycles, for example.
func (ly *Layer) TargToExt(ctx *Context) {
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			if !NrnHasFlag(ctx, ni, di, NeuronHasTarg) { // will be clamped in plus phase
				continue
			}
			SetNrnV(ctx, ni, di, Ext, NrnV(ctx, ni, di, Target))
			NrnSetFlag(ctx, ni, di, NeuronHasExt)
			SetNrnV(ctx, ni, di, ISI, -1) // get fresh update on plus phase output acts
			SetNrnV(ctx, ni, di, ISIAvg, -1)
		}
	}
}

// ClearTargExt clears external inputs Ext that were set from target values Target.
// This can be called to simulate alpha cycles within theta cycles, for example.
func (ly *Layer) ClearTargExt(ctx *Context) {
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			if !NrnHasFlag(ctx, ni, di, NeuronHasTarg) { // will be clamped in plus phase
				continue
			}
			SetNrnV(ctx, ni, di, Ext, 0)
			NrnClearFlag(ctx, ni, di, NeuronHasExt)
			SetNrnV(ctx, ni, di, ISI, -1) // get fresh update on plus phase output acts
			SetNrnV(ctx, ni, di, ISIAvg, -1)
		}
	}
}

// SpkSt1 saves current activation state in SpkSt1 variables (using CaP)
func (ly *Layer) SpkSt1(ctx *Context) {
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			SetNrnV(ctx, ni, di, SpkSt1, NrnV(ctx, ni, di, CaSpkP))
		}
	}
}

// SpkSt2 saves current activation state in SpkSt2 variables (using CaP)
func (ly *Layer) SpkSt2(ctx *Context) {
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			SetNrnV(ctx, ni, di, SpkSt2, NrnV(ctx, ni, di, CaSpkP))
		}
	}
}

// CorSimFmActs computes the correlation similarity
// (centered cosine aka normalized dot product)
// in activation state between minus and plus phases.
func (ly *Layer) CorSimFmActs(ctx *Context) {
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		vals := ly.LayerVals(di)
		lpl := ly.Pool(0, di)
		avgM := lpl.AvgMax.Act.Minus.Avg
		avgP := lpl.AvgMax.Act.Plus.Avg
		cosv := float32(0)
		ssm := float32(0)
		ssp := float32(0)
		nn := ly.NNeurons
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIdx + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			ap := NrnV(ctx, ni, di, ActP) - avgP // zero mean = correl
			am := NrnV(ctx, ni, di, ActM) - avgM
			cosv += ap * am
			ssm += am * am
			ssp += ap * ap
		}
		dist := mat32.Sqrt(ssm * ssp)
		if dist != 0 {
			cosv /= dist
		}
		vals.CorSim.Cor = cosv

		ly.Params.Acts.Dt.AvgVarUpdt(&vals.CorSim.Avg, &vals.CorSim.Var, vals.CorSim.Cor)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learning

// DWt computes the weight change (learning), based on
// synaptically-integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (ly *Layer) DWt(ctx *Context, si uint32) {
	for _, pj := range ly.SndPrjns {
		if pj.IsOff() {
			continue
		}
		pj.DWt(ctx, si)
	}
}

// DWtSubMean computes subtractive normalization of the DWts
func (ly *Layer) DWtSubMean(ctx *Context, ri uint32) {
	for _, pj := range ly.RcvPrjns {
		if pj.IsOff() {
			continue
		}
		pj.DWtSubMean(ctx, ri)
	}
}

// WtFmDWt updates weight values from delta weight changes
func (ly *Layer) WtFmDWt(ctx *Context, si uint32) {
	for _, pj := range ly.SndPrjns {
		if pj.IsOff() {
			continue
		}
		pj.WtFmDWt(ctx, si)
	}
}

// DTrgSubMean subtracts the mean from DTrgAvg values
// Called by TrgAvgFmD
func (ly *Layer) DTrgSubMean(ctx *Context) {
	submean := ly.Params.Learn.TrgAvgAct.SubMean
	if submean == 0 {
		return
	}
	if ly.HasPoolInhib() && ly.Params.Learn.TrgAvgAct.Pool.IsTrue() {
		np := ly.NPools
		for pi := uint32(1); pi < np; pi++ {
			pl := ly.Pool(pi, 0) // only for idxs
			nn := 0
			avg := float32(0)
			for lni := pl.StIdx; lni < pl.EdIdx; lni++ {
				ni := ly.NeurStIdx + lni
				if NrnIsOff(ctx, ni) {
					continue
				}
				avg += NrnAvgV(ctx, ni, DTrgAvg)
				nn++
			}
			if nn == 0 {
				continue
			}
			avg /= float32(nn)
			avg *= submean
			for lni := pl.StIdx; lni < pl.EdIdx; lni++ {
				ni := ly.NeurStIdx + lni
				if NrnIsOff(ctx, ni) {
					continue
				}
				AddNrnAvgV(ctx, ni, DTrgAvg, -avg)
			}
		}
	} else {
		nn := 0
		avg := float32(0)
		for lni := uint32(0); lni < ly.NNeurons; lni++ {
			ni := ly.NeurStIdx + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			avg += NrnAvgV(ctx, ni, DTrgAvg)
			nn++
		}
		if nn == 0 {
			return
		}
		avg /= float32(nn)
		avg *= submean
		for lni := uint32(0); lni < ly.NNeurons; lni++ {
			ni := ly.NeurStIdx + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			AddNrnAvgV(ctx, ni, DTrgAvg, -avg)
		}
	}
}

// TrgAvgFmD updates TrgAvg from DTrgAvg -- called in PlusPhasePost
func (ly *Layer) TrgAvgFmD(ctx *Context) {
	lr := ly.Params.LearnTrgAvgErrLRate()
	if lr == 0 {
		return
	}
	ly.DTrgSubMean(ctx)
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIdx + lni
		if NrnIsOff(ctx, ni) {
			continue
		}
		ntrg := NrnAvgV(ctx, ni, TrgAvg) + NrnAvgV(ctx, ni, DTrgAvg)
		ntrg = ly.Params.Learn.TrgAvgAct.TrgRange.ClipVal(ntrg)
		SetNrnAvgV(ctx, ni, TrgAvg, ntrg)
		SetNrnAvgV(ctx, ni, DTrgAvg, 0)
	}
}

// WtFmDWtLayer does weight update at the layer level.
// does NOT call main projection-level WtFmDWt method.
// in base, only calls TrgAvgFmD
func (ly *Layer) WtFmDWtLayer(ctx *Context) {
	ly.TrgAvgFmD(ctx)
}

// SlowAdapt is the layer-level slow adaptation functions.
// Calls AdaptInhib and AvgDifFmTrgAvg for Synaptic Scaling.
// Does NOT call projection-level methods.
func (ly *Layer) SlowAdapt(ctx *Context) {
	ly.AdaptInhib(ctx)
	ly.AvgDifFmTrgAvg(ctx)
	// note: prjn level call happens at network level
}

// AdaptInhib adapts inhibition
func (ly *Layer) AdaptInhib(ctx *Context) {
	if ly.Params.Inhib.ActAvg.AdaptGi.IsFalse() || ly.Params.IsInput() {
		return
	}
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		vals := ly.LayerVals(di)
		ly.Params.Inhib.ActAvg.Adapt(&vals.ActAvg.GiMult, vals.ActAvg.ActMAvg)
	}
}

// AvgDifFmTrgAvg updates neuron-level AvgDif values from AvgPct - TrgAvg
// which is then used for synaptic scaling of LWt values in Prjn SynScale.
func (ly *Layer) AvgDifFmTrgAvg(ctx *Context) {
	sp := uint32(0)
	if ly.NPools > 1 {
		sp = 1
	}
	np := ly.NPools
	for pi := sp; pi < np; pi++ {
		pl := ly.Pool(pi, 0)
		plavg := float32(0)
		nn := 0
		for lni := pl.StIdx; lni < pl.EdIdx; lni++ {
			ni := ly.NeurStIdx + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			plavg += NrnAvgV(ctx, ni, ActAvg)
			nn++
		}
		if nn == 0 {
			continue
		}
		plavg /= float32(nn)
		if plavg < 0.0001 { // gets unstable below here
			continue
		}
		pl.AvgDif.Init()
		for lni := pl.StIdx; lni < pl.EdIdx; lni++ {
			ni := ly.NeurStIdx + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			apct := NrnAvgV(ctx, ni, ActAvg) / plavg
			adif := apct - NrnAvgV(ctx, ni, TrgAvg)
			SetNrnAvgV(ctx, ni, AvgPct, apct)
			SetNrnAvgV(ctx, ni, AvgDif, adif)
			pl.AvgDif.UpdateVal(mat32.Abs(adif))
		}
		ppi := pi
		SetAvgMaxFloatFromIntErr(func() {
			fmt.Printf("AvgDifFmTrgAvg Pool Layer: %s  pool: %d\n", ly.Nm, ppi)
		})
		pl.AvgDif.Calc(int32(ly.Idx))                       // ref in case of crash
		for di := uint32(1); di < ctx.NetIdxs.NData; di++ { // copy to other datas
			pld := ly.Pool(pi, di)
			pld.AvgDif = pl.AvgDif
		}
	}
	if sp == 1 { // update layer pool
		lpl := ly.Pool(0, 0)
		lpl.AvgDif.Init()
		for lni := lpl.StIdx; lni < lpl.EdIdx; lni++ {
			ni := ly.NeurStIdx + lni
			if NrnIsOff(ctx, ni) {
				continue
			}
			lpl.AvgDif.UpdateVal(mat32.Abs(NrnAvgV(ctx, ni, AvgDif)))
		}
		SetAvgMaxFloatFromIntErr(func() {
			fmt.Printf("AvgDifFmTrgAvg LayPool Layer: %s  pool: %d\n", ly.Nm)
		})
		lpl.AvgDif.Calc(int32(ly.Idx))

		for di := uint32(1); di < ctx.NetIdxs.NData; di++ { // copy to other datas
			lpld := ly.Pool(0, di)
			lpld.AvgDif = lpl.AvgDif
		}
	}
}

// SynFail updates synaptic weight failure only -- normally done as part of DWt
// and WtFmDWt, but this call can be used during testing to update failing synapses.
func (ly *Layer) SynFail(ctx *Context) {
	for _, pj := range ly.SndPrjns {
		if pj.IsOff() {
			continue
		}
		pj.SynFail(ctx)
	}
}

// LRateMod sets the LRate modulation parameter for Prjns, which is
// for dynamic modulation of learning rate (see also LRateSched).
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LRateMod(mod float32) {
	for _, pj := range ly.RcvPrjns {
		// if pj.IsOff() { // keep all sync'd
		// 	continue
		// }
		pj.LRateMod(mod)
	}
}

// LRateSched sets the schedule-based learning rate multiplier.
// See also LRateMod.
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LRateSched(sched float32) {
	for _, pj := range ly.RcvPrjns {
		// if pj.IsOff() { // keep all sync'd
		// 	continue
		// }
		pj.LRateSched(sched)
	}
}

// SetSubMean sets the SubMean parameters in all the layers in the network
// trgAvg is for Learn.TrgAvgAct.SubMean
// prjn is for the prjns Learn.Trace.SubMean
// in both cases, it is generally best to have both parameters set to 0
// at the start of learning
func (ly *Layer) SetSubMean(trgAvg, prjn float32) {
	ly.Params.Learn.TrgAvgAct.SubMean = trgAvg
	for _, pj := range ly.RcvPrjns {
		// if pj.IsOff() { // keep all sync'd
		// 	continue
		// }
		pj.Params.Learn.Trace.SubMean = prjn
	}
}
