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
		bi := pj.Params.Com.ReadIdx(ni, ctx.CycleTot)
		gRaw := pj.GBuf[bi]
		pj.GBuf[bi] = 0
		pj.Params.GatherSpikes(ctx, ly.Params, ni, nrn, gRaw, &pj.GSyns[ni])
	}
}

// GiFmSpikes gets the Spike, GeRaw and GeExt from neurons in the pools
// where Spike drives FBsRaw -- raw feedback signal,
// GeRaw drives FFsRaw -- aggregate feedforward excitatory spiking input
// GeExt represents extra excitatory input from other sources.
// Then integrates new inhibitory conductances therefrom,
// at the layer and pool level.
// Called separately by Network.CycleImpl on all Layers
// Also updates all AvgMax values at the Cycle level.
func (ly *Layer) GiFmSpikes(ctx *Context) {
	lpl := &ly.Pools[0]
	subPools := (len(ly.Pools) > 1)
	lpl.AvgMax.Init()
	for ni := range ly.Neurons { // note: layer-level iterating across neurons
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pl := &ly.Pools[nrn.SubPool]
		ly.Params.GeToPool(ctx, uint32(ni), nrn, pl, lpl, subPools)
		lpl.AvgMax.UpdateVals(nrn, int32(ni))
	}
	lpl.AvgMax.CalcAvg()
	ly.Params.LayPoolGiFmSpikes(ctx, lpl, ly.Vals)
	// ly.PoolGiFmSpikes(ctx) // note: this is now called as a second pass
	// so that we can do between-layer inhibition
}

// PoolGiFmSpikes computes inhibition Gi from Spikes within relevant Pools
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
		pl.AvgMax.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			pl.AvgMax.UpdateVals(nrn, int32(ni))
		}
		pl.AvgMax.CalcAvg()
	}
}

// BetweenLayerGi computes inhibition Gi between layers
func (ly *Layer) BetweenLayerGi(ctx *Context) {
	lpl := &ly.Pools[0]
	maxGi := lpl.Inhib.Gi
	net := ly.Network.(AxonNetwork).AsAxon()
	maxGi = ly.BetweenLayerGiMax(maxGi, net, ly.Params.LayInhib1Idx)
	maxGi = ly.BetweenLayerGiMax(maxGi, net, ly.Params.LayInhib2Idx)
	maxGi = ly.BetweenLayerGiMax(maxGi, net, ly.Params.LayInhib3Idx)
	maxGi = ly.BetweenLayerGiMax(maxGi, net, ly.Params.LayInhib4Idx)
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

	saveVal := ly.Params.SpecialPreGs(ctx, ni, nrn, drvGe, nonDrvPct)

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

// RSalAChMaxLayAct returns the updated maxAct value using
// LayVals.ActAvg.CaSpkP.Max from given layer index,
// subject to any relevant RewThr thresholding.
func (ly *Layer) RSalAChMaxLayAct(maxAct float32, net *Network, layIdx int32) float32 {
	if layIdx < 0 {
		return maxAct
	}
	lay := net.Layers[layIdx].(AxonLayer).AsAxon()
	lpl := &lay.Pools[0]
	act := ly.Params.RSalACh.Thr(lpl.AvgMax.Act.Cycle.Max) // use Act -- otherwise too variable
	if act > maxAct {
		maxAct = act
	}
	return maxAct
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
		maxAct := float32(0)
		if ly.Params.RSalACh.Rew.IsTrue() {
			if ctx.NeuroMod.HasRew.IsTrue() {
				maxAct = 1
			}
		}
		if ly.Params.RSalACh.RewPred.IsTrue() {
			rpAct := ly.Params.RSalACh.Thr(ctx.NeuroMod.RewPred)
			if rpAct > maxAct {
				maxAct = rpAct
			}
		}
		maxAct = ly.RSalAChMaxLayAct(maxAct, net, ly.Params.RSalACh.SrcLay1Idx)
		maxAct = ly.RSalAChMaxLayAct(maxAct, net, ly.Params.RSalACh.SrcLay2Idx)
		maxAct = ly.RSalAChMaxLayAct(maxAct, net, ly.Params.RSalACh.SrcLay3Idx)
		maxAct = ly.RSalAChMaxLayAct(maxAct, net, ly.Params.RSalACh.SrcLay4Idx)
		maxAct = ly.RSalAChMaxLayAct(maxAct, net, ly.Params.RSalACh.SrcLay5Idx)
		ctx.NeuroMod.AChRaw = maxAct // raw value this trial
		ctx.NeuroMod.ACh = mat32.Max(ctx.NeuroMod.ACh, ctx.NeuroMod.AChRaw)
	case RWDaLayer:
		net := ly.Network.(AxonNetwork).AsAxon()
		pvals := net.LayVals[ly.Params.RWDa.RWPredLayIdx]
		pred := pvals.Special.V1 - pvals.Special.V2
		ctx.NeuroMod.RewPred = pred // record
		da := float32(0)
		if ctx.NeuroMod.HasRew.IsTrue() {
			da = ctx.NeuroMod.Rew - pred
		}
		ctx.NeuroMod.DA = da // updates global value that will be copied to layers next cycle.
		ly.Vals.NeuroMod.DA = da
	case TDPredLayer:
		if ctx.PlusPhase.IsTrue() {
			pred := ly.Vals.Special.V1 - ly.Vals.Special.V2
			ctx.NeuroMod.PrevPred = pred
		}
	case TDIntegLayer:
		net := ly.Network.(AxonNetwork).AsAxon()
		pvals := net.LayVals[ly.Params.TDInteg.TDPredLayIdx]
		rew := float32(0)
		if ctx.NeuroMod.HasRew.IsTrue() {
			rew = ctx.NeuroMod.Rew
		}
		rpval := float32(0)
		if ctx.PlusPhase.IsTrue() {
			pred := pvals.Special.V1 - pvals.Special.V2 // neuron0 (pos) - neuron1 (neg)
			rpval = rew + ly.Params.TDInteg.Discount*ly.Params.TDInteg.PredGain*pred
			ly.Vals.Special.V2 = rpval // plus phase
		} else {
			rpval = ly.Params.TDInteg.PredGain * ctx.NeuroMod.PrevPred
			ly.Vals.Special.V1 = rpval // minus phase is *previous trial*
		}
		ctx.NeuroMod.RewPred = rpval // global value will be copied to layers next cycle
	case TDDaLayer:
		net := ly.Network.(AxonNetwork).AsAxon()
		ivals := net.LayVals[ly.Params.TDDa.TDIntegLayIdx]
		da := ivals.Special.V2 - ivals.Special.V1
		if ctx.PlusPhase.IsFalse() {
			da = 0
		}
		ctx.NeuroMod.DA = da // updates global value that will be copied to layers next cycle.
		ly.Vals.NeuroMod.DA = da
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Phase-level

// NewState handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
// Does NOT call InitGScale()
func (ly *Layer) NewState(ctx *Context) {
	lpl := &ly.Pools[0]
	ly.Params.Inhib.ActAvg.AvgFmAct(&ly.Vals.ActAvg.ActMAvg, lpl.AvgMax.Act.Minus.Avg, ly.Params.Act.Dt.LongAvgDt)
	ly.Params.Inhib.ActAvg.AvgFmAct(&ly.Vals.ActAvg.ActPAvg, lpl.AvgMax.Act.Plus.Avg, ly.Params.Act.Dt.LongAvgDt)

	// todo: combine pool-level calls with decaystatelayer below
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		if ly.Params.Act.Clamp.Add.IsFalse() && ly.Params.Act.Clamp.IsTarget.IsTrue() {
			pl.Inhib.Clamped.SetBool(false)
		}
	}

	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		// note: this calls the basic neuron-level DecayState
		ly.Params.NewState(ctx, uint32(ni), nrn, &ly.Pools[nrn.SubPool], ly.Vals)
	}
	// todo: must do on GPU:
	ly.DecayStateLayer(ctx, ly.Params.Act.Decay.Act, ly.Params.Act.Decay.Glong)
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
	// todo: must do on GPU!
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

// AvgGeM computes the average and max GeM stats, updated in MinusPhase
func (ly *Layer) AvgGeM(ctx *Context) {
	lpl := &ly.Pools[0]
	ly.Vals.ActAvg.AvgMaxGeM += ly.Params.Act.Dt.LongAvgDt * (lpl.AvgMax.Ge.Minus.Max - ly.Vals.ActAvg.AvgMaxGeM)
	ly.Vals.ActAvg.AvgMaxGiM += ly.Params.Act.Dt.LongAvgDt * (lpl.AvgMax.Gi.Minus.Max - ly.Vals.ActAvg.AvgMaxGiM)
}

// MinusPhase does updating at end of the minus phase
func (ly *Layer) MinusPhase(ctx *Context) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.AvgMax.CycleToMinus()
		if ly.Params.Act.Clamp.Add.IsFalse() && ly.Params.Act.Clamp.IsTarget.IsTrue() {
			pl.Inhib.Clamped.SetBool(true)
		}
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.MinusPhase(ctx, uint32(ni), nrn, &ly.Pools[nrn.SubPool], ly.Vals)
	}
	ly.AvgGeM(ctx)
}

// PlusPhase does updating at end of the plus phase
func (ly *Layer) PlusPhase(ctx *Context) {
	// todo: see if it is faster to just grab pool info now, then do everything below on CPU
	for pi := range ly.Pools { // gpu_cycletoplus
		pl := &ly.Pools[pi]
		pl.AvgMax.CycleToPlus()
	}
	for ni := range ly.Neurons { // gpu_plusphase
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pl := &ly.Pools[nrn.SubPool]
		lpl := &ly.Pools[0]
		ly.Params.PlusPhase(ctx, uint32(ni), nrn, pl, lpl, ly.Vals)
	}
	ly.AxonLay.CorSimFmActs() // todo: on GPU?
	// sync pools -> GPU -> CPU
	ly.PlusPhasePost(ctx) // special
	// copy CPU -> GPU with gated info..  matrix is expensive!
}

// PlusPhasePost does special algorithm processing at end of plus
func (ly *Layer) PlusPhasePost(ctx *Context) {
	switch ly.LayerType() {
	case MatrixLayer:
		ly.MatrixGated()
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

// IsTarget returns true if this layer is a Target layer.
// By default, returns true for layers of Type == emer.Target
// Other Target layers include the TRCLayer in deep predictive learning.
// It is used in SynScale to not apply it to target layers.
// In both cases, Target layers are purely error-driven.
func (ly *Layer) IsTarget() bool {
	switch ly.LayerType() {
	case TargetLayer:
		return true
	case PulvinarLayer:
		return true
	default:
		return false
	}
}

// IsInput returns true if this layer is an Input layer.
// By default, returns true for layers of Type == emer.Input
// Used to prevent adapting of inhibition or TrgAvg values.
func (ly *Layer) IsInput() bool {
	switch ly.LayerType() {
	case InputLayer:
		return true
	default:
		return false
	}
}

// IsInputOrTarget returns true if this layer is either an Input
// or a Target layer.
func (ly *Layer) IsInputOrTarget() bool {
	return (ly.AxonLay.IsTarget() || ly.AxonLay.IsInput())
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learning

func (ly *Layer) IsLearnTrgAvg() bool {
	if ly.AxonLay.IsTarget() || ly.AxonLay.IsInput() || ly.Params.Learn.TrgAvgAct.On.IsFalse() {
		return false
	}
	return true
}

// DWtLayer does weight change at the layer level.
// does NOT call main projection-level DWt method.
// in base, only calls DTrgAvgFmErr
func (ly *Layer) DWtLayer(ctx *Context) {
	ly.DTrgAvgFmErr()
}

// DTrgAvgFmErr computes change in TrgAvg based on unit-wise error signal
// Called by DWtLayer at the layer level
func (ly *Layer) DTrgAvgFmErr() {
	if !ly.IsLearnTrgAvg() {
		return
	}
	lr := ly.Params.Learn.TrgAvgAct.ErrLRate
	if lr == 0 {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.DTrgAvg += lr * (nrn.CaSpkP - nrn.CaSpkD) // CaP - CaD almost as good in ra25 -- todo explore
	}
}

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

// TrgAvgFmD updates TrgAvg from DTrgAvg
// it is called by WtFmDWtLayer
func (ly *Layer) TrgAvgFmD() {
	if !ly.IsLearnTrgAvg() || ly.Params.Learn.TrgAvgAct.ErrLRate == 0 {
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
	if ly.Params.Inhib.ActAvg.AdaptGi.IsFalse() || ly.AxonLay.IsInput() {
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
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif), int32(ni))
		}
		pl.AvgDif.CalcAvg()
	}
	if sp == 1 { // update stats
		pl := &ly.Pools[0]
		pl.AvgDif.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif), int32(ni))
		}
		pl.AvgDif.CalcAvg()
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
