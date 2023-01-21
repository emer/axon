// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"

	"github.com/emer/etable/minmax"
	"github.com/goki/gosl/sltype"
	"github.com/goki/mat32"
)

// layer_compute.go has the core computational methods, which are also called by GPU

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle
//  note: these are calls to LayerParams methods that have the core computation

// Prjns.PrjnGatherSpikes called first, at Network level for all prjns

// GiFmSpikes integrates new inhibitory conductances from Spikes
// at the layer and pool level.
// Called separately by Network.CycleImpl on all Layers
func (ly *Layer) GiFmSpikes(ctx *Context) {
	lpl := &ly.Pools[0]
	subPools := (len(ly.Pools) > 1)
	for ni := range ly.Neurons { // note: layer-level iterating across neurons
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pl := &ly.Pools[nrn.SubPool]
		ly.Params.GeExtToPool(ctx, uint32(ni), nrn, pl, lpl, subPools) // todo: can this be done in send spike?
		// todo: deal with this in plus phase and new state methods.
		// if ly.Params.Act.Clamp.Add.IsFalse() && nrn.HasFlag(NeuronHasExt) {
		// 	pl.Inhib.Clamped.SetBool(true)
		// 	lpl.Inhib.Clamped.SetBool(true)
		// }
	}
	ly.Params.LayPoolGiFmSpikes(ctx, lpl, ly.Vals)
	ly.PoolGiFmSpikes(ctx)
}

// PoolGiFmSpikes computes inhibition Gi from Spikes within relevant Pools
func (ly *Layer) PoolGiFmSpikes(ctx *Context) {
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

// NeuronGatherSpikes integrates G*Raw and G*Syn values for given neuron
// from the Prjn-level GSyn integrated values.
func (ly *Layer) NeuronGatherSpikes(ctx *Context, ni uint32, nrn *Neuron) {
	ly.Params.NeuronGatherSpikesInit(ctx, ni, nrn)
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.AsAxon()
		gv := pj.GVals[ni]
		pj.Params.NeuronGatherSpikesPrjn(ctx, gv, ni, nrn)
	}
}

func (ly *Layer) PulvinarDriver(ni uint32) (drvGe, nonDrvPct float32) {
	dly := ly.Network.Layer(int(ly.Params.Pulv.DriveLayIdx)).(AxonLayer).AsAxon()
	drvMax := dly.Vals.ActAvg.CaSpkP.Max
	nonDrvPct = ly.Params.Pulv.NonDrivePct(drvMax) // how much non-driver to keep
	dneur := dly.Neurons[ni]
	if dly.LayerType() == SuperLayer {
		drvGe = ly.Params.Pulv.DriveGe(dneur.Burst)
	} else {
		drvGe = ly.Params.Pulv.DriveGe(dneur.CaSpkP)
	}
	return
}

// GInteg integrates conductances G over time (Ge, NMDA, etc).
// calls NeuronGatherSpikes, GFmRawSyn, GiInteg
func (ly *Layer) GInteg(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, vals *LayerVals, randctr *sltype.Uint2) {
	ly.NeuronGatherSpikes(ctx, ni, nrn)

	drvGe := float32(0)
	nonDrvPct := float32(0)
	if ly.LayerType() == PulvinarLayer {
		drvGe, nonDrvPct = ly.PulvinarDriver(ni)
	}

	saveVal := ly.Params.SpecialPreGs(ctx, ni, nrn, drvGe, nonDrvPct, randctr)

	ly.Params.GFmRawSyn(ctx, ni, nrn, randctr)
	ly.Params.GiInteg(ctx, ni, nrn, pl, vals.ActAvg.GiMult)
	ly.Params.GNeuroMod(ctx, ni, nrn, vals)

	ly.Params.SpecialPostGs(ctx, ni, nrn, randctr, saveVal)
}

// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
func (ly *Layer) SpikeFmG(ctx *Context, ni uint32, nrn *Neuron) {
	ly.Params.SpikeFmG(ctx, ni, nrn)
}

// CycleNeuron does one cycle (msec) of updating at the neuron level
func (ly *Layer) CycleNeuron(ctx *Context, ni uint32, nrn *Neuron) {
	randctr := ctx.RandCtr.Uint2() // use local var so updates are local
	ly.AxonLay.GInteg(ctx, ni, nrn, &ly.Pools[nrn.SubPool], ly.Vals, &randctr)
	ly.AxonLay.SpikeFmG(ctx, ni, nrn)
	ly.AxonLay.PostSpike(ctx, ni, nrn)
}

// PostSpike does updates at neuron level after spiking has been computed.
// This is where special layer types add extra code.
func (ly *Layer) PostSpike(ctx *Context, ni uint32, nrn *Neuron) {
	ly.Params.PostSpike(ctx, ni, nrn, ly.Vals)
}

// SendSpike sends spike to receivers for all neurons that spiked
// last step in Cycle, integrated the next time around.
func (ly *Layer) SendSpike(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Spike == 0 {
			continue
		}
		ly.Pools[nrn.SubPool].Inhib.FBsRaw += 1.0 // note: this is immediate..
		if nrn.SubPool > 0 {
			ly.Pools[0].Inhib.FBsRaw += 1.0
		}
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			sp.SendSpike(ni)
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
	act := ly.Params.RSalACh.Thr(net.LayVals[layIdx].ActAvg.CaSpkP.Max)
	if act > maxAct {
		maxAct = act
	}
	return act
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
			maxAct = ly.Params.RSalACh.Thr(ctx.NeuroMod.Rew)
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
		ctx.NeuroMod.ACh = maxAct
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
func (ly *Layer) NewState() {
	pl := &ly.Pools[0]
	ly.Params.Inhib.ActAvg.AvgFmAct(&ly.Vals.ActAvg.ActMAvg, pl.ActM.Avg, ly.Params.Act.Dt.LongAvgDt)
	ly.Params.Inhib.ActAvg.AvgFmAct(&ly.Vals.ActAvg.ActPAvg, pl.ActP.Avg, ly.Params.Act.Dt.LongAvgDt)

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
		nrn.SpkPrv = nrn.CaSpkD
		nrn.GeSynPrv = nrn.GeSynMax
		nrn.SpkMax = 0
		nrn.SpkMaxCa = 0
		nrn.GeSynPrv = 0
	}
	ly.AxonLay.DecayState(ly.Params.Act.Decay.Act, ly.Params.Act.Decay.Glong)
}

// DecayState decays activation state by given proportion
// (default decay values are ly.Params.Act.Decay.Act, Glong)
func (ly *Layer) DecayState(decay, glong float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Params.Act.DecayState(nrn, decay, glong)
		// ly.Params.Learn.DecayCaLrnSpk(nrn, glong) // NOT called by default
		// Note: synapse-level Ca decay happens in DWt
	}
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
		am.UpdateVal(vl, int(ni))
	}
	am.CalcAvg()
	return am
}

// AvgGeM computes the average and max GeM stats, updated in MinusPhase
func (ly *Layer) AvgGeM(ctx *Context) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.ActM = ly.AvgMaxVarByPool("ActM", pi)
		pl.GeM = ly.AvgMaxVarByPool("GeM", pi)
		pl.GiM = ly.AvgMaxVarByPool("GiM", pi)
	}
	lpl := &ly.Pools[0]
	ly.Vals.ActAvg.AvgMaxGeM += ly.Params.Act.Dt.LongAvgDt * (lpl.GeM.Max - ly.Vals.ActAvg.AvgMaxGeM)
	ly.Vals.ActAvg.AvgMaxGiM += ly.Params.Act.Dt.LongAvgDt * (lpl.GiM.Max - ly.Vals.ActAvg.AvgMaxGiM)
}

// MinusPhase does updating at end of the minus phase
func (ly *Layer) MinusPhase(ctx *Context) {
	ly.Vals.ActAvg.CaSpkPM = ly.AvgMaxVarByPool("CaSpkP", 0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.ActM = nrn.ActInt
		nrn.CaSpkPM = nrn.CaSpkP
		if nrn.HasFlag(NeuronHasTarg) { // will be clamped in plus phase
			nrn.Ext = nrn.Target
			nrn.SetFlag(NeuronHasExt)
			nrn.ISI = -1 // get fresh update on plus phase output acts
			nrn.ISIAvg = -1
			nrn.ActInt = ly.Params.Act.Init.Act // reset for plus phase
		}
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		if ly.Params.Act.Clamp.Add.IsFalse() && ly.Params.Act.Clamp.IsTarget.IsTrue() {
			pl.Inhib.Clamped.SetBool(true)
		}
	}
	ly.AvgGeM(ctx)
}

// PlusPhase does updating at end of the plus phase
func (ly *Layer) PlusPhase(ctx *Context) {
	ly.Vals.ActAvg.CaSpkP = ly.AvgMaxVarByPool("CaSpkP", 0)
	ly.Vals.ActAvg.CaSpkD = ly.AvgMaxVarByPool("CaSpkD", 0)

	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.ActP = nrn.ActInt
		mlr := ly.Params.Learn.RLRate.RLRateSigDeriv(nrn.CaSpkD, ly.Vals.ActAvg.CaSpkD.Max)
		dlr := ly.Params.Learn.RLRate.RLRateDiff(nrn.CaSpkP, nrn.CaSpkD)
		modlr := ly.Params.Learn.NeuroMod.LRMod(ly.Vals.NeuroMod.DA, ly.Vals.NeuroMod.ACh)
		nrn.RLRate = mlr * dlr * modlr
		nrn.ActAvg += ly.Params.Act.Dt.LongAvgDt * (nrn.ActM - nrn.ActAvg)
		var tau float32
		ly.Params.Act.Sahp.NinfTauFmCa(nrn.SahpCa, &nrn.SahpN, &tau)
		nrn.SahpCa = ly.Params.Act.Sahp.CaInt(nrn.SahpCa, nrn.CaSpkD)
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.ActP = ly.AvgMaxVarByPool("ActP", pi)
	}
	ly.AxonLay.CorSimFmActs()
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
	avgM := lpl.ActM.Avg
	avgP := lpl.ActP.Avg
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
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif), int(ni))
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
			pl.AvgDif.UpdateVal(mat32.Abs(nrn.AvgDif), int(ni))
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
