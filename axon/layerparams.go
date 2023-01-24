// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"

	"github.com/goki/gosl/sltype"
)

//gosl: hlsl layerparams
// #include "layertypes.hlsl"
// #include "act.hlsl"
// #include "inhib.hlsl"
// #include "learn_neur.hlsl"
// #include "deep_layers.hlsl"
// #include "rl_layers.hlsl"
// #include "pvlv_layers.hlsl"
// #include "pcore_layers.hlsl"
// #include "pool.hlsl"
// #include "layervals.hlsl"
//gosl: end layerparams

//gosl: start layerparams

// global projection param arrays
// var SendPrjns []PrjnParams // [Layer][SendPrjns]
// var RecvPrjns []PrjnParams // [Layer][RecvPrjns]

// LayerIdxs contains index access into global arrays for GPU.
type LayerIdxs struct {
	PoolSt uint32 `inactive:"+" desc:"start of pools for this layer -- first one is always the layer-wide pool"`
	NeurSt uint32 `inactive:"+" desc:"start of neurons for this layer in global array (same as Layer.NeurStIdx)"`
	NeurN  uint32 `inactive:"+" desc:"number of neurons in layer"`
	RecvSt uint32 `inactive:"+" desc:"start index into RecvPrjns global array"`
	RecvN  uint32 `inactive:"+" desc:"number of recv projections"`
	SendSt uint32 `inactive:"+" desc:"start index into SendPrjns global array"`
	SendN  uint32 `inactive:"+" desc:"number of send projections"`

	pad uint32
}

// SetNeuronExtPosNeg sets neuron Ext value based on neuron index
// with positive values going in first unit, negative values rectified
// to positive in 2nd unit
func SetNeuronExtPosNeg(ni uint32, nrn *Neuron, val float32) {
	if ni == 0 {
		if val >= 0 {
			nrn.Ext = val
		} else {
			nrn.Ext = 0
		}
	} else {
		if val >= 0 {
			nrn.Ext = 0
		} else {
			nrn.Ext = -val
		}
	}
}

// LayerParams contains all of the layer parameters.
// These values must remain constant over the course of computation.
// On the GPU, they are loaded into a uniform.
type LayerParams struct {
	LayType LayerTypes `desc:"functional type of layer -- determines functional code path for specialized layer types, and is synchronized with the Layer.Typ value"`

	pad, pad1, pad2 int32

	Act   ActParams       `view:"add-fields" desc:"Activation parameters and methods for computing activations"`
	Inhib InhibParams     `view:"add-fields" desc:"Inhibition parameters and methods for computing layer-level inhibition"`
	Learn LearnNeurParams `view:"add-fields" desc:"Learning parameters and methods that operate at the neuron level"`

	//////////////////////////////////////////
	//  Specialized layer type parameters
	//     each applies to a specific layer type.
	//     use the `viewif` field tag to condition on LayType.

	Burst   BurstParams   `viewif:"LayType=SuperLayer" view:"inline" desc:"BurstParams determine how the 5IB Burst activation is computed from CaSpkP integrated spiking values in Super layers -- thresholded."`
	CT      CTParams      `viewif:"LayType=CTLayer" view:"inline" desc:"params for the CT corticothalamic layer that generates predictions over the Pulvinar using context -- uses the CtxtGe excitatory input plus stronger NMDA channels to maintain context trace"`
	Pulv    PulvParams    `viewif:"LayType=PulvinarLayer" view:"inline" desc:"provides parameters for how the plus-phase (outcome) state of Pulvinar thalamic relay cell neurons is computed from the corresponding driver neuron Burst activation (or CaSpkP if not Super)"`
	RSalACh RSalAChParams `viewif:"LayType=RSalienceAChLayer" view:"inline" desc:"parameterizes reward salience as ACh global neuromodulatory signal as a function of the MAX activation of its inputs."`
	RWPred  RWPredParams  `viewif:"LayType=RWPredLayer" view:"inline" desc:"parameterizes reward prediction for a simple Rescorla-Wagner learning dynamic (i.e., PV learning in the PVLV framework)."`
	RWDa    RWDaParams    `viewif:"LayType=RWDaLayer" view:"inline" desc:"parameterizes reward prediction dopamine for a simple Rescorla-Wagner learning dynamic (i.e., PV learning in the PVLV framework)."`
	TDInteg TDIntegParams `viewif:"LayType=TDIntegLayer" view:"inline" desc:"parameterizes TD reward integration layer"`
	TDDa    TDDaParams    `viewif:"LayType=TDDaLayer" view:"inline" desc:"parameterizes dopamine (DA) signal as the temporal difference (TD) between the TDIntegLayer activations in the minus and plus phase."`
	BLA     BLAParams     `viewif:"LayType=TDDaLayer" view:"inline" desc:"parameterizes basolateral amygdala -- most of which is implemented by the NeuroMod settings for DA and ACh modulation."`
	Matrix  MatrixParams  `viewif:"LayType=MatrixLayer" view:"inline" desc:"parameters for BG Striatum Matrix MSN layers, which are the main Go / NoGo gating units in BG."`
	GP      GPParams      `viewif:"LayType=GPLayer" view:"inline" desc:"type of GP Layer."`

	Idxs LayerIdxs `view:"-" desc:"recv and send projection array access info"`
}

func (ly *LayerParams) Update() {
	ly.Act.Update()
	ly.Inhib.Update()
	ly.Learn.Update()

	ly.Burst.Update()
	ly.CT.Update()
	ly.Pulv.Update()

	ly.RSalACh.Update()
	ly.RWPred.Update()
	ly.RWDa.Update()
	ly.TDInteg.Update()
	ly.TDDa.Update()

	ly.BLA.Update()

	ly.Matrix.Update()
	ly.GP.Update()
}

func (ly *LayerParams) Defaults() {
	ly.Act.Defaults()
	ly.Inhib.Defaults()
	ly.Learn.Defaults()
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 1.0
	ly.Inhib.Pool.Gi = 1.0

	ly.Burst.Defaults()
	ly.CT.Defaults()
	ly.Pulv.Defaults()

	ly.RSalACh.Defaults()
	ly.RWPred.Defaults()
	ly.RWDa.Defaults()
	ly.TDInteg.Defaults()
	ly.TDDa.Defaults()

	ly.BLA.Defaults()

	ly.Matrix.Defaults()
	ly.GP.Defaults()
}

// AllParams returns a listing of all parameters in the Layer
func (ly *LayerParams) AllParams() string {
	str := ""
	// todo: replace with a custom reflection crawler that generates
	// the right output directly and filters based on LayType etc.

	b, _ := json.MarshalIndent(&ly.Act, "", " ")
	str += "Act: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Inhib, "", " ")
	str += "Inhib: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Learn, "", " ")
	str += "Learn: {\n " + JsonToParams(b)

	switch ly.LayType {
	case SuperLayer:
		b, _ = json.MarshalIndent(&ly.Burst, "", " ")
		str += "Burst: {\n " + JsonToParams(b)
	case CTLayer:
		b, _ = json.MarshalIndent(&ly.CT, "", " ")
		str += "CT:    {\n " + JsonToParams(b)
	case PulvinarLayer:
		b, _ = json.MarshalIndent(&ly.Pulv, "", " ")
		str += "Pulv:  {\n " + JsonToParams(b)

	case RSalienceAChLayer:
		b, _ = json.MarshalIndent(&ly.RSalACh, "", " ")
		str += "RSalACh: {\n " + JsonToParams(b)
	case RWPredLayer:
		b, _ = json.MarshalIndent(&ly.RWPred, "", " ")
		str += "RWPred: {\n " + JsonToParams(b)
	case RWDaLayer:
		b, _ = json.MarshalIndent(&ly.RWDa, "", " ")
		str += "RWDa:   {\n " + JsonToParams(b)
	case TDIntegLayer:
		b, _ = json.MarshalIndent(&ly.TDInteg, "", " ")
		str += "TDInteg: {\n " + JsonToParams(b)
	case TDDaLayer:
		b, _ = json.MarshalIndent(&ly.TDDa, "", " ")
		str += "TDDa:   {\n " + JsonToParams(b)

	case BLALayer:
		b, _ = json.MarshalIndent(&ly.BLA, "", " ")
		str += "BLA:    {\n " + JsonToParams(b)

	case MatrixLayer:
		b, _ = json.MarshalIndent(&ly.Matrix, "", " ")
		str += "Matrix: {\n " + JsonToParams(b)
	case GPLayer:
		b, _ = json.MarshalIndent(&ly.GP, "", " ")
		str += "GP:     {\n " + JsonToParams(b)
	}
	return str
}

//////////////////////////////////////////////////////////////////////////////////////
//  GeExtToPool

// GeExtToPool adds GeExt from each neuron into the Pools
func (ly *LayerParams) GeExtToPool(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, lpl *Pool, subPool bool) {
	pl.Inhib.GeExtRaw += nrn.GeExt // note: from previous cycle..
	if subPool {
		lpl.Inhib.GeExtRaw += nrn.GeExt
	}
}

// LayPoolGiFmSpikes computes inhibition Gi from Spikes for layer-level pool
func (ly *LayerParams) LayPoolGiFmSpikes(ctx *Context, lpl *Pool, vals *LayerVals) {
	vals.NeuroMod = ctx.NeuroMod
	lpl.Inhib.SpikesFmRaw(lpl.NNeurons())
	ly.Inhib.Layer.Inhib(&lpl.Inhib, vals.ActAvg.GiMult)
}

// SubPoolGiFmSpikes computes inhibition Gi from Spikes within a sub-pool
// pl is guaranteed not to be the overall layer pool
func (ly *LayerParams) SubPoolGiFmSpikes(ctx *Context, pl *Pool, lpl *Pool, lyInhib bool, giMult float32) {
	pl.Inhib.SpikesFmRaw(pl.NNeurons())
	ly.Inhib.Pool.Inhib(&pl.Inhib, giMult)
	if lyInhib {
		pl.Inhib.LayerMax(lpl.Inhib.Gi) // note: this requires lpl inhib to have been computed before!
	} else {
		lpl.Inhib.PoolMax(pl.Inhib.Gi) // display only
		lpl.Inhib.SaveOrig()           // effective GiOrig
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  CycleNeuron methods

////////////////////////
//  GInteg

// NeuronGatherSpikesInit initializes G*Raw and G*Syn values for given neuron
// prior to integration
func (ly *LayerParams) NeuronGatherSpikesInit(ctx *Context, ni uint32, nrn *Neuron) {
	nrn.GeRaw = 0
	nrn.GiRaw = 0
	nrn.GModRaw = 0
	nrn.GModSyn = 0
	nrn.GeSyn = nrn.GeBase
	nrn.GiSyn = nrn.GiBase
}

// See prjnparams for NeuronGatherSpikesPrjn

// SpecialPreGs is used for special layer types to do things to the
// conductance values prior to doing the standard updates in GFmRawSyn
// drvAct is for Pulvinar layers, activation of driving neuron
func (ly *LayerParams) SpecialPreGs(ctx *Context, ni uint32, nrn *Neuron, drvGe float32, nonDrvPct float32, randctr *sltype.Uint2) float32 {
	var saveVal float32 // sometimes we need to use a value computed here, for the post Gs step
	switch ly.LayType {
	case CTLayer:
		geCtxt := ly.CT.GeGain * nrn.CtxtGe
		nrn.GeRaw += geCtxt
		if ly.CT.DecayDt > 0 {
			nrn.CtxtGe -= ly.CT.DecayDt * nrn.CtxtGe
			ctxExt := ly.Act.Dt.GeSynFmRawSteady(geCtxt)
			nrn.GeSyn += ctxExt
			saveVal = ctxExt // used In PostGs to set nrn.GeExt
		}
	case PulvinarLayer:
		if ctx.PlusPhase.IsFalse() {
			break
		}
		nrn.GeRaw = nonDrvPct*nrn.GeRaw + drvGe
		nrn.GeSyn = nonDrvPct*nrn.GeSyn + ly.Act.Dt.GeSynFmRawSteady(drvGe)
	case RewLayer:
		nrn.SetFlag(NeuronHasExt)
		SetNeuronExtPosNeg(ni, nrn, ctx.NeuroMod.Rew) // Rew must be set in Context!
	case RSalienceAChLayer:
		nrn.GeRaw = ctx.NeuroMod.ACh // straight up
		nrn.GeSyn = ly.Act.Dt.GeSynFmRawSteady(nrn.GeRaw)
	case RWDaLayer:
		nrn.GeRaw = ly.RWDa.GeFmDA(ctx.NeuroMod.DA)
		nrn.GeSyn = ly.Act.Dt.GeSynFmRawSteady(nrn.GeRaw)
	case TDDaLayer:
		nrn.GeRaw = ly.TDDa.GeFmDA(ctx.NeuroMod.DA)
		nrn.GeSyn = ly.Act.Dt.GeSynFmRawSteady(nrn.GeRaw)
	case TDIntegLayer:
		nrn.SetFlag(NeuronHasExt)
		SetNeuronExtPosNeg(ni, nrn, ctx.NeuroMod.RewPred)
	}
	return saveVal
}

// SpecialPostGs is used for special layer types to do things
// after the standard updates in GFmRawSyn.
// It is passed the saveVal from SpecialPreGs
func (ly *LayerParams) SpecialPostGs(ctx *Context, ni uint32, nrn *Neuron, randctr *sltype.Uint2, saveVal float32) {
	switch ly.LayType {
	case CTLayer:
		nrn.GeExt = saveVal // todo: it is not clear if this really does anything?  next time around?
	}
}

// GFmRawSyn computes overall Ge and GiSyn conductances for neuron
// from GeRaw and GeSyn values, including NMDA, VGCC, AMPA, and GABA-A channels.
// drvAct is for Pulvinar layers, activation of driving neuron
func (ly *LayerParams) GFmRawSyn(ctx *Context, ni uint32, nrn *Neuron, randctr *sltype.Uint2) {
	if ly.LayType == PTMaintLayer {
		nrn.GeRaw += ly.Act.Dend.ModGain * nrn.GModRaw
		nrn.GeSyn += ly.Act.Dend.ModGain * nrn.GModSyn
	} else if ly.Act.Dend.HasMod.IsTrue() {
		mod := ly.Act.Dend.ModGain * nrn.GModSyn
		if mod > 1 {
			mod = 1
		}
		nrn.GeRaw *= mod
		nrn.GeSyn *= mod
	}

	if nrn.GeSyn > nrn.GeSynMax {
		nrn.GeSynMax = nrn.GeSyn
	}
	geRaw := nrn.GeRaw
	geSyn := nrn.GeSyn
	if ly.LayType == PPTgLayer {
		geSyn = (nrn.GeSyn - nrn.GeSynPrv)
		if geSyn < 0 {
			geSyn = 0
		}
		geRawPrev := nrn.GeSynPrv * ly.Act.Dt.GeDt
		geRaw = (nrn.GeRaw - geRawPrev)
		if geRaw < 0 {
			geRaw = 0
		}
	}
	ly.Act.NMDAFmRaw(nrn, geRaw)
	ly.Learn.LrnNMDAFmRaw(nrn, geRaw)
	ly.Act.GvgccFmVm(nrn)
	ly.Act.GeFmSyn(ni, nrn, geSyn, nrn.Gnmda+nrn.Gvgcc, randctr) // sets nrn.GeExt too
	ly.Act.GkFmVm(nrn)
	ly.Act.GSkCaFmCa(nrn)
	nrn.GiSyn = ly.Act.GiFmSyn(ni, nrn, nrn.GiSyn, randctr)
}

// GiInteg adds Gi values from all sources including SubPool computed inhib
// and updates GABAB as well
func (ly *LayerParams) GiInteg(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, vals *LayerVals) {
	// pl := &ly.Pools[nrn.SubPool]
	nrn.Gi = vals.ActAvg.GiMult*pl.Inhib.Gi + nrn.GiSyn + nrn.GiNoise + ly.Learn.NeuroMod.GiFmACh(vals.NeuroMod.ACh)
	nrn.SSGi = pl.Inhib.SSGi
	nrn.SSGiDend = 0
	if !(ly.Act.Clamp.IsInput.IsTrue() || ly.Act.Clamp.IsTarget.IsTrue()) {
		nrn.SSGiDend = ly.Act.Dend.SSGi * pl.Inhib.SSGi
	}
	ly.Act.GABAB.GABAB(nrn.GABAB, nrn.GABABx, nrn.Gi, &nrn.GABAB, &nrn.GABABx)
	nrn.GgabaB = ly.Act.GABAB.GgabaB(nrn.GABAB, nrn.VmDend)
	nrn.Gk += nrn.GgabaB // Gk was already init
}

// GNeuroMod does neuromodulation of conductances
func (ly *LayerParams) GNeuroMod(ctx *Context, ni uint32, nrn *Neuron, vals *LayerVals) {
	ggain := ly.Learn.NeuroMod.GGain(vals.NeuroMod.DA)
	nrn.Ge *= ggain
	nrn.Gi *= ggain
}

////////////////////////
//  SpikeFmG

// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
func (ly *LayerParams) SpikeFmG(ctx *Context, ni uint32, nrn *Neuron) {
	ly.Act.VmFmG(nrn)
	ly.Act.SpikeFmVm(nrn)
	ly.Learn.CaFmSpike(nrn)
	if ctx.Cycle >= ly.Act.Dt.MaxCycStart {
		nrn.SpkMaxCa += ly.Learn.CaSpk.Dt.PDt * (nrn.CaSpkM - nrn.SpkMaxCa)
		if nrn.SpkMaxCa > nrn.SpkMax {
			nrn.SpkMax = nrn.SpkMaxCa
		}
	}
}

// PostSpikeSpecial does updates at neuron level after spiking has been computed.
// This is where special layer types add extra code.
// It also updates the CaSpkPCyc stats.
func (ly *LayerParams) PostSpikeSpecial(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, lpl *Pool, vals *LayerVals) {
	switch ly.LayType {
	case SuperLayer:
		if ctx.PlusPhase.IsTrue() {
			actMax := lpl.AvgMax.CaSpkP.Cycle.Max
			actAvg := lpl.AvgMax.CaSpkP.Cycle.Avg
			thr := ly.Burst.ThrFmAvgMax(actAvg, actMax)
			burst := float32(0)
			if nrn.CaSpkP > thr {
				burst = nrn.CaSpkP
			}
			nrn.Burst = burst
		}
	case RewLayer:
		nrn.Act = ctx.NeuroMod.Rew
	case RSalienceAChLayer:
		nrn.Act = ctx.NeuroMod.AChRaw // integrated into ActInt
	case RWPredLayer:
		nrn.Act = ly.RWPred.PredRange.ClipVal(nrn.Ge) // clipped linear
		if ni == 0 {
			vals.Special.V1 = nrn.ActInt
		} else {
			vals.Special.V2 = nrn.ActInt
		}
	case RWDaLayer:
		nrn.Act = ctx.NeuroMod.DA // I presumably set this last time..
	case TDPredLayer:
		nrn.Act = nrn.Ge // linear
		if ni == 0 {
			vals.Special.V1 = nrn.ActInt
		} else {
			vals.Special.V2 = nrn.ActInt
		}
	case TDIntegLayer:
		nrn.Act = ctx.NeuroMod.RewPred
	case TDDaLayer:
		nrn.Act = ctx.NeuroMod.DA // I presumably set this last time..
	case PPTgLayer:
		nrn.Act = nrn.CaSpkP - nrn.SpkPrv
		if nrn.Act < 0 {
			nrn.Act = 0
		}
	case MatrixLayer:
		if ly.Learn.NeuroMod.DAMod == D2Mod && !(ly.Learn.NeuroMod.AChDisInhib > 0 && vals.NeuroMod.ACh < 0.2) && ctx.Cycle >= ly.Act.Dt.MaxCycStart {
			if nrn.Ge > nrn.SpkMax {
				nrn.SpkMax = ly.Matrix.NoGoGeLrn * nrn.Ge
			}
		}
	}
}

// PostSpike does updates at neuron level after spiking has been computed.
// it is called *after* PostSpikeSpecial.
// It also updates the CaSpkPCyc stats.
func (ly *LayerParams) PostSpike(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, vals *LayerVals) {
	intdt := ly.Act.Dt.IntDt
	if ctx.PlusPhase.IsTrue() {
		intdt *= 3.0
	}
	nrn.ActInt += intdt * (nrn.Act - nrn.ActInt) // using reg act here now
	if ctx.PlusPhase.IsFalse() {
		nrn.GeM += ly.Act.Dt.IntDt * (nrn.Ge - nrn.GeM)
		nrn.GiM += ly.Act.Dt.IntDt * (nrn.GiSyn - nrn.GiM)
	}
}

/////////////////////////////////////////////////////////////////////////
//  Phase timescale

// NewState handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
func (ly *LayerParams) NewState(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, vals *LayerVals) {
	nrn.SpkPrv = nrn.CaSpkD
	nrn.GeSynPrv = nrn.GeSynMax
	nrn.SpkMax = 0
	nrn.SpkMaxCa = 0
	nrn.GeSynPrv = 0

	ly.Act.DecayState(nrn, ly.Act.Decay.Act, ly.Act.Decay.Glong)
	// ly.Learn.DecayCaLrnSpk(nrn, glong) // NOT called by default
	// Note: synapse-level Ca decay happens in DWt
}

// MinusPhase does neuron level minus-phase updating
func (ly *LayerParams) MinusPhase(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, vals *LayerVals) {
	nrn.ActM = nrn.ActInt
	nrn.CaSpkPM = nrn.CaSpkP
	if nrn.HasFlag(NeuronHasTarg) { // will be clamped in plus phase
		nrn.Ext = nrn.Target
		nrn.SetFlag(NeuronHasExt)
		nrn.ISI = -1 // get fresh update on plus phase output acts
		nrn.ISIAvg = -1
		nrn.ActInt = ly.Act.Init.Act // reset for plus phase
	}
}

// PlusPhase does neuron level plus-phase updating
func (ly *LayerParams) PlusPhase(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, lpl *Pool, vals *LayerVals) {
	nrn.ActP = nrn.ActInt
	mlr := ly.Learn.RLRate.RLRateSigDeriv(nrn.CaSpkD, lpl.AvgMax.CaSpkD.Plus.Max)
	dlr := float32(0)
	if ly.LayType == BLALayer {
		dlr = ly.Learn.RLRate.RLRateDiff(nrn.CaSpkP, nrn.SpkPrv) // delta on previous trial
	} else {
		dlr = ly.Learn.RLRate.RLRateDiff(nrn.CaSpkP, nrn.CaSpkD)
	}
	modlr := ly.Learn.NeuroMod.LRMod(vals.NeuroMod.DA, vals.NeuroMod.ACh)
	nrn.RLRate = mlr * dlr * modlr
	nrn.ActAvg += ly.Act.Dt.LongAvgDt * (nrn.ActM - nrn.ActAvg)
	var tau float32
	ly.Act.Sahp.NinfTauFmCa(nrn.SahpCa, &nrn.SahpN, &tau)
	nrn.SahpCa = ly.Act.Sahp.CaInt(nrn.SahpCa, nrn.CaSpkD)
}

//gosl: end layerparams
