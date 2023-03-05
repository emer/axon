// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"

	"github.com/goki/mat32"
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

// LayerIdxs contains index access into network global arrays for GPU.
type LayerIdxs struct {
	PoolSt uint32 `inactive:"+" desc:"start of pools for this layer -- first one is always the layer-wide pool"`
	NeurSt uint32 `inactive:"+" desc:"start of neurons for this layer in global array (same as Layer.NeurStIdx)"`
	NeurN  uint32 `inactive:"+" desc:"number of neurons in layer"`
	RecvSt uint32 `inactive:"+" desc:"start index into RecvPrjns global array"`
	RecvN  uint32 `inactive:"+" desc:"number of recv projections"`
	SendSt uint32 `inactive:"+" desc:"start index into RecvPrjns global array"`
	SendN  uint32 `inactive:"+" desc:"number of recv projections"`
	ExtsSt uint32 `inactive:"+" desc:"starting index in network global Exts list of external input for this layer -- only for Input / Target / Compare layer types"`
	ShpPlY int32  `inactive:"+" desc:"layer shape Pools Y dimension -- 1 for 2D"`
	ShpPlX int32  `inactive:"+" desc:"layer shape Pools X dimension -- 1 for 2D"`
	ShpUnY int32  `inactive:"+" desc:"layer shape Units Y dimension"`
	ShpUnX int32  `inactive:"+" desc:"layer shape Units X dimension"`
}

// LayerInhibIdxs contains indexes of layers for between-layer inhibition
type LayerInhibIdxs struct {
	Idx1 int32 `inactive:"+" desc:"idx of Layer to get layer-level inhibition from -- set during Build from BuildConfig LayInhib1Name if present -- -1 if not used"`
	Idx2 int32 `inactive:"+" desc:"idx of Layer to get layer-level inhibition from -- set during Build from BuildConfig LayInhib2Name if present -- -1 if not used"`
	Idx3 int32 `inactive:"+" desc:"idx of Layer to get layer-level inhibition from -- set during Build from BuildConfig LayInhib3Name if present -- -1 if not used"`
	Idx4 int32 `inactive:"+" desc:"idx of Layer to geta layer-level inhibition from -- set during Build from BuildConfig LayInhib4Name if present -- -1 if not used"`
}

// note: the following must appear above LayerParams for GPU usage which is order sensitive

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

	Act      ActParams       `view:"add-fields" desc:"Activation parameters and methods for computing activations"`
	Inhib    InhibParams     `view:"add-fields" desc:"Inhibition parameters and methods for computing layer-level inhibition"`
	LayInhib LayerInhibIdxs  `view:"inline" desc:"indexes of layers that contribute between-layer inhibition to this layer -- set these indexes via BuildConfig LayInhibXName (X = 1, 2...)"`
	Learn    LearnNeurParams `view:"add-fields" desc:"Learning parameters and methods that operate at the neuron level"`

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
//  ApplyExt

// ApplyExtFlags gets the clear mask and set mask for updating neuron flags
// based on layer type, and whether input should be applied to Target (else Ext)
func (ly *LayerParams) ApplyExtFlags(clearMask, setMask *NeuronFlags, toTarg *bool) {
	*clearMask = NeuronHasExt | NeuronHasTarg | NeuronHasCmpr
	*toTarg = false
	switch ly.LayType {
	case TargetLayer:
		*setMask = NeuronHasTarg
		*toTarg = true
	case CompareLayer:
		*setMask = NeuronHasCmpr
		*toTarg = true
	default:
		*setMask = NeuronHasExt
	}
	return
}

// InitExt initializes external input state for given neuron
func (ly *LayerParams) InitExt(ni uint32, nrn *Neuron) {
	nrn.Ext = 0
	nrn.Target = 0
	nrn.ClearFlag(NeuronHasExt | NeuronHasTarg | NeuronHasCmpr)
}

// ApplyExtVal applies given external value to given neuron,
// setting flags based on type of layer.
// Should only be called on Input, Target, Compare layers.
// Negative values are not valid, and will be interpreted as missing inputs.
func (ly *LayerParams) ApplyExtVal(ni uint32, nrn *Neuron, val float32) {
	if val < 0 {
		return
	}
	var clearMask, setMask NeuronFlags
	var toTarg bool
	ly.ApplyExtFlags(&clearMask, &setMask, &toTarg)
	if toTarg {
		nrn.Target = val
	} else {
		nrn.Ext = val
	}
	nrn.ClearFlag(clearMask)
	nrn.SetFlag(setMask)
}

// IsTarget returns true if this layer is a Target layer.
// By default, returns true for layers of Type == TargetLayer
// Other Target layers include the TRCLayer in deep predictive learning.
// It is used in SynScale to not apply it to target layers.
// In both cases, Target layers are purely error-driven.
func (ly *LayerParams) IsTarget() bool {
	switch ly.LayType {
	case TargetLayer:
		return true
	case PulvinarLayer:
		return true
	default:
		return false
	}
}

// IsInput returns true if this layer is an Input layer.
// By default, returns true for layers of Type == axon.InputLayer
// Used to prevent adapting of inhibition or TrgAvg values.
func (ly *LayerParams) IsInput() bool {
	switch ly.LayType {
	case InputLayer:
		return true
	default:
		return false
	}
}

// IsInputOrTarget returns true if this layer is either an Input
// or a Target layer.
func (ly *LayerParams) IsInputOrTarget() bool {
	return (ly.IsTarget() || ly.IsInput())
}

// IsLearnTrgAvg returns true if this layer has Learn.TrgAvgAct.On set for learning
// adjustments based on target average activity levels, and the layer is not an
// input or target layer.
func (ly *LayerParams) IsLearnTrgAvg() bool {
	if ly.Act.Clamp.IsInput.IsTrue() || ly.Act.Clamp.IsTarget.IsTrue() || ly.Learn.TrgAvgAct.On.IsFalse() {
		return false
	}
	return true
}

// LearnTrgAvgErrLRate returns the effective error-driven learning rate for adjusting
// target average activity levels.  This is 0 if !IsLearnTrgAvg() and otherwise
// is Learn.TrgAvgAct.ErrLRate
func (ly *LayerParams) LearnTrgAvgErrLRate() float32 {
	if !ly.IsLearnTrgAvg() {
		return 0
	}
	return ly.Learn.TrgAvgAct.ErrLRate
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle methods

// LayPoolGiFmSpikes computes inhibition Gi from Spikes for layer-level pool.
// Also grabs updated Context NeuroMod values into LayerVals
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

// GatherSpikesInit initializes G*Raw and G*Syn values for given neuron
// prior to integration
func (ly *LayerParams) GatherSpikesInit(nrn *Neuron) {
	nrn.GeRaw = 0
	nrn.GiRaw = 0
	nrn.GModRaw = 0
	nrn.GModSyn = 0
	nrn.CtxtGeRaw = 0
	nrn.GeSyn = nrn.GeBase
	nrn.GiSyn = nrn.GiBase
}

////////////////////////
//  GInteg

// todo: pass in pool -- allows VSPatch to set context VSPatch values from pool vals
// needs to be able to set one for each type (pos / neg D1 / D2)

// SpecialPreGs is used for special layer types to do things to the
// conductance values prior to doing the standard updates in GFmRawSyn
// drvAct is for Pulvinar layers, activation of driving neuron
func (ly *LayerParams) SpecialPreGs(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, vals *LayerVals, drvGe float32, nonDrvPct float32) float32 {
	saveVal := float32(0)        // sometimes we need to use a value computed here, for the post Gs step
	pi := int32(nrn.SubPool) - 1 // 0-n pool index
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

	case USLayer:
		us := float32(0)
		if ly.Learn.NeuroMod.Valence == Positive {
			us = ctx.DrivePVLV.USpos.Get(pi)
		} else {
			us = ctx.DrivePVLV.USneg.Get(pi)
		}
		nrn.GeRaw = mat32.Abs(us)
		nrn.GeSyn = ly.Act.Dt.GeSynFmRawSteady(nrn.GeRaw)
	case PVLayer:
		pv := float32(0)
		if ly.Learn.NeuroMod.Valence == Positive {
			pv = ctx.DrivePVLV.VTA.Vals.PVpos
		} else {
			pv = ctx.DrivePVLV.VTA.Vals.PVneg
		}
		pc := ly.Act.PopCode.EncodeVal(ni, ly.Idxs.NeurN, pv)
		nrn.GeRaw = pc
		nrn.GeSyn = ly.Act.Dt.GeSynFmRawSteady(nrn.GeRaw)
	case VTALayer:
		ctx.DrivePVLV.DA(ctx.NeuroMod.PPTg)
		ctx.NeuroMod.DA = ctx.DrivePVLV.VTA.Vals.DA
		nrn.GeRaw = ly.RWDa.GeFmDA(ctx.DrivePVLV.VTA.Vals.DA)
		nrn.GeSyn = ly.Act.Dt.GeSynFmRawSteady(nrn.GeRaw)
	case LHbLayer:
		if ni == 0 {
			nrn.GeRaw = mat32.Abs(ctx.DrivePVLV.LHb.LHbDip)
		} else {
			nrn.GeRaw = mat32.Abs(ctx.DrivePVLV.LHb.LHbBurst)
		}
		nrn.GeSyn = ly.Act.Dt.GeSynFmRawSteady(nrn.GeRaw)
	case DrivesLayer:
		dr := ctx.DrivePVLV.Drive.Drives.Get(pi)
		pc := dr
		if dr > 0 {
			pni := nrn.NeurIdx - pl.StIdx
			pc = ly.Act.PopCode.EncodeVal(pni, uint32(pl.NNeurons()), dr)
		}
		nrn.GeRaw = pc
		nrn.GeSyn = ly.Act.Dt.GeSynFmRawSteady(nrn.GeRaw)
	}
	return saveVal
}

// SpecialPostGs is used for special layer types to do things
// after the standard updates in GFmRawSyn.
// It is passed the saveVal from SpecialPreGs
func (ly *LayerParams) SpecialPostGs(ctx *Context, ni uint32, nrn *Neuron, saveVal float32) {
	switch ly.LayType {
	case CTLayer:
		nrn.GeExt = saveVal // todo: it is not clear if this really does anything?  next time around?
	}
}

// GFmRawSyn computes overall Ge and GiSyn conductances for neuron
// from GeRaw and GeSyn values, including NMDA, VGCC, AMPA, and GABA-A channels.
// drvAct is for Pulvinar layers, activation of driving neuron
func (ly *LayerParams) GFmRawSyn(ctx *Context, ni uint32, nrn *Neuron) {
	extraRaw := float32(0)
	extraSyn := float32(0)
	if ly.LayType == PTMaintLayer {
		extraRaw = ly.Act.Dend.ModGain * nrn.GModRaw
		extraSyn = ly.Act.Dend.ModGain * nrn.GModSyn
	} else if ly.Act.Dend.HasMod.IsTrue() {
		mod := ly.Act.Dend.ModGain * nrn.GModSyn
		if mod > 1 {
			mod = 1
		}
		nrn.GeRaw *= mod
		nrn.GeSyn *= mod
	}

	geRaw := nrn.GeRaw
	geSyn := nrn.GeSyn
	ly.Act.NMDAFmRaw(nrn, geRaw+extraRaw)
	ly.Learn.LrnNMDAFmRaw(nrn, geRaw)
	ly.Act.GvgccFmVm(nrn)
	ly.Act.GeFmSyn(ctx, ni, nrn, geSyn, nrn.Gnmda+nrn.Gvgcc+extraSyn) // sets nrn.GeExt too
	ly.Act.GkFmVm(nrn)
	ly.Act.GSkCaFmCa(nrn)
	nrn.GiSyn = ly.Act.GiFmSyn(ctx, ni, nrn, nrn.GiSyn)
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
	nrn.Burst = nrn.CaSpkP
	pi := int32(nrn.SubPool) - 1 // 0-n pool index
	switch ly.LayType {
	case SuperLayer:
		if ctx.PlusPhase.IsTrue() {
			actMax := lpl.AvgMax.CaSpkP.Cycle.Max
			actAvg := lpl.AvgMax.CaSpkP.Cycle.Avg
			thr := ly.Burst.ThrFmAvgMax(actAvg, actMax)
			if nrn.CaSpkP < thr {
				nrn.Burst = 0
			}
		}
	case CTLayer:
		if ctx.Cycle == ctx.ThetaCycles-1 {
			nrn.CtxtGe += nrn.CtxtGeRaw
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
	case MatrixLayer:
		if ly.Learn.NeuroMod.DAMod == D2Mod && !(ly.Learn.NeuroMod.AChDisInhib > 0 && vals.NeuroMod.ACh < 0.2) && ctx.Cycle >= ly.Act.Dt.MaxCycStart {
			if nrn.Ge > nrn.SpkMax {
				nrn.SpkMax = ly.Matrix.NoGoGeLrn * nrn.Ge
			}
		}

	case PPTgLayer:
		if ni == 0 {
			ctx.NeuroMod.PPTg = lpl.AvgMax.Act.Cycle.Max // todo: use CaSpkP instead of Act?
		}
	case USLayer:
		us := float32(0)
		if ly.Learn.NeuroMod.Valence == Positive {
			us = ctx.DrivePVLV.USpos.Get(pi)
		} else {
			us = ctx.DrivePVLV.USneg.Get(pi)
		}
		nrn.Act = us
	case PVLayer:
		pv := float32(0)
		if ly.Learn.NeuroMod.Valence == Positive {
			pv = ctx.DrivePVLV.VTA.Vals.PVpos
		} else {
			pv = ctx.DrivePVLV.VTA.Vals.PVneg
		}
		pc := ly.Act.PopCode.EncodeVal(ni, ly.Idxs.NeurN, pv)
		nrn.Act = pc
	case VTALayer:
		nrn.Act = ctx.DrivePVLV.VTA.Vals.DA
	case LHbLayer:
		if ni == 0 {
			nrn.Act = ctx.DrivePVLV.LHb.LHbDip
		} else {
			nrn.Act = ctx.DrivePVLV.LHb.LHbBurst
		}
		nrn.GeSyn = ly.Act.Dt.GeSynFmRawSteady(nrn.GeRaw)
	case DrivesLayer:
		dr := ctx.DrivePVLV.Drive.Drives.Get(pi)
		pc := dr
		if dr > 0 {
			pni := nrn.NeurIdx - pl.StIdx
			pc = ly.Act.PopCode.EncodeVal(pni, uint32(pl.NNeurons()), dr)
		}
		nrn.Act = pc
	case VSPatchLayer:
		if nrn.NeurIdx == pl.StIdx {
			val := pl.AvgMax.CaSpkD.Cycle.Avg / ly.Inhib.ActAvg.Nominal
			ctx.DrivePVLV.VSPatchVals.SetVal(val, pi, ly.Learn.NeuroMod.Valence, ly.Learn.NeuroMod.DAMod)
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
//  Special CyclePost methods for different layer types

// RSalAChMaxLayAct returns the updated maxAct value using
// LayVals.ActAvg.CaSpkP.Max from given layer index,
// subject to any relevant RewThr thresholding.
func (ly *LayerParams) RSalAChMaxLayAct(maxAct, layMaxAct float32) float32 {
	act := ly.RSalACh.Thr(layMaxAct) // use Act -- otherwise too variable
	if act > maxAct {
		maxAct = act
	}
	return maxAct
}

func (ly *LayerParams) CyclePostRSalAChLayer(ctx *Context, vals *LayerVals, lay1MaxAct, lay2MaxAct, lay3MaxAct, lay4MaxAct, lay5MaxAct float32) {
	maxAct := float32(0)
	if ly.RSalACh.Rew.IsTrue() {
		if ctx.NeuroMod.HasRew.IsTrue() {
			maxAct = 1
		}
	}
	if ly.RSalACh.RewPred.IsTrue() {
		rpAct := ly.RSalACh.Thr(ctx.NeuroMod.RewPred)
		if rpAct > maxAct {
			maxAct = rpAct
		}
	}
	maxAct = ly.RSalAChMaxLayAct(maxAct, lay1MaxAct)
	maxAct = ly.RSalAChMaxLayAct(maxAct, lay2MaxAct)
	maxAct = ly.RSalAChMaxLayAct(maxAct, lay3MaxAct)
	maxAct = ly.RSalAChMaxLayAct(maxAct, lay4MaxAct)
	maxAct = ly.RSalAChMaxLayAct(maxAct, lay5MaxAct)
	vals.NeuroMod.AChRaw = maxAct
	vals.NeuroMod.AChFmRaw(ly.Act.Dt.IntDt)

	ctx.NeuroMod.AChRaw = vals.NeuroMod.AChRaw
	ctx.NeuroMod.ACh = vals.NeuroMod.ACh
}

func (ly *LayerParams) CyclePostRWDaLayer(ctx *Context, vals *LayerVals, pvals *LayerVals) {
	pred := pvals.Special.V1 - pvals.Special.V2
	ctx.NeuroMod.RewPred = pred // record
	da := float32(0)
	if ctx.NeuroMod.HasRew.IsTrue() {
		da = ctx.NeuroMod.Rew - pred
	}
	ctx.NeuroMod.DA = da // updates global value that will be copied to layers next cycle.
	vals.NeuroMod.DA = da
}

func (ly *LayerParams) CyclePostTDPredLayer(ctx *Context, vals *LayerVals) {
	if ctx.PlusPhase.IsTrue() {
		pred := vals.Special.V1 - vals.Special.V2
		ctx.NeuroMod.PrevPred = pred
	}
}

func (ly *LayerParams) CyclePostTDIntegLayer(ctx *Context, vals *LayerVals, pvals *LayerVals) {
	rew := float32(0)
	if ctx.NeuroMod.HasRew.IsTrue() {
		rew = ctx.NeuroMod.Rew
	}
	rpval := float32(0)
	if ctx.PlusPhase.IsTrue() {
		pred := pvals.Special.V1 - pvals.Special.V2 // neuron0 (pos) - neuron1 (neg)
		rpval = rew + ly.TDInteg.Discount*ly.TDInteg.PredGain*pred
		vals.Special.V2 = rpval // plus phase
	} else {
		rpval = ly.TDInteg.PredGain * ctx.NeuroMod.PrevPred
		vals.Special.V1 = rpval // minus phase is *previous trial*
	}
	ctx.NeuroMod.RewPred = rpval // global value will be copied to layers next cycle
}

func (ly *LayerParams) CyclePostTDDaLayer(ctx *Context, vals *LayerVals, ivals *LayerVals) {
	da := ivals.Special.V2 - ivals.Special.V1
	if ctx.PlusPhase.IsFalse() {
		da = 0
	}
	ctx.NeuroMod.DA = da // updates global value that will be copied to layers next cycle.
	vals.NeuroMod.DA = da
}

/////////////////////////////////////////////////////////////////////////
//  Phase timescale

// ActAvgFmAct computes the LayerVals ActAvg from act values -- at start of new state
func (ly *LayerParams) ActAvgFmAct(ctx *Context, lpl *Pool, vals *LayerVals) {
	ly.Inhib.ActAvg.AvgFmAct(&vals.ActAvg.ActMAvg, lpl.AvgMax.Act.Minus.Avg, ly.Act.Dt.LongAvgDt)
	ly.Inhib.ActAvg.AvgFmAct(&vals.ActAvg.ActPAvg, lpl.AvgMax.Act.Plus.Avg, ly.Act.Dt.LongAvgDt)
}

func (ly *LayerParams) NewStateLayer(ctx *Context, lpl *Pool, vals *LayerVals) {
	ly.ActAvgFmAct(ctx, lpl, vals)
	ly.Act.Clamp.IsInput.SetBool(ly.IsInput())
	ly.Act.Clamp.IsTarget.SetBool(ly.IsTarget())
}

func (ly *LayerParams) NewStatePool(ctx *Context, pl *Pool) {
	pl.Inhib.Clamped.SetBool(false)
	if ly.Act.Clamp.Add.IsFalse() && ly.Act.Clamp.IsInput.IsTrue() {
		pl.Inhib.Clamped.SetBool(true)
	}
	pl.Inhib.Decay(ly.Act.Decay.Act)
}

// NewStateNeuron handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
func (ly *LayerParams) NewStateNeuron(ctx *Context, ni uint32, nrn *Neuron, vals *LayerVals) {
	nrn.BurstPrv = nrn.Burst
	nrn.SpkPrv = nrn.CaSpkD
	nrn.SpkMax = 0
	nrn.SpkMaxCa = 0

	ly.Act.DecayState(nrn, ly.Act.Decay.Act, ly.Act.Decay.Glong)
	// ly.Learn.DecayCaLrnSpk(nrn, glong) // NOT called by default
	// Note: synapse-level Ca decay happens in DWt
}

func (ly *LayerParams) MinusPhasePool(ctx *Context, pl *Pool) {
	pl.AvgMax.CycleToMinus()
	if ly.Act.Clamp.Add.IsFalse() && ly.Act.Clamp.IsTarget.IsTrue() {
		pl.Inhib.Clamped.SetBool(true)
	}
}

// AvgGeM computes the average and max GeM stats, updated in MinusPhase
func (ly *LayerParams) AvgGeM(ctx *Context, lpl *Pool, vals *LayerVals) {
	vals.ActAvg.AvgMaxGeM += ly.Act.Dt.LongAvgDt * (lpl.AvgMax.Ge.Minus.Max - vals.ActAvg.AvgMaxGeM)
	vals.ActAvg.AvgMaxGiM += ly.Act.Dt.LongAvgDt * (lpl.AvgMax.Gi.Minus.Max - vals.ActAvg.AvgMaxGiM)
}

// MinusPhaseNeuron does neuron level minus-phase updating
func (ly *LayerParams) MinusPhaseNeuron(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, lpl *Pool, vals *LayerVals) {
	nrn.ActM = nrn.ActInt
	nrn.CaSpkPM = nrn.CaSpkP
}

// PlusPhaseStartNeuron does neuron level plus-phase start:
// applies Target inputs as External inputs.
func (ly *LayerParams) PlusPhaseStartNeuron(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, lpl *Pool, vals *LayerVals) {
	if nrn.HasFlag(NeuronHasTarg) { // will be clamped in plus phase
		nrn.Ext = nrn.Target
		nrn.SetFlag(NeuronHasExt)
		nrn.ISI = -1 // get fresh update on plus phase output acts
		nrn.ISIAvg = -1
		nrn.ActInt = ly.Act.Init.Act // reset for plus phase
	}
}

func (ly *LayerParams) PlusPhasePool(ctx *Context, pl *Pool) {
	pl.AvgMax.CycleToPlus()
}

// PlusPhaseNeuron does neuron level plus-phase updating
func (ly *LayerParams) PlusPhaseNeuron(ctx *Context, ni uint32, nrn *Neuron, pl *Pool, lpl *Pool, vals *LayerVals) {
	nrn.ActP = nrn.ActInt
	mlr := ly.Learn.RLRate.RLRateSigDeriv(nrn.CaSpkD, lpl.AvgMax.CaSpkD.Cycle.Max)
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
	nrn.DTrgAvg += ly.LearnTrgAvgErrLRate() * (nrn.CaSpkP - nrn.CaSpkD)
}

//gosl: end layerparams
