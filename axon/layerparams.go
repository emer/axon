// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"

	"cogentcore.org/core/mat32"
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

// LayerIndexes contains index access into network global arrays for GPU.
type LayerIndexes struct {

	// layer index
	LayIndex uint32 `edit:"-"`

	// maximum number of data parallel elements
	MaxData uint32 `edit:"-"`

	// start of pools for this layer -- first one is always the layer-wide pool
	PoolSt uint32 `edit:"-"`

	// start of neurons for this layer in global array (same as Layer.NeurStIndex)
	NeurSt uint32 `edit:"-"`

	// number of neurons in layer
	NeurN uint32 `edit:"-"`

	// start index into RecvPrjns global array
	RecvSt uint32 `edit:"-"`

	// number of recv projections
	RecvN uint32 `edit:"-"`

	// start index into RecvPrjns global array
	SendSt uint32 `edit:"-"`

	// number of recv projections
	SendN uint32 `edit:"-"`

	// starting index in network global Exts list of external input for this layer -- only for Input / Target / Compare layer types
	ExtsSt uint32 `edit:"-"`

	// layer shape Pools Y dimension -- 1 for 2D
	ShpPlY int32 `edit:"-"`

	// layer shape Pools X dimension -- 1 for 2D
	ShpPlX int32 `edit:"-"`

	// layer shape Units Y dimension
	ShpUnY int32 `edit:"-"`

	// layer shape Units X dimension
	ShpUnX int32 `edit:"-"`

	pad, pad1 uint32
}

// PoolIndex returns the global network index for pool with given
// pool (0 = layer pool, 1+ = subpools) and data parallel indexes
func (lx *LayerIndexes) PoolIndex(pi, di uint32) uint32 {
	return lx.PoolSt + pi*lx.MaxData + di
}

// ValuesIndex returns the global network index for LayerValues with given
// data parallel index.
func (lx *LayerIndexes) ValuesIndex(di uint32) uint32 {
	return lx.LayIndex*lx.MaxData + di
}

// ExtIndex returns the index for accessing Exts values: [Neuron][Data]
// Neuron is *layer-relative* lni index -- add the ExtsSt for network level access.
func (lx *LayerIndexes) ExtIndex(ni, di uint32) uint32 {
	return ni*lx.MaxData + di
}

// LayerInhibIndexes contains indexes of layers for between-layer inhibition
type LayerInhibIndexes struct {

	// idx of Layer to get layer-level inhibition from -- set during Build from BuildConfig LayInhib1Name if present -- -1 if not used
	Index1 int32 `edit:"-"`

	// idx of Layer to get layer-level inhibition from -- set during Build from BuildConfig LayInhib2Name if present -- -1 if not used
	Index2 int32 `edit:"-"`

	// idx of Layer to get layer-level inhibition from -- set during Build from BuildConfig LayInhib3Name if present -- -1 if not used
	Index3 int32 `edit:"-"`

	// idx of Layer to geta layer-level inhibition from -- set during Build from BuildConfig LayInhib4Name if present -- -1 if not used
	Index4 int32 `edit:"-"`
}

// note: the following must appear above LayerParams for GPU usage which is order sensitive

// SetNeuronExtPosNeg sets neuron Ext value based on neuron index
// with positive values going in first unit, negative values rectified
// to positive in 2nd unit
func SetNeuronExtPosNeg(ctx *Context, ni, di uint32, val float32) {
	if ni == 0 {
		if val >= 0 {
			SetNrnV(ctx, ni, di, Ext, val)
		} else {
			SetNrnV(ctx, ni, di, Ext, 0)
		}
	} else {
		if val >= 0 {
			SetNrnV(ctx, ni, di, Ext, 0)
		} else {
			SetNrnV(ctx, ni, di, Ext, -val)
		}
	}
}

// LayerParams contains all of the layer parameters.
// These values must remain constant over the course of computation.
// On the GPU, they are loaded into a uniform.
type LayerParams struct {

	// functional type of layer -- determines functional code path for specialized layer types, and is synchronized with the Layer.Typ value
	LayType LayerTypes

	pad, pad1, pad2 int32

	// Activation parameters and methods for computing activations
	Acts ActParams `view:"add-fields"`

	// Inhibition parameters and methods for computing layer-level inhibition
	Inhib InhibParams `view:"add-fields"`

	// indexes of layers that contribute between-layer inhibition to this layer -- set these indexes via BuildConfig LayInhibXName (X = 1, 2...)
	LayInhib LayerInhibIndexes `view:"inline"`

	// Learning parameters and methods that operate at the neuron level
	Learn LearnNeurParams `view:"add-fields"`

	// BurstParams determine how the 5IB Burst activation is computed from CaSpkP integrated spiking values in Super layers -- thresholded.
	Bursts BurstParams `view:"inline"`

	// ] params for the CT corticothalamic layer and PTPred layer that generates predictions over the Pulvinar using context -- uses the CtxtGe excitatory input plus stronger NMDA channels to maintain context trace
	CT CTParams `view:"inline"`

	// provides parameters for how the plus-phase (outcome) state of Pulvinar thalamic relay cell neurons is computed from the corresponding driver neuron Burst activation (or CaSpkP if not Super)
	Pulv PulvParams `view:"inline"`

	// parameters for BG Striatum Matrix MSN layers, which are the main Go / NoGo gating units in BG.
	Matrix MatrixParams `view:"inline"`

	// type of GP Layer.
	GP GPParams `view:"inline"`

	// parameters for VSPatch learning
	VSPatch VSPatchParams `view:"inline"`

	// parameterizes laterodorsal tegmentum ACh salience neuromodulatory signal, driven by superior colliculus stimulus novelty, US input / absence, and OFC / ACC inhibition
	LDT LDTParams `view:"inline"`

	// parameterizes computing overall VTA DA based on LHb PVDA (primary value -- at US time, computed at start of each trial and stored in LHbPVDA global value) and Amygdala (CeM) CS / learned value (LV) activations, which update every cycle.
	VTA VTAParams `view:"inline"`

	// parameterizes reward prediction for a simple Rescorla-Wagner learning dynamic (i.e., PV learning in the PVLV framework).
	RWPred RWPredParams `view:"inline"`

	// parameterizes reward prediction dopamine for a simple Rescorla-Wagner learning dynamic (i.e., PV learning in the PVLV framework).
	RWDa RWDaParams `view:"inline"`

	// parameterizes TD reward integration layer
	TDInteg TDIntegParams `view:"inline"`

	// parameterizes dopamine (DA) signal as the temporal difference (TD) between the TDIntegLayer activations in the minus and plus phase.
	TDDa TDDaParams `view:"inline"`

	// recv and send projection array access info
	Indexes LayerIndexes
}

func (ly *LayerParams) Update() {
	ly.Acts.Update()
	ly.Inhib.Update()
	ly.Learn.Update()

	ly.Bursts.Update()
	ly.CT.Update()
	ly.Pulv.Update()

	ly.Matrix.Update()
	ly.GP.Update()

	ly.VSPatch.Update()
	ly.LDT.Update()
	ly.VTA.Update()

	ly.RWPred.Update()
	ly.RWDa.Update()
	ly.TDInteg.Update()
	ly.TDDa.Update()
}

func (ly *LayerParams) Defaults() {
	ly.Acts.Defaults()
	ly.Inhib.Defaults()
	ly.Learn.Defaults()
	ly.Inhib.Layer.On.SetBool(true)
	ly.Inhib.Layer.Gi = 1.0
	ly.Inhib.Pool.Gi = 1.0

	ly.Bursts.Defaults()
	ly.CT.Defaults()
	ly.Pulv.Defaults()

	ly.Matrix.Defaults()
	ly.GP.Defaults()

	ly.VSPatch.Defaults()
	ly.LDT.Defaults()
	ly.VTA.Defaults()

	ly.RWPred.Defaults()
	ly.RWDa.Defaults()
	ly.TDInteg.Defaults()
	ly.TDDa.Defaults()
}

func (ly *LayerParams) ShouldShow(field string) bool {
	switch field {
	case "Bursts":
		return ly.LayType == SuperLayer
	case "CT":
		return ly.LayType == CTLayer || ly.LayType == PTPredLayer || ly.LayType == BLALayer
	case "Pulv":
		return ly.LayType == PulvinarLayer
	case "Matrix":
		return ly.LayType == MatrixLayer
	case "GP":
		return ly.LayType == GPLayer
	case "VSPatch":
		return ly.LayType == VSPatchLayer
	case "LDT":
		return ly.LayType == LDTLayer
	case "VTA":
		return ly.LayType == VTALayer
	case "RWPred":
		return ly.LayType == RWPredLayer
	case "RWDa":
		return ly.LayType == RWDaLayer
	case "TDInteg":
		return ly.LayType == TDIntegLayer
	case "TDDa":
		return ly.LayType == TDDaLayer
	default:
		return true
	}
}

// AllParams returns a listing of all parameters in the Layer
func (ly *LayerParams) AllParams() string {
	str := ""
	// todo: replace with a custom reflection crawler that generates
	// the right output directly and filters based on LayType etc.

	b, _ := json.MarshalIndent(&ly.Acts, "", " ")
	str += "Act: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Inhib, "", " ")
	str += "Inhib: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Learn, "", " ")
	str += "Learn: {\n " + JsonToParams(b)

	switch ly.LayType {
	case SuperLayer:
		b, _ = json.MarshalIndent(&ly.Bursts, "", " ")
		str += "Burst:   {\n " + JsonToParams(b)
	case CTLayer, PTPredLayer, BLALayer:
		b, _ = json.MarshalIndent(&ly.CT, "", " ")
		str += "CT:      {\n " + JsonToParams(b)
	case PulvinarLayer:
		b, _ = json.MarshalIndent(&ly.Pulv, "", " ")
		str += "Pulv:    {\n " + JsonToParams(b)

	case MatrixLayer:
		b, _ = json.MarshalIndent(&ly.Matrix, "", " ")
		str += "Matrix:  {\n " + JsonToParams(b)
	case GPLayer:
		b, _ = json.MarshalIndent(&ly.GP, "", " ")
		str += "GP:      {\n " + JsonToParams(b)

	case VSPatchLayer:
		b, _ = json.MarshalIndent(&ly.VSPatch, "", " ")
		str += "VSPatch: {\n " + JsonToParams(b)
	case LDTLayer:
		b, _ = json.MarshalIndent(&ly.LDT, "", " ")
		str += "LDT: {\n " + JsonToParams(b)
	case VTALayer:
		b, _ = json.MarshalIndent(&ly.VTA, "", " ")
		str += "VTA: {\n " + JsonToParams(b)

	case RWPredLayer:
		b, _ = json.MarshalIndent(&ly.RWPred, "", " ")
		str += "RWPred:  {\n " + JsonToParams(b)
	case RWDaLayer:
		b, _ = json.MarshalIndent(&ly.RWDa, "", " ")
		str += "RWDa:    {\n " + JsonToParams(b)
	case TDIntegLayer:
		b, _ = json.MarshalIndent(&ly.TDInteg, "", " ")
		str += "TDInteg: {\n " + JsonToParams(b)
	case TDDaLayer:
		b, _ = json.MarshalIndent(&ly.TDDa, "", " ")
		str += "TDDa:    {\n " + JsonToParams(b)
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
func (ly *LayerParams) InitExt(ctx *Context, ni, di uint32) {
	SetNrnV(ctx, ni, di, Ext, 0)
	SetNrnV(ctx, ni, di, Target, 0)
	NrnClearFlag(ctx, ni, di, NeuronHasExt|NeuronHasTarg|NeuronHasCmpr)
}

// ApplyExtVal applies given external value to given neuron,
// setting flags based on type of layer.
// Should only be called on Input, Target, Compare layers.
// Negative values are not valid, and will be interpreted as missing inputs.
func (ly *LayerParams) ApplyExtValue(ctx *Context, ni, di uint32, val float32) {
	if val < 0 {
		return
	}
	var clearMask, setMask NeuronFlags
	var toTarg bool
	ly.ApplyExtFlags(&clearMask, &setMask, &toTarg)
	if toTarg {
		SetNrnV(ctx, ni, di, Target, val)
	} else {
		SetNrnV(ctx, ni, di, Ext, val)
	}
	NrnClearFlag(ctx, ni, di, clearMask)
	NrnSetFlag(ctx, ni, di, setMask)
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

// IsLearnTrgAvg returns true if this layer has Learn.TrgAvgAct.RescaleOn set for learning
// adjustments based on target average activity levels, and the layer is not an
// input or target layer.
func (ly *LayerParams) IsLearnTrgAvg() bool {
	if ly.Acts.Clamp.IsInput.IsTrue() || ly.Acts.Clamp.IsTarget.IsTrue() || ly.Learn.TrgAvgAct.RescaleOn.IsFalse() {
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
// Also grabs updated Context NeuroMod values into LayerValues
func (ly *LayerParams) LayPoolGiFmSpikes(ctx *Context, lpl *Pool, vals *LayerValues) {
	lpl.Inhib.SpikesFmRaw(lpl.NNeurons())
	ly.Inhib.Layer.Inhib(&lpl.Inhib, vals.ActAvg.GiMult)
}

// SubPoolGiFmSpikes computes inhibition Gi from Spikes within a sub-pool
// pl is guaranteed not to be the overall layer pool
func (ly *LayerParams) SubPoolGiFmSpikes(ctx *Context, di uint32, pl *Pool, lpl *Pool, lyInhib bool, giMult float32) {
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
func (ly *LayerParams) GatherSpikesInit(ctx *Context, ni, di uint32) {
	SetNrnV(ctx, ni, di, GeRaw, 0)
	SetNrnV(ctx, ni, di, GiRaw, 0)
	SetNrnV(ctx, ni, di, GModRaw, 0)
	SetNrnV(ctx, ni, di, GModSyn, 0)
	SetNrnV(ctx, ni, di, GMaintRaw, 0)
	SetNrnV(ctx, ni, di, CtxtGeRaw, 0)
	SetNrnV(ctx, ni, di, GeSyn, NrnAvgV(ctx, ni, GeBase))
	SetNrnV(ctx, ni, di, GiSyn, NrnAvgV(ctx, ni, GiBase))
}

////////////////////////
//  GInteg

// SpecialPreGs is used for special layer types to do things to the
// conductance values prior to doing the standard updates in GFmRawSyn
// drvAct is for Pulvinar layers, activation of driving neuron
func (ly *LayerParams) SpecialPreGs(ctx *Context, ni, di uint32, pl *Pool, vals *LayerValues, drvGe float32, nonDrivePct float32) float32 {
	saveVal := float32(0)               // sometimes we need to use a value computed here, for the post Gs step
	pi := NrnI(ctx, ni, NrnSubPool) - 1 // 0-n pool index
	pni := NrnI(ctx, ni, NrnNeurIndex) - pl.StIndex
	nrnCtxtGe := NrnV(ctx, ni, di, CtxtGe)
	nrnGeRaw := NrnV(ctx, ni, di, GeRaw)
	switch ly.LayType {
	case CTLayer:
		fallthrough
	case PTPredLayer:
		geCtxt := ly.CT.GeGain * nrnCtxtGe
		AddNrnV(ctx, ni, di, GeRaw, geCtxt)
		if ly.CT.DecayDt > 0 {
			AddNrnV(ctx, ni, di, CtxtGe, -ly.CT.DecayDt*nrnCtxtGe)
		}
		ctxExt := ly.Acts.Dt.GeSynFmRawSteady(geCtxt)
		AddNrnV(ctx, ni, di, GeSyn, ctxExt)
		saveVal = ctxExt // used In PostGs to set nrn.GeExt
	case PTMaintLayer:
		if ly.Acts.SMaint.On.IsTrue() {
			saveVal = ly.Acts.SMaint.Inhib * NrnV(ctx, ni, di, GMaintRaw) // used In PostGs to set nrn.GeExt
		}
	case PulvinarLayer:
		if ctx.PlusPhase.IsFalse() {
			break
		}
		// geSyn, goes into nrn.GeExt in PostGs, so inhibition gets it
		saveVal = nonDrivePct*NrnV(ctx, ni, di, GeSyn) + ly.Acts.Dt.GeSynFmRawSteady(drvGe)
		SetNrnV(ctx, ni, di, GeRaw, nonDrivePct*nrnGeRaw+drvGe)
		SetNrnV(ctx, ni, di, GeSyn, saveVal)
	case VSGatedLayer:
		dr := float32(0)
		if pi == 0 {
			dr = GlbV(ctx, di, GvVSMatrixJustGated)
		} else {
			dr = GlbV(ctx, di, GvVSMatrixHasGated)
		}
		dr = mat32.Abs(dr)
		SetNrnV(ctx, ni, di, GeRaw, dr)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(dr))

	case BLALayer:
		// only for ext type:
		if ly.Learn.NeuroMod.IsBLAExt() {
			geCtxt := GlbV(ctx, di, GvACh) * ly.CT.GeGain * NrnV(ctx, ni, di, CtxtGeOrig)
			AddNrnV(ctx, ni, di, GeRaw, geCtxt)
			ctxExt := ly.Acts.Dt.GeSynFmRawSteady(geCtxt)
			AddNrnV(ctx, ni, di, GeSyn, ctxExt)
			saveVal = ctxExt // used In PostGs to set nrn.GeExt
		}
	case LHbLayer:
		geRaw := float32(0)
		if ni == 0 {
			geRaw = 0.2 * mat32.Abs(GlbV(ctx, di, GvLHbDip))
		} else {
			geRaw = 0.2 * mat32.Abs(GlbV(ctx, di, GvLHbBurst))
		}
		SetNrnV(ctx, ni, di, GeRaw, geRaw)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(geRaw))
	case DrivesLayer:
		dr := GlbUSposV(ctx, di, GvDrives, uint32(pi))
		geRaw := dr
		if dr > 0 {
			geRaw = ly.Acts.PopCode.EncodeGe(pni, uint32(pl.NNeurons()), dr)
		}
		SetNrnV(ctx, ni, di, GeRaw, geRaw)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(geRaw))
	case UrgencyLayer:
		ur := GlbV(ctx, di, GvUrgency)
		geRaw := ur
		if ur > 0 {
			geRaw = ly.Acts.PopCode.EncodeGe(pni, uint32(pl.NNeurons()), ur)
		}
		SetNrnV(ctx, ni, di, GeRaw, geRaw)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(geRaw))
	case USLayer:
		us := PVLVUSStimValue(ctx, di, pi, ly.Learn.NeuroMod.Valence)
		geRaw := us
		if us > 0 {
			geRaw = ly.Acts.PopCode.EncodeGe(pni, uint32(pl.NNeurons()), us)
		}
		SetNrnV(ctx, ni, di, GeRaw, geRaw)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(geRaw))
	case PVLayer:
		pv := float32(0)
		if ly.Learn.NeuroMod.Valence == Positive {
			pv = GlbV(ctx, di, GvPVpos)
		} else {
			pv = GlbV(ctx, di, GvPVneg)
		}
		pc := ly.Acts.PopCode.EncodeGe(pni, ly.Indexes.NeurN, pv)
		SetNrnV(ctx, ni, di, GeRaw, pc)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(pc))
	case LDTLayer:
		geRaw := 0.4 * GlbV(ctx, di, GvACh)
		SetNrnV(ctx, ni, di, GeRaw, geRaw)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(geRaw))
	case VTALayer:
		geRaw := ly.RWDa.GeFmDA(GlbV(ctx, di, GvVtaDA))
		SetNrnV(ctx, ni, di, GeRaw, geRaw)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(geRaw))

	case RewLayer:
		NrnSetFlag(ctx, ni, di, NeuronHasExt)
		SetNeuronExtPosNeg(ctx, ni, di, GlbV(ctx, di, GvRew)) // Rew must be set in Context!
	case RWDaLayer:
		geRaw := ly.RWDa.GeFmDA(GlbV(ctx, di, GvDA))
		SetNrnV(ctx, ni, di, GeRaw, geRaw)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(geRaw))
	case TDDaLayer:
		geRaw := ly.TDDa.GeFmDA(GlbV(ctx, di, GvDA))
		SetNrnV(ctx, ni, di, GeRaw, geRaw)
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(geRaw))
	case TDIntegLayer:
		NrnSetFlag(ctx, ni, di, NeuronHasExt)
		SetNeuronExtPosNeg(ctx, ni, di, GlbV(ctx, di, GvRewPred))
	}
	return saveVal
}

// SpecialPostGs is used for special layer types to do things
// after the standard updates in GFmRawSyn.
// It is passed the saveVal from SpecialPreGs
func (ly *LayerParams) SpecialPostGs(ctx *Context, ni, di uint32, saveVal float32) {
	switch ly.LayType {
	case BLALayer:
		fallthrough
	case CTLayer:
		fallthrough
	case PTMaintLayer:
		fallthrough
	case PulvinarLayer:
		SetNrnV(ctx, ni, di, GeExt, saveVal)
	case PTPredLayer:
		SetNrnV(ctx, ni, di, GeExt, saveVal)
		orig := NrnV(ctx, ni, di, CtxtGeOrig)
		if orig < 0.05 {
			SetNrnV(ctx, ni, di, Ge, 0) // gated by context input
		}
	}
}

// GFmRawSyn computes overall Ge and GiSyn conductances for neuron
// from GeRaw and GeSyn values, including NMDA, VGCC, AMPA, and GABA-A channels.
// drvAct is for Pulvinar layers, activation of driving neuron
func (ly *LayerParams) GFmRawSyn(ctx *Context, ni, di uint32) {
	extraRaw := float32(0)
	extraSyn := float32(0)
	nrnGModRaw := NrnV(ctx, ni, di, GModRaw)
	nrnGModSyn := NrnV(ctx, ni, di, GModSyn)
	ach := GlbV(ctx, di, GvACh)
	switch ly.LayType {
	case PTMaintLayer:
		mod := ly.Acts.Dend.ModGain * nrnGModSyn
		if ly.Acts.Dend.ModACh.IsTrue() {
			mod *= ach
		}
		mod += ly.Acts.Dend.ModBase
		MulNrnV(ctx, ni, di, GeRaw, mod) // key: excluding GModMaint here, so active maintenance can persist
		MulNrnV(ctx, ni, di, GeSyn, mod)
		extraRaw = ly.Acts.Dend.ModGain * nrnGModRaw
		if ly.Acts.Dend.ModACh.IsTrue() {
			extraRaw *= ach
		}
		extraSyn = mod
	case BLALayer:
		extraRaw = ach * nrnGModRaw * ly.Acts.Dend.ModGain
		extraSyn = ach * nrnGModSyn * ly.Acts.Dend.ModGain
	default:
		if ly.Acts.Dend.HasMod.IsTrue() {
			mod := ly.Acts.Dend.ModBase + ly.Acts.Dend.ModGain*nrnGModSyn
			if mod > 1 {
				mod = 1
			}
			MulNrnV(ctx, ni, di, GeRaw, mod)
			MulNrnV(ctx, ni, di, GeSyn, mod)
		}
	}
	geRaw := NrnV(ctx, ni, di, GeRaw)
	geSyn := NrnV(ctx, ni, di, GeSyn)
	ly.Acts.NMDAFmRaw(ctx, ni, di, geRaw+extraRaw)
	ly.Acts.MaintNMDAFmRaw(ctx, ni, di) // uses GMaintRaw directly
	ly.Learn.LrnNMDAFmRaw(ctx, ni, di, geRaw)
	ly.Acts.GvgccFmVm(ctx, ni, di)
	ege := NrnV(ctx, ni, di, Gnmda) + NrnV(ctx, ni, di, GnmdaMaint) + NrnV(ctx, ni, di, Gvgcc) + extraSyn
	ly.Acts.GeFmSyn(ctx, ni, di, geSyn, ege) // sets nrn.GeExt too
	ly.Acts.GkFmVm(ctx, ni, di)
	ly.Acts.GSkCaFmCa(ctx, ni, di)
	SetNrnV(ctx, ni, di, GiSyn, ly.Acts.GiFmSyn(ctx, ni, di, NrnV(ctx, ni, di, GiSyn)))
}

// GiInteg adds Gi values from all sources including SubPool computed inhib
// and updates GABAB as well
func (ly *LayerParams) GiInteg(ctx *Context, ni, di uint32, pl *Pool, vals *LayerValues) {
	gi := vals.ActAvg.GiMult*pl.Inhib.Gi + NrnV(ctx, ni, di, GiSyn) + NrnV(ctx, ni, di, GiNoise) + ly.Learn.NeuroMod.GiFmACh(GlbV(ctx, di, GvACh))
	SetNrnV(ctx, ni, di, Gi, gi)
	SetNrnV(ctx, ni, di, SSGi, pl.Inhib.SSGi)
	SetNrnV(ctx, ni, di, SSGiDend, 0)
	if ctx.PlusPhase.IsTrue() && ly.LayType == PulvinarLayer {
		ext := NrnV(ctx, ni, di, Ext) // nonDrivePct
		SetNrnV(ctx, ni, di, SSGiDend, ext*ly.Acts.Dend.SSGi*pl.Inhib.SSGi)
	} else {
		if !(ly.Acts.Clamp.IsInput.IsTrue() || ly.Acts.Clamp.IsTarget.IsTrue()) {
			SetNrnV(ctx, ni, di, SSGiDend, ly.Acts.Dend.SSGi*pl.Inhib.SSGi)
		}
	}
	vm := NrnV(ctx, ni, di, VmDend)
	nrnGABAB := NrnV(ctx, ni, di, GABAB)
	nrnGABABx := NrnV(ctx, ni, di, GABABx)
	ly.Acts.GabaB.GABAB(gi, &nrnGABAB, &nrnGABABx)
	SetNrnV(ctx, ni, di, GABAB, nrnGABAB)
	SetNrnV(ctx, ni, di, GABABx, nrnGABABx)
	nrnGgabaB := ly.Acts.GabaB.GgabaB(nrnGABAB, vm)
	SetNrnV(ctx, ni, di, GgabaB, nrnGgabaB)
	AddNrnV(ctx, ni, di, Gk, nrnGgabaB) // Gk was already init
}

// GNeuroMod does neuromodulation of conductances
func (ly *LayerParams) GNeuroMod(ctx *Context, ni, di uint32, vals *LayerValues) {
	ggain := ly.Learn.NeuroMod.GGain(GlbV(ctx, di, GvDA) + GlbV(ctx, di, GvDAtonic))
	MulNrnV(ctx, ni, di, Ge, ggain)
	MulNrnV(ctx, ni, di, Gi, ggain)
}

////////////////////////
//  SpikeFmG

// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
func (ly *LayerParams) SpikeFmG(ctx *Context, ni, di uint32, lpl *Pool) {
	ly.Acts.VmFmG(ctx, ni, di)
	ly.Acts.SpikeFmVm(ctx, ni, di)
	ly.Learn.CaFmSpike(ctx, ni, di)
	lmax := lpl.AvgMax.GeInt.Cycle.Max
	if lmax > 0 {
		SetNrnV(ctx, ni, di, GeIntNorm, NrnV(ctx, ni, di, GeInt)/lmax)
	} else {
		SetNrnV(ctx, ni, di, GeIntNorm, NrnV(ctx, ni, di, GeInt))
	}
	if ctx.Cycle >= ly.Acts.Dt.MaxCycStart {
		AddNrnV(ctx, ni, di, SpkMaxCa, ly.Learn.CaSpk.Dt.PDt*(NrnV(ctx, ni, di, CaSpkM)-NrnV(ctx, ni, di, SpkMaxCa)))
		spkmax := NrnV(ctx, ni, di, SpkMaxCa)
		if spkmax > NrnV(ctx, ni, di, SpkMax) {
			SetNrnV(ctx, ni, di, SpkMax, spkmax)
		}
	}
}

// PostSpikeSpecial does updates at neuron level after spiking has been computed.
// This is where special layer types add extra code.
// warning: if more than 1 layer writes to vals, gpu will fail!
func (ly *LayerParams) PostSpikeSpecial(ctx *Context, ni, di uint32, pl *Pool, lpl *Pool, vals *LayerValues) {
	SetNrnV(ctx, ni, di, Burst, NrnV(ctx, ni, di, CaSpkP))
	pi := NrnI(ctx, ni, NrnSubPool) - 1 // 0-n pool index
	pni := NrnI(ctx, ni, NrnNeurIndex) - pl.StIndex
	switch ly.LayType {
	case SuperLayer:
		if ctx.PlusPhase.IsTrue() {
			actMax := lpl.AvgMax.CaSpkP.Cycle.Max
			actAvg := lpl.AvgMax.CaSpkP.Cycle.Avg
			thr := ly.Bursts.ThrFmAvgMax(actAvg, actMax)
			if NrnV(ctx, ni, di, CaSpkP) < thr {
				SetNrnV(ctx, ni, di, Burst, 0)
			}
		}
	case CTLayer:
		fallthrough
	case PTPredLayer:
		if ctx.Cycle == ctx.ThetaCycles-1 {
			if ly.CT.DecayTau == 0 {
				SetNrnV(ctx, ni, di, CtxtGe, NrnV(ctx, ni, di, CtxtGeRaw))
			} else {
				AddNrnV(ctx, ni, di, CtxtGe, NrnV(ctx, ni, di, CtxtGeRaw))
			}
			SetNrnV(ctx, ni, di, CtxtGeOrig, NrnV(ctx, ni, di, CtxtGe))
		}
	case VSGatedLayer:
		dr := float32(0)
		if pi == 0 {
			dr = GlbV(ctx, di, GvVSMatrixJustGated)
		} else {
			dr = GlbV(ctx, di, GvVSMatrixHasGated)
		}
		SetNrnV(ctx, ni, di, Act, dr)

	case BLALayer:
		if ctx.Cycle == ctx.ThetaCycles-1 {
			if GlbV(ctx, di, GvHasRew) > 0 {
				SetNrnV(ctx, ni, di, CtxtGe, 0)
				SetNrnV(ctx, ni, di, CtxtGeOrig, 0)
			} else if GlbV(ctx, di, GvACh) > 0.1 {
				SetNrnV(ctx, ni, di, CtxtGe, NrnV(ctx, ni, di, CtxtGeRaw))
				SetNrnV(ctx, ni, di, CtxtGeOrig, NrnV(ctx, ni, di, CtxtGe))
			}
		}
	case LHbLayer:
		if pni == 0 {
			SetNrnV(ctx, ni, di, Act, GlbV(ctx, di, GvLHbDip))
		} else {
			SetNrnV(ctx, ni, di, Act, GlbV(ctx, di, GvLHbBurst))
		}
		SetNrnV(ctx, ni, di, GeSyn, ly.Acts.Dt.GeSynFmRawSteady(NrnV(ctx, ni, di, GeRaw)))
	case DrivesLayer:
		dr := GlbUSposV(ctx, di, GvDrives, uint32(pi))
		act := dr
		if dr > 0 {
			act = ly.Acts.PopCode.EncodeValue(pni, uint32(pl.NNeurons()), dr)
		}
		SetNrnV(ctx, ni, di, Act, act)
	case UrgencyLayer:
		ur := GlbV(ctx, di, GvUrgency)
		act := ur
		if ur > 0 {
			act = ly.Acts.PopCode.EncodeValue(pni, uint32(pl.NNeurons()), ur)
		}
		SetNrnV(ctx, ni, di, Act, act)
	case USLayer:
		us := PVLVUSStimValue(ctx, di, pi, ly.Learn.NeuroMod.Valence)
		act := us
		if us > 0 {
			act = ly.Acts.PopCode.EncodeValue(pni, uint32(pl.NNeurons()), us)
		}
		SetNrnV(ctx, ni, di, Act, act)
	case PVLayer:
		pv := float32(0)
		if ly.Learn.NeuroMod.Valence == Positive {
			pv = GlbV(ctx, di, GvPVpos)
		} else {
			pv = GlbV(ctx, di, GvPVneg)
		}
		act := ly.Acts.PopCode.EncodeValue(pni, ly.Indexes.NeurN, pv)
		SetNrnV(ctx, ni, di, Act, act)
	case LDTLayer:
		SetNrnV(ctx, ni, di, Act, GlbV(ctx, di, GvAChRaw)) // I set this in CyclePost
	case VTALayer:
		SetNrnV(ctx, ni, di, Act, GlbV(ctx, di, GvVtaDA)) // I set this in CyclePost

	case RewLayer:
		SetNrnV(ctx, ni, di, Act, GlbV(ctx, di, GvRew))
	case RWPredLayer:
		SetNrnV(ctx, ni, di, Act, ly.RWPred.PredRange.ClipValue(NrnV(ctx, ni, di, Ge))) // clipped linear
		if pni == 0 {
			vals.Special.V1 = NrnV(ctx, ni, di, ActInt) // warning: if more than 1 layer writes to vals, gpu will fail!
		} else {
			vals.Special.V2 = NrnV(ctx, ni, di, ActInt)
		}
	case RWDaLayer:
		SetNrnV(ctx, ni, di, Act, GlbV(ctx, di, GvDA)) // I set this in CyclePost
	case TDPredLayer:
		SetNrnV(ctx, ni, di, Act, NrnV(ctx, ni, di, Ge)) // linear
		if pni == 0 {
			vals.Special.V1 = NrnV(ctx, ni, di, ActInt) // warning: if more than 1 layer writes to vals, gpu will fail!
		} else {
			vals.Special.V2 = NrnV(ctx, ni, di, ActInt)
		}
	case TDIntegLayer:
		SetNrnV(ctx, ni, di, Act, GlbV(ctx, di, GvRewPred))
	case TDDaLayer:
		SetNrnV(ctx, ni, di, Act, GlbV(ctx, di, GvDA)) // I set this in CyclePost
	}
}

// PostSpike does updates at neuron level after spiking has been computed.
// it is called *after* PostSpikeSpecial.
// It also updates the CaSpkPCyc stats.
func (ly *LayerParams) PostSpike(ctx *Context, ni, di uint32, pl *Pool, vals *LayerValues) {
	intdt := ly.Acts.Dt.IntDt
	AddNrnV(ctx, ni, di, GeInt, intdt*(NrnV(ctx, ni, di, Ge)-NrnV(ctx, ni, di, GeInt)))
	AddNrnV(ctx, ni, di, GiInt, intdt*(NrnV(ctx, ni, di, GiSyn)-NrnV(ctx, ni, di, GiInt)))
	// act int is reset at start of the plus phase -- needs faster integration:
	if ctx.PlusPhase.IsTrue() {
		intdt *= 3.0
	}
	AddNrnV(ctx, ni, di, ActInt, intdt*(NrnV(ctx, ni, di, Act)-NrnV(ctx, ni, di, ActInt))) // using reg act here now
}

/////////////////////////////////////////////////////////////////////////
//  Special CyclePost methods for different layer types
//  call these in layer_compute.go/CyclePost and
//  gpu_hlsl/gpu_cyclepost.hlsl

// CyclePostLayer is called for all layer types
func (ly *LayerParams) CyclePostLayer(ctx *Context, di uint32, lpl *Pool, vals *LayerValues) {
	if ctx.Cycle >= ly.Acts.Dt.MaxCycStart && lpl.AvgMax.CaSpkP.Cycle.Max > 0.5 { // todo: ly.Acts.AttnMod.RTThr {
		if vals.RT <= 0 {
			vals.RT = float32(ctx.Cycle)
		}
	}
}

func (ly *LayerParams) CyclePostLDTLayer(ctx *Context, di uint32, vals *LayerValues, srcLay1Act, srcLay2Act, srcLay3Act, srcLay4Act float32) {
	ach := ly.LDT.ACh(ctx, di, srcLay1Act, srcLay2Act, srcLay3Act, srcLay4Act)

	SetGlbV(ctx, di, GvAChRaw, ach)
	if ach > GlbV(ctx, di, GvACh) { // instant up
		SetGlbV(ctx, di, GvACh, ach)
	} else {
		AddGlbV(ctx, di, GvACh, ly.Acts.Dt.IntDt*(ach-GlbV(ctx, di, GvACh)))
	}
}

func (ly *LayerParams) CyclePostRWDaLayer(ctx *Context, di uint32, vals *LayerValues, pvals *LayerValues) {
	pred := pvals.Special.V1 - pvals.Special.V2
	SetGlbV(ctx, di, GvRewPred, pred) // record
	da := float32(0)
	if GlbV(ctx, di, GvHasRew) > 0 {
		da = GlbV(ctx, di, GvRew) - pred
	}
	SetGlbV(ctx, di, GvDA, da) // updates global value that will be copied to layers next cycle.
}

func (ly *LayerParams) CyclePostTDPredLayer(ctx *Context, di uint32, vals *LayerValues) {
	if ctx.PlusPhase.IsTrue() {
		pred := vals.Special.V1 - vals.Special.V2
		SetGlbV(ctx, di, GvPrevPred, pred)
	}
}

func (ly *LayerParams) CyclePostTDIntegLayer(ctx *Context, di uint32, vals *LayerValues, pvals *LayerValues) {
	rew := float32(0)
	if GlbV(ctx, di, GvHasRew) > 0 {
		rew = GlbV(ctx, di, GvRew)
	}
	rpval := float32(0)
	if ctx.PlusPhase.IsTrue() {
		pred := pvals.Special.V1 - pvals.Special.V2 // neuron0 (pos) - neuron1 (neg)
		rpval = rew + ly.TDInteg.Discount*ly.TDInteg.PredGain*pred
		vals.Special.V2 = rpval // plus phase
	} else {
		rpval = ly.TDInteg.PredGain * GlbV(ctx, di, GvPrevPred)
		vals.Special.V1 = rpval // minus phase is *previous trial*
	}
	SetGlbV(ctx, di, GvRewPred, rpval) // global value will be copied to layers next cycle
}

func (ly *LayerParams) CyclePostTDDaLayer(ctx *Context, di uint32, vals *LayerValues, ivals *LayerValues) {
	da := ivals.Special.V2 - ivals.Special.V1
	if ctx.PlusPhase.IsFalse() {
		da = 0
	}
	SetGlbV(ctx, di, GvDA, da) // updates global value that will be copied to layers next cycle.
}

func (ly *LayerParams) CyclePostCeMLayer(ctx *Context, di uint32, lpl *Pool) {
	if ly.Learn.NeuroMod.Valence == Positive {
		SetGlbV(ctx, di, GvCeMpos, lpl.AvgMax.CaSpkD.Cycle.Max)
	} else {
		SetGlbV(ctx, di, GvCeMneg, lpl.AvgMax.CaSpkD.Cycle.Max)
	}
}

func (ly *LayerParams) CyclePostVTALayer(ctx *Context, di uint32) {
	ly.VTA.VTADA(ctx, di, GlbV(ctx, di, GvACh), (GlbV(ctx, di, GvHasRew) > 0))
}

// note: needs to iterate over sub-pools in layer!
func (ly *LayerParams) CyclePostVSPatchLayer(ctx *Context, di uint32, pi int32, pl *Pool, vals *LayerValues) {
	val := pl.AvgMax.CaSpkD.Cycle.Avg
	if ly.Learn.NeuroMod.DAMod == D1Mod {
		SetGlbUSposV(ctx, di, GvVSPatchD1, uint32(pi-1), val)
	} else {
		SetGlbUSposV(ctx, di, GvVSPatchD2, uint32(pi-1), val)
	}
}

/////////////////////////////////////////////////////////////////////////
//  Phase timescale

// NewStateLayerActAvg updates ActAvg.ActMAvg and ActPAvg based on current values
// that have been averaged across NData already.
func (ly *LayerParams) NewStateLayerActAvg(ctx *Context, vals *LayerValues, actMinusAvg, actPlusAvg float32) {
	ly.Inhib.ActAvg.AvgFmAct(&vals.ActAvg.ActMAvg, actMinusAvg, ly.Acts.Dt.LongAvgDt)
	ly.Inhib.ActAvg.AvgFmAct(&vals.ActAvg.ActPAvg, actPlusAvg, ly.Acts.Dt.LongAvgDt)
}

func (ly *LayerParams) NewStateLayer(ctx *Context, lpl *Pool, vals *LayerValues) {
	ly.Acts.Clamp.IsInput.SetBool(ly.IsInput())
	ly.Acts.Clamp.IsTarget.SetBool(ly.IsTarget())
	vals.RT = -1
}

func (ly *LayerParams) NewStatePool(ctx *Context, pl *Pool) {
	pl.Inhib.Clamped.SetBool(false)
	if ly.Acts.Clamp.Add.IsFalse() && ly.Acts.Clamp.IsInput.IsTrue() {
		pl.Inhib.Clamped.SetBool(true)
	}
	pl.Inhib.Decay(ly.Acts.Decay.Act)
	pl.Gated.SetBool(false)
}

// NewStateNeuron handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
func (ly *LayerParams) NewStateNeuron(ctx *Context, ni, di uint32, vals *LayerValues, pl *Pool) {
	SetNrnV(ctx, ni, di, BurstPrv, NrnV(ctx, ni, di, Burst))
	SetNrnV(ctx, ni, di, SpkPrv, NrnV(ctx, ni, di, CaSpkD))
	SetNrnV(ctx, ni, di, SpkMax, 0)
	SetNrnV(ctx, ni, di, SpkMaxCa, 0)

	ly.Acts.DecayState(ctx, ni, di, ly.Acts.Decay.Act, ly.Acts.Decay.Glong, ly.Acts.Decay.AHP)
	// Note: synapse-level Ca decay happens in DWt
	ly.Acts.KNaNewState(ctx, ni, di)
}

func (ly *LayerParams) MinusPhasePool(ctx *Context, pl *Pool) {
	pl.AvgMax.CycleToMinus()
	if ly.Acts.Clamp.Add.IsFalse() && ly.Acts.Clamp.IsTarget.IsTrue() {
		pl.Inhib.Clamped.SetBool(true)
	}
}

// AvgGeM computes the average and max GeInt, GiInt in minus phase
// (AvgMaxGeM, AvgMaxGiM) stats, updated in MinusPhase,
// using values that already max across NData.
func (ly *LayerParams) AvgGeM(ctx *Context, vals *LayerValues, geIntMinusMax, giIntMinusMax float32) {
	vals.ActAvg.AvgMaxGeM += ly.Acts.Dt.LongAvgDt * (geIntMinusMax - vals.ActAvg.AvgMaxGeM)
	vals.ActAvg.AvgMaxGiM += ly.Acts.Dt.LongAvgDt * (giIntMinusMax - vals.ActAvg.AvgMaxGiM)
}

// MinusPhaseNeuron does neuron level minus-phase updating
func (ly *LayerParams) MinusPhaseNeuron(ctx *Context, ni, di uint32, pl *Pool, lpl *Pool, vals *LayerValues) {
	SetNrnV(ctx, ni, di, ActM, NrnV(ctx, ni, di, ActInt))
	SetNrnV(ctx, ni, di, CaSpkPM, NrnV(ctx, ni, di, CaSpkP))
}

// PlusPhaseStartNeuron does neuron level plus-phase start:
// applies Target inputs as External inputs.
func (ly *LayerParams) PlusPhaseStartNeuron(ctx *Context, ni, di uint32, pl *Pool, lpl *Pool, vals *LayerValues) {
	if NrnHasFlag(ctx, ni, di, NeuronHasTarg) { // will be clamped in plus phase
		SetNrnV(ctx, ni, di, Ext, NrnV(ctx, ni, di, Target))
		NrnSetFlag(ctx, ni, di, NeuronHasExt)
		SetNrnV(ctx, ni, di, ISI, -1) // get fresh update on plus phase output acts
		SetNrnV(ctx, ni, di, ISIAvg, -1)
		SetNrnV(ctx, ni, di, ActInt, ly.Acts.Init.Act) // reset for plus phase
	}
}

func (ly *LayerParams) PlusPhasePool(ctx *Context, pl *Pool) {
	pl.AvgMax.CycleToPlus()
}

// PlusPhaseNeuronSpecial does special layer type neuron level plus-phase updating
func (ly *LayerParams) PlusPhaseNeuronSpecial(ctx *Context, ni, di uint32, pl *Pool, lpl *Pool, vals *LayerValues) {
}

// PlusPhaseNeuron does neuron level plus-phase updating
func (ly *LayerParams) PlusPhaseNeuron(ctx *Context, ni, di uint32, pl *Pool, lpl *Pool, vals *LayerValues) {
	SetNrnV(ctx, ni, di, ActP, NrnV(ctx, ni, di, ActInt))
	nrnCaSpkP := NrnV(ctx, ni, di, CaSpkP)
	nrnCaSpkD := NrnV(ctx, ni, di, CaSpkD)
	da := GlbV(ctx, di, GvDA)
	ach := GlbV(ctx, di, GvACh)
	mlr := ly.Learn.RLRate.RLRateSigDeriv(nrnCaSpkD, lpl.AvgMax.CaSpkD.Cycle.Max)
	modlr := ly.Learn.NeuroMod.LRMod(da, ach)
	dlr := float32(1)
	hasRew := (GlbV(ctx, di, GvHasRew) > 0)

	switch ly.LayType {
	case BLALayer:
		dlr = ly.Learn.RLRate.RLRateDiff(nrnCaSpkP, NrnV(ctx, ni, di, SpkPrv)) // delta on previous trial
		if !ly.Learn.NeuroMod.IsBLAExt() && pl.StIndex == 0 {                  // first pool
			dlr = 0 // first pool is novelty / curiosity -- no learn
		}
	case VSPatchLayer:
		mlr = ly.Learn.RLRate.RLRateSigDeriv(NrnV(ctx, ni, di, SpkPrv), 1) // note: don't have proper max here
		dlr = ly.Learn.RLRate.RLRateDiff(nrnCaSpkP, nrnCaSpkD)
		if !hasRew && da < 0 && da > -ly.VSPatch.SmallNegDAThr { // for negative da, increase lrate
			mlr *= ly.VSPatch.SmallNegDALRate
		}
	case MatrixLayer:
		if hasRew { // reward time
			mlr = 1 // don't use dig deriv
		} else {
			modlr = 1 // don't use mod
		}
	default:
		dlr = ly.Learn.RLRate.RLRateDiff(nrnCaSpkP, nrnCaSpkD)
	}
	SetNrnV(ctx, ni, di, RLRate, mlr*dlr*modlr)
	var tau float32
	sahpN := NrnV(ctx, ni, di, SahpN)
	nrnSaphCa := NrnV(ctx, ni, di, SahpCa)
	ly.Acts.Sahp.NinfTauFmCa(nrnSaphCa, &sahpN, &tau)
	nrnSaphCa = ly.Acts.Sahp.CaInt(nrnSaphCa, nrnCaSpkD)
	SetNrnV(ctx, ni, di, SahpN, sahpN)
	SetNrnV(ctx, ni, di, SahpCa, nrnSaphCa)
	SetNrnV(ctx, ni, di, Gsahp, ly.Acts.Sahp.GsAHP(sahpN))
}

//gosl: end layerparams
