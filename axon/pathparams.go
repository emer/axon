// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"
	"strings"

	"cogentcore.org/core/math32"
)

//gosl:hlsl pathparams
// #include "pathtypes.hlsl"
// #include "act_path.hlsl"
// #include "learn.hlsl"
// #include "deep_paths.hlsl"
// #include "rl_paths.hlsl"
// #include "rubicon_paths.hlsl"
// #include "pcore_paths.hlsl"
// #include "hip_paths.hlsl"

//gosl:end pathparams

//gosl:start pathparams

// StartN holds a starting offset index and a number of items
// arranged from Start to Start+N (exclusive).
// This is not 16 byte padded and only for use on CPU side.
type StartN struct {

	// starting offset
	Start uint32

	// number of items --
	N uint32

	pad, pad1 uint32 // todo: see if we can do without these?
}

// PathIndexes contains path-level index information into global memory arrays
type PathIndexes struct {
	PathIndex  uint32 // index of the pathway in global path list: [Layer][SendPaths]
	RecvLay    uint32 // index of the receiving layer in global list of layers
	RecvNeurSt uint32 // starting index of neurons in recv layer -- so we don't need layer to get to neurons
	RecvNeurN  uint32 // number of neurons in recv layer
	SendLay    uint32 // index of the sending layer in global list of layers
	SendNeurSt uint32 // starting index of neurons in sending layer -- so we don't need layer to get to neurons
	SendNeurN  uint32 // number of neurons in send layer
	SynapseSt  uint32 // start index into global Synapse array: [Layer][SendPaths][Synapses]
	SendConSt  uint32 // start index into global PathSendCon array: [Layer][SendPaths][SendNeurons]
	RecvConSt  uint32 // start index into global PathRecvCon array: [Layer][RecvPaths][RecvNeurons]
	RecvSynSt  uint32 // start index into global sender-based Synapse index array: [Layer][SendPaths][Synapses]
	GBufSt     uint32 // start index into global PathGBuf global array: [Layer][RecvPaths][RecvNeurons][MaxDelay+1]
	GSynSt     uint32 // start index into global PathGSyn global array: [Layer][RecvPaths][RecvNeurons]

	pad, pad1, pad2 uint32
}

// RecvNIndexToLayIndex converts a neuron's index in network level global list of all neurons
// to receiving layer-specific index-- e.g., for accessing GBuf and GSyn values.
// Just subtracts RecvNeurSt -- docu-function basically..
func (pi *PathIndexes) RecvNIndexToLayIndex(ni uint32) uint32 {
	return ni - pi.RecvNeurSt
}

// SendNIndexToLayIndex converts a neuron's index in network level global list of all neurons
// to sending layer-specific index.  Just subtracts SendNeurSt -- docu-function basically..
func (pi *PathIndexes) SendNIndexToLayIndex(ni uint32) uint32 {
	return ni - pi.SendNeurSt
}

// GScaleValues holds the conductance scaling values.
// These are computed once at start and remain constant thereafter,
// and therefore belong on Params and not on PathValues.
type GScaleValues struct {

	// scaling factor for integrating synaptic input conductances (G's), originally computed as a function of sending layer activity and number of connections, and typically adapted from there -- see Path.PathScale adapt params
	Scale float32 `edit:"-"`

	// normalized relative proportion of total receiving conductance for this pathway: PathScale.Rel / sum(PathScale.Rel across relevant paths)
	Rel float32 `edit:"-"`

	pad, pad1 float32
}

// PathParams contains all of the path parameters.
// These values must remain constant over the course of computation.
// On the GPU, they are loaded into a uniform.
type PathParams struct {

	// functional type of path, which determines functional code path
	// for specialized layer types, and is synchronized with the Path.Typ value
	PathType PathTypes

	pad, pad1, pad2 int32

	// recv and send neuron-level pathway index array access info
	Indexes PathIndexes `view:"-"`

	// synaptic communication parameters: delay, probability of failure
	Com SynComParams `view:"inline"`

	// pathway scaling parameters for computing GScale:
	// modulates overall strength of pathway, using both
	// absolute and relative factors, with adaptation option to maintain target max conductances
	PathScale PathScaleParams `view:"inline"`

	// slowly adapting, structural weight value parameters,
	// which control initial weight values and slower outer-loop adjustments
	SWts SWtParams `view:"add-fields"`

	// synaptic-level learning parameters for learning in the fast LWt values.
	Learn LearnSynParams `view:"add-fields"`

	// conductance scaling values
	GScale GScaleValues `view:"inline"`

	// Params for RWPath and TDPredPath for doing dopamine-modulated learning
	// for reward prediction: Da * Send activity.
	// Use in RWPredLayer or TDPredLayer typically to generate reward predictions.
	// If the Da sign is positive, the first recv unit learns fully; for negative,
	// second one learns fully.
	// Lower lrate applies for opposite cases.  Weights are positive-only.
	RLPred RLPredPathParams `view:"inline"`

	// for trace-based learning in the MatrixPath. A trace of synaptic co-activity
	// is formed, and then modulated by dopamine whenever it occurs.
	// This bridges the temporal gap between gating activity and subsequent activity,
	// and is based biologically on synaptic tags.
	// Trace is reset at time of reward based on ACh level from CINs.
	Matrix MatrixPathParams `view:"inline"`

	// Basolateral Amygdala pathway parameters.
	BLA BLAPathParams `view:"inline"`

	// Hip bench parameters.
	Hip HipPathParams `view:"inline"`
}

func (pj *PathParams) Defaults() {
	pj.Com.Defaults()
	pj.SWts.Defaults()
	pj.PathScale.Defaults()
	pj.Learn.Defaults()
	pj.RLPred.Defaults()
	pj.Matrix.Defaults()
	pj.BLA.Defaults()
	pj.Hip.Defaults()
}

func (pj *PathParams) Update() {
	pj.Com.Update()
	pj.PathScale.Update()
	pj.SWts.Update()
	pj.Learn.Update()
	pj.RLPred.Update()
	pj.Matrix.Update()
	pj.BLA.Update()
	pj.Hip.Update()

	if pj.PathType == CTCtxtPath {
		pj.Com.GType = ContextG
	}
}

func (pj *PathParams) ShouldShow(field string) bool {
	switch field {
	case "RLPred":
		return pj.PathType == RWPath || pj.PathType == TDPredPath
	case "Matrix":
		return pj.PathType == VSMatrixPath || pj.PathType == DSMatrixPath
	case "BLA":
		return pj.PathType == BLAPath
	case "Hip":
		return pj.PathType == HipPath
	default:
		return true
	}
}

func (pj *PathParams) AllParams() string {
	str := ""
	b, _ := json.MarshalIndent(&pj.Com, "", " ")
	str += "Com: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pj.PathScale, "", " ")
	str += "PathScale: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pj.SWts, "", " ")
	str += "SWt: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pj.Learn, "", " ")
	str += "Learn: {\n " + strings.Replace(JsonToParams(b), " LRate: {", "\n  LRate: {", -1)

	switch pj.PathType {
	case RWPath, TDPredPath:
		b, _ = json.MarshalIndent(&pj.RLPred, "", " ")
		str += "RLPred: {\n " + JsonToParams(b)
	case VSMatrixPath, DSMatrixPath:
		b, _ = json.MarshalIndent(&pj.Matrix, "", " ")
		str += "Matrix: {\n " + JsonToParams(b)
	case BLAPath:
		b, _ = json.MarshalIndent(&pj.BLA, "", " ")
		str += "BLA: {\n " + JsonToParams(b)
	case HipPath:
		b, _ = json.MarshalIndent(&pj.BLA, "", " ")
		str += "Hip: {\n " + JsonToParams(b)
	}
	return str
}

func (pj *PathParams) IsInhib() bool {
	return pj.Com.GType == InhibitoryG
}

func (pj *PathParams) IsExcitatory() bool {
	return pj.Com.GType == ExcitatoryG
}

// SetFixedWts sets parameters for fixed, non-learning weights
// with a default of Mean = 0.8, Var = 0 strength
func (pj *PathParams) SetFixedWts() {
	pj.SWts.Init.SPct = 0
	pj.Learn.Learn.SetBool(false)
	pj.SWts.Adapt.On.SetBool(false)
	pj.SWts.Adapt.SigGain = 1
	pj.SWts.Init.Mean = 0.8
	pj.SWts.Init.Var = 0.0
	pj.SWts.Init.Sym.SetBool(false)
}

// SynRecvLayIndex converts the Synapse RecvIndex of recv neuron's index
// in network level global list of all neurons to receiving
// layer-specific index.
func (pj *PathParams) SynRecvLayIndex(ctx *Context, syni uint32) uint32 {
	return pj.Indexes.RecvNIndexToLayIndex(SynI(ctx, syni, SynRecvIndex))
}

// SynSendLayIndex converts the Synapse SendIndex of sending neuron's index
// in network level global list of all neurons to sending
// layer-specific index.
func (pj *PathParams) SynSendLayIndex(ctx *Context, syni uint32) uint32 {
	return pj.Indexes.SendNIndexToLayIndex(SynI(ctx, syni, SynSendIndex))
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GatherSpikes integrates G*Raw and G*Syn values for given neuron
// from the given Path-level GRaw value, first integrating
// pathway-level GSyn value.
func (pj *PathParams) GatherSpikes(ctx *Context, ly *LayerParams, ni, di uint32, gRaw float32, gSyn *float32) {
	switch pj.Com.GType {
	case ExcitatoryG:
		*gSyn = ly.Acts.Dt.GeSynFromRaw(*gSyn, gRaw)
		AddNrnV(ctx, ni, di, GeRaw, gRaw)
		AddNrnV(ctx, ni, di, GeSyn, *gSyn)
	case InhibitoryG:
		*gSyn = ly.Acts.Dt.GiSynFromRaw(*gSyn, gRaw)
		AddNrnV(ctx, ni, di, GiRaw, gRaw)
		AddNrnV(ctx, ni, di, GiSyn, *gSyn)
	case ModulatoryG:
		*gSyn = ly.Acts.Dt.GeSynFromRaw(*gSyn, gRaw)
		AddNrnV(ctx, ni, di, GModRaw, gRaw)
		AddNrnV(ctx, ni, di, GModSyn, *gSyn)
	case MaintG:
		*gSyn = ly.Acts.Dt.GeSynFromRaw(*gSyn, gRaw)
		AddNrnV(ctx, ni, di, GMaintRaw, gRaw)
		// note: Syn happens via NMDA in Act
	case ContextG:
		AddNrnV(ctx, ni, di, CtxtGeRaw, gRaw)
	}
}

///////////////////////////////////////////////////
// SynCa

// DoSynCa returns false if should not do synaptic-level calcium updating.
// Done by default in Cortex, not for some other special pathway types.
func (pj *PathParams) DoSynCa() bool {
	if pj.PathType == RWPath || pj.PathType == TDPredPath || pj.PathType == VSMatrixPath ||
		pj.PathType == DSMatrixPath || pj.PathType == VSPatchPath || pj.PathType == BLAPath {
		return false
	}
	return true
}

// SynCaSyn updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking, threaded over neurons.
func (pj *PathParams) SynCaSyn(ctx *Context, syni uint32, ni, di uint32, otherCaSyn, updtThr float32) {
	if NrnV(ctx, ni, di, CaSpkP) < updtThr && NrnV(ctx, ni, di, CaSpkD) < updtThr {
		return
	}
	caUpT := SynCaV(ctx, syni, di, CaUpT)
	syCaM := SynCaV(ctx, syni, di, CaM)
	syCaP := SynCaV(ctx, syni, di, CaP)
	syCaD := SynCaV(ctx, syni, di, CaD)
	pj.Learn.KinaseCa.CurCa(ctx.SynCaCtr-1, caUpT, &syCaM, &syCaP, &syCaD)
	ca := NrnV(ctx, ni, di, CaSyn) * otherCaSyn
	pj.Learn.KinaseCa.FromCa(ca, &syCaM, &syCaP, &syCaD)
	SetSynCaV(ctx, syni, di, CaM, syCaM)
	SetSynCaV(ctx, syni, di, CaP, syCaP)
	SetSynCaV(ctx, syni, di, CaD, syCaD)
	SetSynCaV(ctx, syni, di, CaUpT, ctx.SynCaCtr)
}

///////////////////////////////////////////////////
// DWt

// DWtSyn is the overall entry point for weight change (learning) at given synapse.
// It selects appropriate function based on pathway type.
// rpl is the receiving layer SubPool
func (pj *PathParams) DWtSyn(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool, isTarget bool) {
	switch pj.PathType {
	case RWPath:
		pj.DWtSynRWPred(ctx, syni, si, ri, di, layPool, subPool)
	case TDPredPath:
		pj.DWtSynTDPred(ctx, syni, si, ri, di, layPool, subPool)
	case VSMatrixPath:
		pj.DWtSynVSMatrix(ctx, syni, si, ri, di, layPool, subPool)
	case DSMatrixPath:
		pj.DWtSynDSMatrix(ctx, syni, si, ri, di, layPool, subPool)
	case VSPatchPath:
		pj.DWtSynVSPatch(ctx, syni, si, ri, di, layPool, subPool)
	case BLAPath:
		pj.DWtSynBLA(ctx, syni, si, ri, di, layPool, subPool)
	case HipPath:
		pj.DWtSynHip(ctx, syni, si, ri, di, layPool, subPool, isTarget) // by default this is the same as DWtSynCortex (w/ unused Hebb component in the algorithm) except that it uses WtFromDWtSynNoLimits
	default:
		if pj.Learn.Hebb.On.IsTrue() {
			pj.DWtSynHebb(ctx, syni, si, ri, di, layPool, subPool)
		} else {
			pj.DWtSynCortex(ctx, syni, si, ri, di, layPool, subPool, isTarget)
		}
	}
}

// DWtSynCortex computes the weight change (learning) at given synapse for cortex.
// Uses synaptically integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *PathParams) DWtSynCortex(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool, isTarget bool) {
	// credit assignment part
	caUpT := SynCaV(ctx, syni, di, CaUpT)                                // time of last update
	syCaM := SynCaV(ctx, syni, di, CaM)                                  // fast time scale
	syCaP := SynCaV(ctx, syni, di, CaP)                                  // slower but still fast time scale, drives Potentiation
	syCaD := SynCaV(ctx, syni, di, CaD)                                  // slow time scale, drives Depression (one trial = 200 cycles)
	pj.Learn.KinaseCa.CurCa(ctx.SynCaCtr, caUpT, &syCaM, &syCaP, &syCaD) // always update, getting current Ca (just optimization)
	dtr := syCaD                                                         // delta trace, caD reflects entire window
	if pj.PathType == CTCtxtPath {                                       // layer 6 CT pathway
		dtr = NrnV(ctx, si, di, BurstPrv)
	}
	SetSynCaV(ctx, syni, di, DTr, dtr)                            // save delta trace for GUI
	tr := pj.Learn.Trace.TrFromCa(SynCaV(ctx, syni, di, Tr), dtr) // TrFromCa(prev-multiTrial Integrated Trace, deltaTrace), as a mixing func
	SetSynCaV(ctx, syni, di, Tr, tr)                              // save new trace, updated w/ credit assignment (dependent on Tau in the TrFromCa function)
	if SynV(ctx, syni, Wt) == 0 {                                 // failed con, no learn
		return
	}

	// error-driven learning
	var err float32
	if isTarget {
		err = syCaP - syCaD // for target layers, syn Ca drives error signal directly
	} else {
		err = tr * (NrnV(ctx, ri, di, NrnCaP) - NrnV(ctx, ri, di, NrnCaD)) // hiddens: recv NMDA Ca drives error signal w/ trace credit
	}
	// note: trace ensures that nothing changes for inactive synapses..
	// sb immediately -- enters into zero sum.
	// also other types might not use, so need to do this per learning rule
	lwt := SynV(ctx, syni, LWt) // linear weight
	if err > 0 {
		err *= (1 - lwt)
	} else {
		err *= lwt
	}
	if pj.PathType == CTCtxtPath { // rn.RLRate IS needed for other pathways, just not the context one
		SetSynCaV(ctx, syni, di, DiDWt, pj.Learn.LRate.Eff*err)
	} else {
		SetSynCaV(ctx, syni, di, DiDWt, NrnV(ctx, ri, di, RLRate)*pj.Learn.LRate.Eff*err)
	}
}

// DWtSynHebb computes the weight change (learning) at given synapse for cortex.
// Uses synaptically integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *PathParams) DWtSynHebb(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool) {
	rNrnCaP := NrnV(ctx, ri, di, NrnCaP)
	sNrnCap := NrnV(ctx, si, di, NrnCaP)
	lwt := SynV(ctx, syni, LWt) // linear weight
	hebb := rNrnCaP * (pj.Learn.Hebb.Up*sNrnCap*(1-lwt) - pj.Learn.Hebb.Down*(1-sNrnCap)*lwt)

	SetSynCaV(ctx, syni, di, DiDWt, pj.Learn.LRate.Eff*hebb) // not: NrnV(ctx, ri, di, RLRate)*
}

// DWtSynHip computes the weight change (learning) at given synapse for cortex + Hip (CPCA Hebb learning).
// Uses synaptically integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
// Adds proportional CPCA learning rule for hip-specific paths
func (pj *PathParams) DWtSynHip(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool, isTarget bool) {
	// credit assignment part
	caUpT := SynCaV(ctx, syni, di, CaUpT)                                // time of last update
	syCaM := SynCaV(ctx, syni, di, CaM)                                  // fast time scale
	syCaP := SynCaV(ctx, syni, di, CaP)                                  // slower but still fast time scale, drives Potentiation
	syCaD := SynCaV(ctx, syni, di, CaD)                                  // slow time scale, drives Depression (one trial = 200 cycles)
	pj.Learn.KinaseCa.CurCa(ctx.SynCaCtr, caUpT, &syCaM, &syCaP, &syCaD) // always update, getting current Ca (just optimization)
	dtr := syCaD                                                         // delta trace, caD reflects entire window
	SetSynCaV(ctx, syni, di, DTr, dtr)                                   // save delta trace for GUI
	tr := pj.Learn.Trace.TrFromCa(SynCaV(ctx, syni, di, Tr), dtr)        // TrFromCa(prev-multiTrial Integrated Trace, deltaTrace), as a mixing func
	SetSynCaV(ctx, syni, di, Tr, tr)                                     // save new trace, updated w/ credit assignment (dependent on Tau in the TrFromCa function)
	if SynV(ctx, syni, Wt) == 0 {                                        // failed con, no learn
		return
	}

	// error-driven learning part
	rNrnCaP := NrnV(ctx, ri, di, NrnCaP)
	rNrnCaD := NrnV(ctx, ri, di, NrnCaD)
	var err float32
	if isTarget {
		err = syCaP - syCaD // for target layers, syn Ca drives error signal directly
	} else {
		err = tr * (rNrnCaP - rNrnCaD) // hiddens: recv NMDA Ca drives error signal w/ trace credit
	}
	// note: trace ensures that nothing changes for inactive synapses..
	// sb immediately -- enters into zero sum.
	// also other types might not use, so need to do this per learning rule
	lwt := SynV(ctx, syni, LWt) // linear weight
	if err > 0 {
		err *= (1 - lwt)
	} else {
		err *= lwt
	}

	// hebbian-learning part
	sNrnCap := NrnV(ctx, si, di, NrnCaP)
	savg := 0.5 + pj.Hip.SAvgCor*(pj.Hip.SNominal-0.5)
	savg = 0.5 / math32.Max(pj.Hip.SAvgThr, savg) // keep this Sending Average Correction term within bounds (SAvgThr)
	hebb := rNrnCaP * (sNrnCap*(savg-lwt) - (1-sNrnCap)*lwt)

	// setting delta weight (note: impossible to be CTCtxtPath)
	dwt := NrnV(ctx, ri, di, RLRate) * pj.Learn.LRate.Eff * (pj.Hip.Hebb*hebb + pj.Hip.Err*err)
	SetSynCaV(ctx, syni, di, DiDWt, dwt)
}

// DWtSynBLA computes the weight change (learning) at given synapse for BLAPath type.
// Like the BG Matrix learning rule, a synaptic tag "trace" is established at CS onset (ACh)
// and learning at US / extinction is a function of trace * delta from US activity
// (temporal difference), which limits learning.
func (pj *PathParams) DWtSynBLA(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool) {
	dwt := float32(0)
	ach := GlbV(ctx, di, GvACh)
	if GlbV(ctx, di, GvHasRew) > 0 { // learn and reset
		ract := NrnV(ctx, ri, di, CaSpkD)
		if ract < pj.Learn.Trace.LearnThr {
			ract = 0
		}
		tr := SynCaV(ctx, syni, di, Tr)
		ustr := pj.BLA.USTrace
		tr = ustr*NrnV(ctx, si, di, Burst) + (1.0-ustr)*tr
		delta := NrnV(ctx, ri, di, CaSpkP) - NrnV(ctx, ri, di, SpkPrv)
		if delta < 0 { // neg delta learns slower in Acq, not Ext
			delta *= pj.BLA.NegDeltaLRate
		}
		dwt = tr * delta * ract
		SetSynCaV(ctx, syni, di, Tr, 0.0)
	} else if ach > pj.BLA.AChThr {
		// note: the former NonUSLRate parameter is not used -- Trace update Tau replaces it..  elegant
		dtr := ach * NrnV(ctx, si, di, Burst)
		SetSynCaV(ctx, syni, di, DTr, dtr)
		tr := pj.Learn.Trace.TrFromCa(SynCaV(ctx, syni, di, Tr), dtr)
		SetSynCaV(ctx, syni, di, Tr, tr)
	} else {
		SetSynCaV(ctx, syni, di, DTr, 0.0)
	}
	lwt := SynV(ctx, syni, LWt)
	if dwt > 0 {
		dwt *= (1 - lwt)
	} else {
		dwt *= lwt
	}
	SetSynCaV(ctx, syni, di, DiDWt, NrnV(ctx, ri, di, RLRate)*pj.Learn.LRate.Eff*dwt)
}

// DWtSynRWPred computes the weight change (learning) at given synapse,
// for the RWPredPath type
func (pj *PathParams) DWtSynRWPred(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool) {
	// todo: move all of this into rn.RLRate
	lda := GlbV(ctx, di, GvDA)
	da := lda
	lr := pj.Learn.LRate.Eff
	eff_lr := lr
	if NrnI(ctx, ri, NrnNeurIndex) == 0 {
		if NrnV(ctx, ri, di, Ge) > NrnV(ctx, ri, di, Act) && da > 0 { // clipped at top, saturate up
			da = 0
		}
		if NrnV(ctx, ri, di, Ge) < NrnV(ctx, ri, di, Act) && da < 0 { // clipped at bottom, saturate down
			da = 0
		}
		if da < 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	} else {
		eff_lr = -eff_lr                                              // negative case
		if NrnV(ctx, ri, di, Ge) > NrnV(ctx, ri, di, Act) && da < 0 { // clipped at top, saturate up
			da = 0
		}
		if NrnV(ctx, ri, di, Ge) < NrnV(ctx, ri, di, Act) && da > 0 { // clipped at bottom, saturate down
			da = 0
		}
		if da >= 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	}

	dwt := da * NrnV(ctx, si, di, CaSpkP) // no recv unit activation
	SetSynCaV(ctx, syni, di, DiDWt, eff_lr*dwt)
}

// DWtSynTDPred computes the weight change (learning) at given synapse,
// for the TDRewPredPath type
func (pj *PathParams) DWtSynTDPred(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool) {
	// todo: move all of this into rn.RLRate
	lda := GlbV(ctx, di, GvDA)
	da := lda
	lr := pj.Learn.LRate.Eff
	eff_lr := lr
	ni := NrnI(ctx, ri, NrnNeurIndex)
	if ni == 0 {
		if da < 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	} else {
		eff_lr = -eff_lr
		if da >= 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	}

	dwt := da * NrnV(ctx, si, di, SpkPrv) // no recv unit activation, prior trial act
	SetSynCaV(ctx, syni, di, DiDWt, eff_lr*dwt)
}

// DWtSynVSMatrix computes the weight change (learning) at given synapse,
// for the VSMatrixPath type.
func (pj *PathParams) DWtSynVSMatrix(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool) {
	// note: rn.RLRate already has BurstGain * ACh * DA * (D1 vs. D2 sign reversal) factored in.

	hasRew := GlbV(ctx, di, GvHasRew) > 0
	ach := GlbV(ctx, di, GvACh)
	if !hasRew && ach < 0.1 {
		SetSynCaV(ctx, syni, di, DTr, 0.0)
		return
	}
	rlr := NrnV(ctx, ri, di, RLRate)

	rplus := NrnV(ctx, ri, di, CaSpkP)
	rminus := NrnV(ctx, ri, di, CaSpkD)
	sact := NrnV(ctx, si, di, CaSpkD)
	dtr := ach * (pj.Matrix.Delta * sact * (rplus - rminus))
	if rminus > pj.Learn.Trace.LearnThr { // key: prevents learning if < threshold
		dtr += ach * (pj.Matrix.Credit * sact * rminus)
	}
	if hasRew {
		tr := SynCaV(ctx, syni, di, Tr)
		if pj.Matrix.VSRewLearn.IsTrue() {
			tr += (1 - GlbV(ctx, di, GvGoalMaint)) * dtr
		}
		dwt := rlr * pj.Learn.LRate.Eff * tr
		SetSynCaV(ctx, syni, di, DiDWt, dwt)
		SetSynCaV(ctx, syni, di, Tr, 0.0)
		SetSynCaV(ctx, syni, di, DTr, 0.0)
	} else {
		dtr *= rlr
		SetSynCaV(ctx, syni, di, DTr, dtr)
		AddSynCaV(ctx, syni, di, Tr, dtr)
	}
}

// DWtSynDSMatrix computes the weight change (learning) at given synapse,
// for the DSMatrixPath type.
func (pj *PathParams) DWtSynDSMatrix(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool) {
	// note: rn.RLRate already has ACh * DA * (D1 vs. D2 sign reversal) factored in.

	rlr := NrnV(ctx, ri, di, RLRate)
	if GlbV(ctx, di, GvHasRew) > 0 { // US time -- use DA and current recv activity
		tr := SynCaV(ctx, syni, di, Tr)
		dwt := rlr * pj.Learn.LRate.Eff * tr
		SetSynCaV(ctx, syni, di, DiDWt, dwt)
		SetSynCaV(ctx, syni, di, Tr, 0.0)
		SetSynCaV(ctx, syni, di, DTr, 0.0)
	} else {
		pfmod := pj.Matrix.BasePF + NrnV(ctx, ri, di, GModSyn)
		rplus := NrnV(ctx, ri, di, CaSpkP)
		rminus := NrnV(ctx, ri, di, CaSpkD)
		sact := NrnV(ctx, si, di, CaSpkD)
		dtr := rlr * (pj.Matrix.Delta * sact * (rplus - rminus))
		if rminus > pj.Learn.Trace.LearnThr { // key: prevents learning if < threshold
			dtr += rlr * (pj.Matrix.Credit * pfmod * sact * rminus)
		}
		SetSynCaV(ctx, syni, di, DTr, dtr)
		AddSynCaV(ctx, syni, di, Tr, dtr)
	}
}

// DWtSynVSPatch computes the weight change (learning) at given synapse,
// for the VSPatchPath type.
func (pj *PathParams) DWtSynVSPatch(ctx *Context, syni, si, ri, di uint32, layPool, subPool *Pool) {
	ract := NrnV(ctx, ri, di, SpkPrv) // t-1
	if ract < pj.Learn.Trace.LearnThr {
		ract = 0
	}
	// note: rn.RLRate already has ACh * DA * (D1 vs. D2 sign reversal) factored in.
	// and also the logic that non-positive DA leads to weight decreases.
	sact := NrnV(ctx, si, di, SpkPrv) // t-1
	dwt := NrnV(ctx, ri, di, RLRate) * pj.Learn.LRate.Eff * sact * ract
	SetSynCaV(ctx, syni, di, DiDWt, dwt)
}

///////////////////////////////////////////////////
// WtFromDWt

// DWtFromDiDWtSyn updates DWt from data parallel DiDWt values
func (pj *PathParams) DWtFromDiDWtSyn(ctx *Context, syni uint32) {
	dwt := float32(0)
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		dwt += SynCaV(ctx, syni, di, DiDWt)
	}
	AddSynV(ctx, syni, DWt, dwt)
}

// WtFromDWtSyn is the overall entry point for updating weights from weight changes.
func (pj *PathParams) WtFromDWtSyn(ctx *Context, syni uint32) {
	switch pj.PathType {
	case RWPath:
		pj.WtFromDWtSynNoLimits(ctx, syni)
	case TDPredPath:
		pj.WtFromDWtSynNoLimits(ctx, syni)
	case BLAPath:
		pj.WtFromDWtSynNoLimits(ctx, syni)
	case HipPath:
		pj.WtFromDWtSynNoLimits(ctx, syni)
	default:
		pj.WtFromDWtSynCortex(ctx, syni)
	}
}

// WtFromDWtSynCortex updates weights from dwt changes
func (pj *PathParams) WtFromDWtSynCortex(ctx *Context, syni uint32) {
	dwt := SynV(ctx, syni, DWt)
	AddSynV(ctx, syni, DSWt, dwt)
	wt := SynV(ctx, syni, Wt)
	lwt := SynV(ctx, syni, LWt)

	pj.SWts.WtFromDWt(&wt, &lwt, dwt, SynV(ctx, syni, SWt))
	SetSynV(ctx, syni, DWt, 0)
	SetSynV(ctx, syni, Wt, wt)
	SetSynV(ctx, syni, LWt, lwt)
	// pj.Com.Fail(&sy.Wt, sy.SWt) // skipping for now -- not useful actually
}

// WtFromDWtSynNoLimits -- weight update without limits
func (pj *PathParams) WtFromDWtSynNoLimits(ctx *Context, syni uint32) {
	dwt := SynV(ctx, syni, DWt)
	if dwt == 0 {
		return
	}
	AddSynV(ctx, syni, Wt, dwt)
	if SynV(ctx, syni, Wt) < 0 {
		SetSynV(ctx, syni, Wt, 0)
	}
	SetSynV(ctx, syni, LWt, SynV(ctx, syni, Wt))
	SetSynV(ctx, syni, DWt, 0)
}

//gosl:end pathparams
