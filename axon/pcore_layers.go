// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"
	"strings"

	"github.com/goki/ki/kit"
)

//gosl: start pcore_layers

// MatrixParams has parameters for BG Striatum Matrix MSN layers
// These are the main Go / NoGo gating units in BG.
// DA, ACh learning rate modulation is pre-computed on the recv neuron
// RLRate variable via NeuroMod.  Also uses Pool.Gated for InvertNoGate,
// updated in PlusPhase prior to DWt call.
// Must set Learn.NeuroMod.DAMod = D1Mod or D2Mod via SetBuildConfig("DAMod").
type MatrixParams struct {
	// GPHasPools     bool        `desc:"do the GP pathways that we drive have separate pools that compete for selecting one out of multiple options in parallel (true) or is it a single big competition for Go vs. No (false).  If true, then Matrix must also have pools "`
	GateThr        float32 `desc:"threshold on layer Avg SpkMax for Matrix Go and VThal layers to count as having gated"`
	NoGoGeLrn      float32 `desc:"multiplier on Ge in NoGo (D2) neurons to provide a baseline level of learning, so that if a negative DA outcome occurs, there is some activity in NoGo for learning to work on.  Strong values increase amount of NoGo learning.  Shows up in SpkMax value which is what drives learning."`
	OtherMatrixIdx int32   `inactive:"+" desc:"index of other matrix (Go if we are NoGo and vice-versa).    Set during Build from BuildConfig OtherMatrixName"`
	ThalLay1Idx    int32   `inactive:"+" desc:"index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay1Name if present -- -1 if not used"`
	ThalLay2Idx    int32   `inactive:"+" desc:"index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay2Name if present -- -1 if not used"`
	ThalLay3Idx    int32   `inactive:"+" desc:"index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay3Name if present -- -1 if not used"`
	ThalLay4Idx    int32   `inactive:"+" desc:"index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay4Name if present -- -1 if not used"`
	ThalLay5Idx    int32   `inactive:"+" desc:"index of thalamus layer that we gate.  needed to get gating information.  Set during Build from BuildConfig ThalLay5Name if present -- -1 if not used"`
}

func (mp *MatrixParams) Defaults() {
	mp.GateThr = 0.01
	mp.NoGoGeLrn = 1
}

func (mp *MatrixParams) Update() {
}

// GPLayerTypes is a GPLayer axon-specific layer type enum.
type GPLayerTypes int32

// The GPLayer types
const (
	// GPeOut is Outer layer of GPe neurons, receiving inhibition from MtxGo
	GPeOut GPLayerTypes = iota

	// GPeIn is Inner layer of GPe neurons, receiving inhibition from GPeOut and MtxNo
	GPeIn

	// GPeTA is arkypallidal layer of GPe neurons, receiving inhibition from GPeIn
	// and projecting inhibition to Mtx
	GPeTA

	// GPi is the inner globus pallidus, functionally equivalent to SNr,
	// receiving from MtxGo and GPeIn, and sending inhibition to VThal
	GPi

	GPLayerTypesN
)

// GPLayer represents a globus pallidus layer, including:
// GPeOut, GPeIn, GPeTA (arkypallidal), and GPi (see GPType for type).
// Typically just a single unit per Pool representing a given stripe.
type GPParams struct {
	GPType GPLayerTypes `viewif:"LayType=GPLayer" view:"inline" desc:"type of GP Layer -- must set during config using SetBuildConfig of GPType."`

	pad, pad1, pad2 uint32
}

func (gp *GPParams) Defaults() {
}

func (gp *GPParams) Update() {
}

//gosl: end pcore_layers

// todo: original has this:

// func (ly *MatrixLayer) DecayState(decay, glong float32) {
// 	ly.Layer.DecayState(decay, glong)
// 	for ni := range ly.Neurons {
// 		nrn := &ly.Neurons[ni]
// 		if nrn.IsOff() {
// 			continue
// 		}
// 		ly.Params.Learn.DecayCaLrnSpk(nrn, glong) // ?
// 	}
// 	ly.InitMods()
// }

func (ly *Layer) MatrixDefaults() {
	ly.Params.Act.Decay.Act = 0
	ly.Params.Act.Decay.Glong = 0
	ly.Params.Inhib.Pool.On.SetBool(false)
	ly.Params.Inhib.Layer.On.SetBool(true)
	ly.Params.Inhib.Layer.Gi = 0.5
	ly.Params.Inhib.Layer.FB = 0
	ly.Params.Inhib.Pool.FB = 0
	ly.Params.Inhib.Pool.Gi = 0.5
	ly.Params.Inhib.ActAvg.Nominal = 0.25

	// ly.Params.Learn.NeuroMod.DAMod needs to be set via BuildConfig
	ly.Params.Learn.NeuroMod.DALrateMod.SetBool(true)
	ly.Params.Learn.NeuroMod.AChLrateMod.SetBool(true)
	ly.Params.Learn.NeuroMod.DALratePct = 1
	ly.Params.Learn.NeuroMod.AChLratePct = 1
	ly.Params.Learn.NeuroMod.AChDisInhib = 1

	// important: user needs to adjust wt scale of some PFC inputs vs others:
	// drivers vs. modulators

	for _, pji := range ly.RcvPrjns {
		pj := pji.(AxonPrjn).AsAxon()
		pj.Params.SWt.Init.SPct = 0
		if pj.Send.(AxonLayer).LayerType() == GPLayer { // From GPe TA or In
			pj.Params.PrjnScale.Abs = 1
			pj.Params.Learn.Learn.SetBool(false)
			pj.Params.SWt.Adapt.SigGain = 1
			pj.Params.SWt.Init.Mean = 0.75
			pj.Params.SWt.Init.Var = 0.0
			pj.Params.SWt.Init.Sym.SetBool(false)
			if strings.HasSuffix(pj.Send.Name(), "GPeIn") { // GPeInToMtx
				pj.Params.PrjnScale.Abs = 0.5 // counterbalance for GPeTA to reduce oscillations
			} else if strings.HasSuffix(pj.Send.Name(), "GPeTA") { // GPeTAToMtx
				if strings.HasSuffix(ly.Nm, "MtxGo") {
					pj.Params.PrjnScale.Abs = 2 // was .8
				} else {
					pj.Params.PrjnScale.Abs = 1 // was .3 GPeTAToMtxNo must be weaker to prevent oscillations, even with GPeIn offset
				}
			}
		}
	}
}

// MatrixGated is called after std PlusPhase, on CPU, has Pool info
// downloaded from GPU
func (ly *Layer) MatrixGated() bool {
	if ly.Params.Learn.NeuroMod.DAMod != D1Mod { // copy from go
		oly := ly.Network.Layer(int(ly.Params.Matrix.OtherMatrixIdx)).(AxonLayer).AsAxon()
		for pi := range ly.Pools {
			pl := &ly.Pools[pi]
			opl := &oly.Pools[pi]
			pl.Gated = opl.Gated
		}
		return ly.Pools[0].Gated.IsTrue()
	}
	mtxGated := ly.GatedFmSpkMax(ly.Params.Matrix.GateThr)

	thalGated := false
	if ly.Params.Matrix.ThalLay1Idx >= 0 {
		tly := ly.Network.Layer(int(ly.Params.Matrix.ThalLay1Idx)).(AxonLayer).AsAxon()
		thalGated = tly.GatedFmSpkMax(ly.Params.Matrix.GateThr) || thalGated
	}
	if ly.Params.Matrix.ThalLay2Idx >= 0 {
		tly := ly.Network.Layer(int(ly.Params.Matrix.ThalLay2Idx)).(AxonLayer).AsAxon()
		thalGated = tly.GatedFmSpkMax(ly.Params.Matrix.GateThr) || thalGated
	}
	if ly.Params.Matrix.ThalLay3Idx >= 0 {
		tly := ly.Network.Layer(int(ly.Params.Matrix.ThalLay3Idx)).(AxonLayer).AsAxon()
		thalGated = tly.GatedFmSpkMax(ly.Params.Matrix.GateThr) || thalGated
	}
	if ly.Params.Matrix.ThalLay4Idx >= 0 {
		tly := ly.Network.Layer(int(ly.Params.Matrix.ThalLay4Idx)).(AxonLayer).AsAxon()
		thalGated = tly.GatedFmSpkMax(ly.Params.Matrix.GateThr) || thalGated
	}
	if ly.Params.Matrix.ThalLay5Idx >= 0 {
		tly := ly.Network.Layer(int(ly.Params.Matrix.ThalLay5Idx)).(AxonLayer).AsAxon()
		thalGated = tly.GatedFmSpkMax(ly.Params.Matrix.GateThr) || thalGated
	}

	mtxGated = mtxGated && thalGated

	// note: in principle with multi-pool GP, could try to establish
	// a correspondence between thal and matrix pools, such that
	// a failure to gate at the thal level for a given pool would veto
	// just the one corresponding pool.  However, we're not really sure
	// that this will make sense and not doing yet..

	if !mtxGated { // nobody did if thal didn't
		for pi := range ly.Pools {
			pl := &ly.Pools[pi]
			pl.Gated.SetBool(false)
		}
	}
	return mtxGated
}

// GatedFmSpkMax updates the Gated state in Pools of given layer,
// based on Avg SpkMax being above given threshold.
// returns true if any gated.
func (ly *Layer) GatedFmSpkMax(thr float32) bool {
	anyGated := false
	if ly.Is4D() {
		for pi := 1; pi < len(ly.Pools); pi++ {
			pl := &ly.Pools[pi]
			spkavg := pl.AvgMax.SpkMax.Cycle.Avg
			gthr := spkavg > thr
			if gthr {
				anyGated = true
			}
			pl.Gated.SetBool(gthr)
		}
	} else {
		spkavg := ly.Pools[0].AvgMax.SpkMax.Cycle.Avg
		if spkavg > thr {
			anyGated = true
		}
	}
	ly.Pools[0].Gated.SetBool(anyGated)
	return anyGated
}

// AnyGated returns true if the layer-level pool Gated flag is true,
// which indicates if any of the layers gated.
func (ly *Layer) AnyGated() bool {
	return ly.Pools[0].Gated.IsTrue()
}

func (ly *Layer) MatrixPostBuild() {
	ly.Params.Matrix.ThalLay1Idx = ly.BuildConfigFindLayer("ThalLay1Name", false) // optional
	ly.Params.Matrix.ThalLay2Idx = ly.BuildConfigFindLayer("ThalLay2Name", false) // optional
	ly.Params.Matrix.ThalLay3Idx = ly.BuildConfigFindLayer("ThalLay3Name", false) // optional
	ly.Params.Matrix.ThalLay4Idx = ly.BuildConfigFindLayer("ThalLay4Name", false) // optional
	ly.Params.Matrix.ThalLay5Idx = ly.BuildConfigFindLayer("ThalLay5Name", false) // optional

	ly.Params.Matrix.OtherMatrixIdx = ly.BuildConfigFindLayer("OtherMatrixName", true)

	dm, err := ly.BuildConfigByName("DAMod")
	if err == nil {
		err = ly.Params.Learn.NeuroMod.DAMod.FromString(dm)
		if err != nil {
			log.Println(err)
		}
	}
}

////////////////////////////////////////////////////////////////////
//  GP

func (ly *Layer) GPDefaults() {
	// GP is tonically self-active and has no FFFB inhibition
	ly.Params.Act.Init.Ge = 0.3
	ly.Params.Act.Init.GeVar = 0.1
	ly.Params.Act.Init.GiVar = 0.1
	ly.Params.Act.Decay.Act = 0
	ly.Params.Act.Decay.Glong = 0
	ly.Params.Inhib.ActAvg.Nominal = 1 // very active!
	ly.Params.Inhib.Layer.On.SetBool(false)
	ly.Params.Inhib.Pool.On.SetBool(false)

	for _, pjii := range ly.RcvPrjns {
		pji := pjii.(AxonPrjn)
		pj := pji.AsAxon()
		pj.Params.Learn.Learn.SetBool(false)
		pj.Params.SWt.Adapt.SigGain = 1
		pj.Params.SWt.Init.SPct = 0
		pj.Params.SWt.Init.Mean = 0.75
		pj.Params.SWt.Init.Var = 0.25
		pj.Params.SWt.Init.Sym.SetBool(false)
		if pj.Send.(AxonLayer).LayerType() == MatrixLayer {
			pj.Params.PrjnScale.Abs = 0.5
		} else if pj.Send.(AxonLayer).LayerType() == STNLayer {
			pj.Params.PrjnScale.Abs = 1 // STNpToGPTA -- default level for GPeOut and GPeTA -- weaker to not oppose GPeIn surge
		}
		switch ly.Params.GP.GPType {
		case GPeIn:
			if pj.Send.(AxonLayer).LayerType() == MatrixLayer { // MtxNoToGPeIn -- primary NoGo pathway
				pj.Params.PrjnScale.Abs = 1
			} else if pj.Send.(AxonLayer).LayerType() == GPLayer { // GPeOutToGPeIn
				pj.Params.PrjnScale.Abs = 0.3 // was 0.5
			}
			if pj.Send.(AxonLayer).LayerType() == STNLayer { // STNpToGPeIn -- stronger to drive burst of activity
				pj.Params.PrjnScale.Abs = 1 // was 0.5
			}
		case GPeOut:
			if pj.Send.(AxonLayer).LayerType() == STNLayer { // STNpToGPeOut
				pj.Params.PrjnScale.Abs = 0.1
			}
		case GPeTA:
			if pj.Send.(AxonLayer).LayerType() == GPLayer { // GPeInToGPeTA
				pj.Params.PrjnScale.Abs = 0.7 // was 0.9 -- just enough to knock down to near-zero at baseline
			}
		}
	}

	if ly.Params.GP.GPType == GPi {
		ly.GPiDefaults()
	}
}

func (ly *Layer) GPiDefaults() {
	ly.Params.Act.Init.Ge = 0.6
	// note: GPLayer took care of STN input prjns

	for _, pji := range ly.RcvPrjns {
		pj := pji.(AxonPrjn).AsAxon()
		pj.Params.SWt.Adapt.SigGain = 1
		pj.Params.SWt.Init.SPct = 0
		pj.Params.SWt.Init.Mean = 0.75
		pj.Params.SWt.Init.Var = 0.25
		pj.Params.SWt.Init.Sym.SetBool(false)
		pj.Params.Learn.Learn.SetBool(false)
		if pj.Send.(AxonLayer).LayerType() == MatrixLayer { // MtxGoToGPi
			pj.Params.PrjnScale.Abs = 0.8 // slightly weaker than GPeIn
		} else if pj.Send.(AxonLayer).LayerType() == GPLayer { // GPeInToGPi
			pj.Params.PrjnScale.Abs = 1 // stronger because integrated signal, also act can be weaker
		} else if strings.HasSuffix(pj.Send.Name(), "STNp") { // STNpToGPi
			pj.Params.PrjnScale.Abs = 1
		} else if strings.HasSuffix(pj.Send.Name(), "STNs") { // STNsToGPi
			pj.Params.PrjnScale.Abs = 0.3
		}
	}
}

//go:generate stringer -type=GPLayerTypes

var KiT_GPLayerTypes = kit.Enums.AddEnum(GPLayerTypesN, kit.NotBitFlag, nil)

func (ev GPLayerTypes) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *GPLayerTypes) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

func (ly *Layer) GPPostBuild() {
	gpnm, err := ly.BuildConfigByName("GPType")
	if err == nil {
		err = ly.Params.GP.GPType.FromString(gpnm)
		if err != nil {
			log.Println(err)
		}
	}
}

////////////////////////////////////////////////////////////////////
//  STN

func (ly *Layer) STNDefaults() {
	// STN is tonically self-active and has no FFFB inhibition
	ly.Params.Act.SKCa.Gbar = 2
	ly.Params.Act.Decay.Act = 0
	ly.Params.Act.Decay.Glong = 0
	ly.Params.Act.Decay.Act = 0
	ly.Params.Act.Decay.Glong = 0
	ly.Params.Act.Dend.SSGi = 0
	ly.Params.Inhib.Layer.On.SetBool(true) // true = important for real-world cases
	ly.Params.Inhib.Layer.Gi = 0.2
	ly.Params.Inhib.Pool.On.SetBool(false)
	ly.Params.Inhib.ActAvg.Nominal = 0.15

	if strings.HasSuffix(ly.Nm, "STNp") {
		ly.Params.Act.SKCa.CaD.SetBool(false)
		ly.Params.Act.SKCa.CaScale = 4
	} else {
		ly.Params.Act.SKCa.CaD.SetBool(true)
		ly.Params.Act.SKCa.CaScale = 3
		ly.Params.Act.Init.Ge = 0.2
		ly.Params.Act.Init.GeVar = 0.2
	}

	for _, pji := range ly.RcvPrjns {
		pj := pji.(AxonPrjn).AsAxon()
		pj.Params.Learn.Learn.SetBool(false)
		pj.Params.SWt.Adapt.SigGain = 1
		pj.Params.SWt.Init.SPct = 0
		pj.Params.SWt.Init.Mean = 0.75
		pj.Params.SWt.Init.Var = 0.25
		pj.Params.SWt.Init.Sym.SetBool(false)
		if strings.HasSuffix(ly.Nm, "STNp") {
			if pj.Send.(AxonLayer).LayerType() == GPLayer { // GPeInToSTNp
				pj.Params.PrjnScale.Abs = 0.1
			}
		} else { // STNs
			if pj.Send.(AxonLayer).LayerType() == GPLayer { // GPeInToSTNs
				pj.Params.PrjnScale.Abs = 0.1 // note: not currently used -- interferes with threshold-based Ca self-inhib dynamics
			} else {
				pj.Params.PrjnScale.Abs = 0.2 // weaker inputs
			}
		}
	}
}

////////////////////////////////////////////////////////////////////
//  VThal

func (ly *Layer) VThalDefaults() {
	// note: not tonically active
	ly.Params.Act.Dend.SSGi = 0
	ly.Params.Inhib.Layer.On.SetBool(false)
	ly.Params.Inhib.Pool.On.SetBool(false)
	ly.Params.Inhib.ActAvg.Nominal = 0.25

	for _, pji := range ly.RcvPrjns {
		pj := pji.(AxonPrjn).AsAxon()
		pj.Params.Learn.Learn.SetBool(false)
		pj.Params.SWt.Adapt.SigGain = 1
		pj.Params.SWt.Init.SPct = 0
		pj.Params.SWt.Init.Mean = 0.75
		pj.Params.SWt.Init.Var = 0.0
		pj.Params.SWt.Init.Sym.SetBool(false)
		if strings.HasSuffix(pj.Send.Name(), "GPi") { // GPiToVThal
			pj.Params.PrjnScale.Abs = 2 // was 2.5 for agate model..
		}
	}
}
