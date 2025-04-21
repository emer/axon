// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"reflect"

	"cogentcore.org/core/base/reflectx"
	"github.com/emer/emergent/v2/params"
)

//gosl:start

// LayerIndexes contains index access into network global arrays for GPU.
type LayerIndexes struct {
	// NPools is the total number of pools for this layer, including layer-wide.
	NPools uint32 `edit:"-"`

	// start of neurons for this layer in global array (same as Layer.NeurStIndex)
	NeurSt uint32 `edit:"-"`

	// number of neurons in layer
	NNeurons uint32 `edit:"-"`

	// start index into RecvPaths global array
	RecvSt uint32 `edit:"-"`

	// number of recv pathways
	RecvN uint32 `edit:"-"`

	// start index into RecvPaths global array
	SendSt uint32 `edit:"-"`

	// number of recv pathways
	SendN uint32 `edit:"-"`

	// starting neuron index in global Exts list of external input for this layer.
	// Only for Input / Target / Compare layer types
	ExtsSt uint32 `edit:"-"`

	// layer shape Pools Y dimension -- 1 for 2D
	ShpPlY int32 `edit:"-"`

	// layer shape Pools X dimension -- 1 for 2D
	ShpPlX int32 `edit:"-"`

	// layer shape Units Y dimension
	ShpUnY int32 `edit:"-"`

	// layer shape Units X dimension
	ShpUnX int32 `edit:"-"`
}

// LayerInhibIndexes contains indexes of layers for between-layer inhibition.
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

// LayerParams contains all of the layer parameters.
// These values must remain constant over the course of computation.
// On the GPU, they are loaded into a uniform.
type LayerParams struct {

	// Type is the functional type of layer, which determines the code path
	// for specialized layer types, and is synchronized with [Layer.Type].
	Type LayerTypes

	// Index of this layer in [Layers] list.
	Index uint32 `edit:"-"`

	// MaxData is the maximum number of data parallel elements.
	MaxData uint32 `display:"-"`

	// PoolSt is the start of pools for this layer; first one is always the layer-wide pool.
	PoolSt uint32 `display:"-"`

	// Activation parameters and methods for computing activations
	Acts ActParams `display:"add-fields"`

	// Inhibition parameters and methods for computing layer-level inhibition
	Inhib InhibParams `display:"add-fields"`

	// LayInhib has indexes of layers that contribute between-layer inhibition
	//  to this layer. Set these indexes via BuildConfig LayInhibXName (X = 1, 2...).
	LayInhib LayerInhibIndexes `display:"inline"`

	// Learn has learning parameters and methods that operate at the neuron level.
	Learn LearnNeuronParams `display:"add-fields"`

	// Bursts has [BurstParams] that determine how the 5IB Burst activation
	// is computed from CaP integrated spiking values in Super layers.
	Bursts BurstParams `display:"inline"`

	// CT has params for the CT corticothalamic layer and PTPred layer that
	// generates predictions over the Pulvinar using context. Uses the CtxtGe
	// excitatory input plus stronger NMDA channels to maintain context trace.
	CT CTParams `display:"inline"`

	// Pulv has parameters for how the plus-phase (outcome) state of Pulvinar
	// thalamic relay cell neurons is computed from the corresponding driver
	// neuron Burst activation (or CaP if not Super).
	Pulv PulvParams `display:"inline"`

	// Matrix has parameters for BG Striatum Matrix MSN layers, which are
	// the main Go / NoGo gating units in BG. GateThr also used in BGThal.
	Matrix MatrixParams `display:"inline"`

	// GP has params for GP (globus pallidus) of the BG layers.
	GP GPParams `display:"inline"`

	// LDT has parameters for laterodorsal tegmentum ACh salience neuromodulatory
	// signal, driven by superior colliculus stimulus novelty, US input / absence,
	// and OFC / ACC inhibition.
	LDT LDTParams `display:"inline"`

	// VTA has parameters for ventral tegmental area dopamine (DA) based on
	// LHb PVDA (primary value -- at US time, computed at start of each trial
	// and stored in LHbPVDA global value) and Amygdala (CeM) CS / learned
	// value (LV) activations, which update every cycle.
	VTA VTAParams `display:"inline"`

	// RWPred has parameters for reward prediction using a simple Rescorla-Wagner
	// learning rule (i.e., PV learning in the Rubicon framework).
	RWPred RWPredParams `display:"inline"`

	// RWDa has parameters for reward prediction dopamine using a simple
	// Rescorla-Wagner learning rule (i.e., PV learning in the Rubicon framework).
	RWDa RWDaParams `display:"inline"`

	// TDInteg has parameters for temporal differences (TD) reward integration layer.
	TDInteg TDIntegParams `display:"inline"`

	// TDDa has parameters for dopamine (DA) signal as the temporal difference
	// (TD) between the TDIntegLayer activations in the minus and plus phase.
	TDDa TDDaParams `display:"inline"`

	// Indexes has recv and send pathway array access info.
	Indexes LayerIndexes `display:"-"`
}

// PoolIndex returns the global network index for pool with given
// pool (0 = layer pool, 1+ = subpools): just PoolSt + pi
func (ly *LayerParams) PoolIndex(pi uint32) uint32 {
	return ly.PoolSt + pi
}

// HasPoolInhib returns true if the layer is using pool-level inhibition (implies 4D too).
// This is the proper check for using pool-level target average activations, for example.
func (ly *LayerParams) HasPoolInhib() bool {
	return ly.Inhib.Pool.On.IsTrue()
}

//gosl:end

// StyleClass implements the [params.Styler] interface for parameter setting,
// and must only be called after the network has been built, and is current,
// because it uses the global CurrentNetwork variable.
func (ly *LayerParams) StyleClass() string {
	lay := CurrentNetwork.Layers[ly.Index]
	return ly.Type.String() + " " + lay.Class
}

// StyleName implements the [params.Styler] interface for parameter setting,
// and must only be called after the network has been built, and is current,
// because it uses the global CurrentNetwork variable.
func (ly *LayerParams) StyleName() string {
	lay := CurrentNetwork.Layers[ly.Index]
	return lay.Name
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

	ly.LDT.Defaults()
	ly.VTA.Defaults()

	ly.RWPred.Defaults()
	ly.RWDa.Defaults()
	ly.TDInteg.Defaults()
	ly.TDDa.Defaults()
}

func (ly *LayerParams) ShouldDisplay(field string) bool {
	switch field {
	case "Bursts":
		return ly.Type == SuperLayer
	case "CT":
		return ly.Type == CTLayer || ly.Type == PTPredLayer || ly.Type == BLALayer
	case "Pulv":
		return ly.Type == PulvinarLayer
	case "Matrix":
		return ly.Type == MatrixLayer || ly.Type == BGThalLayer
	case "GP":
		return ly.Type == GPLayer
	case "LDT":
		return ly.Type == LDTLayer
	case "VTA":
		return ly.Type == VTALayer
	case "RWPred":
		return ly.Type == RWPredLayer
	case "RWDa":
		return ly.Type == RWDaLayer
	case "TDInteg":
		return ly.Type == TDIntegLayer
	case "TDDa":
		return ly.Type == TDDaLayer
	default:
		return true
	}
}

// ParamsString returns a listing of all parameters in the Layer and
// pathways within the layer. If nonDefault is true, only report those
// not at their default values.
func (ly *LayerParams) ParamsString(nonDefault bool) string {
	ltyp := ly.Type
	return params.PrintStruct(ly, 1, func(path string, ft reflect.StructField, fv any) bool {
		if ft.Tag.Get("display") == "-" {
			return false
		}
		if nonDefault {
			if def := ft.Tag.Get("default"); def != "" {
				if reflectx.ValueIsDefault(reflect.ValueOf(fv), def) {
					return false
				}
			} else {
				if reflectx.NonPointerType(ft.Type).Kind() != reflect.Struct {
					return false
				}
			}
		}
		switch path {
		case "Bursts":
			return ltyp == SuperLayer
		case "CT":
			return ltyp == CTLayer || ltyp == PTPredLayer || ltyp == BLALayer
		case "Pulv":
			return ltyp == PulvinarLayer
		case "Matrix":
			return ltyp == MatrixLayer || ltyp == BGThalLayer
		case "GP":
			return ltyp == GPLayer
		case "LDT":
			return ltyp == LDTLayer
		case "VTA":
			return ltyp == VTALayer
		case "RWPred":
			return ltyp == RWPredLayer
		case "RWDa":
			return ltyp == RWDaLayer
		case "TDInteg":
			return ltyp == TDIntegLayer
		case "TDDa":
			return ltyp == TDDaLayer
		}
		return true
	},
		func(path string, ft reflect.StructField, fv any) string {
			if nonDefault {
				if def := ft.Tag.Get("default"); def != "" {
					return reflectx.ToString(fv) + " [" + def + "]"
				}
			}
			return ""
		})
}
