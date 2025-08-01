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

const (
	// StartOff is the starting offset.
	StartOff int32 = iota

	// Number of items.
	Nitems

	// Number of StartN elements.
	StartNN
)

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
	// RecvLayer is the index of the receiving layer in global list of layers.
	RecvLayer uint32

	// RecvNeurSt is the starting index of neurons in recv layer,
	// so we don't need layer to get to neurons.
	RecvNeurSt uint32

	// RecvNeurN is the number of neurons in recv layer.
	RecvNeurN uint32

	// SendLayer is the index of the sending layer in global list of layers.
	SendLayer uint32

	// SendNeurSt is the starting index of neurons in sending layer,
	// so we don't need layer to get to neurons.
	SendNeurSt uint32

	// SendNeurN is the number of neurons in send layer
	SendNeurN uint32

	// SynapseSt is the start index into global Synapse array.
	// [Layer][SendPaths][Synapses].
	SynapseSt uint32

	// SendConSt is the start index into global PathSendCon array.
	// [Layer][SendPaths][SendNeurons]
	SendConSt uint32

	// RecvConSt is the start index into global PathRecvCon array.
	// [Layer][RecvPaths][RecvNeurons]
	RecvConSt uint32

	// RecvSynSt is the start index into global sender-based Synapse index array.
	// [Layer][SendPaths][Synapses]
	RecvSynSt uint32

	// NPathNeurSt is the start NPathNeur index into PathGBuf, PathGSyns global arrays.
	// [Layer][RecvPaths][RecvNeurons]
	NPathNeurSt uint32

	pad uint32
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
// On the GPU, they are loaded into a read-only storage buffer.
type PathParams struct {

	// Type is the functional type of path, which determines the code path
	// for specialized types, and is synchronized with [Path.Type].
	Type PathTypes

	// Index is the index of the pathway in global path list: [Layer][SendPaths]
	Index uint32 `edit:"-"`

	pad, pad1 int32

	// recv and send neuron-level pathway index array access info
	Indexes PathIndexes `display:"-"`

	// synaptic communication parameters: delay, probability of failure
	Com SynComParams `display:"inline"`

	// pathway scaling parameters for computing GScale:
	// modulates overall strength of pathway, using both
	// absolute and relative factors, with adaptation option to maintain target max conductances
	PathScale PathScaleParams `display:"inline"`

	// slowly adapting, structural weight value parameters,
	// which control initial weight values and slower outer-loop adjustments
	SWts SWtParams `display:"add-fields"`

	// synaptic-level learning parameters for learning in the fast LWt values.
	Learn LearnSynParams `display:"add-fields"`

	// conductance scaling values
	GScale GScaleValues `display:"inline"`

	// Params for RWPath and TDPredPath for doing dopamine-modulated learning
	// for reward prediction: Da * Send activity.
	// Use in RWPredLayer or TDPredLayer typically to generate reward predictions.
	// If the Da sign is positive, the first recv unit learns fully; for negative,
	// second one learns fully.
	// Lower lrate applies for opposite cases.  Weights are positive-only.
	RLPred RLPredPathParams `display:"inline"`

	// VSMatrix has parameters for trace-based learning in the MatrixPath.
	// A trace of synaptic co-activity is formed, and then modulated by
	// dopamine whenever it occurs.
	// This bridges the temporal gap between gating activity and subsequent activity,
	// and is based biologically on synaptic tags.
	// DSPatch provides modulation of trace activity based on local critic signal.
	VSMatrix VSMatrixPathParams `display:"inline"`

	// DSMatrix has parameters for trace-based learning in the MatrixPath.
	// A trace of synaptic co-activity is formed, and then modulated by
	// dopamine whenever it occurs.
	// This bridges the temporal gap between gating activity and subsequent activity,
	// and is based biologically on synaptic tags.
	// DSPatch provides modulation of trace activity based on local critic signal.
	DSMatrix DSMatrixPathParams `display:"inline"`

	// Basolateral Amygdala pathway parameters.
	BLA BLAPathParams `display:"inline"`

	// Hip bench parameters.
	Hip HipPathParams `display:"inline"`
}

func (pt *PathParams) IsInhib() bool {
	return pt.Com.GType == InhibitoryG
}

func (pt *PathParams) IsExcitatory() bool {
	return pt.Com.GType == ExcitatoryG
}

// SetFixedWts sets parameters for fixed, non-learning weights
// with a default of Mean = 0.8, Var = 0 strength
func (pt *PathParams) SetFixedWts() {
	pt.SWts.Init.SPct = 0
	pt.Learn.Learn.SetBool(false)
	pt.SWts.Adapt.On.SetBool(false)
	pt.SWts.Adapt.SigGain = 1
	pt.SWts.Init.Mean = 0.8
	pt.SWts.Init.Var = 0.0
	pt.SWts.Init.Sym.SetBool(false)
}

//gosl:end

// StyleClass implements the [params.Styler] interface for parameter setting,
// and must only be called after the network has been built, and is current,
// because it uses the global CurrentNetwork variable.
func (pt *PathParams) StyleClass() string {
	pth := CurrentNetwork.Paths[pt.Index]
	return pt.Type.String() + " " + pth.Class
}

// StyleName implements the [params.Styler] interface for parameter setting,
// and must only be called after the network has been built, and is current,
// because it uses the global CurrentNetwork variable.
func (pt *PathParams) StyleName() string {
	pth := CurrentNetwork.Paths[pt.Index]
	return pth.Name
}

func (pt *PathParams) Defaults() {
	pt.Com.Defaults()
	pt.SWts.Defaults()
	pt.PathScale.Defaults()
	pt.Learn.Defaults()
	pt.RLPred.Defaults()
	pt.VSMatrix.Defaults()
	pt.DSMatrix.Defaults()
	pt.BLA.Defaults()
	pt.Hip.Defaults()
}

func (pt *PathParams) Update() {
	pt.Com.Update()
	pt.PathScale.Update()
	pt.SWts.Update()
	pt.Learn.Update()
	pt.RLPred.Update()
	pt.VSMatrix.Update()
	pt.DSMatrix.Update()
	pt.BLA.Update()
	pt.Hip.Update()

	if pt.Type == CTCtxtPath {
		pt.Com.GType = ContextG
	}
}

func (pt *PathParams) ShouldDisplay(field string) bool {
	switch field {
	case "RLPred":
		return pt.Type == RWPath || pt.Type == TDPredPath
	case "VSMatrix":
		return pt.Type == VSMatrixPath
	case "DSMatrix":
		return pt.Type == DSMatrixPath
	case "BLA":
		return pt.Type == BLAPath
	case "Hip":
		return pt.Type == HipPath
	default:
		return true
	}
}

// ParamsString returns a listing of all parameters in the Pathway.
// If nonDefault is true, only report those not at their default values.
func (pt *PathParams) ParamsString(nonDefault bool) string {
	ptyp := pt.Type
	return params.PrintStruct(pt, 2, func(path string, ft reflect.StructField, fv any) bool {
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
		case "RLPred":
			return ptyp == RWPath || ptyp == TDPredPath
		case "Matrix":
			return ptyp == VSMatrixPath || ptyp == DSMatrixPath
		case "BLA":
			return ptyp == BLAPath
		case "Hip":
			return ptyp == HipPath
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
